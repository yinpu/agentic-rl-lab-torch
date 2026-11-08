"""GSPO：固定 rollout 旧概率，按题组分四个 mini-batch 依次更新。"""
import argparse
import os
import random
from pathlib import Path
import sys
import torch
from lightning.fabric import Fabric
from lightning.fabric.strategies import DDPStrategy
from lightning.fabric.loggers import TensorBoardLogger
from peft import LoraConfig, get_peft_model

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "experiments.gspo"
from .task import (DEFAULT_MODEL, load_model, load_rows, sample_group,
                   batch_completion_logprobs, sequence_objective)


def training_size(count, batch_size, mini_batches, world_size):
    if batch_size % (mini_batches * world_size):
        raise ValueError("rollout batch must be divisible by num-mini-batches * world-size")
    size = count // batch_size * batch_size
    if not size:
        raise ValueError("not enough training data for one global rollout batch")
    return size


def cache_old(model, samples, micro):
    model.eval()
    with torch.no_grad():
        for start in range(0, len(samples), micro):
            chunk = samples[start:start + micro]
            for sample, logp in zip(chunk, batch_completion_logprobs(model, chunk), strict=True):
                sample["old"] = logp.detach().cpu()


def update_policy(model, optimizer, samples, args, fabric):
    """每回答等权；DDP 梯度平均通过 world_size/global_count 抵消。"""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    counts = fabric.all_reduce(torch.tensor([len(samples), sum(s["advantage"] != 0 for s in samples)],
                                           device=fabric.device), reduce_op="sum")
    if counts[1] == 0:
        return {"skipped": 1, "loss": 0.}
    totals = torch.zeros(3, device=fabric.device)
    for start in range(0, len(samples), args.micro_batch_size):
        chunk = fabric.to_device(samples[start:start + args.micro_batch_size])
        with fabric.no_backward_sync(model, enabled=start + args.micro_batch_size < len(samples)):
            current = batch_completion_logprobs(model, chunk)
            values = [sequence_objective(p, s["old"], s["advantage"])
                      for p, s in zip(current, chunk, strict=True)]
            loss = torch.stack([v[0] for v in values]).sum()
            fabric.backward(loss * fabric.world_size / counts[0])
        totals += torch.stack([loss.detach(), torch.stack([v[1] for v in values]).sum(),
                               torch.stack([v[2] for v in values]).float().sum()])
    norm = fabric.clip_gradients(model, optimizer, max_norm=args.max_grad_norm)
    optimizer.step()
    totals = fabric.all_reduce(totals, reduce_op="sum") / counts[0]
    return dict(loss=totals[0].item(), seq_ratio=totals[1].item(),
                clip_fraction=totals[2].item(), grad_norm=norm.item(), skipped=0)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--data", type=Path, default=Path("experiments/gspo/datasets/train.jsonl"))
    p.add_argument("--dev", type=Path, default=Path("experiments/gspo/datasets/dev.jsonl"))
    p.add_argument("--output", type=Path, default=Path("experiments/gspo/runs/gspo"))
    for name, default in [("rollout-batch-size", 8), ("num-mini-batches", 4), ("group-size", 8),
                          ("micro-batch-size", 1), ("generation-micro-batch-size", 1),
                          ("max-prompt-tokens", 2048), ("max-tokens", 4096),
                          ("epochs", 1), ("lora-rank", 32), ("lora-alpha", 64)]:
        p.add_argument("--" + name, type=int, default=default)
    for name, default in [("steps", 100), ("train-limit", 0), ("save-every", 20)]:
        p.add_argument("--" + name, type=int, default=default)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--learning-rate", type=float, default=4e-5)
    p.add_argument("--max-grad-norm", type=float, default=1.)
    p.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    a = p.parse_args()
    for key, value in vars(a).items():
        if isinstance(value, int) and key != "seed":
            if value < (0 if key in {"steps", "train_limit", "save_every"} else 1):
                p.error(f"invalid {key}")
    if a.group_size < 2:
        p.error("group-size must be >= 2")
    if not 0 < a.learning_rate < float("inf") or not 0 < a.max_grad_norm < float("inf"):
        p.error("learning-rate/max-grad-norm must be positive finite")
    return a


def main(a):
    world = int(os.environ.get("WORLD_SIZE", "1"))
    fabric = Fabric(accelerator="cuda" if torch.cuda.is_available() else "cpu", devices=world,
                    strategy=DDPStrategy(broadcast_buffers=False) if world > 1 else "auto",
                    precision="bf16-mixed" if a.precision == "bf16" else "32-true",
                    loggers=TensorBoardLogger(str(a.output), name="tensorboard", version=""))
    fabric.launch()
    occupied = a.output.exists() and any(a.output.iterdir()) if fabric.is_global_zero else False
    if fabric.broadcast(occupied, src=0):
        raise ValueError("use a new output directory")
    rows, dev = load_rows(a.data), load_rows(a.dev)
    key = lambda row: " ".join(row["question"].split())
    if {key(r) for r in rows} & {key(r) for r in dev}:
        raise ValueError("train/dev question overlap")
    if len({key(r) for r in rows}) != len(rows):
        raise ValueError("duplicate training questions; run prepare_data.py")
    rows = rows[:a.train_limit] if a.train_limit else rows
    size = training_size(len(rows), a.rollout_batch_size, a.num_mini_batches, world)
    fabric.seed_everything(a.seed)
    model, tokenizer = load_model(a.model, torch.bfloat16 if a.precision == "bf16" else torch.float32)
    model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", r=a.lora_rank,
                          lora_alpha=a.lora_alpha, target_modules="all-linear", lora_dropout=0.))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=a.learning_rate, betas=(.9, .95), weight_decay=0.)
    model, optimizer = fabric.setup(model, optimizer)
    raw = model.module
    fabric.seed_everything(a.seed + fabric.global_rank)
    if fabric.is_global_zero:
        fabric.logger.log_hyperparams({k: str(v) if isinstance(v, Path) else v for k,v in vars(a).items()})

    def save(name):
        if fabric.is_global_zero:
            raw.save_pretrained(a.output / name)
            tokenizer.save_pretrained(a.output / name)
        fabric.barrier()

    step = 0
    for epoch in range(a.epochs):
        order = list(rows)
        random.Random(a.seed + epoch).shuffle(order)
        for start in range(0, size, a.rollout_batch_size):
            groups = []
            with fabric.autocast():
                for i in range(fabric.global_rank, a.rollout_batch_size, world):
                    groups.append(sample_group(raw, tokenizer, order[start + i], a))
                samples = [s for g in groups for s in g]
                # 整批缓存后才允许第一次 optimizer.step；后续 mini-batch 不重新计算 old。
                cache_old(raw, samples, a.micro_batch_size)
            stats = torch.tensor([sum(s["reward"] for s in samples),
                                  sum(len(s["tokens"]) for s in samples),
                                  sum(all(s["advantage"] == 0 for s in g) for g in groups)],
                                 device=fabric.device, dtype=torch.float32)
            stats = fabric.all_reduce(stats, reduce_op="sum")
            fabric.log_dict({"rollout/reward": (stats[0] / (a.rollout_batch_size*a.group_size)).item(),
                             "rollout/mean_tokens": (stats[1] / (a.rollout_batch_size*a.group_size)).item(),
                             "rollout/degenerate_fraction": (stats[2] / a.rollout_batch_size).item()}, step=step+1)
            local_mini = len(groups) // a.num_mini_batches
            for mini in range(a.num_mini_batches):
                chunk = [s for g in groups[mini*local_mini:(mini+1)*local_mini] for s in g]
                metrics = update_policy(model, optimizer, chunk, a, fabric)
                fabric.log_dict({"update/"+k: v for k,v in metrics.items()},
                                step=step*a.num_mini_batches+mini+1)
                if fabric.is_global_zero:
                    print(f"rollout={step+1} mini={mini+1} {metrics}", flush=True)
            step += 1
            if a.save_every and step % a.save_every == 0:
                save(f"adapter-{step}")
            if a.steps and step >= a.steps:
                break
        if a.steps and step >= a.steps:
            break
    save("adapter")
    fabric.logger.finalize("success")


if __name__ == "__main__":
    main(parse_args())
