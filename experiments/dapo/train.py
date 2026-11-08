"""DAPO：动态补采完整题组，按全局 token 平均 clipped loss，更新 LoRA。"""
import argparse
import json
import os
import random
from pathlib import Path
import sys
import torch
import torch.distributed as dist
from lightning.fabric import Fabric
from lightning.fabric.strategies import DDPStrategy
from lightning.fabric.loggers import TensorBoardLogger
from peft import LoraConfig, get_peft_model

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "experiments.dapo"
from .task import (DEFAULT_MODEL, load_model, load_rows, sample_group, informative_group,
                   batch_completion_logprobs, token_loss, write_json)
from .evaluate import evaluate_model


def gather_groups(local):
    """收集 CPU 记录；候选位置排序使筛选顺序不依赖进程分配。"""
    if not dist.is_initialized():
        return sorted(local, key=lambda pair: pair[0])
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local)
    return sorted([pair for records in gathered for pair in records], key=lambda pair: pair[0])


def select_batch(groups, batch_size):
    selected = [group for _, group in groups if informative_group(group)]
    return selected[:batch_size] if len(selected) >= batch_size else []


def training_size(count, batch_size, world_size):
    if batch_size % world_size:
        raise ValueError("global batch size must be divisible by world size")
    size = count // batch_size * batch_size
    if not size:
        raise ValueError("not enough training data for one global batch")
    return size


def cache_old(model, samples, micro):
    with torch.no_grad():
        for start in range(0, len(samples), micro):
            chunk = samples[start:start + micro]
            for sample, logp in zip(chunk, batch_completion_logprobs(model, chunk), strict=True):
                sample["old"] = logp.cpu()


def update_policy(model, optimizer, samples, args, fabric):
    """DDP 平均梯度，因此本卡 loss sum 乘 world_size / global_token_count。"""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    count = fabric.all_reduce(torch.tensor(sum(len(s["tokens"]) for s in samples),
                                          device=fabric.device), reduce_op="sum")
    total = torch.zeros((), device=fabric.device)
    for start in range(0, len(samples), args.micro_batch_size):
        chunk = fabric.to_device(samples[start:start + args.micro_batch_size])
        with fabric.no_backward_sync(model, enabled=start + args.micro_batch_size < len(samples)):
            current = batch_completion_logprobs(model, chunk)
            loss_sum = torch.stack([token_loss(logp, s["old"], s["advantage"]).sum()
                                    for s, logp in zip(chunk, current, strict=True)]).sum()
            fabric.backward(loss_sum * fabric.world_size / count)
        total += loss_sum.detach()
    norm = fabric.clip_gradients(model, optimizer, max_norm=args.max_grad_norm)
    optimizer.step()
    return {"loss": (fabric.all_reduce(total, reduce_op="sum") / count).item(),
            "grad_norm": norm.item()}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--data", type=Path, default=Path("experiments/dapo/datasets/train.jsonl"))
    p.add_argument("--dev", type=Path, default=Path("experiments/dapo/datasets/dev.jsonl"))
    p.add_argument("--output", type=Path, default=Path("experiments/dapo/runs/dapo"))
    for name, default in [("batch-size", 4), ("group-size", 8), ("micro-batch-size", 1),
                          ("generation-micro-batch-size", 1), ("max-candidate-multiplier", 2),
                          ("max-prompt-tokens", 512), ("max-tokens", 4096), ("overlong-cache", 1024),
                          ("epochs", 1), ("lora-rank", 32), ("lora-alpha", 64),
                          ("eval-max-tokens", 4096)]:
        p.add_argument("--" + name, type=int, default=default)
    for name, default in [("steps", 0), ("train-limit", 0), ("eval-limit", 50), ("eval-every", 25)]:
        p.add_argument("--" + name, type=int, default=default)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--learning-rate", type=float, default=4e-5)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    a = p.parse_args()
    for k, v in vars(a).items():
        if isinstance(v, int) and k != "seed":
            if v < (0 if k in {"steps", "train_limit", "eval_limit", "eval_every"} else 1):
                p.error(f"invalid {k}: {v}")
    if a.group_size < 2 or a.overlong_cache > a.max_tokens:
        p.error("group-size >= 2 and overlong-cache <= max-tokens required")
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
    fabric.seed_everything(a.seed)
    rows, dev = load_rows(a.data), load_rows(a.dev)
    key = lambda row: " ".join(row["question"].split())
    if {key(r) for r in rows} & {key(r) for r in dev}:
        raise ValueError("train/dev question overlap")
    if len({key(r) for r in rows}) != len(rows):
        raise ValueError("duplicate training questions; run prepare_data.py")
    rows = rows[:a.train_limit] if a.train_limit else rows
    dev = dev[:a.eval_limit] if a.eval_limit else dev
    size = training_size(len(rows), a.batch_size, world)
    model, tokenizer = load_model(a.model, torch.bfloat16 if a.precision == "bf16" else torch.float32)
    model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", r=a.lora_rank,
                           lora_alpha=a.lora_alpha, target_modules="all-linear", lora_dropout=0.0))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=a.learning_rate, betas=(0.9, 0.95), weight_decay=0)
    model, optimizer = fabric.setup(model, optimizer)
    raw = model.module
    fabric.seed_everything(a.seed + fabric.global_rank)
    if fabric.is_global_zero:
        a.output.mkdir(parents=True, exist_ok=True)
        fabric.logger.log_hyperparams({k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()})
    fabric.barrier()

    def validation(name, step):
        with fabric.autocast():
            metrics = evaluate_model(raw, tokenizer, dev, fabric, a.eval_max_tokens,
                                     1, a.generation_micro_batch_size, 4095,
                                     a.output / f"{name}.json")
        fabric.log_dict({"validation/" + k: v for k, v in metrics.items()}, step=step)
        return metrics

    before = validation("validation-before", 0)
    attempts = updates = 0
    for epoch in range(a.epochs):
        order = list(rows)
        random.Random(a.seed + epoch).shuffle(order)
        cursor = 0
        while cursor + a.batch_size <= size and (not a.steps or attempts < a.steps):
            groups = []
            # 每轮补采一个完整全局候选批；所有进程使用相同游标、预算与停止条件。
            for _ in range(a.max_candidate_multiplier):
                if cursor + a.batch_size > size:
                    break
                local = []
                with fabric.autocast():
                    for i in range(fabric.global_rank, a.batch_size, world):
                        local.append((cursor + i, sample_group(raw, tokenizer, order[cursor + i], a)))
                groups.extend(gather_groups(local))
                cursor += a.batch_size
                selected = select_batch(groups, a.batch_size)
                if selected:
                    break
            selected = select_batch(groups, a.batch_size)
            attempts += 1
            records = [s for _, group in groups for s in group]
            metrics = {"accuracy": sum(s["correct"] for s in records) / len(records),
                       "reward": sum(s["reward"] for s in records) / len(records),
                       "length_penalty": sum(s["length_penalty"] for s in records) / len(records),
                       "mean_tokens": sum(len(s["tokens"]) for s in records) / len(records),
                       "truncation_rate": sum(s["truncated"] for s in records) / len(records),
                       "effective_group_fraction": sum(informative_group(g) for _, g in groups) / len(groups),
                       "skipped": int(not selected)}
            if selected:
                samples = [s for g in selected[fabric.global_rank::world] for s in g]
                with fabric.autocast():
                    cache_old(raw, samples, a.micro_batch_size)
                metrics.update(update_policy(model, optimizer, samples, a, fabric))
                updates += 1
            fabric.log_dict({"train/" + k: v for k, v in metrics.items()}, step=attempts)
            if fabric.is_global_zero:
                with (a.output / "rollouts.jsonl").open("a") as f:
                    f.write(json.dumps({"attempt": attempts, "metrics": metrics,
                            "samples": [{k: v for k, v in s.items() if k != "old"} for s in records]},
                            ensure_ascii=False) + "\n")
                print(f"attempt={attempts} updates={updates} {metrics}", flush=True)
            if a.eval_every and attempts % a.eval_every == 0:
                validation(f"validation-{attempts}", attempts)
        if a.steps and attempts >= a.steps:
            break
    after = validation("validation-after", attempts)
    if fabric.is_global_zero:
        raw.save_pretrained(a.output / "adapter")
        tokenizer.save_pretrained(a.output / "adapter")
        write_json(a.output / "summary.json", dict(before=before, after=after,
                   attempts=attempts, optimizer_steps=updates))
    fabric.barrier()
    fabric.logger.finalize("success")


if __name__ == "__main__":
    main(parse_args())
