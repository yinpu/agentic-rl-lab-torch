"""LoRA masked SFT。一个全局 batch 内按案例累积，按决策字段数归一。"""
import argparse
import os
import json
import hashlib
import math
from pathlib import Path
import random
import sys

import torch
import torch.nn.functional as F
from lightning.fabric import Fabric
from lightning.fabric.strategies import DDPStrategy
from lightning.fabric.loggers import TensorBoardLogger
from peft import LoraConfig, get_peft_model

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "experiments.jev"

from .data import Encoder, load_jsonl
from .modeling import DEFAULT_MODEL, batch_decision_logits, load_model


def update(model, optimizer, examples, fabric, max_grad_norm=0.0, micro_batch_size=1):
    """按全局字段数归一；micro-batch 只改变并行计算，不改变更新目标。"""
    count = sum(len(e["targets"]) for e in examples)
    slots = math.ceil(len(examples) / fabric.world_size)
    local = examples[fabric.global_rank::fabric.world_size]
    loss_sum = torch.zeros((), device=fabric.device)
    optimizer.zero_grad()
    for start in range(0, slots, micro_batch_size):
        # 所有 rank 使用相同槽位数，尾批空槽仍参与反向但权重为零。
        chunk = [local[i] if i < len(local) else examples[0]
                 for i in range(start, min(start + micro_batch_size, slots))]
        with fabric.no_backward_sync(model, enabled=start + micro_batch_size < slots):
            logits = batch_decision_logits(model, chunk, fabric.device)
            losses = []
            for i, (example, values) in enumerate(zip(chunk, logits, strict=True), start):
                targets = torch.tensor(example["targets"], device=fabric.device)
                loss = F.cross_entropy(values, targets, reduction="sum")
                real = i < len(local)
                losses.append(loss * (fabric.world_size / count if real else 0.0))
                if real:
                    loss_sum += loss.detach()
            fabric.backward(torch.stack(losses).sum())
    norm = fabric.clip_gradients(model, optimizer, max_norm=max_grad_norm or float("inf"))
    if not torch.isfinite(norm):
        raise RuntimeError("non-finite gradient; optimizer step cancelled")
    optimizer.step()
    loss = fabric.all_reduce(loss_sum, reduce_op="sum") / count
    return {"loss": loss.item(), "supervised_tokens": count, "grad_norm": norm.item()}


def main(args):
    # torchrun 提供进程数；直接 python 启动时使用单设备。
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    fabric = Fabric(
        accelerator="cuda" if torch.cuda.is_available() else "cpu", devices=world_size,
        strategy=DDPStrategy(broadcast_buffers=False) if world_size > 1 else "auto",
        precision="bf16-mixed" if args.precision == "bf16" else "32-true",
        loggers=TensorBoardLogger(str(args.output), name="tensorboard", version=""),
    )
    fabric.launch()
    fabric.seed_everything(args.seed)
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError(f"output is not empty: {args.output}; choose a new directory")
    records = load_jsonl(args.data)
    if args.max_samples:
        records = records[:args.max_samples]
    if not records:
        raise ValueError("empty training set")
    model, tokenizer = load_model(args.model, args.precision)
    encoder = Encoder(tokenizer)
    examples = [encoder.build_example(r, args.max_length) for r in records]
    model = get_peft_model(model, LoraConfig(
        task_type="CAUSAL_LM", r=args.lora_rank, lora_alpha=args.lora_alpha,
        lora_dropout=0.0, target_modules="all-linear", bias="none",
    ))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if any("lora_" not in n for n, p in model.named_parameters() if p.requires_grad):
        raise RuntimeError("only LoRA parameters should be trainable")
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=0.0)
    model, optimizer = fabric.setup(model, optimizer)
    if fabric.is_global_zero:
        args.output.mkdir(parents=True, exist_ok=True)
        config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        config.update(world_size=fabric.world_size, records=len(records),
                      fields=sum(len(e["targets"]) for e in examples),
                      trainable_parameters=sum(p.numel() for p in trainable),
                      data_sha256=hashlib.sha256(args.data.read_bytes()).hexdigest(),
                      record_ids=[r.get("id") for r in records])
        (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        print(json.dumps(config, ensure_ascii=False), flush=True)
    fabric.barrier()

    def save(name):
        fabric.barrier()
        if fabric.is_global_zero:
            folder = args.output / name
            model.module.save_pretrained(folder, safe_serialization=True)
            tokenizer.save_pretrained(folder)
        fabric.barrier()

    step = 0
    model.train()
    for epoch in range(args.epochs):
        # 与参考脚本一致：在上个 epoch 顺序上再次 shuffle。
        random.Random(args.seed + epoch).shuffle(examples)
        for start in range(0, len(examples), args.batch_size):
            batch = examples[start:start + args.batch_size]
            metrics = update(model, optimizer, batch, fabric, args.max_grad_norm, args.micro_batch_size)
            step += 1
            if fabric.is_global_zero:
                metrics.update(step=step, epoch=epoch + 1, records=len(batch))
                with (args.output / "train.jsonl").open("a") as f:
                    f.write(json.dumps(metrics) + "\n")
                fabric.log_dict({f"train/{k}": v for k, v in metrics.items() if k != "step"}
                                | {"train/learning_rate": optimizer.param_groups[0]["lr"]}, step=step)
                print(json.dumps(metrics), flush=True)
            if args.save_every and step % args.save_every == 0:
                save(f"step-{step}")
            if args.max_steps and step >= args.max_steps:
                break
        save(f"epoch-{epoch + 1}")
        if args.max_steps and step >= args.max_steps:
            break
    save("final")
    for logger in fabric.loggers:
        logger.finalize("success")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--data", type=Path, default=Path(__file__).parent / "datasets/train.jsonl")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "runs/jev")
    parser.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--micro-batch-size", type=int, default=1, help="每进程一次训练前向的样本数；不改变有效 batch")
    parser.add_argument("--batch-size", type=int, default=16, help="全局案例数；各卡逐条累积")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--max-grad-norm", type=float, default=0.0, help="0 不裁剪")
    parser.add_argument("--max-length", type=int, default=8192, help="超长报错，不截断；0 无上限")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    for key in ("micro_batch_size", "epochs", "batch_size", "lora_rank", "lora_alpha"):
        if getattr(args, key) <= 0:
            parser.error(f"--{key.replace('_', '-')} must be positive")
    for key in ("max_samples", "max_steps", "save_every", "max_length", "max_grad_norm"):
        if getattr(args, key) < 0:
            parser.error(f"--{key.replace('_', '-')} must be nonnegative")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    return args


if __name__ == "__main__":
    main(parse_args())
