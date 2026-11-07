"""LoRA masked SFT。一个全局 batch 内按案例累积，按决策字段数归一。"""
import argparse
import os
import json
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

from .jev import DEFAULT_MODEL, Encoder, batch_decision_logits, load_jsonl, load_model


def training_size(count, batch_size, world_size):
    """只处理完整全局 batch；多卡时每卡分到相同数量的样本。"""
    if world_size > 1 and batch_size % world_size:
        raise ValueError("global batch size must be divisible by world size")
    size = count // batch_size * batch_size
    if size == 0:
        raise ValueError("not enough training data for one batch; reduce batch size or add data")
    return size


def update(model, optimizer, examples, fabric, max_grad_norm=0.0, micro_batch_size=1):
    """按全局字段数归一；micro-batch 只改变并行计算，不改变更新目标。"""
    count = sum(len(e["targets"]) for e in examples)
    local = examples[fabric.global_rank::fabric.world_size]
    loss_sum = torch.zeros((), device=fabric.device)
    optimizer.zero_grad()
    for start in range(0, len(local), micro_batch_size):
        chunk = local[start:start + micro_batch_size]
        with fabric.no_backward_sync(model, enabled=start + micro_batch_size < len(local)):
            logits = batch_decision_logits(model, chunk, fabric.device)
            losses = []
            for example, values in zip(chunk, logits, strict=True):
                targets = torch.tensor(example["targets"], device=fabric.device)
                loss = F.cross_entropy(values, targets, reduction="sum")
                losses.append(loss * fabric.world_size / count)
                loss_sum += loss.detach()
            fabric.backward(torch.stack(losses).sum())
    norm = fabric.clip_gradients(model, optimizer, max_norm=max_grad_norm or float("inf"))
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
    train_size = training_size(len(records), args.batch_size, fabric.world_size)
    model, tokenizer = load_model(args.model, args.precision)
    encoder = Encoder(tokenizer)
    examples = [encoder.build_example(r, args.max_length) for r in records]
    model = get_peft_model(model, LoraConfig(
        task_type="CAUSAL_LM", r=args.lora_rank, lora_alpha=args.lora_alpha,
        lora_dropout=0.0, target_modules="all-linear", bias="none",
    ))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=0.0)
    model, optimizer = fabric.setup(model, optimizer)
    if fabric.is_global_zero:
        args.output.mkdir(parents=True, exist_ok=True)
        fabric.logger.log_hyperparams({
            key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
        } | {"world_size": fabric.world_size})
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
        for start in range(0, train_size, args.batch_size):
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
    fabric.logger.finalize("success")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--data", type=Path, default=Path(__file__).parent / "datasets/train.jsonl")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "runs/jev")
    parser.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--micro-batch-size", type=int, default=1, help="每进程一次训练前向的样本数；不改变有效 batch")
    parser.add_argument("--batch-size", type=int, default=16, help="全局案例数；各卡按 micro-batch 累积")
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
