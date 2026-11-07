"""同步 OPD：学生采样 → 教师同轨迹打分 → token 优势 → LoRA 更新。"""
import argparse
import os
from pathlib import Path
import torch
from lightning.fabric import Fabric
from lightning.fabric.strategies import DDPStrategy
from lightning.fabric.loggers import TensorBoardLogger
from peft import LoraConfig, get_peft_model
from task import (load_model, check_tokenizers, load_deepmath, make_split, training_size,
                  rollout, update_policy, write_json)
from evaluate import evaluate


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--student", default=".cache/models/Qwen3.5-4B")
    p.add_argument("--teacher", default=".cache/models/Qwen3.6-27B")
    p.add_argument("--output", type=Path, default=Path("experiments/general-opd/runs/opd"))
    for name, default in [("batch-size", 2), ("group-size", 2), ("micro-batch-size", 1),
                          ("max-tokens", 512), ("validation-size", 64), ("eval-max-tokens", 512),
                          ("eval-batch-size", 1), ("num-shards", 1), ("lora-rank", 32), ("lora-alpha", 64)]:
        p.add_argument("--" + name, type=int, default=default)
    for name, default in [("steps", 10), ("train-limit", 1000), ("eval-limit", 64), ("seed", 42)]:
        p.add_argument("--" + name, type=int, default=default)
    p.add_argument("--learning-rate", type=float, default=4e-5)
    p.add_argument("--beta", type=float, default=1.)
    p.add_argument("--max-grad-norm", type=float, default=1.)
    p.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    args = p.parse_args()
    for k, v in vars(args).items():
        if isinstance(v, (int, float)) and (v < 0 or (k not in {"steps", "train_limit", "eval_limit", "seed"} and not 0 < v < float("inf"))):
            p.error(f"invalid {k}")
    if args.num_shards > 10:
        p.error("num-shards must be 1..10")
    if args.output.exists() and any(args.output.iterdir()):
        p.error("use a new output directory")
    return args


def main(args):
    world = int(os.environ.get("WORLD_SIZE", "1"))
    fabric = Fabric(accelerator="cuda" if torch.cuda.is_available() else "cpu", devices=world,
                    strategy=DDPStrategy(broadcast_buffers=False) if world > 1 else "auto",
                    precision="bf16-mixed" if args.precision == "bf16" else "32-true",
                    loggers=TensorBoardLogger(str(args.output), name="tensorboard", version=""))
    fabric.launch()
    fabric.seed_everything(args.seed)
    with fabric.rank_zero_first():
        data = load_deepmath(args.num_shards)
    train, validation = make_split(data, args.seed, args.validation_size, args.train_limit)
    size = training_size(len(train), args.batch_size, world)
    if fabric.is_global_zero:
        write_json(args.output / "splits.json", {"validation": validation})
        fabric.logger.log_hyperparams({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    validation = validation[:args.eval_limit] if args.eval_limit else validation
    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float32
    student, tokenizer = load_model(args.student, dtype)
    teacher, teacher_tokenizer = load_model(args.teacher, dtype)
    check_tokenizers(tokenizer, teacher_tokenizer)
    teacher.requires_grad_(False).eval().to(fabric.device)
    student = get_peft_model(student, LoraConfig(task_type="CAUSAL_LM", r=args.lora_rank,
                             lora_alpha=args.lora_alpha, target_modules="all-linear", lora_dropout=0.))
    student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    optimizer = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad],
                                 lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=0.)
    model, optimizer = fabric.setup(student, optimizer)
    raw = model.module
    fabric.seed_everything(args.seed + fabric.global_rank)

    def assess(candidate, name, step):
        # 评测仅 rank 0 使用未包装模型；其他 rank 等待，不发生 DDP forward collective。
        if fabric.is_global_zero:
            with fabric.autocast():
                result = evaluate(candidate, tokenizer, validation, args.eval_max_tokens, args.eval_batch_size)
            write_json(args.output / f"{name}.json", result)
            fabric.log_dict({f"{name}/{k}": v for k, v in result["metrics"].items()}, step=step)
        fabric.barrier()

    assess(raw, "student-before", 0)
    assess(teacher, "teacher", 0)
    last_step = 0
    for step, start in enumerate(range(0, size, args.batch_size), 1):
        if args.steps and step > args.steps:
            break
        rows = train[start:start + args.batch_size][fabric.global_rank::world]
        with fabric.autocast():
            samples = rollout(raw, teacher, tokenizer, rows, args)
        metrics = update_policy(model, optimizer, samples, args, fabric)
        stats = torch.tensor([sum((s["old"] - s["teacher"]).sum().item() for s in samples),
                              sum(len(s["tokens"]) for s in samples), len(samples),
                              sum(s["truncated"] for s in samples)], device=fabric.device)
        stats = fabric.all_reduce(stats, reduce_op="sum")
        metrics.update(reverse_kl=(stats[0] / stats[1]).item(), mean_tokens=(stats[1] / stats[2]).item(),
                       truncation_rate=(stats[3] / stats[2]).item())
        fabric.log_dict({f"train/{k}": v for k, v in metrics.items()}, step=step)
        fabric.print(f"step={step} {metrics}")
        last_step = step
    assess(raw, "student-after", last_step)
    if fabric.is_global_zero:
        raw.save_pretrained(args.output / "adapter")
        tokenizer.save_pretrained(args.output / "adapter")
        if torch.cuda.is_available():
            write_json(args.output / "memory.json", {"peak_allocated_bytes": torch.cuda.max_memory_allocated()})
    fabric.barrier()


if __name__ == "__main__":
    main(parse_args())
