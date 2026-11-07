"""GSM8K 上的显式 GRPO 循环：生成一组回答 → 组内优势 → clipped loss → LoRA 更新。"""

import argparse
import os
from importlib.metadata import version
import json
from pathlib import Path
import sys

import torch
from lightning.fabric import Fabric
from lightning.fabric.strategies import DDPStrategy
from lightning.fabric.loggers import TensorBoardLogger
from peft import LoraConfig, get_peft_model

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "experiments.grpo"

from .evaluate import evaluate_model, gather_records
from .task import (DEFAULT_MODEL, TASK_INSTRUCTION, batch_completion_logprobs, completion_logprobs,
                  eos_ids, generate, ground_truth,
                  group_advantages, grpo_loss, load_gsm8k, load_model, make_split,
                  prompt_ids, rows_at, training_rewards, write_json)


def parse_args():
    """读取训练配置；batch/step 均以“采样题目批次”为单位，区别于优化器更新。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "runs/grpo")
    parser.add_argument("--steps", type=int, default=0, help="采样批次数上限；0 为训练集一遍")
    parser.add_argument("--train-limit", type=int, default=0, help="排除验证题后取前 N 题；0 为全部")
    # W 个进程每批最多采样 batch_size * W 道题，每题生成 group_size 条回答。
    parser.add_argument("--micro-batch-size", type=int, default=1, help="每进程一次训练前向的样本数；不改变有效 batch")
    parser.add_argument("--batch-size", type=int, default=4, help="每进程每个采样批次的题数")
    parser.add_argument("--group-size", type=int, default=8, help="同一道题的回答数，至少为 2")
    parser.add_argument("--max-tokens", type=int, default=1024)  # 每条回答的新增 token 上限。
    parser.add_argument("--updates-per-batch", type=int, default=1, help="固定本批旧 logprob，重复优化次数")
    parser.add_argument("--standardize-advantages", action="store_true", help="额外除以组内总体标准差；默认仅减均值")
    parser.add_argument("--beta", type=float, default=0.0, help="reference KL 系数；0 不计算 KL")
    parser.add_argument("--format-reward-weight", type=float, default=0.0,
                        help="格式奖励权重 λ：总奖励=正确性+λ×格式合规；0 关闭，实验可取 0.5")
    parser.add_argument("--clip", type=float, default=0.2)  # GRPO 概率比裁剪 ε。
    parser.add_argument("--learning-rate", type=float, default=4e-5)
    parser.add_argument("--lora-rank", type=int, default=32)  # 低秩更新 ΔW 的秩 r。
    parser.add_argument("--lora-alpha", type=int, default=64)  # LoRA 更新缩放为 alpha / r。
    parser.add_argument("--max-grad-norm", type=float, default=1.0)  # 梯度总范数裁剪，区别于概率比裁剪。
    parser.add_argument("--seed", type=int, default=42)  # 控制划分、LoRA 初始化和采样随机性。
    parser.add_argument("--validation-size", type=int, default=256)  # 从官方 train 留出的题数。
    parser.add_argument("--eval-limit", type=int, default=64, help="固定验证题数；0 为全部验证集")
    parser.add_argument("--eval-every", type=int, default=50, help="每 N 个采样批次验证；0 仅训练前后验证")
    parser.add_argument("--eval-max-tokens", type=int, default=1024)  # 验证时每条回答的长度上限。
    parser.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    args = parser.parse_args()
    for name in ["micro_batch_size", "batch_size", "max_tokens", "updates_per_batch", "lora_rank", "lora_alpha",
                 "eval_max_tokens", "validation_size"]:
        if getattr(args, name) < 1:
            parser.error(f"{name} must be >= 1")
    if args.group_size < 2 or not 0 < args.clip < 1:
        parser.error("group-size must be >= 2 and clip must be in (0, 1)")
    if any(getattr(args, name) < 0 for name in ["steps", "train_limit", "eval_limit", "eval_every", "beta"]):
        parser.error("steps/train-limit/eval-limit/eval-every/beta must be >= 0")
    if not 0 <= args.format_reward_weight < float("inf"):
        parser.error("format-reward-weight must be finite and >= 0")
    if args.learning_rate <= 0 or args.max_grad_norm <= 0:
        parser.error("learning-rate and max-grad-norm must be > 0")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error(f"Use a new output directory: {args.output}")
    return args


def rollout(model, tokenizer, rows, args):
    """用更新前的策略采样、判分，返回本进程的回答记录列表 samples。

    model 是解开 DDP 包装的 LoRA 模型；rows 为含 index/question/answer 的题目列表。
    args 使用 max_tokens、group_size、standardize_advantages、beta 和 format_reward_weight。
    B 道题返回 B*G 条记录，同题的 G 条回答连续排列。关键字段：
      prompt/tokens：提示词/回答 ID 列表，长度分别为 P/T；
      old/reference：冻结的逐 token logprob，CPU 张量 [T]，reference 可为 None；
      advantage/reward：该回答的组内优势和总奖励，均为标量；
      correct/format_reward/think_format：0/1 正确性、加权格式奖励、格式合规标记。
    整批采样完成后才更新参数，因此所有 old 都来自同一个旧策略。
    """
    model.eval()
    samples = []
    for row in rows:
        prompt = prompt_ids(tokenizer, row["question"])
        # 对同一道题从当前策略采样 G 个回答，构成一个比较组。
        completions = generate(model, tokenizer, prompt, args.max_tokens, args.group_size)
        texts = [tokenizer.decode(tokens, skip_special_tokens=True) for tokens, _ in completions]
        # 逐回答打分：正确性奖励加可选的格式奖励，得到 R_i。
        scores = [training_rewards(text, ground_truth(row["answer"]), args.format_reward_weight)
                  for text in texts]
        # R_i = 正确性_i + λ * 格式合规_i；先加奖励，再在同一道题内计算优势。
        # KL 不加进 R_i，而是由 grpo_loss 直接加到 loss。
        rewards = [score["reward"] for score in scores]
        # 每一道题单独构成一组；不能把不同题目的奖励混在一起计算均值。
        advantages = group_advantages(rewards, args.standardize_advantages)
        for (tokens, truncated), text, score, advantage in zip(completions, texts, scores, advantages, strict=True):
            with torch.no_grad():
                # πold 只在采样阶段计算一次，多次 update 时也不重算。
                old = completion_logprobs(model, prompt, tokens).cpu()
                reference = None
                if args.beta:
                    # 基础权重始终冻结；临时关闭 LoRA，即得到固定的 πref。
                    # 它不同于每轮重新采样前的 πold。
                    with model.disable_adapter():
                        reference = completion_logprobs(model, prompt, tokens).cpu()
            samples.append({"prompt": prompt, "tokens": tokens, "old": old, "reference": reference,
                            "advantage": advantage.item(), "index": row["index"],
                            "response": text, "truncated": truncated, **score})
    return samples


def update_policy(model, optimizer, samples, real_count, global_count, args, fabric):
    """逐 micro-batch 累积梯度，执行一次 optimizer.step()，返回全局平均的日志指标。

    model 是 Fabric 包装后的策略；optimizer 只更新 LoRA 参数。
    samples 是 rollout 记录，尾部可有用于同步的占位回答；real_count 是本进程
    真实回答数，global_count 是所有进程的真实回答总数，均不含占位项。
    args 使用 beta、clip、max_grad_norm、micro_batch_size；fabric 负责反传、同步和归约。
    对每条回答先平均 token loss，再平均全局回答；一次只保留一个 micro-batch 的计算图。
    """
    model.train()  # dropout 已关闭；train 模式用于启用 gradient checkpointing。
    optimizer.zero_grad(set_to_none=True)
    metrics_sum = {}
    micro = args.micro_batch_size
    for start in range(0, len(samples), micro):
        chunk = samples[start:start + micro]
        # 只有最后一次反向同步梯度；context 同时覆盖 forward 和 backward。
        with fabric.no_backward_sync(model, enabled=start + micro < len(samples)):
            currents = batch_completion_logprobs(model, chunk)
            weighted_losses = []
            for i, (sample, current) in enumerate(zip(chunk, currents, strict=True), start):
                real = i < real_count
                old = sample["old"].to(fabric.device)
                reference = sample["reference"]
                if reference is not None:
                    reference = reference.to(fabric.device)
                loss, metrics = grpo_loss(current, old, sample["advantage"], reference,
                                          args.beta if real else 0.0, args.clip)
                # DDP 平均 W 份梯度：每条回答乘 W/N，保持全局回答等权。
                weighted_losses.append(loss * (fabric.world_size / global_count if real else 0.0))
                if real:
                    for key, value in metrics.items():
                        metrics_sum[key] = metrics_sum.get(key, 0.0) + value
            fabric.backward(torch.stack(weighted_losses).sum())
    # 整批梯度累积完后统一裁剪；返回的 grad_norm 是裁剪前的总范数。
    grad_norm = fabric.clip_gradients(model, optimizer, max_norm=args.max_grad_norm)
    if not torch.isfinite(grad_norm):
        raise RuntimeError("Non-finite gradient; stop without saving updated weights")
    optimizer.step()
    # 空槽也使用相同的 key，保证所有进程参与相同的 collective。
    keys = ["policy_loss", "ratio_mean", "clip_fraction"] + (["kl"] if args.beta else [])
    values = torch.tensor([metrics_sum.get(k, 0.0) for k in keys], device=fabric.device)
    values = fabric.all_reduce(values, reduce_op="sum") / global_count
    return {"grad_norm": grad_norm.item(), **dict(zip(keys, values.tolist(), strict=True))}


def pad_samples(samples, prompt_count, group_size, dummy_prompt, stop_id, fabric):
    """原地补齐各进程的 forward/backward 次数；真实回答数由调用方在补齐前保存。

    尾批不重复真实回答，空槽只用于 DDP 同步，更新时权重为 0。
    每进程的槽位数为 ceil(全局题数 / 进程数) * 每题回答数。
    """
    slots = ((prompt_count + fabric.world_size - 1) // fabric.world_size) * group_size
    while len(samples) < slots:
        samples.append({"prompt": dummy_prompt, "tokens": [stop_id],
                        "old": torch.zeros(1), "reference": None, "advantage": 0.0})


def should_update(samples, beta, fabric):
    """全局有非零优势或启用 KL 时更新；所有进程共同决定，防止同步分支不一致。"""
    signal = torch.tensor(sum(s["advantage"] != 0 for s in samples), device=fabric.device)
    global_signal = fabric.all_reduce(signal, reduce_op="sum").item()
    # 总奖励相同时优势为 0；无 KL 则跳过，避免 Adam 历史动量继续改变参数。
    return bool(global_signal or beta)


def log_batch(samples, real_count, prompt_count, group_size, batch_index,
              optimizer_steps, update_metrics, output, fabric):
    """所有进程收集真实回答，主进程写入原有日志字段；占位回答不参与统计。"""
    # 日志只收集真实回答，不保存大张量，也不计入同步用的占位回答。
    records = gather_records([{k: s[k] for k in ["index", "reward", "correct", "format_reward",
                                               "think_format", "advantage", "response", "truncated"]}
                             | {"tokens": len(s["tokens"])} for s in samples[:real_count]])
    if fabric.is_global_zero:
        groups = [records[i:i + group_size] for i in range(0, len(records), group_size)]
        # frac_degenerate 是优势全零的题组比例；signal_tokens 统计有非零优势的
        # 回答 token 数，便于判断奖励是否提供了区分信号，而不是衡量梯度大小。
        log = {"batch": batch_index, "optimizer_steps": optimizer_steps, "prompts": prompt_count,
               "reward": sum(s["reward"] for s in records) / len(records),
               "accuracy": sum(s["correct"] for s in records) / len(records),
               "format_reward": sum(s["format_reward"] for s in records) / len(records),
               "think_format_rate": sum(s["think_format"] for s in records) / len(records),
               "frac_degenerate": sum(all(s["advantage"] == 0 for s in g) for g in groups) / len(groups),
               "mean_tokens": sum(s["tokens"] for s in records) / len(records),
               "truncation_rate": sum(s["truncated"] for s in records) / len(records),
               "signal_tokens": sum(s["tokens"] for s in records if s["advantage"] != 0),
               "updates": update_metrics}
        with (output / "train.jsonl").open("a") as f:
            f.write(json.dumps(log, ensure_ascii=False) + "\n")
        with (output / "rollouts.jsonl").open("a") as f:
            f.write(json.dumps({"batch": batch_index, "samples": records}, ensure_ascii=False) + "\n")
        fabric.log_dict({f"train/{k}": v for k, v in log.items() if k not in ("batch", "updates")},
                        step=batch_index)
        print(json.dumps(log, ensure_ascii=False), flush=True)


def save_adapter(model, tokenizer, output, fabric):
    """只由主进程保存 adapter 与 tokenizer；调用方负责需要时的进程同步。"""
    if fabric.is_global_zero:
        model.save_pretrained(output)
        tokenizer.save_pretrained(output)


def main(args):
    """组织数据、LoRA、采样与更新；将逐批日志、验证结果和最终 adapter 写入 output。"""
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
    # 准备 GSM8K 训练题和固定验证题，验证题不参与策略更新。
    data = load_gsm8k()
    split = make_split(data["train"], args.seed, args.validation_size, args.train_limit)
    train_rows = rows_at(data["train"], split["train"])
    validation = rows_at(data["train"], split["validation"])
    if args.eval_limit:
        validation = validation[:args.eval_limit]
    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float32
    model, tokenizer = load_model(args.model, dtype)
    # 初始化策略 πθ：冻结基础模型，仅通过 LoRA 学习策略增量。
    model = get_peft_model(model, LoraConfig(
        task_type="CAUSAL_LM", r=args.lora_rank, lora_alpha=args.lora_alpha,
        target_modules="all-linear", lora_dropout=0.0, bias="none",
    ))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    trainable = [p for p in model.parameters() if p.requires_grad]
    if any("lora_" not in name for name, p in model.named_parameters() if p.requires_grad):
        raise RuntimeError("Only LoRA parameters should be trainable")
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=0)
    model, optimizer = fabric.setup(model, optimizer)
    fabric.seed_everything(args.seed + fabric.global_rank)
    raw_model = model.module
    if fabric.is_global_zero:
        args.output.mkdir(parents=True, exist_ok=True)
        write_json(args.output / "splits.json", split)
        model_source = Path(args.model) / "source.json"
        if model_source.is_file():
            write_json(args.output / "model-source.json", json.loads(model_source.read_text()))
        write_json(args.output / "config.json", vars(args) | {
            "output": str(args.output), "world_size": fabric.world_size,
            "global_prompts_per_batch": args.batch_size * fabric.world_size,
            "reference_source_commit": "8493321013c02eef6cf79ca2594811f53c6fdac1",
            "trainable_parameters": sum(p.numel() for p in trainable),
            "task_instruction": TASK_INSTRUCTION,
        })
        write_json(args.output / "environment.json", {
            name: version(name) for name in ["torch", "transformers", "peft", "lightning", "datasets", "tensorboard"]
        } | {"cuda": torch.version.cuda, "device": str(fabric.device),
             "python": sys.version,
             "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None})
    fabric.barrier()

    # 训练前评测：记录初始策略在固定验证题上的正确率。
    before = evaluate_model(raw_model, tokenizer, validation, fabric, args.eval_max_tokens,
                            args.output / "validation-before.json")
    fabric.log_dict({f"validation/{k}": v for k, v in before.items()}, step=0)
    optimizer_steps = 0
    completed_batches = 0
    last_validation_batch = 0
    global_batch_size = args.batch_size * fabric.world_size
    for batch_index, start in enumerate(range(0, len(train_rows), global_batch_size), 1):
        if args.steps and batch_index > args.steps:
            break
        batch = train_rows[start:start + global_batch_size]
        local_rows = batch[fabric.global_rank::fabric.world_size]
        # 采样：每题生成 G 个回答，计算奖励和组内优势，并固定本批 πold 的概率。
        with fabric.autocast():
            samples = rollout(raw_model, tokenizer, local_rows, args)
        real_count = len(samples)
        pad_samples(samples, len(batch), args.group_size,
                    prompt_ids(tokenizer, batch[0]["question"]), eos_ids(tokenizer)[0], fabric)
        update_metrics = []
        # 有非零优势或启用 KL 时才更新；否则本批不执行优化器步。
        if should_update(samples, args.beta, fabric):
            # 策略更新：重算 πθ 的概率，用优势、概率比、clip 和可选 KL 计算 loss。
            # 同批多次更新始终固定 πold；下一批才用更新后的策略重新采样。
            for update in range(args.updates_per_batch):
                metrics = update_policy(model, optimizer, samples, real_count,
                                        len(batch) * args.group_size, args, fabric)
                optimizer_steps += 1
                update_metrics.append({"update": update + 1, **metrics})
                fabric.log_dict({f"update/{k}": v for k, v in metrics.items()} | {
                    "update/loss": metrics["policy_loss"] + args.beta * metrics.get("kl", 0.0),
                    "update/learning_rate": optimizer.param_groups[0]["lr"],
                }, step=optimizer_steps)
        log_batch(samples, real_count, len(batch), args.group_size, batch_index,
                  optimizer_steps, update_metrics, args.output, fabric)
        completed_batches = batch_index
        if args.eval_every and batch_index % args.eval_every == 0:
            # 定期评测：用相同验证题和贪心解码观察当前策略的表现。
            save_adapter(raw_model, tokenizer, args.output / f"adapter-batch-{batch_index}", fabric)
            fabric.barrier()
            metrics = evaluate_model(raw_model, tokenizer, validation, fabric, args.eval_max_tokens,
                                     args.output / f"validation-batch-{batch_index}.json")
            fabric.log_dict({f"validation/{k}": v for k, v in metrics.items()}, step=batch_index)
            last_validation_batch = batch_index
    # 训练后评测：与初始策略比较，检查留出题目的正确率变化。
    after = evaluate_model(raw_model, tokenizer, validation, fabric, args.eval_max_tokens,
                           args.output / "validation-after.json")
    if completed_batches != last_validation_batch:
        fabric.log_dict({f"validation/{k}": v for k, v in after.items()}, step=completed_batches)
    fabric.barrier()
    save_adapter(raw_model, tokenizer, args.output / "adapter", fabric)
    if fabric.is_global_zero:
        write_json(args.output / "summary.json", {"before": before, "after": after,
            "optimizer_steps": optimizer_steps,
            "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else None,
            "peak_cuda_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3 if torch.cuda.is_available() else None,
        })
    fabric.barrier()
    for logger in fabric.loggers:
        logger.finalize("success")


if __name__ == "__main__":
    main(parse_args())
