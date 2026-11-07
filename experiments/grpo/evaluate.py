"""固定贪心解码，保存逐题结果；与训练共用提示词、奖励和 EOS 处理。"""

import argparse
import os
import sys
import json
from pathlib import Path

import torch
from lightning.fabric import Fabric
from torch.distributed import all_gather_object
from peft import PeftConfig, PeftModel

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "experiments.grpo"

from .task import (DEFAULT_MODEL, TASK_INSTRUCTION, batch_completion_logprobs,
                  extract_boxed, follows_think_format, generate, ground_truth, load_gsm8k,
                  load_model, make_split, prompt_ids, reward, rows_at, write_json)


def gather_records(records):
    """各 rank 的变长日志列表；不收集计算图或 GPU 张量。"""
    if not torch.distributed.is_initialized():
        return records
    gathered = [None] * torch.distributed.get_world_size()
    all_gather_object(gathered, records)
    return [record for part in gathered for record in part]


def evaluate_model(model, tokenizer, rows, fabric, max_tokens, output, measure_kl=False, batch_size=8):
    """贪心评测指定题目，返回各进程一致的全局 metrics 字典，并保存逐题结果。

    model 是未包装 DDP/Fabric 的模型，tokenizer 可以是基础模型，也可以是已加载 LoRA 的模型及其 tokenizer。
    rows 为非空题目列表，每项含唯一 index、question 和 GSM8K 原始 answer；
    所有进程传入相同列表，fabric 按进程号分配题目并汇总结果。
    batch_size 是每进程一次生成的题数；不足一批的尾部仍参与评测。
    max_tokens 限制每题新生成的 token 数。output 是最终 JSON 路径，旁边还会
    生成各进程的 .rank-N.jsonl；使用新路径，避免覆盖已有逐题记录。
    measure_kl=True 要求模型带 LoRA，额外记录相对冻结基础模型的概率偏离。

    metrics 中 accuracy 为 0/1 奖励的题均值；boxed_rate/think_format_rate
    单独监控答案可提取率/格式合规率，格式不影响奖励；各比例取值均为 [0, 1]。
    """
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    model.eval()
    results = []
    # 每题只交给一个进程，不通过会补齐样本的 DistributedSampler。
    local_rows = rows[fabric.global_rank::fabric.world_size]
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(f".rank-{fabric.global_rank}.jsonl")
    partial.touch(exist_ok=False)  # 逐题保存；中止也能复核，重跑请使用新 output。
    for start in range(0, len(local_rows), batch_size):
        chunk = local_rows[start:start + batch_size]
        prompts = [prompt_ids(tokenizer, row["question"]) for row in chunk]
        with fabric.autocast():
            completions = generate(model, tokenizer, prompts, max_tokens, sample=False)
        kls = [None] * len(chunk)
        if measure_kl:
            samples = [{"prompt": prompt, "tokens": tokens}
                       for prompt, (tokens, _) in zip(prompts, completions, strict=True)]
            with torch.no_grad(), fabric.autocast():
                currents = batch_completion_logprobs(model, samples)
                with model.disable_adapter():
                    references = batch_completion_logprobs(model, samples)
                # 贪心回答路径上的诊断，并非完整策略分布的精确 KL。
                deltas = [ref - cur for ref, cur in zip(references, currents, strict=True)]
                kls = [(delta.exp() - delta - 1).mean().item() for delta in deltas]
        for row, prompt, (tokens, truncated), kl in zip(chunk, prompts, completions, kls, strict=True):
            text = tokenizer.decode(tokens, skip_special_tokens=True)
            # prompt 保存含特殊标记的实际模型输入，response 是去掉特殊标记的回答；
            # tokens 记录有效生成长度（含 EOS），answer 是 #### 后的标准答案。
            result = {"index": row["index"], "question": row["question"], "response": text,
                      "prompt": tokenizer.decode(prompt, skip_special_tokens=False),
                      "answer": ground_truth(row["answer"]), "tokens": len(tokens),
                      "truncated": truncated, "boxed": extract_boxed(text) is not None,
                      "think_format": follows_think_format(text)}
            result["correct"] = reward(text, result["answer"])
            if measure_kl:
                result["greedy_token_kl"] = kl
            results.append(result)
            with partial.open("a") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
        completed = len(results)
        correct = sum(r["correct"] for r in results)
        print(f"{Path(output).name} rank {fabric.global_rank}: "
              f"{completed}/{len(local_rows)}, correct {correct:.0f}, "
              f"accuracy {correct / completed:.2%}", flush=True)
    # 汇总后按原始行号排序；题目集合仍是 rows 指定的集合，不依赖进程数。
    results = sorted(gather_records(results), key=lambda x: x["index"])
    if len(results) != len(rows) or len({r["index"] for r in results}) != len(rows):
        raise RuntimeError("Evaluation has missing or duplicated questions")
    n = len(results)
    if not n:
        raise ValueError("Empty evaluation set")
    # 所有指标按题求平均；准确率包含未输出 boxed 而记 0 的题目。
    metrics = {"questions": n, "accuracy": sum(r["correct"] for r in results) / n,
               "boxed_rate": sum(r["boxed"] for r in results) / n,
               "think_format_rate": sum(r["think_format"] for r in results) / n,
               "mean_tokens": sum(r["tokens"] for r in results) / n,
               "truncation_rate": sum(r["truncated"] for r in results) / n}
    if measure_kl:
        metrics["greedy_token_kl"] = sum(r["greedy_token_kl"] for r in results) / n
    if fabric.is_global_zero:
        write_json(output, {"decoding": "greedy", "max_new_tokens": max_tokens,
                            "task_instruction": TASK_INSTRUCTION,
                            "metrics": metrics, "results": results})
        print(f"Evaluation {output}: {json.dumps(metrics, ensure_ascii=False)}", flush=True)
    fabric.barrier()
    return metrics


def main():
    """解析评测参数，确定数据划分并加载模型；不训练或更新模型参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="默认读取 adapter 的基础模型路径；无 adapter 时使用本章默认模型")
    parser.add_argument("--adapter", help="训练输出的 adapter 目录；省略则评测基础模型")
    # validation 是从官方 train 留出的集合；test 才是官方 GSM8K 测试集。
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--split-file", type=Path, help="复用已有划分；省略则按 seed 创建并保存验证划分")
    parser.add_argument("--seed", type=int, default=42)  # 新建验证划分时的抽样种子。
    parser.add_argument("--validation-size", type=int, default=256)  # 新建划分时的留出题数。
    parser.add_argument("--limit", type=int, default=64, help="0 表示该划分全部题目")
    parser.add_argument("--batch-size", type=int, default=8, help="每进程一次评测生成的题数")
    parser.add_argument("--max-tokens", type=int, default=1024)  # 回答长度上限，不含提示词。
    parser.add_argument("--measure-kl", action="store_true", help="相对关闭 LoRA 的基础模型，诊断贪心回答上的 KL")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")  # 权重与前向计算精度。
    args = parser.parse_args()
    if args.limit < 0 or args.max_tokens < 1 or args.batch_size < 1:
        parser.error("limit must be >= 0 and max-tokens/batch-size must be >= 1")
    if args.measure_kl and not args.adapter:
        parser.error("--measure-kl requires --adapter")
    if args.output.exists() or args.output.with_suffix(".config.json").exists():
        parser.error(f"Output already exists: {args.output}")
    if args.model is None:
        args.model = PeftConfig.from_pretrained(args.adapter).base_model_name_or_path if args.adapter else DEFAULT_MODEL
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    fabric = Fabric(accelerator="cuda" if torch.cuda.is_available() else "cpu", devices=world_size,
                    strategy="ddp" if world_size > 1 else "auto",
                    precision="bf16-mixed" if args.precision == "bf16" else "32-true")
    fabric.launch()
    data = load_gsm8k()
    if args.split == "validation":
        # 指定 split-file 时以保存的索引为准，seed/validation-size 不重新划分数据。
        if args.split_file:
            split = json.loads(args.split_file.read_text())
        else:
            split = make_split(data["train"], args.seed, args.validation_size)
            args.split_file = args.output.with_suffix(".splits.json")
            if args.split_file.exists():
                parser.error(f"Split output already exists: {args.split_file}")
            if fabric.is_global_zero:
                write_json(args.split_file, split)
            fabric.barrier()
        # 数据内容必须与训练时一致，避免同一索引指向不同题目。
        current = make_split(data["train"], split["seed"], len(split["validation"]))
        if current["train_sha256"] != split["train_sha256"]:
            raise ValueError("Dataset content differs from the saved split")
        rows = rows_at(data["train"], split["validation"])
    else:
        rows = rows_at(data["test"], range(len(data["test"])))
    # limit 只截取已经确定的题目列表，不改变完整划分文件；0 表示评测全部。
    if args.limit:
        rows = rows[:args.limit]
    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float32
    model, tokenizer = load_model(args.model, dtype)
    # adapter 只提供训练得到的增量参数，基础模型由 --model 或 adapter 配置确定。
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model = fabric.to_device(model)
    # 评测没有梯度同步需求，不包 DDP，避免各进程题数不同时阻塞。
    if fabric.is_global_zero:
        write_json(args.output.with_suffix(".config.json"), vars(args) | {
            "output": str(args.output), "split_file": str(args.split_file) if args.split_file else None,
            "task_instruction": TASK_INSTRUCTION,
        })
    evaluate_model(model, tokenizer, rows, fabric, args.max_tokens, args.output, args.measure_kl, args.batch_size)


if __name__ == "__main__":
    main()
