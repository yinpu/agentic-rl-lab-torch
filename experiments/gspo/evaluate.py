"""基础模型或 LoRA 在 dev/AIME25 上的平均回答正确率。"""
import argparse
from pathlib import Path
import sys
import torch
import torch.distributed as dist
from lightning.fabric import Fabric
from peft import PeftModel
from datasets import load_from_disk
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "experiments.gspo"
from .task import DEFAULT_MODEL, load_model, load_rows, prompt_ids, generate, score_answer, write_json


def evaluate_model(model, tokenizer, rows, fabric, max_tokens, repeats, micro, prompt_limit, output):
    records = []
    for i in range(fabric.global_rank, len(rows), fabric.world_size):
        row = rows[i]
        prompt = prompt_ids(tokenizer, row["question"], prompt_limit)
        for start in range(0, repeats, micro):
            for tokens, truncated in generate(model, tokenizer, [prompt], max_tokens,
                                               min(micro, repeats - start)):
                text = tokenizer.decode(tokens, skip_special_tokens=True)
                score = score_answer(text, row["answer"])
                records.append(dict(index=i, response=text, **score,
                                    tokens=len(tokens), truncated=truncated))
    if dist.is_initialized():
        parts = [None] * fabric.world_size
        dist.all_gather_object(parts, records)
        records = [r for part in parts for r in part]
    records.sort(key=lambda r: r["index"])
    if not records:
        raise ValueError("empty evaluation dataset")
    metrics = {f"pass@{repeats}": sum(any(r["correct"] for r in records if r["index"] == i)
                                     for i in range(len(rows))) / len(rows),
               "format_rate": sum(r["valid_format"] for r in records) / len(records),
               f"average@{repeats}": sum(r["correct"] for r in records) / len(records),
               "mean_tokens": sum(r["tokens"] for r in records) / len(records),
               "truncation_rate": sum(r["truncated"] for r in records) / len(records)}
    if fabric.is_global_zero:
        write_json(output, dict(metrics=metrics, records=records))
    return metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--adapter")
    p.add_argument("--data", type=Path, default=Path("experiments/gspo/datasets/aime_2025"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--repeats", type=int, default=12)
    p.add_argument("--micro-batch-size", type=int, default=1)
    p.add_argument("--max-tokens", type=int, default=8192)
    p.add_argument("--max-prompt-tokens", type=int, default=4095)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    a = p.parse_args()
    if min(a.repeats, a.micro_batch_size, a.max_tokens, a.max_prompt_tokens) < 1 or a.limit < 0:
        p.error("invalid evaluation sizes")
    rows = (load_rows(a.data) if a.data.suffix == ".jsonl" else
            [dict(question=r["problem"], answer=str(r["answer"])) for r in load_from_disk(str(a.data))])
    if a.limit:
        rows = rows[:a.limit]
    fabric = Fabric(accelerator="cuda" if torch.cuda.is_available() else "cpu", devices=1,
                    precision="bf16-mixed" if a.precision == "bf16" else "32-true")
    fabric.launch()
    fabric.seed_everything(a.seed)
    model, tokenizer = load_model(a.model, torch.bfloat16 if a.precision == "bf16" else torch.float32)
    if a.adapter:
        model = PeftModel.from_pretrained(model, a.adapter)
    model = fabric.setup_module(model)
    with fabric.autocast():
        metrics = evaluate_model(model.module, tokenizer, rows, fabric, a.max_tokens, a.repeats,
                                 a.micro_batch_size, a.max_prompt_tokens, a.output)
    print(metrics)


if __name__ == "__main__":
    main()
