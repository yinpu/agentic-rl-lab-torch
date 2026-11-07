"""复用固定留出题目，比较学生 adapter、基础学生和教师。"""
import argparse
import json
from pathlib import Path
import re
import torch
from peft import PeftModel
from task import load_model, prompt_ids, generate, write_json


def extract_boxed(text):
    start = text.rfind("\\boxed{")
    if start < 0:
        return None
    start += 7
    depth = 1
    for end in range(start, len(text)):
        depth += (text[end] == "{") - (text[end] == "}")
        if depth == 0:
            return text[start:end].strip() or None
    return None


def normalize(text):
    return re.sub(r"\s+", "", text.strip().strip("$")).replace(r"\left", "").replace(r"\right", "")


@torch.no_grad()
def evaluate(model, tokenizer, rows, max_tokens=512, batch_size=1):
    records = []
    for start in range(0, len(rows), batch_size):
        chunk = rows[start:start + batch_size]
        outputs = generate(model, tokenizer, [prompt_ids(tokenizer, r["question"]) for r in chunk],
                           max_tokens, sample=False)
        for row, (tokens, truncated) in zip(chunk, outputs, strict=True):
            response = tokenizer.decode(tokens, skip_special_tokens=True)
            predicted = extract_boxed(response)
            target = extract_boxed(row["answer"]) or row["answer"]
            records.append({**row, "response": response, "boxed": predicted is not None,
                            "correct": predicted is not None and normalize(predicted) == normalize(target),
                            "tokens": len(tokens), "truncated": truncated})
    if not records:
        raise ValueError("evaluation set is empty")
    metrics = {"exact_match": sum(r["correct"] for r in records) / len(records),
               "boxed_rate": sum(r["boxed"] for r in records) / len(records),
               "mean_tokens": sum(r["tokens"] for r in records) / len(records),
               "truncation_rate": sum(r["truncated"] for r in records) / len(records)}
    return {"metrics": metrics, "records": records}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=".cache/models/Qwen3.5-4B")
    p.add_argument("--adapter")
    p.add_argument("--split-file", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    args = p.parse_args()
    if args.max_tokens < 1 or args.batch_size < 1 or args.limit < 0:
        p.error("invalid evaluation size")
    rows = json.loads(args.split_file.read_text())["validation"]
    rows = rows[:args.limit] if args.limit else rows
    model, tokenizer = load_model(args.model, torch.bfloat16 if args.precision == "bf16" else torch.float32)
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    with torch.autocast(next(model.parameters()).device.type, dtype=torch.bfloat16,
                        enabled=args.precision == "bf16"):
        result = evaluate(model, tokenizer, rows, args.max_tokens, args.batch_size)
    result["config"] = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    write_json(args.output, result)
    print(result["metrics"])
