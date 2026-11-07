"""一次前向得到全部决策字段的精确候选概率，不生成答案文本。"""
import argparse
import json
import math
import sys
from pathlib import Path

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "experiments.jev"

from .data import Encoder, check_length, compile_record, load_jsonl
from .modeling import decision_logits, load_model, model_arguments


@torch.inference_mode()
def candidate_logits(model, encoder, record, device, max_length=8192):
    encoded = encoder.encode(record, include_targets=False)
    check_length(encoded["ids"], max_length, record.get("id"))
    example = {
        "input_ids": encoded["ids"][:-1],
        "positions": [encoded["markers"][f] - 1 for f in encoded["fields"]],
    }
    logits = decision_logits(model, example, device)
    return {
        field: logits[i, [encoder.symbol_token_id(s) for s, _, _ in options]].cpu().double()
        for i, (field, options) in enumerate(encoded["options"].items())
    }


def build_decisions(record, logits, temperature=1.0):
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    encoded = compile_record(record, include_targets=False)
    results = {}
    for field, options in encoded["options"].items():
        probabilities = torch.softmax(logits[field] / temperature, dim=0).tolist()
        by_value = {v: p for (_, v, _), p in zip(options, probabilities, strict=True)}
        kind = record["questions"][field]["type"]
        result = {"type": kind, "probabilities": by_value,
                  "prediction": max(by_value, key=by_value.get)}
        if kind == "noul":
            result["p_yes"] = by_value["yes"]
            # 与参考 noul 平局规则一致。
            result["prediction"] = "yes" if by_value["yes"] >= by_value["no"] else "no"
        if kind == "score":
            result["expected_score"] = sum(float(v) * p for v, p in by_value.items())
        results[field] = result
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    model_arguments(parser)
    parser.add_argument("--input", type=Path, required=True, help="canonical JSONL")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=3, help="0 表示全部")
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be nonnegative")
    model, tokenizer = load_model(args.model, args.precision, args.adapter)
    model.to(args.device).eval()
    encoder = Encoder(tokenizer)
    records = load_jsonl(args.input)
    if args.limit:
        records = records[:args.limit]
    outputs = []
    for record in records:
        logits = candidate_logits(model, encoder, record, args.device, args.max_length)
        result = {"id": record.get("id"), "decisions": build_decisions(record, logits, args.temperature)}
        outputs.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in outputs))


if __name__ == "__main__":
    main()
