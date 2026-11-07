"""dev 选优与五项基准评测，输出字段级预测和 accuracy / ECE / Brier / NLL。"""
import argparse
import json
import sys
from pathlib import Path

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "experiments.jev"

from .data import ANSWER_SYMBOLS, Encoder, compile_record, load_jsonl, normalize_label
from .inference import build_decisions, candidate_logits
from .modeling import identity, load_model, model_arguments

SUITES = ("jevbench_easy", "jevbench_original", "jevbench_hard", "typed_decisions", "toolace")


def adapt_jevbench(row):
    question = dict(row["question"])
    if question["type"] == "choice":
        question["criteria"] = {label: question["criteria"][label] for label in row["labels"]}
    return {"id": row["id"], "state": row["state"], "questions": {"decision": question},
            "targets": {"decision": {"label": row["expected"]}}}


def metrics(predictions):
    if not predictions:
        raise ValueError("no evaluated fields")
    correct = torch.tensor([r["prediction"] == r["gold"] for r in predictions], dtype=torch.float64)
    confidence = torch.tensor([max(r["probabilities"].values()) for r in predictions], dtype=torch.float64)
    # [0,.1), ... [.9,1]，整数分桶避免浮点边界的重复计数。
    bins = (confidence * 10).long().clamp(max=9)
    ece = 0.0
    for i in range(10):
        mask = bins == i
        if mask.any():
            ece += (mask.double().mean() * (correct[mask].mean() - confidence[mask].mean()).abs()).item()
    brier = sum(sum((p - float(v == r["gold"])) ** 2 for v, p in r["probabilities"].items())
                for r in predictions) / len(predictions)
    return {"decisions": len(predictions), "accuracy": correct.mean().item(), "ece": ece,
            "brier": brier, "nll": sum(r["nll"] for r in predictions) / len(predictions)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    model_arguments(parser)
    parser.add_argument("--dataset-dir", type=Path, default=Path(__file__).parent / "datasets")
    parser.add_argument("--suite", choices=["dev", "all", *SUITES], default="dev")
    parser.add_argument("--calibration", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0, help="每项最多案例数，0 全部")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be nonnegative")
    model_identity = identity(args.model, args.adapter)
    temperature = 1.0
    if args.calibration:
        calibration = json.loads(args.calibration.read_text())
        if calibration["identity"] != model_identity:
            raise ValueError("calibration was fitted for a different model/adapter")
        temperature = calibration["temperature"]
    model, tokenizer = load_model(args.model, args.precision, args.adapter)
    model.to(args.device).eval()
    encoder = Encoder(tokenizer)
    results = []
    for suite in SUITES if args.suite == "all" else [args.suite]:
        path = args.dataset_dir / ("dev.jsonl" if suite == "dev" else f"eval/{suite}.jsonl")
        records = load_jsonl(path)
        if args.limit:
            records = records[:args.limit]
        predictions = []
        for i, record in enumerate(records):
            if suite.startswith("jevbench"):
                record = adapt_jevbench(record)
            logits = candidate_logits(model, encoder, record, args.device, args.max_length)
            decisions = build_decisions(record, logits, temperature)
            targets = compile_record(record)["targets"]
            for field, decision in decisions.items():
                gold = normalize_label(record["questions"][field], record["targets"][field]["label"])
                z = logits[field] / temperature
                nll = (torch.logsumexp(z, dim=0) - z[ANSWER_SYMBOLS.index(targets[field])]).item()
                predictions.append({"id": record.get("id"), "field": field, "gold": gold,
                                    "nll": nll, **decision})
            print(f"{suite} {i + 1}/{len(records)}", flush=True)
        result = {"suite": suite, "records": len(records), **metrics(predictions), "predictions": predictions}
        results.append(result)
        print(json.dumps({k: v for k, v in result.items() if k != "predictions"}), flush=True)
    output = {"identity": model_identity, "temperature": temperature,
              "limit": args.limit, "results": results}
    if args.suite == "all":
        output["macro_accuracy"] = sum(r["accuracy"] for r in results) / len(results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
