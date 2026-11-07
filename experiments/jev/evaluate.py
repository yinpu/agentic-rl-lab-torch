"""Jev 温度校准与评测：calibrate 拟合温度，eval 运行 dev 或五项基准。"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "experiments.jev"

from .jev import (ANSWER_SYMBOLS, Encoder, build_decisions, candidate_logits,
                  compile_record, load_jsonl, load_model,
                  model_arguments, normalize_label)

SUITES = ("jevbench_easy", "jevbench_original", "jevbench_hard", "typed_decisions", "toolace")


def identity(model_name, adapter):
    name = str(Path(model_name).resolve()) if Path(model_name).exists() else model_name
    result = {"model": name, "adapter_sha256": None}
    if adapter:
        result["adapter_sha256"] = hashlib.sha256((Path(adapter) / "adapter_model.safetensors").read_bytes()).hexdigest()
    return result


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


def mean_nll(instances, temperature):
    return sum((torch.logsumexp(z / temperature, dim=0) - z[y] / temperature).item()
               for z, y in instances) / len(instances)


def fit_temperature(instances):
    if not instances:
        raise ValueError("empty calibration set")
    # NLL 对逆温度 beta 凸；导数为 E_p[z] - z_gold，二分找零点。
    def derivative(beta):
        return sum((torch.softmax(z * beta, dim=0).dot(z) - z[y]).item()
                   for z, y in instances) / len(instances)
    lo, hi = 0.01, 100.0
    if derivative(lo) >= 0:
        return 1 / lo
    if derivative(hi) <= 0:
        return 1 / hi
    for _ in range(80):
        mid = (lo + hi) / 2
        if derivative(mid) > 0:
            hi = mid
        else:
            lo = mid
    return 1 / ((lo + hi) / 2)


def evaluate(args):
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


def calibrate(args):
    records = load_jsonl(args.data)
    if args.limit:
        records = records[:args.limit]
    if not records:
        raise ValueError("empty calibration data")
    model, tokenizer = load_model(args.model, args.precision, args.adapter)
    model.to(args.device).eval()
    encoder = Encoder(tokenizer)
    instances, cached = [], []
    for i, record in enumerate(records):
        logits = candidate_logits(model, encoder, record, args.device, args.max_length)
        targets = compile_record(record)["targets"]
        for field, z in logits.items():
            gold = ANSWER_SYMBOLS.index(targets[field])
            instances.append((z, gold))
            cached.append({"id": record.get("id"), "field": field, "logits": z.tolist(), "gold": gold})
        print(f"calibration {i + 1}/{len(records)}", flush=True)
    temperature = fit_temperature(instances)
    result = {"identity": identity(args.model, args.adapter), "temperature": temperature,
              "data": str(args.data.resolve()), "records": len(records), "instances": len(instances),
              "nll_t1": mean_nll(instances, 1), "nll_fitted": mean_nll(instances, temperature),
              "predictions": cached}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "predictions"}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    evaluation = commands.add_parser("eval", help="dev 选优或测试集评测")
    calibration = commands.add_parser("calibrate", help="仅在 calibration 划分拟合温度")
    for command in (evaluation, calibration):
        model_arguments(command)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--limit", type=int, default=0, help="每项最多案例数；0 全部")
    evaluation.add_argument("--dataset-dir", type=Path, default=Path(__file__).parent / "datasets")
    evaluation.add_argument("--suite", choices=["dev", "all", *SUITES], default="dev")
    evaluation.add_argument("--calibration", type=Path, default=None)
    calibration.add_argument("--data", type=Path, default=Path(__file__).parent / "datasets/calibration.jsonl")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be nonnegative")
    if args.command == "calibrate":
        calibrate(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
