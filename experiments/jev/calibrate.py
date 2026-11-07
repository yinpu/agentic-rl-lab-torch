"""仅在 calibration 划分上拟合一个温度；缓存候选 logits，供审查。"""
import argparse
import json
import sys
from pathlib import Path

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "experiments.jev"

from .data import ANSWER_SYMBOLS, Encoder, compile_record, load_jsonl
from .inference import candidate_logits
from .modeling import identity, load_model, model_arguments


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    model_arguments(parser)
    parser.add_argument("--data", type=Path, default=Path(__file__).parent / "datasets/calibration.jsonl")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent / "runs/jev/calibration.json")
    parser.add_argument("--limit", type=int, default=0, help="仅流程检查时限量，0 全部")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be nonnegative")
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


if __name__ == "__main__":
    main()
