"""下载并准备训练/评测数据。

改编自 datawhalechina/agentic-rl-lab 的 pytrio-jev/00-prepare-data.py（Apache-2.0）。
增加版本与文件指纹记录、输入检查；保留上游切分和采样规则。

用法（仓库根目录）：
python experiments/jev/prepare_data.py

产物（experiments/jev/datasets/）：
- train.jsonl       typed-decisions train 切出 960 案例 + ToolACE 1200 条
- dev.jsonl         typed-decisions 案例级切分 120 案例（选优用）
- calibration.jsonl typed-decisions 案例级切分 120 案例（温度拟合用）
- eval/*.jsonl      五个评测基准文件
"""

from __future__ import annotations

import hashlib
import argparse
import json
import random
import re
import urllib.request
from pathlib import Path

from datasets import load_dataset

SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_DIR = SCRIPT_DIR / "datasets"
EVAL_DIR = DATASET_DIR / "eval"

TYPED_DECISIONS_REPO = "LocalLLaMA/typed-decisions"
TOOLACE_REPO = "Team-ACE/ToolACE"
TOOLACE_TRAIN_SIZE = 1200
SPLIT_SEED = "pytrio-jev"
SAMPLE_SEED = 42
TD_REVISION = "f5affa5f05859bd8486bc15e1e352a04f67994cf"
TOOLACE_REVISION = "6bda777c88d21e5a204703c1ee45597a8fa4f734"

EVAL_BASE = (
    "https://raw.githubusercontent.com/InternLM/Intern-Decision/"
    "565350244af860e1c276768cd80bdd9b8be93913/benchmarks/accuracy-v1"
)
EVAL_FILES = {
    "jevbench_easy.jsonl": "jevbench/easy.jsonl",
    "jevbench_original.jsonl": "jevbench/original.jsonl",
    "jevbench_hard.jsonl": "jevbench/hard.jsonl",
    "typed_decisions.jsonl": "typed_decisions/test.jsonl",
    "toolace.jsonl": "toolace/test.jsonl",
}


def download_eval_files() -> None:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    for name, remote in EVAL_FILES.items():
        target = EVAL_DIR / name
        if target.exists():
            continue
        with urllib.request.urlopen(f"{EVAL_BASE}/{remote}", timeout=60) as response:
            contents = response.read()
        # 完整读取成功后才替换，避免中断留下半个评测文件。
        temporary = target.with_suffix(".tmp")
        temporary.write_bytes(contents)
        temporary.replace(target)
        print(f"下载 {name}: {sum(1 for _ in open(target))} 行")


def convert_typed_decisions(row: dict) -> dict:
    """HF 行 → canonical record；gold 取硬标签（软分布在训练里不用）。"""
    questions = json.loads(row["questions"])
    gold = json.loads(row["gold"])
    return {
        "id": row["id"],
        "state": json.loads(row["state"]),
        "questions": questions,
        "targets": {field: {"label": gold[field]["label"]} for field in questions},
    }


def split_typed_decisions() -> tuple[list[dict], list[dict], list[dict]]:
    """案例级确定性切分：每个 workflow 240 train / 30 dev / 30 calibration。"""
    dataset = load_dataset(TYPED_DECISIONS_REPO, "all", split="train", revision=TD_REVISION)
    print(f"typed-decisions train: {len(dataset)} 案例")
    by_workflow: dict[str, list[dict]] = {}
    for row in dataset:
        by_workflow.setdefault(row["workflow"], []).append(row)

    train, dev, calibration = [], [], []
    for workflow, rows in sorted(by_workflow.items()):
        if len(rows) != 300:
            raise ValueError(f"{workflow}: expected 300 cases, got {len(rows)}")
        rows = sorted(
            rows,
            key=lambda r: hashlib.sha256(f"{SPLIT_SEED}:{r['id']}".encode()).hexdigest(),
        )
        train.extend(convert_typed_decisions(r) for r in rows[:240])
        dev.extend(convert_typed_decisions(r) for r in rows[240:270])
        calibration.extend(convert_typed_decisions(r) for r in rows[270:300])
        print(f"  {workflow}: {len(rows)} → 240 train / 30 dev / 30 calibration")
    return train, dev, calibration


def load_eval_request_keys() -> set[str]:
    """ToolACE 310 条 eval 行的请求文本指纹，训练采样要避开它们。"""
    keys = set()
    with open(EVAL_DIR / "toolace.jsonl", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            keys.add(" ".join(row["state"]["request"].lower().split()))
    return keys


def convert_toolace(row: dict) -> dict | None:
    """raw ToolACE 行 → canonical record；只保留标准格式且单工具调用的样本。"""
    system = row["system"]
    marker = "Here is a list of functions in JSON format that you can invoke:\n"
    if marker not in system:
        return None
    try:
        tools, _ = json.JSONDecoder().raw_decode(system.split(marker, 1)[1].strip())
    except json.JSONDecodeError:
        return None
    tool_names = [tool["name"] for tool in tools]
    if not 2 <= len(tool_names) <= 62 or len(set(tool_names)) != len(tool_names):
        return None

    conversations = row["conversations"]
    for index, message in enumerate(conversations):
        if message["from"] != "user" or index + 1 >= len(conversations):
            continue
        reply = conversations[index + 1]
        if reply["from"] != "assistant":
            continue
        called = [
            name
            for name in tool_names
            if re.search(r"(?:^|\[|, )" + re.escape(name) + r"\(", reply["value"])
        ]
        if len(called) != 1:
            continue
        request = message["value"]
        return {
            "id": "",
            "state": {
                "request": request,
                "tools": [
                    {"name": t["name"], "description": t.get("description", "")}
                    for t in tools
                ],
            },
            "questions": {
                "decision": {
                    "type": "choice",
                    "instructions": "Which single tool from the catalog should be called for this request?",
                    "criteria": {
                        t["name"]: t.get("description", "") for t in tools
                    },
                }
            },
            "targets": {"decision": {"label": called[0]}},
            "_request_key": " ".join(request.lower().split()),
        }
    return None


def sample_toolace() -> list[dict]:
    eval_keys = load_eval_request_keys()
    dataset = load_dataset(TOOLACE_REPO, split="train", revision=TOOLACE_REVISION)
    print(f"ToolACE train: {len(dataset)} 行")
    pool = []
    for row in dataset:
        record = convert_toolace(row)
        if record is None or record["_request_key"] in eval_keys:
            continue
        pool.append(record)
    print(f"可转换且不与 eval 重叠: {len(pool)} 条")
    random.Random(SAMPLE_SEED).shuffle(pool)
    selected = pool[:TOOLACE_TRAIN_SIZE]
    if len(selected) != TOOLACE_TRAIN_SIZE:
        raise ValueError("not enough ToolACE training records")
    for index, record in enumerate(selected):
        record["id"] = f"toolace-{index:05d}"
        del record["_request_key"]
    return selected


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"写出 {path.name}: {len(rows)} 行")


def main() -> None:
    global DATASET_DIR, EVAL_DIR, TD_REVISION, TOOLACE_REVISION
    parser = argparse.ArgumentParser(description="准备原任务数据与确定性案例切分")
    parser.add_argument("--output", type=Path, default=DATASET_DIR)
    parser.add_argument("--typed-revision", default=TD_REVISION)
    parser.add_argument("--toolace-revision", default=TOOLACE_REVISION)
    args = parser.parse_args()
    DATASET_DIR, EVAL_DIR = args.output, args.output / "eval"
    TD_REVISION, TOOLACE_REVISION = args.typed_revision, args.toolace_revision
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    download_eval_files()

    train_td, dev, calibration = split_typed_decisions()
    train_toolace = sample_toolace()

    train = train_td + train_toolace
    random.Random(SAMPLE_SEED).shuffle(train)
    write_jsonl(DATASET_DIR / "train.jsonl", train)
    write_jsonl(DATASET_DIR / "dev.jsonl", dev)
    write_jsonl(DATASET_DIR / "calibration.jsonl", calibration)
    if len(train_td) != 960 or len(dev) != 120 or len(calibration) != 120:
        raise ValueError("unexpected typed-decisions split sizes")
    id_sets = [set(r["id"] for r in rows) for rows in (train_td, dev, calibration)]
    if any(id_sets[i] & id_sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("typed-decisions split overlap")
    manifest = {
        "typed_decisions_revision": TD_REVISION, "toolace_revision": TOOLACE_REVISION,
        "benchmark_base": EVAL_BASE, "split_seed": SPLIT_SEED, "sample_seed": SAMPLE_SEED,
        "counts": {"train": len(train), "dev": len(dev), "calibration": len(calibration)},
        "sha256": {str(p.relative_to(DATASET_DIR)): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in sorted(DATASET_DIR.rglob("*.jsonl"))},
    }
    (DATASET_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
