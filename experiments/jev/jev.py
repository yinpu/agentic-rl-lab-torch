"""Jev 决策：编译决策 schema → 定位 marker → 模型前向 → 候选概率。

编码逻辑改编自 datawhalechina/agentic-rl-lab 的 pytrio-jev/data.py（Apache-2.0）。
使用已有单 token marker，不扩展词表；训练仅监督 marker 前一位置。
直接运行本文件可对 canonical JSONL 执行一次前向决策。
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM_PROMPT = (
    "You are a careful decision assistant. Use the state and decision schema in "
    "the user message to make the requested decisions. For every field, choose "
    "exactly one answer symbol (e.g. A, B, C, ...) from its listed options and "
    "return one valid JSON object mapping each field name to its chosen symbol. "
    "Use the field names and symbols exactly as given. Do not include "
    "explanations, Markdown, or extra text."
)
ANSWER_SYMBOLS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
MARKER = "decision"
NOUL_YES = "The answer is yes (affirmative, or align with the claim)."
NOUL_NO = "The answer is no (negative, or disagree with the claim)."
TEMPLATE_PATH = Path(__file__).resolve().parent / "templates" / "qwen3_5_jev.jinja"


def question_options(question: dict) -> list[tuple[str, str]]:
    """按题型展开为有序的 (value, description) 列表。"""
    kind = question["type"]
    criteria = question.get("criteria")
    if kind == "choice":
        return [(str(k), str(v)) for k, v in criteria.items()]
    if kind == "score":
        if isinstance(criteria, list):
            return [(str(i), str(v)) for i, v in enumerate(criteria)]
        return [(str(k), str(v)) for k, v in criteria.items()]
    if kind == "noul":
        descriptions = criteria or {}
        yes = next(
            (str(v) for k, v in descriptions.items() if str(k).lower() in {"yes", "true", "1"}),
            NOUL_YES,
        )
        no = next(
            (str(v) for k, v in descriptions.items() if str(k).lower() in {"no", "false", "0"}),
            NOUL_NO,
        )
        return [("no", no), ("yes", yes)]
    raise ValueError(f"unsupported question type: {kind!r}")


def normalize_label(question: dict, label: object) -> str:
    """noul 的 true/false/1/0 归一到 yes/no，其余题型转字符串。"""
    if question["type"] == "noul":
        value = str(label).lower()
        if value not in {"yes", "true", "1", "no", "false", "0"}:
            raise ValueError(f"invalid noul label: {label!r}")
        return "yes" if value in {"yes", "true", "1"} else "no"
    return str(label)


def compile_record(record: dict, include_targets: bool = True) -> dict:
    """把一条 canonical record 编译成 messages 和符号映射。

    返回 fields / options（field → [(symbol, value, description)]）/ targets（field → 符号）。
    """
    questions = record["questions"]
    if not questions:
        raise ValueError("questions cannot be empty")
    fields: list[str] = []
    options: dict[str, list[tuple[str, str, str]]] = {}
    targets: dict[str, str] = {}
    schema_lines: list[str] = []
    for field, question in questions.items():
        opts = question_options(question)
        if not 1 <= len(opts) <= len(ANSWER_SYMBOLS):
            raise ValueError(f"{field}: expected 1..62 candidates, got {len(opts)}")
        field = str(field)
        fields.append(field)
        symbols = ANSWER_SYMBOLS[: len(opts)]
        options[field] = [
            (symbol, value, description)
            for symbol, (value, description) in zip(symbols, opts)
        ]
        schema_lines.append(f"{field}: {question['instructions']}")
        for symbol, value, description in options[field]:
            schema_lines.append(f"    {symbol} = {value}: {description}")
        if include_targets:
            label = normalize_label(question, record["targets"][field]["label"])
            values = [value for _, value, _ in options[field]]
            targets[field] = symbols[values.index(label)]

    state = json.dumps(record.get("state"), ensure_ascii=False, indent=2)
    user_text = (
        "Return one answer for every field using the supplied answer symbols.\n\n"
        f"## State\n{state}\n"
        "## Decision schema\n" + "\n".join(schema_lines)
    )
    skeleton = json.dumps(dict.fromkeys(fields, MARKER), ensure_ascii=False, indent=4)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": skeleton},
    ]
    return {
        "messages": messages,
        "fields": fields,
        "options": options,
        "targets": targets if include_targets else None,
    }


class Encoder:
    """绑定 tokenizer 与钉死的 chat template，负责 tokenization 和 marker 定位。"""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.template = TEMPLATE_PATH.read_text()
        marker_ids = tokenizer.encode(MARKER, add_special_tokens=False)
        if len(marker_ids) != 1:
            raise ValueError("marker 必须是单 token")
        self.marker_id = marker_ids[0]
        self.symbol_ids = {}
        for symbol in ANSWER_SYMBOLS:
            ids = tokenizer.encode(symbol, add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError(f"answer symbol must be one token: {symbol!r}")
            self.symbol_ids[symbol] = ids[0]
        if len(set(self.symbol_ids.values())) != len(ANSWER_SYMBOLS):
            raise ValueError("answer symbols must have distinct token IDs")

    def encode(self, record: dict, include_targets: bool = True) -> dict:
        """渲染完整训练/推理文本，返回 token ids 与各字段的 marker 索引。

        marker 索引靠"prefix 之外、骨架之内"定位，不扫全文，因此 instructions 或
        state 里出现单词 decision 也不会误判。
        """
        compiled = compile_record(record, include_targets)
        messages = compiled["messages"]
        ids = self.tokenizer.encode(
            self.tokenizer.apply_chat_template(
                messages, chat_template=self.template, tokenize=False
            ),
            add_special_tokens=False,
        )
        prefix_ids = self.tokenizer.encode(
            self.tokenizer.apply_chat_template(
                messages[:-1],
                chat_template=self.template,
                tokenize=False,
                add_generation_prompt=True,
            ),
            add_special_tokens=False,
        )
        if ids[:len(prefix_ids)] != prefix_ids:
            raise ValueError("prefix/skeleton 边界 tokenization 不稳定")
        # 按 JSON value 的字符范围定位；字段名、state 中的 decision 不参与匹配。
        # offset_mapping 与实际整段 tokenization 一致，也能处理引号和转义字段名。
        text = self.tokenizer.apply_chat_template(
            messages, chat_template=self.template, tokenize=False
        )
        prefix = self.tokenizer.apply_chat_template(
            messages[:-1], chat_template=self.template, tokenize=False,
            add_generation_prompt=True,
        )
        tokenized = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        if tokenized["input_ids"] != ids:
            raise ValueError("offset tokenization differs from model input")
        spans = [
            (len(prefix) + m.start(1), len(prefix) + m.end(1))
            for m in re.finditer(r': "(decision)"', messages[-1]["content"])
        ]
        if len(spans) != len(compiled["fields"]):
            raise ValueError("JSON marker count mismatch")
        markers = {}
        for field, span in zip(compiled["fields"], spans, strict=True):
            hits = [i for i, offset in enumerate(tokenized["offset_mapping"]) if tuple(offset) == span]
            if len(hits) != 1 or ids[hits[0]] != self.marker_id:
                raise ValueError(f"{field}: marker is not a standalone token in context")
            markers[field] = hits[0]
        return {
            **compiled,
            "ids": ids,
            "markers": markers,
        }

    def symbol_token_id(self, symbol: str) -> int:
        return self.symbol_ids[symbol]

    def build_example(self, record: dict, max_length: int = 0) -> dict:
        """输入保留 marker；positions 是用于预测答案符号的 m-1 位置。"""
        encoded = self.encode(record)
        ids = encoded["ids"]
        check_length(ids, max_length, record.get("id"))
        return {
            "input_ids": ids[:-1],
            "positions": [encoded["markers"][f] - 1 for f in encoded["fields"]],
            "targets": [self.symbol_token_id(encoded["targets"][f]) for f in encoded["fields"]],
        }


def check_length(ids, max_length, record_id):
    if max_length and len(ids) > max_length:
        raise ValueError(f"record {record_id}: {len(ids)} tokens exceeds --max-length {max_length}; "
                         "increase the limit; truncation would lose decision fields")


def load_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


DEFAULT_MODEL = "Qwen/Qwen3.5-0.8B"


def model_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--model", default=DEFAULT_MODEL, help="模型 ID 或本地模型目录")
    parser.add_argument("--adapter", type=Path, default=None, help="本地 LoRA adapter")
    parser.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-length", type=int, default=8192, help="超长报错，不截断；0 不设上限")


def load_model(model_name, precision="bf16", adapter=None):
    dtype = torch.bfloat16 if precision == "bf16" else torch.float32
    # Transformers 5 的 AutoModelForCausalLM 为 Qwen3.5 加载文本骨干，跳过视觉塔。
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, attn_implementation="sdpa")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    model.config.use_cache = False
    return model, tokenizer


def decision_logits(model, example, device):
    """只对 m-1 隐状态做 lm_head，仍保留全词表 CE 的分母。"""
    return batch_decision_logits(model, [example], device)[0]


def batch_decision_logits(model, examples, device):
    """批量右补齐；仅投影 marker 前一位置的并集，再取各案例实际字段。"""
    width = max(len(e["input_ids"]) for e in examples)
    ids = torch.zeros((len(examples), width), dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    selected = torch.tensor(sorted({p for e in examples for p in e["positions"]}), device=device)
    for i, example in enumerate(examples):
        n = len(example["input_ids"])
        ids[i, :n] = torch.tensor(example["input_ids"], device=device)
        mask[i, :n] = 1
    logits = model(input_ids=ids, attention_mask=mask, use_cache=False,
                   logits_to_keep=selected).logits
    field_logits = []
    for i, example in enumerate(examples):
        positions = torch.tensor(example["positions"], device=device)
        columns = torch.searchsorted(selected, positions)
        field_logits.append(logits[i, columns].float())
    return field_logits


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
