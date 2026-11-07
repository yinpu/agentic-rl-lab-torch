"""本章的模型加载与 marker 前一位置的全词表 logits。"""
import argparse
import hashlib
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

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
    ids = torch.tensor([example["input_ids"]], device=device)
    positions = torch.tensor(example["positions"], device=device)
    return model(input_ids=ids, use_cache=False, logits_to_keep=positions).logits[0].float()


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



def identity(model_name, adapter):
    name = str(Path(model_name).resolve()) if Path(model_name).exists() else model_name
    result = {"model": name, "adapter_sha256": None}
    if adapter:
        result["adapter_sha256"] = hashlib.sha256((Path(adapter) / "adapter_model.safetensors").read_bytes()).hexdigest()
    return result
