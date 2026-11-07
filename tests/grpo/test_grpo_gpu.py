"""用真实 Qwen2.5-0.5B 和 CUDA 检查奖励、更新及保存重载，不评估训练收益。

显式运行：python -m pytest tests/grpo/test_grpo_gpu.py -q -s。
使用真实 GSM8K 题目，构造四种回答以确定覆盖格式奖励的非零分支；这些回答
是检查用的固定输入，不声称来自模型自由采样，也不改变训练入口的采样方式。
"""

from datetime import datetime
import hashlib
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

from lightning.fabric import Fabric

from peft import LoraConfig, PeftModel, get_peft_model
import pytest
import torch

from experiments.grpo import train
from experiments.grpo.task import (DEFAULT_MODEL, completion_logprobs, eos_ids, generate, ground_truth,
                  load_gsm8k, load_model, make_split, rows_at, write_json)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not Path(DEFAULT_MODEL).is_dir(),
    reason="需要 CUDA GPU 和本章缓存的 Qwen2.5-0.5B 权重",
)


def frozen_hash(model):
    """按原始字节检查全部冻结参数，兼容 BF16，不仅检查少量层或参数范数。"""
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            digest.update(name.encode())
            digest.update(parameter.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


@pytest.mark.parametrize("beta,weight", [(0., 0.), (0.04, 0.), (0., 0.5), (0.04, 0.5)],
                         ids=["baseline", "kl", "format", "kl-format"])
def test_qwen_gpu_reward_update_and_reload(beta, weight, monkeypatch, tmp_path):
    fabric = Fabric(accelerator="cuda", devices=1, precision="bf16-mixed")
    fabric.launch()
    assert fabric.device.type == "cuda" and fabric.world_size == 1
    assert torch.cuda.is_bf16_supported()
    torch.cuda.reset_peak_memory_stats()
    fabric.seed_everything(42)
    data = load_gsm8k()
    split = make_split(data["train"], seed=42, validation_size=256, train_limit=1)
    row, = rows_at(data["train"], split["train"])
    assert row["index"] not in split["validation"]
    model, tokenizer = load_model(DEFAULT_MODEL, torch.bfloat16)
    model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", r=32, lora_alpha=64,
                                            target_modules="all-linear", lora_dropout=0., bias="none"))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert all("lora_" in name for name, p in model.named_parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW(trainable, lr=4e-5, betas=(0.9, 0.95), weight_decay=0)
    model, optimizer = fabric.setup(model, optimizer)
    raw_model = model.module
    initial_frozen_hash = frozen_hash(raw_model)
    initial_lora = [p.detach().cpu().clone() for p in trainable]

    answer = ground_truth(row["answer"])
    wrong_answer = str(int(answer) + 1)
    reasoning = row["answer"].split("####")[0].strip()
    # 覆盖：正确且合规、正确但不合规、错误但合规、错误且不合规。
    texts = [f"<think>\n{reasoning}\n</think>\n\\boxed{{{answer}}}",
             f"{reasoning}\n\\boxed{{{answer}}}",
             f"<think>\n{reasoning}\n</think>\n\\boxed{{{wrong_answer}}}",
             f"{reasoning}\n\\boxed{{{wrong_answer}}}"]
    completions = [(tokenizer.encode(text, add_special_tokens=False) + [eos_ids(tokenizer)[0]], False)
                   for text in texts]
    args = SimpleNamespace(max_tokens=1024, group_size=4,
                           beta=beta, format_reward_weight=weight, clip=0.2, max_grad_norm=1., micro_batch_size=2)
    # 只替换检查输入，仍执行生产代码 rollout 的奖励、优势和旧/参考概率计算。
    with monkeypatch.context() as patch, fabric.autocast():
        patch.setattr(train, "generate", lambda *a, **k: completions)
        samples = train.rollout(raw_model, tokenizer, [row], args)
    assert [s["correct"] for s in samples] == [1., 1., 0., 0.]
    assert [s["format_reward"] for s in samples] == [weight, 0., weight, 0.]
    assert [s["reward"] for s in samples] == [1. + weight, 1., weight, 0.]
    # 四种奖励组合的总体方差为 (1 + weight**2) / 4。
    std = ((1 + weight**2) / 4)**0.5
    expected_advantages = [(0.5 + weight / 2) / std, (0.5 - weight / 2) / std,
                           (-0.5 + weight / 2) / std, (-0.5 - weight / 2) / std]
    assert [s["advantage"] for s in samples] == pytest.approx(expected_advantages)
    old_copies = [s["old"].clone() for s in samples]
    for sample in samples:
        assert sample["old"].shape == (len(sample["tokens"]),)
        assert not sample["old"].requires_grad
        if beta:
            torch.testing.assert_close(sample["reference"], sample["old"], rtol=0, atol=0)
            assert not sample["reference"].requires_grad
        else:
            assert sample["reference"] is None
    with torch.no_grad(), fabric.autocast(), raw_model.disable_adapter():
        initial_reference = completion_logprobs(raw_model, samples[0]["prompt"], samples[0]["tokens"])

    # 固定同一批 old，更新两次：首步检查初始 ratio/KL，第二步实际检验策略变化。
    updates = [train.update_policy(model, optimizer, samples,
                                    args, fabric) for _ in range(2)]
    assert all(torch.isfinite(torch.tensor(list(metrics.values()))).all() for metrics in updates)
    assert all(metrics["grad_norm"] > 0 for metrics in updates)
    # grad_norm 日志是裁剪前的值；实际保留的梯度应满足总范数上限。
    clipped_grad_norm = torch.stack([p.grad.detach().float().norm() for p in trainable
                                    if p.grad is not None]).norm().item()
    assert clipped_grad_norm <= args.max_grad_norm + 1e-5
    if beta:
        # BF16 的批量前向与逐条前向可有舍入差异，初始 KL 不要求精确为零。
        assert updates[0]["kl"] >= 0
        assert updates[1]["kl"] > 0
    else:
        assert all("kl" not in metrics for metrics in updates)
    assert any(not torch.equal(p.detach().cpu(), before) for p, before in zip(trainable, initial_lora, strict=True))
    assert frozen_hash(raw_model) == initial_frozen_hash
    assert all(p.grad is None for p in raw_model.parameters() if not p.requires_grad)
    for sample, old in zip(samples, old_copies, strict=True):
        torch.testing.assert_close(sample["old"], old, rtol=0, atol=0)
        if beta:
            torch.testing.assert_close(sample["reference"], old, rtol=0, atol=0)

    raw_model.eval()
    with torch.no_grad(), fabric.autocast(), raw_model.disable_adapter():
        final_reference = completion_logprobs(raw_model, samples[0]["prompt"], samples[0]["tokens"])
    torch.testing.assert_close(final_reference, initial_reference, rtol=0, atol=0)
    adapter = tmp_path / "adapter"
    raw_model.save_pretrained(adapter)
    tokenizer.save_pretrained(adapter)
    base, restored_tokenizer = load_model(DEFAULT_MODEL, torch.bfloat16)
    restored = PeftModel.from_pretrained(base, adapter).to(fabric.device).eval()
    with torch.no_grad(), fabric.autocast():
        for sample in samples:
            current = completion_logprobs(raw_model, sample["prompt"], sample["tokens"])
            reloaded = completion_logprobs(restored, sample["prompt"], sample["tokens"])
            torch.testing.assert_close(reloaded, current, rtol=0, atol=0)
        current_tokens = generate(raw_model, tokenizer, [samples[0]["prompt"]], 32, sample=False)
        restored_tokens = generate(restored, restored_tokenizer, [samples[0]["prompt"]], 32, sample=False)
    assert current_tokens == restored_tokens
    write_json(tmp_path / "report.json", {
        "verified_at": datetime.now().astimezone().isoformat(), "model": DEFAULT_MODEL,
        "gpu": torch.cuda.get_device_name(), "precision": "bf16", "torch": torch.__version__,
        "transformers": version("transformers"), "beta": beta, "format_reward_weight": weight,
        "input_type": "Controlled GSM8K completions, not free generation",
        "question_index": row["index"], "texts": texts,
        "rewards": [s["reward"] for s in samples], "advantages": [s["advantage"] for s in samples],
        "updates": updates, "frozen_parameters_hash": initial_frozen_hash,
        "last_clipped_grad_norm": clipped_grad_norm,
        "base_weights_unchanged": True, "reference_logprobs_unchanged": True,
        "old_logprobs_unchanged": True, "lora_updated": True,
        "reload_logprobs_exact": True, "reload_greedy_tokens_exact": True,
        "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
    })
