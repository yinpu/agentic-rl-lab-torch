"""torchrun --standalone --nproc_per_node=2 tests/jev/check_ddp.py

用小随机模型检查完整批次中字段数不等的全局归约；不报告任务效果。
"""
from copy import deepcopy
from pathlib import Path
import sys

from lightning.fabric import Fabric
from lightning.fabric.strategies import DDPStrategy
from peft import LoraConfig, get_peft_model
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.jev.train import update


def main():
    torch.set_num_threads(1)
    fabric = Fabric(accelerator="cpu", devices=2, strategy=DDPStrategy(broadcast_buffers=False))
    fabric.launch()
    torch.manual_seed(42)
    model = Qwen2ForCausalLM(Qwen2Config(vocab_size=80, hidden_size=16, intermediate_size=24,
                                      num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2))
    model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", r=2, lora_alpha=2,
                                           target_modules=["q_proj", "v_proj"]))
    reference = deepcopy(model)
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.1)
    ref_optimizer = torch.optim.SGD([p for p in reference.parameters() if p.requires_grad], lr=.1)
    model, optimizer = fabric.setup(model, optimizer)
    many = {"input_ids": [1, 2, 3, 4, 5], "positions": [1, 3], "targets": [61, 62]}
    one = {"input_ids": [6, 7, 8, 9], "positions": [2], "targets": [63]}
    for batch, per_device_batch in [([many, one] * 3, 1), ([many, one], 1),
                                    ([one, many, many, one], 2), ([many, one] * 4, 2)]:
        actual = update(model, optimizer, batch, fabric, per_device_train_batch_size=per_device_batch)
        ref_optimizer.zero_grad()
        from experiments.jev.jev import decision_logits
        import torch.nn.functional as F
        loss = sum(F.cross_entropy(decision_logits(reference, e, "cpu"), torch.tensor(e["targets"]), reduction="sum") for e in batch) / sum(len(e["targets"]) for e in batch)
        loss.backward()
        ref_optimizer.step()
        expected = {"loss": loss.item()}
        assert abs(actual["loss"] - expected["loss"]) < 1e-6
        for p, q in zip(model.module.parameters(), reference.parameters(), strict=True):
            assert torch.allclose(p, q, atol=1e-6), (p - q).abs().max().item()
    fabric.print("PASS: DDP matches single-process field-mean updates for complete batches with unequal field counts")
    for logger in fabric.loggers:
        logger.finalize("success")


if __name__ == "__main__":
    main()
