"""torchrun --standalone --nproc_per_node=2 experiments/jev/check_ddp.py

用小随机模型检查字段数不等、不能整除的 batch、空 rank 占位的归约；不报告任务效果。
"""
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
import sys

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from peft import LoraConfig, get_peft_model
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "experiments.jev"

from .train import update


class SingleProcess:
    num_processes = 1
    process_index = 0
    device = torch.device("cpu")
    no_sync = staticmethod(lambda model: nullcontext())
    backward = staticmethod(lambda loss: loss.backward())
    reduce = staticmethod(lambda value, reduction: value)
    clip_grad_norm_ = staticmethod(torch.nn.utils.clip_grad_norm_)


def main():
    torch.set_num_threads(1)
    accelerator = Accelerator(cpu=True, kwargs_handlers=[DistributedDataParallelKwargs(broadcast_buffers=False)])
    torch.manual_seed(42)
    model = Qwen2ForCausalLM(Qwen2Config(vocab_size=80, hidden_size=16, intermediate_size=24,
                                      num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2))
    model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", r=2, lora_alpha=2,
                                           target_modules=["q_proj", "v_proj"]))
    reference = deepcopy(model)
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.1)
    ref_optimizer = torch.optim.SGD([p for p in reference.parameters() if p.requires_grad], lr=.1)
    model, optimizer = accelerator.prepare(model, optimizer)
    many = {"input_ids": [1, 2, 3, 4, 5], "positions": [1, 3], "targets": [61, 62]}
    one = {"input_ids": [6, 7, 8, 9], "positions": [2], "targets": [63]}
    for batch in [[many, one, many], [one], [many, one]]:
        actual = update(model, optimizer, batch, accelerator)
        expected = update(reference, ref_optimizer, batch, SingleProcess())
        assert abs(actual["loss"] - expected["loss"]) < 1e-6
        for p, q in zip(accelerator.unwrap_model(model).parameters(), reference.parameters(), strict=True):
            assert torch.allclose(p, q, atol=1e-6), (p - q).abs().max().item()
    accelerator.print("PASS: DDP matches single-process field-mean updates for uneven and singleton batches")
    accelerator.end_training()


if __name__ == "__main__":
    main()
