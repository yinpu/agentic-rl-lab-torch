"""torchrun --standalone --nproc_per_node=2 tests/dapo/distributed_check.py"""
import os
import sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM
from lightning.fabric import Fabric
from lightning.fabric.strategies import DDPStrategy
from experiments.dapo.train import update_policy, cache_old, gather_groups, select_batch

def make_model():
    torch.manual_seed(17)
    return Qwen2ForCausalLM(Qwen2Config(vocab_size=23, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2)).eval()

def main():
    f = Fabric(accelerator="cpu", devices=int(os.environ["WORLD_SIZE"]),
               strategy=DDPStrategy(broadcast_buffers=False))
    f.launch()
    samples = [{"prompt": [2,3], "tokens": [4,5,6], "advantage": 1.},
               {"prompt": [7], "tokens": [8], "advantage": -1.},
               {"prompt": [2], "tokens": [9,10], "advantage": .5},
               {"prompt": [3], "tokens": [11,12,13,14], "advantage": -.5}]
    reference = make_model()
    cache_old(reference, samples, 2)
    # 在分布式进程内直接算完整 batch 的参考梯度，避免单设备 Fabric 继承进程环境。
    opt_ref = torch.optim.SGD(reference.parameters(), lr=.01)
    from experiments.dapo.task import batch_completion_logprobs, token_loss
    loss = sum(token_loss(p, s["old"], s["advantage"]).sum()
               for s,p in zip(samples, batch_completion_logprobs(reference, samples))) / 10
    loss.backward()
    torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.)
    opt_ref.step()
    model = make_model()
    opt = torch.optim.SGD(model.parameters(), lr=.01)
    model, opt = f.setup(model, opt)
    update_policy(model, opt, samples[f.global_rank::f.world_size],
                  SimpleNamespace(micro_batch_size=1, max_grad_norm=1.), f)
    for actual, expected in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    good = [{"correct": True}, {"correct": False}]
    bad = [{"correct": False}, {"correct": False}]
    groups = gather_groups([(f.global_rank, good if f.global_rank == 0 else bad)])
    assert not select_batch(groups, 2)
    assert select_batch(groups, 1) == [good]
    if f.is_global_zero:
        print("DDP token normalization, accumulation and common skip: passed")
if __name__ == "__main__":
    main()
