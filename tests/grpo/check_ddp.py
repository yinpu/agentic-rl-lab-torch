"""torchrun --standalone --nproc_per_node=2 tests/grpo/check_ddp.py

使用小随机模型检查全局回答均值及空 rank；不报告任务效果。
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from peft import LoraConfig, get_peft_model
from transformers import Qwen2Config, Qwen2ForCausalLM
from experiments.grpo.task import completion_logprobs, grpo_loss


def tiny_model():
    torch.manual_seed(7)
    return Qwen2ForCausalLM(Qwen2Config(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1,
        tie_word_embeddings=True, pad_token_id=0, eos_token_id=2,
    )).eval()


def distributed_check():
    """用 torchrun 在两个 CPU 进程运行，直接验证训练函数的尾批全局归约。"""
    from types import SimpleNamespace
    from lightning.fabric import Fabric
    from lightning.fabric.strategies import DDPStrategy
    from experiments.grpo.train import pad_samples, should_update, update_policy

    fabric = Fabric(accelerator="cpu", devices=2, strategy=DDPStrategy(broadcast_buffers=False))
    fabric.launch()
    if fabric.world_size != 2:
        raise ValueError("Run this check with torchrun --nproc_per_node=2")
    args = SimpleNamespace(beta=0.04, clip=0.2, max_grad_norm=100., micro_batch_size=3)
    for prompts in [3, 1]:  # 3 题：各进程 2/1 题；1 题：一个进程完全没有真实题。
        model = get_peft_model(tiny_model(), LoraConfig(task_type="CAUSAL_LM", r=2, lora_alpha=4,
                                                       target_modules="all-linear", lora_dropout=0.))
        expected = get_peft_model(tiny_model(), LoraConfig(task_type="CAUSAL_LM", r=2, lora_alpha=4,
                                                          target_modules="all-linear", lora_dropout=0.))
        expected.load_state_dict(model.state_dict())
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(trainable, lr=0.01)
        model, optimizer = fabric.setup(model, optimizer)
        expected_optimizer = torch.optim.SGD([p for p in expected.parameters() if p.requires_grad], lr=0.01)
        all_samples = []
        for i in range(prompts):
            for tokens, advantage in [([7, 2], 0.5), ([8, 9, 2], -0.5)]:
                prompt = [4, 5 + i]
                with torch.no_grad():
                    old = completion_logprobs(expected, prompt, tokens)
                all_samples.append({"prompt": prompt, "tokens": tokens, "advantage": advantage,
                                    "old": old, "reference": old})
        samples = [s for i, s in enumerate(all_samples) if i // 2 % 2 == fabric.global_rank]
        real_count = len(samples)
        pad_samples(samples, prompts, 2, [4, 5], 2, fabric)
        # 包括本地无真实回答的进程，也必须与其他进程作出同一个更新决定。
        assert should_update(samples, 0., fabric)
        zero_samples = [sample | {"advantage": 0.} for sample in samples]
        assert not should_update(zero_samples, 0., fabric)
        assert should_update(zero_samples, args.beta, fabric)
        # 集中式完整 loss 独立计算，包含不同回答长度和冻结的 reference。
        expected.train()
        for sample in all_samples:
            current = completion_logprobs(expected, sample["prompt"], sample["tokens"])
            loss, _ = grpo_loss(current, sample["old"], sample["advantage"], sample["reference"], args.beta)
            (loss / len(all_samples)).backward()
        expected_optimizer.step()
        update_policy(model, optimizer, samples, real_count, len(all_samples), args, fabric)
        for actual, target in zip(model.module.parameters(), expected.parameters(), strict=True):
            torch.testing.assert_close(actual, target, atol=1e-6, rtol=1e-5)
        fabric.barrier()
        fabric.print(f"CPU DDP: {prompts} prompts, tail normalization matches centralized update")
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    distributed_check()
