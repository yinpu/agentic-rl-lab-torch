import torch
import pytest
from experiments.dapo.task import (group_advantages, token_loss, soft_overlong_penalty,
    score_answer, trim_completion, batch_completion_logprobs)
from experiments.dapo.train import select_batch, training_size
from experiments.dapo.prepare_data import deduplicate_rows, split_records


def test_clipping_gradient():
    ratio = torch.tensor([1.4, .7, 1.1, .9], requires_grad=True)
    loss = token_loss(ratio.log(), torch.zeros(4), torch.tensor([1., -1., 1., -1.]))
    torch.testing.assert_close(loss, torch.tensor([-1.28, .8, -1.1, .9]))
    loss.sum().backward()
    torch.testing.assert_close(ratio.grad, torch.tensor([0., 0., -1., 1.]))


def test_rewards_and_group_filter():
    a = group_advantages([1, -1, 1, -1])
    torch.testing.assert_close(a, torch.tensor([1., -1., 1., -1.]) * (3/4)**.5)
    assert not group_advantages([0, 0]).any()
    # 长度奖励有差异但全错，仍然过滤。
    bad = [{"correct": False, "reward": -1}, {"correct": False, "reward": -2}]
    good = [{"correct": True}, {"correct": False}]
    assert select_batch([(0, bad), (1, good)], 2) == []
    assert select_batch([(0, bad), (1, good)], 1) == [good]
    for length, expected in [(3072, 0), (3584, -.5), (4096, -1), (5000, -1)]:
        assert soft_overlong_penalty(length, max_tokens=4096, overlong_cache=1024) == expected
    score = score_answer(r"answer \boxed{\frac{1}{2}}", "0.5", completion_tokens=9,
                         max_tokens=10, overlong_cache=2, enable_overlong=True)
    assert score.correct and score.shaped_reward == .5


def test_mask_alignment_and_microbatch():
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(3)
    model = Qwen2ForCausalLM(Qwen2Config(vocab_size=23, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2)).eval()
    samples = [{"prompt": [2,3], "tokens": [4,5,6]}, {"prompt": [7], "tokens": [8]}]
    actual = batch_completion_logprobs(model, samples)
    for sample, logp in zip(samples, actual):
        ids = torch.tensor([sample["prompt"] + sample["tokens"][:-1]])
        logits = model(ids).logits[0, len(sample["prompt"])-1:].float()
        expected = logits.log_softmax(-1).gather(-1, torch.tensor(sample["tokens"])[:,None]).squeeze(-1)
        torch.testing.assert_close(logp, expected)
    old = [p.detach() for p in actual]
    (-torch.cat(actual).sum()/4).backward()
    expected_grads = [p.grad.clone() for p in model.parameters()]
    model.zero_grad()
    for sample, prev in zip(samples, old):
        logp = batch_completion_logprobs(model, [sample])[0]
        (token_loss(logp, prev, 1).sum()/4).backward()
    for p, expected in zip(model.parameters(), expected_grads):
        torch.testing.assert_close(p.grad, expected, atol=1e-6, rtol=1e-5)
    assert trim_completion([4, 2, 0], [2]) == ([4,2], False)
    assert trim_completion([4, 5], [2]) == ([4,5], True)


def test_data_and_full_batch():
    def row(q, a):
        return {"prompt": [{"content": q}], "reward_model": {"ground_truth": a}}
    from experiments.dapo.prepare_data import normalize_row
    assert normalize_row(0, row("zero", 0))["answer"] == "0"
    records, stats = deduplicate_rows([row("x", "1"), row(" x ", "1"), row("y", "2"),
                                      row("y", "3"), row("z", "4")])
    assert len(records) == 2 and stats.conflicting_questions == 1
    train, dev = split_records(records, dev_size=1, seed=42)
    assert train[0]["question"] != dev[0]["question"]
    assert training_size(11, 4, 2) == 8
    with pytest.raises(ValueError):
        training_size(3, 4, 2)
    with pytest.raises(ValueError):
        training_size(12, 3, 2)
