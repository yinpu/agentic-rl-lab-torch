"""不下载权重的算法检查。随机小网络只用于验证数学和模型接口，不作为章节实验。"""

import pytest
import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import Qwen2Config, Qwen2ForCausalLM

from experiments.grpo.task import (TASK_INSTRUCTION, completion_logprobs,
                  extract_boxed, follows_think_format, ground_truth, group_advantages,
                  grpo_loss, make_split, prompt_ids, reward, training_rewards, trim_completion)


def tiny_model():
    torch.manual_seed(7)
    return Qwen2ForCausalLM(Qwen2Config(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1,
        tie_word_embeddings=True, pad_token_id=0, eos_token_id=2,
    )).eval()


def test_format_monitor_does_not_change_answer_reward():
    correct = r"<think>1 + 1 = 2.</think>\boxed{2}"
    assert follows_think_format(correct) and reward(correct, "2") == 1
    assert follows_think_format(r"<think>Wrong reasoning.</think>\boxed{3}")
    assert reward(r"<think>Wrong reasoning.</think>\boxed{3}", "2") == 0
    assert reward(r"\boxed{2}", "2") == 1
    for text in [r"\boxed{2}", r"<think></think>\boxed{2}",
                 r"<think>unfinished \boxed{2}", r"<think>work</think>",
                 r"<think><think>nested</think></think>\boxed{2}"]:
        assert not follows_think_format(text)


@pytest.mark.parametrize("weight", [0.0, 0.5])
@pytest.mark.parametrize("text,correct,formatted", [
    (r"<think>1 + 1 = 2.</think>\boxed{2}", 1., True),
    (r"\boxed{2}", 1., False),
    (r"<think>Wrong reasoning.</think>\boxed{3}", 0., True),
    (r"\boxed{3}", 0., False),
])
def test_training_reward_keeps_accuracy_and_format_separate(text, correct, formatted, weight):
    scores = training_rewards(text, "2", weight)
    assert scores == {"correct": correct, "think_format": formatted,
                      "format_reward": weight if formatted else 0.,
                      "reward": correct + (weight if formatted else 0.)}
    # 训练可以奖励答错但合规的输出；独立评测的正确性仍为 0。
    assert reward(text, "2") == correct


@pytest.mark.parametrize("weight", ["-0.5", "nan", "inf"])
def test_cli_rejects_invalid_format_reward_weight(monkeypatch, weight):
    from experiments.grpo import train
    monkeypatch.setattr("sys.argv", ["train.py", "--format-reward-weight", weight])
    with pytest.raises(SystemExit) as error:
        train.parse_args()
    assert error.value.code == 2


class RecordingTokenizer:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        self.text = "\n\n".join(m["role"] + ": " + m["content"] for m in messages) + "\nAssistant:\n"
        return self.text

    def encode(self, text, **kwargs):
        return [4, 5, 6]

    def decode(self, tokens, skip_special_tokens=True):
        if list(tokens) == [4, 5, 6]:
            return self.text
        return r"<think>1 + 1 = 2.</think>\boxed{2}" if tokens[0] == 7 else r"\boxed{3}"


def test_prompt_has_only_current_problem():
    tokenizer = RecordingTokenizer()
    prompt_ids(tokenizer, "CURRENT PROBLEM")
    messages, kwargs = tokenizer.calls[-1]
    assert messages == [{"role": "user", "content": TASK_INSTRUCTION + "\n\nProblem:\nCURRENT PROBLEM"}]
    assert kwargs == {"tokenize": False, "add_generation_prompt": True}
    assert not tokenizer.text.endswith("<think>\n")


@pytest.mark.parametrize("beta", [0., 0.04])
@pytest.mark.parametrize("weight", [0., 0.5])
def test_rollout_combines_format_advantages_and_optional_fixed_reference(monkeypatch, beta, weight):
    from types import SimpleNamespace
    from experiments.grpo import train

    model = get_peft_model(tiny_model(), LoraConfig(task_type="CAUSAL_LM", r=2, lora_alpha=4,
                                                   target_modules="all-linear", lora_dropout=0.))
    args = SimpleNamespace(max_tokens=16, group_size=2, standardize_advantages=False,
                           beta=beta, format_reward_weight=weight)
    # 两条回答都错，只有第一条符合格式；因此只有格式奖励能产生组内差异。
    row = {"index": 0, "question": "What is 4 + 5?", "answer": "#### 9"}
    monkeypatch.setattr(train, "generate", lambda *a, **k: [([7, 2], False), ([8, 2], False)])
    samples = train.rollout(model, RecordingTokenizer(), [row], args)
    assert [s["correct"] for s in samples] == [0., 0.]
    assert [s["reward"] for s in samples] == [weight, 0.]
    assert [s["format_reward"] for s in samples] == [weight, 0.]
    assert [s["think_format"] for s in samples] == [True, False]
    assert [s["advantage"] for s in samples] == [weight / 2, -weight / 2]
    for sample in samples:
        assert not sample["old"].requires_grad
        if beta:
            assert not sample["reference"].requires_grad
            torch.testing.assert_close(sample["reference"], sample["old"])
        else:
            assert sample["reference"] is None


def test_training_and_evaluation_share_prompt_and_save_each_answer(monkeypatch, tmp_path):
    import json
    from contextlib import nullcontext
    from types import SimpleNamespace
    from experiments.grpo import evaluate
    from experiments.grpo import train

    tokenizer = RecordingTokenizer()
    model = tiny_model()
    row = {"index": 0, "question": "What is 1 + 1?", "answer": "#### 2"}
    args = SimpleNamespace(max_tokens=16, group_size=2, standardize_advantages=False,
                           beta=0., format_reward_weight=0.)
    monkeypatch.setattr(train, "generate", lambda *a, **k: [([7, 2], False), ([8, 2], False)])
    samples = train.rollout(model, tokenizer, [row], args)
    assert [s["advantage"] for s in samples] == [0.5, -0.5]
    monkeypatch.setattr(evaluate, "generate", lambda *a, **k: [([7, 2], False)])
    fabric = SimpleNamespace(global_rank=0, world_size=1,
                                  is_global_zero=True, autocast=nullcontext,
                                  barrier=lambda: None)
    metrics = evaluate.evaluate_model(model, tokenizer, [row], fabric, 16, tmp_path / "eval.json")
    assert tokenizer.calls[0] == tokenizer.calls[1]
    assert metrics["accuracy"] == metrics["think_format_rate"] == 1.
    final = json.loads((tmp_path / "eval.json").read_text())
    partial = [json.loads(line) for line in (tmp_path / "eval.rank-0.jsonl").read_text().splitlines()]
    assert final["results"] == partial
    assert "CURRENT PROBLEM" not in partial[0]["prompt"]
    assert TASK_INSTRUCTION in partial[0]["prompt"]


def test_boxed_answer_reward():
    assert extract_boxed(r"first \boxed{2}, final \boxed{1,000.}") == "1,000."
    assert ground_truth("reasoning\n#### 1,000") == "1000"
    assert reward(r"\boxed{1,000.}", "1000") == 1
    assert reward("1000", "1000") == 0
    assert reward(r"\boxed{1000.0}", "1000") == 0
    assert reward(r"\boxed{2} then \boxed{3}", "2") == 0
    with pytest.raises(ValueError):
        ground_truth("missing final answer")


def test_group_advantages_and_degenerate_groups():
    torch.testing.assert_close(group_advantages([1, 0, 1, 0]), torch.tensor([0.5, -0.5, 0.5, -0.5]))
    torch.testing.assert_close(group_advantages([1, 0, 1, 0], True), torch.tensor([1., -1., 1., -1.]))
    for scores in [[0, 0], [1, 1]]:
        for standardize in [False, True]:
            torch.testing.assert_close(group_advantages(scores, standardize), torch.zeros(2))


def test_positive_and_negative_advantage_gradient():
    current = torch.tensor([-1., -2.], requires_grad=True)
    loss, _ = grpo_loss(current, current.detach(), 1.)
    loss.backward()
    assert torch.all(current.grad < 0)  # 梯度下降会提高 logprob。
    current.grad = None
    loss, _ = grpo_loss(current, current.detach(), -1.)
    loss.backward()
    assert torch.all(current.grad > 0)


@pytest.mark.parametrize("ratio,advantage,clipped", [(1.3, 1., True), (0.7, -1., True),
                                                  (1.3, -1., False), (0.7, 1., False)])
def test_clip_uses_advantage_sign(ratio, advantage, clipped):
    old = torch.tensor([-2.])
    current = (old + torch.tensor(ratio).log()).requires_grad_()
    loss, metrics = grpo_loss(current, old, advantage)
    loss.backward()
    assert (current.grad.item() == 0) == clipped
    assert metrics["clip_fraction"] == float(clipped)


def test_kl_zero_and_pull_toward_reference():
    reference = torch.tensor([-2.])
    current = reference.clone().requires_grad_()
    loss, metrics = grpo_loss(current, current.detach(), 0., reference, beta=0.04)
    loss.backward()
    assert metrics["kl"] == 0 and current.grad.item() == 0
    current = torch.tensor([-1.], requires_grad=True)
    loss, metrics = grpo_loss(current, current.detach(), 0., reference, beta=0.04)
    loss.backward()
    assert metrics["kl"] > 0 and current.grad.item() > 0
    assert reference.grad is None


def test_eos_is_included_and_padding_is_excluded():
    assert trim_completion([5, 2, 0, 0], [2]) == ([5, 2], False)
    assert trim_completion([5, 6, 7], [2]) == ([5, 6, 7], True)
    assert trim_completion([0, 5, 2, 0], [2]) == ([0, 5, 2], False)


def test_autoregressive_alignment_with_actual_model_class():
    model = tiny_model()
    prompt, completion = [4, 5, 6], [7, 8, 2]
    with torch.no_grad():
        actual = completion_logprobs(model, prompt, completion)
        # 独立的完整 teacher-forcing 计算，只取预测 completion 的位置。
        ids = torch.tensor([prompt + completion])
        logits = model(ids, use_cache=False).logits[0, len(prompt)-1:-1]
        expected = logits.log_softmax(-1).gather(-1, torch.tensor(completion)[:, None]).squeeze(-1)
    torch.testing.assert_close(actual, expected)
    assert len(actual) == len(completion)


def test_lora_update_fixed_reference_and_save_reload(tmp_path):
    base = tiny_model()
    original_state = {key: value.clone() for key, value in base.state_dict().items()}
    model = get_peft_model(base, LoraConfig(task_type="CAUSAL_LM", r=2, lora_alpha=4,
                                          target_modules="all-linear", lora_dropout=0.))
    frozen = {name: p.clone() for name, p in model.named_parameters() if not p.requires_grad}
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    prompt, completion = [4, 5, 6], [7, 8, 2]
    with torch.no_grad():
        old = completion_logprobs(model, prompt, completion)
        with model.disable_adapter():
            reference = completion_logprobs(model, prompt, completion)
    torch.testing.assert_close(old, reference)
    model.train()
    current = completion_logprobs(model, prompt, completion)
    loss, metrics = grpo_loss(current, old, 1., reference, beta=0.04)
    assert metrics["ratio_mean"] == pytest.approx(1., abs=1e-6)
    assert metrics["kl"] == pytest.approx(0., abs=1e-6)
    loss.backward()
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    optimizer.step()
    for name, param in model.named_parameters():
        if name in frozen:
            torch.testing.assert_close(param, frozen[name], rtol=0, atol=0)
    model.eval()
    with torch.no_grad(), model.disable_adapter():
        torch.testing.assert_close(completion_logprobs(model, prompt, completion), reference)
    model.save_pretrained(tmp_path)
    new_base = tiny_model()
    new_base.load_state_dict(original_state)
    reloaded = PeftModel.from_pretrained(new_base, tmp_path).eval()
    with torch.no_grad():
        torch.testing.assert_close(completion_logprobs(reloaded, prompt, completion),
                                   completion_logprobs(model, prompt, completion))


def test_dataset_split_is_reproducible_and_disjoint():
    dataset = Dataset.from_list([{"question": str(i), "answer": f"#### {i}"} for i in range(20)])
    split = make_split(dataset, validation_size=4, train_limit=3)
    assert split == make_split(dataset, validation_size=4, train_limit=3)
    assert not set(split["train"]) & set(split["validation"])
    assert split["train"] == sorted(split["train"])
    assert len(split["train"]) == 3 and len(split["validation"]) == 4



@pytest.mark.parametrize("micro", [1, 2, 3])
@pytest.mark.parametrize("beta", [0., .04])
def test_batch_update_matches_sequence_mean(micro, beta):
    from copy import deepcopy
    from types import SimpleNamespace
    from lightning.fabric import Fabric
    from experiments.grpo.task import batch_completion_logprobs
    from experiments.grpo.train import update_policy
    model = tiny_model()
    reference = deepcopy(model)
    samples = []
    for prompt, tokens, advantage in [([4, 5], [7, 2], .5), ([4], [8, 9, 2], -.5), ([4, 6, 5], [2], 0.)]:
        old = completion_logprobs(reference, prompt, tokens).detach() + .1
        samples.append(dict(prompt=prompt, tokens=tokens, advantage=advantage, old=old, reference=old - .2))
    for current, sample in zip(batch_completion_logprobs(model, samples), samples, strict=True):
        torch.testing.assert_close(current, completion_logprobs(reference, sample["prompt"], sample["tokens"]))
    for sample in samples:
        current = completion_logprobs(reference, sample["prompt"], sample["tokens"])
        loss, _ = grpo_loss(current, sample["old"], sample["advantage"], sample["reference"], beta)
        (loss / 3).backward()
    torch.optim.SGD(reference.parameters(), lr=.01).step()
    fabric = Fabric(accelerator="cpu", devices=1)
    fabric.launch()
    optimizer = torch.optim.SGD(model.parameters(), lr=.01)
    model, optimizer = fabric.setup(model, optimizer)
    update_policy(model, optimizer, samples,
                  SimpleNamespace(beta=beta, clip=.2, max_grad_norm=100., micro_batch_size=micro), fabric)
    for p, q in zip(model.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(p.grad, q.grad, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(p, q, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("count,batch_size,world_size,expected", [
    (7, 4, 1, 4), (7, 4, 2, 4), (8, 4, 2, 8),
])
def test_training_size_uses_complete_batches(count, batch_size, world_size, expected):
    from experiments.grpo.train import training_size
    assert training_size(count, batch_size, world_size) == expected


@pytest.mark.parametrize("count,batch_size,world_size", [(3, 4, 1), (3, 4, 2), (7, 3, 2), (0, 4, 1)])
def test_training_size_rejects_unusable_batches(count, batch_size, world_size):
    from experiments.grpo.train import training_size
    with pytest.raises(ValueError):
        training_size(count, batch_size, world_size)
