import importlib.util
from pathlib import Path
from types import SimpleNamespace
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM
from peft import LoraConfig, get_peft_model
from lightning.fabric import Fabric
import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("opd_task", ROOT / "experiments/general-opd/task.py")
task = importlib.util.module_from_spec(spec)
spec.loader.exec_module(task)


def tiny():
    torch.manual_seed(3)
    return get_peft_model(Qwen2ForCausalLM(Qwen2Config(vocab_size=32, hidden_size=16,
        intermediate_size=32, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1)),
        LoraConfig(task_type="CAUSAL_LM", r=2, lora_alpha=4, target_modules="all-linear"))


def samples(model):
    rows = [{"prompt": [1, 5], "tokens": [7, 2]}, {"prompt": [1, 6, 9], "tokens": [8, 9, 2]}]
    with torch.no_grad():
        for row, logp in zip(rows, task.batch_completion_logprobs(model, rows)):
            row.update(old=logp.clone(), teacher=logp + torch.linspace(-.2, .3, len(logp)))
    return rows


def test_token_alignment():
    m = tiny().eval()
    rows = samples(m)
    for row, actual in zip(rows, task.batch_completion_logprobs(m, rows)):
        ids = torch.tensor([row["prompt"] + row["tokens"]])
        logits = m(ids).logits[0, len(row["prompt"])-1:-1].float()
        expected = logits.log_softmax(-1).gather(1, torch.tensor(row["tokens"])[:, None]).squeeze(1)
        torch.testing.assert_close(actual, expected)
    assert task.trim_completion([7, 2, 0], [2]) == ([7, 2], False)


def test_loss():
    current = torch.tensor([-1., -2.], requires_grad=True)
    old = current.detach().clone().requires_grad_()
    teacher = torch.tensor([-.5, -3.], requires_grad=True)
    loss = task.opd_loss(current, old, teacher)
    torch.testing.assert_close(loss, torch.tensor([-.5, 1.]))
    loss.sum().backward()
    torch.testing.assert_close(current.grad, torch.tensor([-.5, 1.]))
    assert old.grad is None and teacher.grad is None


def test_accumulation():
    fabric = Fabric(accelerator="cpu", devices=1)
    results = []
    for micro in [1, 2]:
        m = tiny()
        rows = samples(m)
        frozen = {k: v.clone() for k, v in m.named_parameters() if not v.requires_grad}
        before = {k: v.clone() for k, v in m.named_parameters() if v.requires_grad}
        opt = torch.optim.SGD([p for p in m.parameters() if p.requires_grad], lr=.1)
        wrapped, opt = fabric.setup(m, opt)
        task.update_policy(wrapped, opt, rows, SimpleNamespace(micro_batch_size=micro,beta=1.,max_grad_norm=100.), fabric)
        assert all(torch.equal(dict(m.named_parameters())[k], v) for k, v in frozen.items())
        assert any(not torch.equal(dict(m.named_parameters())[k], v) for k, v in before.items())
        results.append([p.detach().clone() for p in m.parameters()])
    for a, b in zip(*results):
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)


def test_split_and_tail():
    data = [{"question": q, "final_answer": "1"} for q in ["a", " a ", "b", "c", "d"]]
    train, val = task.make_split(data, 42, 1, 0)
    assert len(train) == 3 and len(val) == 1
    assert not {r['question'].strip() for r in train} & {r['question'].strip() for r in val}
    assert task.training_size(5, 2, 2) == 4
    with pytest.raises(ValueError): task.training_size(1, 2, 1)
    with pytest.raises(ValueError): task.training_size(5, 3, 2)


def test_rollout_freezes_teacher_and_cache(monkeypatch):
    student, teacher = tiny(), tiny().requires_grad_(False)
    before = {k: p.clone() for k, p in teacher.named_parameters()}
    monkeypatch.setattr(task, "prompt_ids", lambda tok, question: [1, 5])
    monkeypatch.setattr(task, "generate", lambda *args: [([7, 2], False), ([8, 9, 2], False)])
    args = SimpleNamespace(max_tokens=3, group_size=2, micro_batch_size=1)
    rows = task.rollout(student, teacher, None, [{"question": "x"}], args)
    assert len(rows) == 2
    assert all(not r[k].requires_grad for r in rows for k in ['old', 'teacher'])
    assert all(torch.equal(before[k], p) and p.grad is None for k, p in teacher.named_parameters())


def test_tokenizer_compatibility():
    class Tokenizer:
        all_special_ids = [0, 1]
        def __init__(self, vocab): self.vocab = vocab
        def get_vocab(self): return self.vocab
    task.check_tokenizers(Tokenizer({'x': 2}), Tokenizer({'x': 2}))
    with pytest.raises(ValueError):
        task.check_tokenizers(Tokenizer({'x': 2}), Tokenizer({'x': 3}))


def test_evaluation_tail_and_nested_box(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, 'task', task)
    spec = importlib.util.spec_from_file_location('opd_eval', ROOT / 'experiments/general-opd/evaluate.py')
    evaluation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluation)
    assert evaluation.extract_boxed(r'\boxed{0} then \boxed{\frac{1}{2}}') == r'\frac{1}{2}'
    assert evaluation.extract_boxed(r'\boxed{unfinished') is None
    monkeypatch.setattr(evaluation, 'prompt_ids', lambda *args: [1])
    monkeypatch.setattr(evaluation, 'generate', lambda m,t,p,*args,**kw: [([2], False) for _ in p])
    tokenizer = SimpleNamespace(decode=lambda *args, **kw: r'\boxed{1}')
    result = evaluation.evaluate(None, tokenizer, [{'question':'x', 'answer':'1'}] * 3, batch_size=2)
    assert len(result['records']) == 3
    assert result['metrics']['exact_match'] == 1


def test_prompt_is_plain_token_ids():
    from transformers import AutoTokenizer
    folder = ROOT / '.cache/models/Qwen3.5-0.8B'
    if not folder.exists():
        pytest.skip('local tokenizer is not cached')
    tokenizer = AutoTokenizer.from_pretrained(folder)
    ids = task.prompt_ids(tokenizer, 'What is 1+1?')
    assert isinstance(ids, list) and all(isinstance(i, int) for i in ids)
    assert task.SUFFIX in tokenizer.decode(ids)
