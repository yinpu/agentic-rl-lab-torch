import copy
from types import SimpleNamespace
import pytest
import torch
from lightning.fabric import Fabric
from transformers import Qwen2Config, Qwen2ForCausalLM
from experiments.gspo.task import (sequence_objective, group_advantages, score_answer,
                                   trim_completion, batch_completion_logprobs)
from experiments.gspo.train import cache_old, update_policy, training_size


def make_model():
    torch.manual_seed(17)
    return Qwen2ForCausalLM(Qwen2Config(vocab_size=23, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2)).eval()


def samples():
    return [dict(prompt=[2,3], tokens=[4,5,6], advantage=1.),
            dict(prompt=[7], tokens=[8], advantage=-1.),
            dict(prompt=[2], tokens=[9,10], advantage=0.),
            dict(prompt=[3], tokens=[11,12,13,14], advantage=0.)]


def test_clipping_and_length():
    for ratio, advantage, expected, grad in [(1.01,1,-1.0004,0),(.99,-1,.9997,0),
                                             (.99,1,-.99,-1),(1.01,-1,1.01,1)]:
        x = torch.tensor(ratio, requires_grad=True)
        loss, _, _ = sequence_objective(x.log().repeat(3), torch.zeros(3), advantage)
        torch.testing.assert_close(loss, torch.tensor(expected))
        loss.backward()
        torch.testing.assert_close(x.grad, torch.tensor(float(grad)))
    x = torch.tensor([.02, -.02], requires_grad=True)
    loss, ratio, _ = sequence_objective(x, torch.zeros(2), 1.)
    torch.testing.assert_close(ratio, torch.tensor(1.))
    loss.backward()
    torch.testing.assert_close(x.grad, torch.tensor([-.5, -.5]))


def test_alignment_accumulation_and_fixed_old():
    data = samples()
    m = make_model()
    for s,p in zip(data, batch_completion_logprobs(m, data)):
        logits = m(torch.tensor([s['prompt']+s['tokens'][:-1]])).logits[0,len(s['prompt'])-1:]
        expected = logits.log_softmax(-1).gather(-1,torch.tensor(s['tokens'])[:,None]).squeeze(-1)
        torch.testing.assert_close(p,expected)
    cache_old(m, data, 2)
    before = [s['old'].clone() for s in data]
    f = Fabric(accelerator='cpu', devices=1)
    f.launch()
    full, micro = make_model(), make_model()
    opt1, opt2 = torch.optim.SGD(full.parameters(),lr=.0001), torch.optim.SGD(micro.parameters(),lr=.0001)
    full,opt1 = f.setup(full,opt1)
    micro,opt2 = f.setup(micro,opt2)
    for _ in range(4):
        for model,opt,size in [(full,opt1,4),(micro,opt2,1)]:
            update_policy(model,opt,data,SimpleNamespace(micro_batch_size=size,max_grad_norm=1.),f)
        for p,q in zip(full.parameters(),micro.parameters()):
            torch.testing.assert_close(p,q,atol=1e-6,rtol=1e-5)
    for s,old in zip(data,before):
        torch.testing.assert_close(s['old'],old,atol=0,rtol=0)
    assert any(not torch.equal(p,q) for p,q in zip(m.parameters(),full.parameters()))
    state = copy.deepcopy(opt1.state_dict())
    zero = [dict(s,advantage=0.) for s in data]
    assert update_policy(full,opt1,zero,SimpleNamespace(micro_batch_size=1,max_grad_norm=1.),f)['skipped']
    assert opt1.state_dict() == state


def test_rewards_and_batches():
    assert score_answer(r'\boxed{\frac{1}{2}}','0.5')['reward'] == 1
    assert score_answer('0.5','0.5')['reward'] == 0
    torch.testing.assert_close(group_advantages([0,1]), torch.tensor([-1.,1.])/2**.5)
    assert not group_advantages([1,1]).any()
    assert trim_completion([4,2,0],[2]) == ([4,2],False)
    assert trim_completion([4,5],[2]) == ([4,5],True)
    assert training_size(19,8,4,2) == 16
    for args in [(7,8,4,1),(16,8,4,4)]:
        with pytest.raises(ValueError):
            training_size(*args)


def test_data_and_full_batch():
    from experiments.gspo.prepare_data import deduplicate_rows, split_records
    def row(q, a):
        return {"prompt": [{"content": q}], "reward_model": {"ground_truth": a}}
    from experiments.gspo.prepare_data import normalize_row
    assert normalize_row(0, row("zero", 0))["answer"] == "0"
    records, stats = deduplicate_rows([row("x", "1"), row(" x ", "1"), row("y", "2"),
                                      row("y", "3"), row("z", "4")])
    assert len(records) == 2 and stats.conflicting_questions == 1
    train, dev = split_records(records, dev_size=1, seed=42)
    assert train[0]["question"] != dev[0]["question"]
