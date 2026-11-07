"""关键算法检查；tokenizer 检查只读取本地缓存，不下载模型。"""
from copy import deepcopy
from pathlib import Path
import json
import sys

import pytest
import torch
import torch.nn.functional as F

from experiments.jev.calibrate import fit_temperature, mean_nll
from experiments.jev.data import ANSWER_SYMBOLS, Encoder, compile_record, normalize_label
from experiments.jev.evaluate import adapt_jevbench, metrics
from experiments.jev.inference import build_decisions
from experiments.jev.modeling import decision_logits
from experiments.jev.prepare_data import convert_toolace, convert_typed_decisions


@pytest.fixture
def record():
    return {"id": "case", "state": {"text": "decision in the state"}, "questions": {
        "decision": {"type": "choice", "instructions": "route", "criteria": {"z": "last", "a": "first"}},
        'quoted"decision': {"type": "score", "instructions": "rate", "criteria": ["low", "high"]},
        "flag": {"type": "noul", "instructions": "approve"}},
        "targets": {"decision": {"label": "a"}, 'quoted"decision': {"label": 1}, "flag": {"label": True}}}


def test_schema_order_and_targets(record):
    compiled = compile_record(record)
    assert compiled["targets"] == {"decision": "B", 'quoted"decision': "B", "flag": "B"}
    assert compiled["options"]["decision"][0] == ("A", "z", "last")
    hidden = deepcopy(record)
    hidden.pop("targets")
    assert compile_record(hidden, False)["messages"] == compiled["messages"]
    assert normalize_label({"type": "noul"}, False) == "no"
    with pytest.raises(ValueError):
        normalize_label({"type": "noul"}, "maybe")


def test_candidate_limits(record):
    record["questions"]["decision"]["criteria"] = {str(i): str(i) for i in range(63)}
    with pytest.raises(ValueError, match="1..62"):
        compile_record(record)


def test_real_tokenizer_marker_and_template(record):
    from transformers import AutoTokenizer
    path = Path(__file__).resolve().parents[2] / ".cache/models/Qwen3.5-0.8B"
    if not (path / "tokenizer.json").exists():
        pytest.skip("local Qwen3.5-0.8B tokenizer absent")
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    encoder = Encoder(tokenizer)
    encoded = encoder.encode(record)
    example = encoder.build_example(record)
    assert len(encoded["markers"]) == 3
    for position, target in zip(example["positions"], example["targets"], strict=True):
        assert example["input_ids"][position + 1] == encoder.marker_id
        assert target == encoder.symbol_token_id("B")
        assert example["input_ids"][position + 1] != target
    messages = encoded["messages"]
    assert tokenizer.apply_chat_template(messages, tokenize=False) == tokenizer.apply_chat_template(
        messages, chat_template=encoder.template, tokenize=False)
    with pytest.raises(ValueError, match="exceeds"):
        encoder.build_example(record, max_length=20)
    # 全部 62 个候选，确保小写和数字也能监督。
    record["questions"] = {"many": {"type": "choice", "instructions": "pick",
                                   "criteria": {str(i): str(i) for i in range(62)}}}
    record["targets"] = {"many": {"label": "61"}}
    assert encoder.build_example(record)["targets"] == [encoder.symbol_token_id(ANSWER_SYMBOLS[-1])]


def test_sparse_logits_matches_full_vocab_shifted_ce():
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(3)
    model = Qwen2ForCausalLM(Qwen2Config(vocab_size=80, hidden_size=16, intermediate_size=24,
                                      num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2))
    example = {"input_ids": [1, 2, 3, 4, 5], "positions": [1, 3], "targets": [61, 62]}
    selected = decision_logits(model, example, "cpu")
    labels = torch.tensor([[-100, -100, 61, -100, 62]])
    reference = model(input_ids=torch.tensor([example["input_ids"]]), labels=labels).loss
    loss = F.cross_entropy(selected, torch.tensor(example["targets"]))
    assert torch.allclose(loss, reference, atol=1e-6)
    loss.backward()
    grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    model.zero_grad()
    model(input_ids=torch.tensor([example["input_ids"]]), labels=labels).loss.backward()
    for n, p in model.named_parameters():
        if p.grad is not None:
            assert torch.allclose(grads[n], p.grad, atol=1e-6)


def test_restricted_probabilities(record):
    logits = {f: torch.tensor([0.0, 2.0], dtype=torch.float64) for f in record["questions"]}
    output = build_decisions(record, logits)
    assert output["decision"]["prediction"] == "a"
    assert output["flag"]["p_yes"] == pytest.approx(0.8807970779)
    assert output['quoted"decision']["expected_score"] == pytest.approx(0.8807970779)
    warmer = build_decisions(record, logits, temperature=2)
    for field in output:
        assert sum(output[field]["probabilities"].values()) == pytest.approx(1)
        assert warmer[field]["prediction"] == output[field]["prediction"]
    with pytest.raises(ValueError):
        build_decisions(record, logits, temperature=0)


def test_temperature_improves_nll_and_preserves_argmax():
    z = torch.tensor([4.0, 0.0], dtype=torch.float64)
    instances = [(z, 0)] * 3 + [(z, 1)]
    temperature = fit_temperature(instances)
    assert temperature == pytest.approx(4 / torch.log(torch.tensor(3.0)).item(), rel=1e-5)
    assert mean_nll(instances, temperature) < mean_nll(instances, 1)
    assert (z / temperature).argmax() == z.argmax()
    with pytest.raises(ValueError):
        fit_temperature([])


def test_metrics_bins_and_brier():
    rows = [{"prediction": "a", "gold": "a", "probabilities": {"a": 1., "b": 0.}, "nll": 0.},
            {"prediction": "a", "gold": "b", "probabilities": {"a": .9, "b": .1}, "nll": 2.3}]
    result = metrics(rows)
    assert result["accuracy"] == .5
    assert result["ece"] == pytest.approx(.45)
    assert result["brier"] == pytest.approx(.81)


def test_benchmark_choice_uses_labels_order():
    row = {"id": "eval", "state": {}, "labels": ["b", "a"], "expected": "b",
           "question": {"type": "choice", "instructions": "choose", "criteria": {"a": "A", "b": "B"}}}
    assert compile_record(adapt_jevbench(row))["targets"]["decision"] == "A"


def test_dataset_conversion_preserves_reference_task():
    row = {"id": "1", "state": '{"x":1}', "questions": '{"f":{"type":"noul"}}',
           "gold": '{"f":{"label":true,"distribution":{"yes":0.8}}}'}
    assert convert_typed_decisions(row)["targets"] == {"f": {"label": True}}
    tools = '[{"name":"a","description":"A"},{"name":"b","description":"B"}]'
    row = {"system": "Here is a list of functions in JSON format that you can invoke:\n" + tools,
           "conversations": [{"from": "user", "value": "  GET   X "}, {"from": "assistant", "value": '[a(x=1)]'}]}
    converted = convert_toolace(row)
    assert converted["_request_key"] == "get x"
    assert converted["targets"]["decision"]["label"] == "a"
    row["conversations"][1]["value"] = '[a(x=1), b(x=2)]'
    assert convert_toolace(row) is None


def test_calibration_identity_is_checked_before_loading(tmp_path, monkeypatch):
    from experiments.jev import evaluate
    calibration = tmp_path / "calibration.json"
    calibration.write_text(json.dumps({"identity": {"model": "different", "adapter_sha256": None},
                                       "temperature": 1.0}))
    monkeypatch.setattr(sys, "argv", ["evaluate.py", "--model", "unused-model", "--calibration",
                                    str(calibration), "--output", str(tmp_path / "eval.json")])
    with pytest.raises(ValueError, match="different model/adapter"):
        evaluate.main()
