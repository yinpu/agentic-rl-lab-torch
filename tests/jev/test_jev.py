"""关键算法检查；tokenizer 检查只读取本地缓存，不下载模型。"""
from copy import deepcopy
from pathlib import Path
import json
import sys

import pytest
import torch
import torch.nn.functional as F

from experiments.jev.evaluate import adapt_jevbench, fit_temperature, mean_nll, metrics
from experiments.jev.jev import (ANSWER_SYMBOLS, MARKER, Encoder, build_decisions, compile_record,
                                 decision_logits, normalize_label)
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


@pytest.fixture
def tokenizer():
    from transformers import AutoTokenizer
    path = Path(__file__).resolve().parents[2] / ".cache/models/Qwen3.5-0.8B"
    if not (path / "tokenizer.json").exists():
        pytest.skip("local Qwen3.5-0.8B tokenizer absent")
    return AutoTokenizer.from_pretrained(path, local_files_only=True)


def test_real_tokenizer_marker_and_template(record, tokenizer):
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
    assert tokenizer.decode(encoded["ids"], clean_up_tokenization_spaces=False) == tokenizer.apply_chat_template(
        messages, chat_template=encoder.template, tokenize=False)
    assert json.loads(messages[-1]["content"]) == dict.fromkeys(record["questions"], MARKER)
    with pytest.raises(ValueError, match="exceeds"):
        encoder.build_example(record, max_length=20)
    # 全部 62 个候选，确保小写和数字也能监督。
    record["questions"] = {"many": {"type": "choice", "instructions": "pick",
                                   "criteria": {str(i): str(i) for i in range(62)}}}
    record["targets"] = {"many": {"label": "61"}}
    assert encoder.build_example(record)["targets"] == [encoder.symbol_token_id(ANSWER_SYMBOLS[-1])]


@pytest.mark.parametrize("merged_text", ['"情感": "decision"', '"decision"'])
def test_marker_stays_separate_when_context_has_a_merged_token(record, tokenizer, merged_text):
    # 在测试词表中构造用户担心的整段 token；生产编码不扩展词表。
    tokenizer.add_tokens([merged_text])
    merged_id = tokenizer.convert_tokens_to_ids(merged_text)
    assert tokenizer.encode(merged_text, add_special_tokens=False) == [merged_id]
    for field in ["情感", '转义"\\\n🧪decision']:
        record["questions"][field] = deepcopy(record["questions"]["decision"])
        record["targets"][field] = {"label": "a"}
    encoder = Encoder(tokenizer)
    encoded = encoder.encode(record)
    rendered = tokenizer.apply_chat_template(encoded["messages"], chat_template=encoder.template, tokenize=False)
    assert merged_id in tokenizer.encode(rendered, add_special_tokens=False)
    assert tokenizer.decode(encoded["ids"], clean_up_tokenization_spaces=False) == rendered
    for index in encoded["markers"].values():
        assert encoded["ids"][index] == encoder.marker_id
        assert tokenizer.decode(encoded["ids"][:index], clean_up_tokenization_spaces=False).endswith(': "')
        assert tokenizer.decode(encoded["ids"][index + 1:], clean_up_tokenization_spaces=False).startswith('"')
    inference_record = deepcopy(record)
    inference_record.pop("targets")
    inference = encoder.encode(inference_record, include_targets=False)
    assert inference["ids"] == encoded["ids"]
    assert inference["markers"] == encoded["markers"]
    example = encoder.build_example(record)
    assert example["positions"] == [i - 1 for i in encoded["markers"].values()]
    assert example["targets"] == [encoder.symbol_token_id("B")] * len(record["questions"])


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
    monkeypatch.setattr(sys, "argv", ["evaluate.py", "eval", "--model", "unused-model", "--calibration",
                                    str(calibration), "--output", str(tmp_path / "eval.json")])
    with pytest.raises(ValueError, match="different model/adapter"):
        evaluate.main()


@pytest.mark.parametrize("per_device_train_batch_size", [1, 2, 4])
def test_batch_update_matches_field_mean(per_device_train_batch_size):
    from copy import deepcopy
    from lightning.fabric import Fabric
    from transformers import Qwen2Config, Qwen2ForCausalLM
    from experiments.jev.jev import batch_decision_logits
    from experiments.jev.train import update
    torch.manual_seed(8)
    model = Qwen2ForCausalLM(Qwen2Config(vocab_size=80, hidden_size=16, intermediate_size=24,
                                      num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2)).eval()
    reference = deepcopy(model)
    examples = [{"input_ids": [1, 2, 3, 4, 5], "positions": [1, 3], "targets": [61, 62]},
                {"input_ids": [6, 7, 8], "positions": [1], "targets": [63]},
                {"input_ids": [9, 10, 11, 12], "positions": [0, 2], "targets": [64, 65]},
                {"input_ids": [13, 14, 15], "positions": [1], "targets": [66]}]
    for actual, e in zip(batch_decision_logits(model, examples, "cpu"), examples, strict=True):
        torch.testing.assert_close(actual, decision_logits(reference, e, "cpu"))
    expected_loss = sum(F.cross_entropy(decision_logits(reference, e, "cpu"), torch.tensor(e["targets"]), reduction="sum") for e in examples) / 6
    expected_loss.backward()
    ref_opt = torch.optim.SGD(reference.parameters(), lr=.1)
    ref_opt.step()
    fabric = Fabric(accelerator="cpu", devices=1)
    fabric.launch()
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    model, optimizer = fabric.setup(model, optimizer)
    metrics = update(model, optimizer, examples, fabric,
                     per_device_train_batch_size=per_device_train_batch_size)
    assert metrics["loss"] == pytest.approx(expected_loss.item(), abs=1e-6)
    for p, q in zip(model.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(p.grad, q.grad, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(p, q, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("count,per_device_batch,accumulation,world_size,expected", [
    (7, 2, 2, 1, 4), (7, 1, 2, 2, 4), (8, 2, 2, 2, 8),
    (15, 2, 2, 2, 8), (16, 2, 2, 2, 16),
])
def test_training_size_uses_complete_batches(count, per_device_batch, accumulation, world_size, expected):
    from experiments.jev.train import training_size
    global_batch_size = per_device_batch * accumulation * world_size
    assert training_size(count, global_batch_size) == expected


@pytest.mark.parametrize("count,global_batch_size", [(3, 4), (7, 8), (0, 4)])
def test_training_size_rejects_unusable_batches(count, global_batch_size):
    from experiments.jev.train import training_size
    with pytest.raises(ValueError):
        training_size(count, global_batch_size)


@pytest.mark.parametrize("option", ["--per-device-train-batch-size", "--gradient-accumulation-steps"])
@pytest.mark.parametrize("value", ["0", "-1"])
def test_batch_arguments_must_be_positive(monkeypatch, option, value):
    from experiments.jev.train import parse_args
    monkeypatch.setattr(sys, "argv", ["train.py", option, value])
    with pytest.raises(SystemExit) as error:
        parse_args()
    assert error.value.code == 2
