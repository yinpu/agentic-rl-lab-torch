"""DAPO 数学任务、采样、逐 token 概率与奖励。"""
import json
from pathlib import Path
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from math_verify import parse, verify
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

DEFAULT_MODEL = ".cache/models/Qwen3.5-4B"

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")

def load_model(model_name, dtype=torch.bfloat16):
    """从本地路径或模型 ID 加载，返回 (model, tokenizer)，暂不搬到 GPU。

    dtype 决定基础权重精度；设备放置由调用方或 Fabric 负责。
    关闭 dropout，使旧/新策略的概率比较只反映参数变化。
    """
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, attn_implementation="sdpa")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # 两次 logprob forward 必须使用相同的无 dropout 策略。
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
        if isinstance(getattr(module, "attention_dropout", None), (int, float)):
            module.attention_dropout = 0.0
    return model, tokenizer

def eos_ids(tokenizer):
    """返回允许结束回答的 token ID 列表，兼顾文本结束和 chat 消息结束标记。"""
    ids = {tokenizer.eos_token_id}
    if "<|im_end|>" in tokenizer.get_vocab():
        ids.add(tokenizer.convert_tokens_to_ids("<|im_end|>"))
    return sorted(i for i in ids if i is not None)

def trim_completion(tokens, stop_ids):
    """输入生成的 ID 列表和结束标记集合，返回 (有效 ID 列表, 是否截断)。

    保留首个 EOS，使结束动作也参与策略学习；删除其后的 padding。
    找不到 EOS 时标记截断，不以 token 是否等于 pad 判断有效长度。
    """
    for i, token in enumerate(tokens):
        if token in stop_ids:
            return tokens[:i + 1], False
    return tokens, True

@torch.no_grad()
def generate(model, tokenizer, prompts, max_tokens, group_size=1, sample=True):
    """左侧 padding 后批量生成，返回按题连续排列的 B*G 个 (tokens, truncated)。

    prompts 是 B 个未补齐的 ID 列表；每题由 num_return_sequences 扩展为 G 条。
    左侧补齐让所有 prompt 的最后一个有效 token 位于同一列，回答统一从
    padded width 开始。attention_mask 屏蔽 padding，生成器据此构造位置编号。
    训练与评测均独立采样，调用方控制回答数。
    返回值仅含回答，保留首个 EOS、删除其后的 padding。
    """
    model.eval()
    device = next(model.parameters()).device
    width = max(map(len, prompts))
    inputs = torch.full((len(prompts), width), tokenizer.pad_token_id,
                        dtype=torch.long, device=device)
    mask = torch.zeros_like(inputs)
    for i, prompt in enumerate(prompts):
        inputs[i, -len(prompt):] = torch.tensor(prompt, device=device)
        mask[i, -len(prompt):] = 1
    # 新建配置，避免继承模型仓库的 top-k、repetition penalty 等默认变换。
    config = GenerationConfig(
        max_new_tokens=max_tokens, do_sample=sample, num_return_sequences=group_size,
        repetition_penalty=1.0,
        eos_token_id=eos_ids(tokenizer), pad_token_id=tokenizer.pad_token_id,
        use_cache=True,
        # 不改变温度或截断词表，使采样分布与后面计算 logprob 的策略一致。
        **({"temperature": 1.0, "top_p": 1.0, "top_k": 0} if sample else {}),
    )
    output = model.generate(input_ids=inputs, attention_mask=mask,
                            generation_config=config)
    # output 为 [B*G, width + 生成长度]；切片使用补齐宽度，不用各题原始长度。
    return [trim_completion(seq[width:].tolist(), eos_ids(tokenizer)) for seq in output]

def batch_completion_logprobs(model, samples):
    """右侧补齐后共同前向；返回每条回答的有效 token logprob，保留 EOS。

    仅在回答预测位置的并集计算词表投影；prompt/padding 不参与 loss。
    """
    if any(not sample["tokens"] for sample in samples):
        raise ValueError("Empty completion")
    device = next(model.parameters()).device
    sequences = [s["prompt"] + s["tokens"][:-1] for s in samples]
    positions = [list(range(len(s["prompt"]) - 1, len(seq)))
                 for s, seq in zip(samples, sequences, strict=True)]
    selected = torch.tensor(sorted({i for row in positions for i in row}), device=device)
    width = max(map(len, sequences))
    ids = torch.zeros((len(samples), width), dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for i, seq in enumerate(sequences):
        ids[i, :len(seq)] = torch.tensor(seq, device=device)
        mask[i, :len(seq)] = 1
    logits = model(input_ids=ids, attention_mask=mask, use_cache=False,
                   logits_to_keep=selected).logits
    logprobs = []
    for i, (sample, row_positions) in enumerate(zip(samples, positions, strict=True)):
        # selected 是升序并集，searchsorted 将原序列位置映射回投影后的列号。
        columns = torch.searchsorted(selected, torch.tensor(row_positions, device=device))
        targets = torch.tensor(sample["tokens"], device=device)
        logprobs.append(-F.cross_entropy(logits[i, columns].float(), targets, reduction="none"))
    return logprobs

ANSWER_WINDOW_CHARS = 300


@dataclass(frozen=True)
class RewardResult:
    """一条 completion 的可解释奖励。"""

    base_reward: float
    length_penalty: float
    shaped_reward: float
    correct: bool
    valid_format: bool
    answer: str | None
    completion_tokens: int


def extract_last_boxed(text: str) -> str | None:
    """从末尾向前找到最后一个完整的 ``\\boxed{...}``，支持嵌套括号。"""
    end = len(text)
    while True:
        marker = text.rfind("\\boxed", 0, end)
        if marker < 0:
            return None
        left = text.find("{", marker)
        if left < 0:
            end = marker
            continue
        depth = 0
        for position in range(left, len(text)):
            if text[position] == "{":
                depth += 1
            elif text[position] == "}":
                depth -= 1
                if depth == 0:
                    return text[left + 1 : position].strip()
        end = marker


def answers_equivalent(prediction: str, reference: str) -> bool:
    """使用 ``math_verify`` 判断两个数学答案是否等价。"""
    try:
        return bool(verify(parse(f"${reference}$"), parse(f"${prediction}$")))
    except Exception:
        return False


def soft_overlong_penalty(
    completion_tokens: int,
    *,
    max_tokens: int,
    overlong_cache: int,
) -> float:
    """按 DAPO 的分段线性规则返回 ``[-1, 0]`` 长度惩罚。"""
    if completion_tokens < 0:
        raise ValueError("completion_tokens must be >= 0")
    if max_tokens < 1:
        raise ValueError("max_tokens must be >= 1")
    if not 1 <= overlong_cache <= max_tokens:
        raise ValueError("overlong_cache must be in [1, max_tokens]")

    penalty_start = max_tokens - overlong_cache
    if completion_tokens <= penalty_start:
        return 0.0
    if completion_tokens <= max_tokens:
        return -(completion_tokens - penalty_start) / overlong_cache
    return -1.0


def score_answer(
    text: str,
    reference: str,
    *,
    completion_tokens: int,
    max_tokens: int,
    overlong_cache: int,
    enable_overlong: bool,
) -> RewardResult:
    """计算 ±1 正确性奖励，并按 preset 可选叠加长度惩罚。"""
    answer = extract_last_boxed(text[-ANSWER_WINDOW_CHARS:])
    correct = (
        answer is not None
        and answers_equivalent(answer.strip(), reference.strip())
    )
    base_reward = 1.0 if correct else -1.0
    length_penalty = (
        soft_overlong_penalty(
            completion_tokens,
            max_tokens=max_tokens,
            overlong_cache=overlong_cache,
        )
        if enable_overlong
        else 0.0
    )
    return RewardResult(
        base_reward=base_reward,
        length_penalty=length_penalty,
        shaped_reward=base_reward + length_penalty,
        correct=correct,
        valid_format=answer is not None,
        answer=answer,
        completion_tokens=completion_tokens,
    )

def prompt_ids(tokenizer, question, max_prompt_tokens=512):
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": question.strip() +
          "\n\nPlease reason step by step, and put your final answer within \\boxed{}."}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids or len(ids) > max_prompt_tokens:
        raise ValueError(f"prompt length {len(ids)} outside [1, {max_prompt_tokens}]")
    return ids


def load_rows(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not rows or any(not str(r.get(k, "")).strip() for r in rows for k in ("question", "answer")):
        raise ValueError(f"empty or invalid dataset: {path}")
    return rows


def group_advantages(rewards):
    rewards = torch.tensor(rewards, dtype=torch.float32)
    if len(rewards) < 2:
        raise ValueError("group size must be >= 2")
    return (rewards - rewards.mean()) / (rewards.std(correction=1) + 1e-8)


def informative_group(samples):
    return 0 < sum(s["correct"] for s in samples) < len(samples)


def token_loss(current, old, advantage):
    """返回逐 token 的负 clipped surrogate；调用方统一用全局 token 数归约。"""
    ratio = (current - old).exp()
    return -torch.minimum(ratio * advantage, ratio.clamp(0.8, 1.28) * advantage)


def sample_group(model, tokenizer, row, args):
    prompt = prompt_ids(tokenizer, row["question"], args.max_prompt_tokens)
    completions = []
    for start in range(0, args.group_size, args.generation_micro_batch_size):
        completions.extend(generate(model, tokenizer, [prompt], args.max_tokens,
                           min(args.generation_micro_batch_size, args.group_size - start)))
    samples = []
    for tokens, truncated in completions:
        text = tokenizer.decode(tokens, skip_special_tokens=True)
        score = score_answer(text, row["answer"], completion_tokens=len(tokens),
                             max_tokens=args.max_tokens, overlong_cache=args.overlong_cache,
                             enable_overlong=True)
        samples.append(dict(prompt=prompt, tokens=tokens, response=text, truncated=truncated,
                            correct=score.correct, reward=score.shaped_reward,
                            length_penalty=score.length_penalty, id=row.get("id")))
    for sample, advantage in zip(samples, group_advantages([s["reward"] for s in samples])):
        sample["advantage"] = advantage.item()
    return samples
