"""本章共用的 GSM8K 任务、模型加载和 GRPO 公式；不依赖远程训练服务。"""

import hashlib
import json
import random
import re
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


DEFAULT_MODEL = ".cache/models/Qwen2.5-0.5B"
# 训练和评测只使用这一份指令；格式要求是输入的一部分，正确性奖励另行定义。
TASK_INSTRUCTION = (
    "Solve the following math problem.\n"
    "Write your reasoning inside <think>...</think>.\n"
    "Then give the final numerical answer inside \\boxed{}."
)
GSM8K_REVISION = "740312add88f781978c0658806c59bc2815b9866"


def normalize_answer(text):
    """去除千位逗号、首尾空白和末尾句点；仍按字符串比较，不做数值等价判断。"""
    return text.replace(",", "").strip().rstrip(".")


def extract_boxed(text):
    r"""从模型回答中提取最后一个非空 \boxed{...} 的内容；没有则返回 None。"""
    matches = re.findall(r"\\boxed\{([^}]+)\}", text)
    return matches[-1].strip() if matches else None


def ground_truth(text):
    """输入 GSM8K 的完整 answer 字段，返回 #### 后规范化的标准答案字符串。"""
    match = re.search(r"####\s*(.+)", text)
    if match is None:
        raise ValueError(f"GSM8K answer has no #### separator: {text!r}")
    return normalize_answer(match.group(1))


def reward(text, answer):
    """输入模型回答 text 和标准答案 answer，返回浮点奖励 0.0 或 1.0。

    只比较最后一个 boxed 答案；缺失 boxed 记 0，think 格式不参与奖励。
    """
    boxed = extract_boxed(text)
    return float(boxed is not None and normalize_answer(boxed) == normalize_answer(answer))


def follows_think_format(text):
    """检查一个非空 think 块后紧跟 boxed 答案；可作监控或可选的训练格式奖励。

    只检查结构，不检查推理或答案是否正确；不改变 reward() 的正确性判定。
    """
    match = re.fullmatch(r"\s*<think>(.*?)</think>\s*\\boxed\{([^{}]+)\}\s*", text, re.S)
    return bool(match and match.group(1).strip() and match.group(2).strip()
                and text.count("<think>") == 1 and text.count("</think>") == 1)


def training_rewards(text, answer, format_reward_weight=0.0):
    """返回一条回答的训练奖励及分项，评测仍单独调用 reward()。

    text/answer 是模型回答和标准答案字符串。format_reward_weight 是非负权重 λ，
    默认 0 保持原实验；例如 0.5 表示格式合规时额外加 0.5，不要求答案正确。
    返回字典含正确性 correct（0/1）、think_format（bool）、加权格式奖励
    format_reward（0 或 λ）和用于组内优势的总奖励 reward = correct + format_reward。
    """
    correct = reward(text, answer)
    think_format = follows_think_format(text)
    format_reward = format_reward_weight * float(think_format)
    return {"correct": correct, "think_format": think_format,
            "format_reward": format_reward, "reward": correct + format_reward}


def prompt_ids(tokenizer, question):
    """将原始题目 question 转成训练和评测共用的 token ID 列表 [P]。

    tokenizer 使用模型自带的 chat template；add_generation_prompt=True
    添加 assistant 开始标记，让模型从回答位置续写，不预填 think 内容。
    """
    messages = [{"role": "user", "content": TASK_INSTRUCTION + "\n\nProblem:\n" + question}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    # 模板已包含特殊标记，编码时不再额外添加，避免重复的开始/结束标记。
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        raise ValueError("Empty prompt")
    return ids


def load_gsm8k():
    """返回固定版本的 DatasetDict，含官方 train/test；验证集另从 train 留出。"""
    return load_dataset("openai/gsm8k", "main", revision=GSM8K_REVISION, cache_dir=".cache/datasets")


def make_split(dataset, seed=42, validation_size=256, train_limit=0):
    """输入官方 train Dataset，返回划分索引和数据来源信息的字典。

    seed 控制抽样，validation_size 是留出题数；train_limit=0 使用其余全部题，
    正数则只取其中前 N 题。train/validation 均保存原始行号，互不重叠。
    """
    if not 0 < validation_size < len(dataset):
        raise ValueError("validation_size must be between 1 and len(train)-1")
    # 保留抽样顺序，使只评测前 64 题时仍是随机子集，不偏向原数据前部。
    validation = random.Random(seed).sample(range(len(dataset)), validation_size)
    held_out = set(validation)
    train = [i for i in range(len(dataset)) if i not in held_out]
    if train_limit:
        train = train[:train_limit]
    # 保存原始问答内容的哈希，评测复用划分时可检查索引对应的数据是否改变。
    digest = hashlib.sha256()
    for row in dataset:
        digest.update(json.dumps([row["question"], row["answer"]], ensure_ascii=False).encode())
    return {"seed": seed, "validation": validation, "train": train, "revision": GSM8K_REVISION,
            "train_sha256": digest.hexdigest(), "dataset_fingerprint": dataset._fingerprint}


def rows_at(dataset, indices):
    """按 indices 取出问答，并附加原始行号 index，供训练日志和逐题评测追溯。"""
    return [{"index": i, **dataset[i]} for i in indices]


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
    训练独立采样 G 条，评测使用 sample=False、group_size=1 贪心生成。
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


def completion_logprobs(model, prompt, completion):
    """返回一条回答每个实际 token 的 log π(c_t | prompt, c_<t)，形状 [T]。

    prompt/completion 是长度 P/T 的 ID 列表；completion 可包含结束 EOS。
    用 prompt + completion[:-1] 做 teacher forcing，仅取预测回答的位置，
    不计算提示词 loss，也不含 padding。返回值保留模型的梯度图；计算旧策略
    或参考策略时，由调用方使用 no_grad() 固定这些概率。
    """
    return batch_completion_logprobs(model, [{"prompt": prompt, "tokens": completion}])[0]


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



def group_advantages(rewards):
    """输入同一道题 G 个标量奖励，返回 CPU float32 优势张量 [G]。

    A_i = (R_i - mean(R)) / (std(R) + 1e-8)，使用组内总体标准差。
    以组内均值作基线，无需训练价值网络。
    同组总奖励全相同时优势为 0，没有奖励差异的策略梯度信号。
    默认仅正确性奖励时，全对或全错都会如此；加格式奖励后还要看格式是否不同。
    """
    rewards = torch.tensor(rewards, dtype=torch.float32)
    return (rewards - rewards.mean()) / (rewards.std(correction=0) + 1e-8)


def grpo_loss(logprobs, old_logprobs, advantage, ref_logprobs=None, beta=0.0, clip=0.2):
    """计算一条回答的 GRPO loss，返回 (可反传的标量 loss, 日志指标字典)。

    logprobs: 当前策略逐 token 的 log πθ，形状 [T]，保留梯度。
    old_logprobs: 本轮采样时旧策略的 log πold，形状 [T]，调用方已冻结。
    advantage: 该回答的标量 A_i，广播到每个 token，不参与求导。
    ref_logprobs: 固定参考策略的 log πref，[T]；beta=0 时可省略。
    beta: KL 惩罚系数；clip: 概率比裁剪的半径 ε，范围为 [1-ε, 1+ε]。

    先对回答内 T 个 token 求均值；跨回答的平均在 update_policy 中完成，
    因而每条回答的权重相等，不会按回答长度给长回答更大权重。
    """
    # ρ_t = πθ / πold = exp(log πθ - log πold)，逐 token 比较新旧策略。
    ratio = torch.exp(logprobs - old_logprobs)
    unclipped = ratio * advantage
    clipped = ratio.clamp(1 - clip, 1 + clip) * advantage
    # 最大化 min(ρ_t A_i, clip(ρ_t) A_i)；取负号转为梯度下降的 loss。
    # 正优势时 ρ 过大、负优势时 ρ 过小，裁剪分支不再奖励继续向该方向偏离。
    policy = -torch.minimum(unclipped, clipped)
    if beta:
        if ref_logprobs is None:
            raise ValueError("beta > 0 requires frozen reference logprobs")
        # 逐 token 的 KL 估计项：πref/πθ - log(πref/πθ) - 1。
        # 回答来自本轮旧策略；这里计算采样 token 上的惩罚，不遍历整个词表。
        delta = ref_logprobs - logprobs
        kl = torch.exp(delta) - delta - 1
        loss = (policy + beta * kl).mean()
    else:
        kl = None
        loss = policy.mean()
    # clip_fraction 统计裁剪项实际成为较小目标的比例，受优势正负号影响。
    # 指标 detach 后只用于日志，不进入反向传播。
    metrics = {"policy_loss": policy.detach().mean().item(),
               "ratio_mean": ratio.detach().mean().item(),
               "clip_fraction": (unclipped > clipped).float().mean().item()}
    if kl is not None:
        metrics["kl"] = kl.detach().mean().item()
    return loss, metrics
