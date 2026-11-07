"""DeepMath 数据、同轨迹概率计算和 OPD token 损失。"""
import json
import random
import re
from pathlib import Path
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

SUFFIX = "Please solve the problem step by step and put the final answer in \\boxed{}."

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



def check_tokenizers(student, teacher):
    if student.get_vocab() != teacher.get_vocab() or student.all_special_ids != teacher.all_special_ids:
        raise ValueError("Student and teacher must share token IDs and special tokens")


def prompt_ids(tokenizer, question):
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": question.strip() + "\n\n" + SUFFIX}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    return tokenizer.encode(text, add_special_tokens=False)


def load_deepmath(num_shards=1):
    import urllib.request
    import urllib.parse
    import shutil
    paths = []
    for i in range(num_shards):
        name = f"data/train-{i:05d}-of-00010.parquet"
        path = Path(".cache/datasets/DeepMath-103K") / name
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            query = urllib.parse.urlencode({"Source": "SDK", "Revision": "master", "FilePath": name, "View": "False"})
            url = "https://www.modelscope.cn/api/v1/datasets/AI-ModelScope/DeepMath-103K/repo?" + query
            tmp = path.with_suffix(".tmp")
            with urllib.request.urlopen(url, timeout=120) as response, tmp.open("wb") as output:
                shutil.copyfileobj(response, output)
            tmp.replace(path)
        paths.append(str(path))
    return load_dataset("parquet", data_files=paths, split="train")


def make_split(data, seed, validation_size, train_limit):
    # 同一道题只保留一次，避免重复题穿越训练/验证边界。
    unique = {}
    for row in data:
        key = " ".join(row["question"].split())
        unique.setdefault(key, {"question": row["question"], "answer": str(row["final_answer"])})
    rows = list(unique.values())
    random.Random(seed).shuffle(rows)
    if not 0 < validation_size < len(rows):
        raise ValueError("Need both training and validation questions")
    validation, train = rows[:validation_size], rows[validation_size:]
    if train_limit:
        train = train[:train_limit]
    return train, validation


def training_size(count, batch_size, world_size):
    if batch_size % world_size:
        raise ValueError("global batch size must be divisible by world size")
    size = count // batch_size * batch_size
    if not size:
        raise ValueError("not enough data for one global batch")
    return size


def opd_loss(current, old, teacher, beta=1.0):
    """返回 token loss 向量；缓存概率不参与求导，归约由全局 token 数决定。"""
    advantage = beta * (teacher.detach() - old.detach())
    return -torch.exp(current - old.detach()) * advantage


@torch.no_grad()
def rollout(student, teacher, tokenizer, rows, args):
    prompts = [prompt_ids(tokenizer, row["question"]) for row in rows]
    generated = generate(student, tokenizer, prompts, args.max_tokens, args.group_size)
    samples = [{"prompt": prompts[i // args.group_size], "tokens": tokens, "truncated": truncated}
               for i, (tokens, truncated) in enumerate(generated)]
    student.eval()
    teacher.eval()
    for start in range(0, len(samples), args.micro_batch_size):
        chunk = samples[start:start + args.micro_batch_size]
        old = batch_completion_logprobs(student, chunk)
        target = batch_completion_logprobs(teacher, chunk)
        for row, a, b in zip(chunk, old, target, strict=True):
            row.update(old=a.cpu(), teacher=b.cpu())
    return samples


def update_policy(model, optimizer, samples, args, fabric):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    count = fabric.all_reduce(torch.tensor(sum(len(s["tokens"]) for s in samples),
                                           device=fabric.device), reduce_op="sum")
    total = torch.zeros((), device=fabric.device)
    for start in range(0, len(samples), args.micro_batch_size):
        chunk = fabric.to_device(samples[start:start + args.micro_batch_size])
        with fabric.no_backward_sync(model, enabled=start + args.micro_batch_size < len(samples)):
            currents = batch_completion_logprobs(model, chunk)
            loss = sum(opd_loss(c, s["old"], s["teacher"], args.beta).sum()
                       for c, s in zip(currents, chunk, strict=True))
            # DDP 平均各卡梯度，乘 world_size 还原全局 token 加权均值。
            fabric.backward(loss * fabric.world_size / count)
            total += loss.detach()
    norm = fabric.clip_gradients(model, optimizer, max_norm=args.max_grad_norm)
    optimizer.step()
    return {"loss": (fabric.all_reduce(total, reduce_op="sum") / count).item(),
            "grad_norm": norm.item()}
