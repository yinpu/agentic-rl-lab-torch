# GRPO：GSM8K 上的 LoRA 强化学习

## 算法原理

GRPO 对同一道题采样多个回答，用组内奖励差异估计优势，再通过裁剪概率比更新策略。本章在 GSM8K 上训练 Qwen2.5-0.5B Base，比较正确性奖励、reference KL 和格式奖励的影响。

### 任务与奖励

训练和评测共用以下提示词，经 tokenizer 的 chat template 编码：

```text
Solve the following math problem.
Write your reasoning inside <think>...</think>.
Then give the final numerical answer inside \boxed{}.

Problem:
{question}
```

标准答案取 GSM8K `answer` 中 `####` 后的内容。正确性奖励 $c_i$ 提取回答最后一个 `\boxed{...}`，去掉逗号、首尾空白和末尾句点后，与标准答案做字符串比较：相同为 1，否则为 0。

可选格式奖励检查回答是否为一个非空 `<think>...</think>` 块，后面紧跟 boxed 答案。令格式合规标记为 $f_i\in\{0,1\}$，总奖励为：

\[
r_i=c_i+\lambda f_i.
\]

默认 $\lambda=0$；设置 `--format-reward-weight 0.5` 时，格式合规额外得 0.5 分。正确性和格式分别打分，独立评测的 accuracy 只使用正确性。

### 组内优势

对题目 $q$，旧策略采样 $G$ 个回答，奖励为 $r_1,\ldots,r_G$：

\[
\bar r=\frac{1}{G}\sum_{j=1}^{G}r_j,
\qquad
A_i=\frac{r_i-\bar r}{\operatorname{std}(r_1,\ldots,r_G)+10^{-8}}.
\]

组内均值充当 baseline，无需训练 critic。这里使用总体标准差，同一回答的所有有效 token 共用 $A_i$。例如奖励 `[1, 0, 1, 0]` 得到优势约 `[1, -1, 1, -1]`；奖励全相同时，优势为零。

### 裁剪目标与 KL

当前策略与采样时旧策略的逐 token 概率比为：

\[
\rho_{i,t}=\exp\left(\log\pi_\theta(o_{i,t}\mid q,o_{i,<t})
-\log\pi_{\mathrm{old}}(o_{i,t}\mid q,o_{i,<t})\right).
\]

策略损失使用 clipped surrogate，默认 $\epsilon=0.2$：

\[
\ell^{\mathrm{policy}}_{i,t}=-\min\left(
\rho_{i,t}A_i,
\operatorname{clip}(\rho_{i,t},1-\epsilon,1+\epsilon)A_i
\right).
\]

正优势提高回答概率，负优势降低回答概率；裁剪限制沿有利方向的过大更新。

可选 KL 约束使用固定基础模型作为 reference：

\[
\Delta_{i,t}=\log\pi_{\mathrm{ref}}-\log\pi_\theta,
\qquad
\widehat D_{i,t}=e^{\Delta_{i,t}}-\Delta_{i,t}-1.
\]

先平均每条回答的有效 token，再对全部 $N$ 条回答等权平均：

\[
L=\frac{1}{N}\sum_{i=1}^{N}\frac{1}{T_i}\sum_{t=1}^{T_i}
\left(\ell^{\mathrm{policy}}_{i,t}+\beta\widehat D_{i,t}\right).
\]

KL 直接加入 loss，组内优势只使用奖励 $r_i$。默认 $\beta=0$，可用 `--beta 0.04` 开启 KL 对照。

任务与判分参考 [agentic-rl-lab 的 GRPO 章节](https://github.com/datawhalechina/agentic-rl-lab/tree/8493321013c02eef6cf79ca2594811f53c6fdac1/experiments/grpo)，算法见 [DeepSeekMath](https://arxiv.org/abs/2402.03300)。本章选用 Qwen2.5-0.5B Base、无示例 think + boxed 提示词和 LoRA；优势在参考代码减组内均值的基础上再除以标准差，训练目标采用上述 clipped loss 与按回答等权归约。

## 关键算法的代码实现

### 一批数据的训练流程

[train.py](train.py) 的 `main` 按以下顺序处理每个全局采样批次：

1. 取出一批题目，分配给各进程；同一道题的全部回答在同一进程采样。
2. `rollout` 用更新前的策略为每题生成 $G$ 个回答，计算奖励、组内优势，并缓存旧策略和可选 reference 的逐 token logprob。
3. `should_update` 汇总全局优势信号，决定本批是否更新。
4. `update_policy` 对这些回答重新计算当前策略 logprob，计算 clipped loss 和可选 KL，累积梯度并更新 LoRA。
5. 同批可重复更新 `updates_per_batch` 次，随后用更新后的策略采样下一批；按 `eval_every` 定期评测。

主循环的核心调用为：

```python
batch = train_rows[start:start + global_batch_size]
local_rows = batch[fabric.global_rank::fabric.world_size]
with fabric.autocast():
    samples = rollout(raw_model, tokenizer, local_rows, args)
if should_update(samples, args.beta, fabric):
    for update in range(args.updates_per_batch):
        metrics = update_policy(model, optimizer, samples, args, fabric)
```

### Rollout：从题目到训练样本

`rollout` 的输入是本进程的题目列表 `rows`，输出是回答记录列表 `samples`。若本进程有 $B$ 道题，输出就有 $B\times G$ 条记录。每道题先编码为 prompt，再一次生成该题的 $G$ 个回答：

```python
prompt = prompt_ids(tokenizer, row["question"])
completions = generate(model, tokenizer, prompt, args.max_tokens, args.group_size)
texts = [tokenizer.decode(tokens, skip_special_tokens=True) for tokens, _ in completions]
scores = [training_rewards(text, ground_truth(row["answer"]), args.format_reward_weight)
          for text in texts]
rewards = [score["reward"] for score in scores]
advantages = group_advantages(rewards)
```

`generate` 返回每条回答的 token ID 和截断标记；解码文本用于判分，原始 token ID 留给后续概率计算。同题的奖励单独标准化为优势。采样设置为 temperature=1、top-p=1、top-k=0、repetition penalty=1，模型和 LoRA 的 dropout 均为 0。

随后遍历该题回答，在无梯度上下文中缓存概率：

```python
with torch.no_grad():
    old = completion_logprobs(model, prompt, tokens).cpu()
    reference = None
    if args.beta:
        with model.disable_adapter():
            reference = completion_logprobs(model, prompt, tokens).cpu()
```

每条记录保存 `prompt`、`tokens`、`reward`、`advantage`、`old` 和 `reference`，以及回答文本和截断标记。`old` 与 `reference` 是长度等于有效回答 token 数的向量，`advantage` 是该回答的标量。

整批 rollout 结束后才开始更新，所以这些 `old` 都来自同一个采样策略。同批重复更新始终复用这份记录；当前策略概率在每次更新时重算，下一批 rollout 才刷新旧概率。reference 来自关闭 LoRA 的冻结基础模型，整个训练过程中固定。

### 奖励与优势

[task.py](task.py) 的 `training_rewards` 在 rollout 判分时组合两项奖励：

```python
correct = reward(text, answer)
think_format = follows_think_format(text)
format_reward = format_reward_weight * float(think_format)
```

返回的 `reward` 为 `correct + format_reward`。`group_advantages` 对同一道题的奖励执行：

```python
rewards = torch.tensor(rewards, dtype=torch.float32)
return (rewards - rewards.mean()) / (rewards.std(correction=0) + 1e-8)
```

### 回答 token 的概率对齐

`task.py::completion_logprobs` 复用批量函数 `batch_completion_logprobs`。模型输入是 `prompt + completion[:-1]`，预测目标是 completion：

```text
输入：p0 p1 p2 c0 c1
取值：      ↓  ↓  ↓
目标：      c0 c1 EOS
```

批量输入右侧补齐，attention mask 标记有效输入。`selected` 收集本批回答预测位置的并集，`logits_to_keep` 只在这些位置执行词表投影：

```python
logits = model(input_ids=ids, attention_mask=mask, use_cache=False,
               logits_to_keep=selected).logits
columns = torch.searchsorted(selected, torch.tensor(row_positions, device=device))
targets = torch.tensor(sample["tokens"], device=device)
logprobs = -F.cross_entropy(logits[i, columns].float(), targets, reduction="none")
```

交叉熵取负得到目标 token 的 logprob。每条回答只保留有效 completion 位置，包含首个 EOS，排除 prompt 和 padding；达到长度上限的回答也参与训练。

### 损失与参数更新

`task.py::grpo_loss` 直接实现概率比和裁剪目标：

```python
ratio = torch.exp(logprobs - old_logprobs)
unclipped = ratio * advantage
clipped = ratio.clamp(1 - clip, 1 + clip) * advantage
policy = -torch.minimum(unclipped, clipped)
```

开启 KL 时继续计算：

```python
delta = ref_logprobs - logprobs
kl = torch.exp(delta) - delta - 1
loss = (policy + beta * kl).mean()
```

关闭 KL 时取 `policy.mean()`。

`train.py::update_policy` 先清空梯度，再按 micro-batch 读取 rollout 记录。每个 micro-batch 批量计算当前策略概率，将缓存的旧概率与 reference 搬到当前设备，逐回答调用损失函数：

```python
currents = batch_completion_logprobs(model, chunk)
for sample, current in zip(chunk, currents, strict=True):
    old = sample["old"].to(fabric.device)
    reference = sample["reference"]
    if reference is not None:
        reference = reference.to(fabric.device)
    loss, metrics = grpo_loss(current, old, sample["advantage"], reference,
                              args.beta, args.clip)
```

`current` 保留梯度，缓存的 `old`、`reference` 和 `advantage` 保持固定。每条回答已在 `grpo_loss` 内完成 token 平均，接着按本卡回答数归一，并对本 micro-batch 反传：

```python
weighted_losses.append(loss / len(samples))
fabric.backward(torch.stack(weighted_losses).sum())
```

各 micro-batch 累积到同一次参数更新，只有最后一次反传同步梯度。各卡回答数相同，DDP 平均本卡梯度后得到全局回答均值。整批反传结束后统一裁剪梯度并更新参数：

```python
grad_norm = fabric.clip_gradients(model, optimizer, max_norm=args.max_grad_norm)
optimizer.step()
```

奖励全相同的组仍保留在损失分母中。`should_update` 汇总全局优势信号：当整批优势均为零且 $\beta=0$ 时跳过更新，避免 Adam 的历史动量继续改变参数。默认每批更新一次；`--updates-per-batch` 可在固定旧概率下重复更新。

## 运行方法

以下命令从仓库根目录执行，使用 Python 3.12。

### 环境与模型

```bash
python -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
.venv/bin/python -m pip install -r requirements.txt

.venv/bin/python -c 'from modelscope import snapshot_download; snapshot_download("Qwen/Qwen2.5-0.5B", local_dir=".cache/models/Qwen2.5-0.5B")'
```

上述 PyTorch 命令使用 CUDA 12.8 wheel。`--model` 默认读取 `.cache/models/Qwen2.5-0.5B`，也接受模型 ID 或其他本地目录。

### 数据与训练

GSM8K 首次运行自动下载到 `.cache/datasets`。从官方 train 按 seed 42 留出 256 题作为验证集，其余题按原始顺序训练，划分保存为 `splits.json`。训练前后及每 50 个采样批次贪心评测一次，默认评测 64 道验证题；`--eval-limit 0` 使用全部验证题。

先运行小子集：

```bash
.venv/bin/python experiments/grpo/train.py \
  --output experiments/grpo/runs/qwen05-smoke \
  --steps 1 --train-limit 2 --batch-size 1 --group-size 2 \
  --max-tokens 128 --eval-limit 2 --eval-max-tokens 128
```

完整训练：

```bash
.venv/bin/python experiments/grpo/train.py \
  --output experiments/grpo/runs/qwen05-main \
  --batch-size 4 --group-size 8 --max-tokens 1024 --eval-limit 0
```

多卡使用同一入口。下面两卡配置与上面的单卡配置均为每批 4 题、32 条回答：

```bash
.venv/bin/torchrun --standalone --nproc_per_node=2 experiments/grpo/train.py \
  --output experiments/grpo/runs/qwen05-ddp \
  --batch-size 2 --group-size 8 --max-tokens 1024 --eval-limit 0
```

| 参数 | 默认值与含义 |
| --- | --- |
| `--batch-size` | 4；每进程每个采样批次的题数，全局回答数为进程数 × batch-size × group-size |
| `--group-size` | 8；同一道题采样的回答数 |
| `--micro-batch-size` | 1；每进程一次训练前向的回答数，不改变有效 batch |
| `--steps` / `--train-limit` | 0 / 0；采样批次数上限 / 训练题数上限，0 使用全部 |
| `--max-tokens` / `--eval-max-tokens` | 1024 / 1024；采样 / 评测的回答 token 上限 |
| `--updates-per-batch` | 1；每批回答复用的优化次数 |
| `--clip` / `--max-grad-norm` | 0.2 / 1；概率比裁剪半径 / 梯度范数上限 |
| `--learning-rate` | 4e-5；AdamW，betas=(0.9, 0.95)，weight decay=0 |
| `--lora-rank` / `--lora-alpha` | 32 / 64；all-linear，dropout=0 |
| `--beta` / `--format-reward-weight` | 0 / 0；KL 系数 / 格式奖励权重 |
| `--precision` | bf16；可设为 fp32 |

单卡和多卡都舍弃不足一个全局 batch 的训练尾部；数据至少需要一批。`--steps 0` 遍历训练集一遍，`--steps N` 限制批次数。评测使用所选的全部题目。

### KL 与格式奖励对照

各组从基础模型初始化，使用相同的数据划分、训练预算和评测参数：

| 组别 | `--beta` | `--format-reward-weight` |
| --- | ---: | ---: |
| baseline | 0 | 0 |
| kl | 0.04 | 0 |
| format | 0 | 0.5 |
| kl-format | 0.04 | 0.5 |

```bash
.venv/bin/python experiments/grpo/train.py \
  --output experiments/grpo/runs/qwen05-baseline --eval-limit 0 \
  --beta 0 --format-reward-weight 0
.venv/bin/python experiments/grpo/train.py \
  --output experiments/grpo/runs/qwen05-kl --eval-limit 0 \
  --beta 0.04 --format-reward-weight 0
.venv/bin/python experiments/grpo/train.py \
  --output experiments/grpo/runs/qwen05-format --eval-limit 0 \
  --beta 0 --format-reward-weight 0.5
.venv/bin/python experiments/grpo/train.py \
  --output experiments/grpo/runs/qwen05-kl-format --eval-limit 0 \
  --beta 0.04 --format-reward-weight 0.5
```

### 独立评测与训练曲线

不传 `--adapter` 评测基础模型，传入训练产物的 `adapter/` 评测 LoRA。验证集可通过 `--split-file` 复用训练划分：

```bash
.venv/bin/python experiments/grpo/evaluate.py \
  --adapter experiments/grpo/runs/qwen05-main/adapter \
  --split validation --split-file experiments/grpo/runs/qwen05-main/splits.json \
  --limit 0 --max-tokens 1024 --output experiments/grpo/runs/qwen05-main/validation.json

.venv/bin/python experiments/grpo/evaluate.py \
  --split test --limit 0 --max-tokens 1024 --output experiments/grpo/runs/test-base.json
.venv/bin/python experiments/grpo/evaluate.py \
  --adapter experiments/grpo/runs/qwen05-main/adapter \
  --split test --limit 0 --max-tokens 1024 --output experiments/grpo/runs/test-grpo.json

.venv/bin/tensorboard --logdir experiments/grpo/runs
```

评测输出 accuracy、boxed 格式率、think 格式率、平均生成长度、截断率和逐题回答。对照分析使用相同划分和生成上限，分别比较正确率、格式率与长度变化。

训练产物写入 `--output`，默认 `experiments/grpo/runs/grpo/`，包括日志、rollout、训练前后验证结果、`summary.json` 和 LoRA adapter。TensorBoard 在 `http://localhost:6006` 展示参数和曲线；`train/*`、`validation/*` 以采样批次为横轴，`update/*` 以实际优化器更新次数为横轴。

## 实验结果与分析

TODO
