# GSPO：按完整回答进行策略优化

## 算法原理

GSPO（Group Sequence Policy Optimization）让奖励、重要性比率和裁剪都作用于完整回答。本章以 [GSPO 论文](https://arxiv.org/abs/2507.18071) 的式 (5)–(7) 和四个 mini-batch 更新安排为主，实现单轮数学问答训练。

对问题 $`x`$，旧策略 $`\pi_{\mathrm{old}}`$ 生成 $`G`$ 条回答。第 $`i`$ 条回答含 $`T_i`$ 个有效 token，奖励 $`r_i`$ 为最终 boxed 答案的数学正确性：正确为 1，错误或缺失为 0。没有额外格式奖励或长度惩罚。组内优势为

$$
A_i=\frac{r_i-\bar r}{\sqrt{\frac{1}{G-1}\sum_{j=1}^{G}(r_j-\bar r)^2}+10^{-8}},
\qquad \bar r=\frac{1}{G}\sum_{j=1}^{G}r_j.
$$

样本标准差和稳定项是本章的具体数值选择。全对或全错的组优势全为零，仍保留在 batch 中，不动态补采。

定义当前与旧策略在实际生成 token 上的 log probability 为 $`\ell_{i,t}(\theta)`$ 和 $`\ell_{i,t}^{\mathrm{old}}`$。序列比率是 token 概率比的几何平均：

$$
s_i(\theta)=\exp\left[\frac{1}{T_i}\sum_{t=1}^{T_i}\left(\ell_{i,t}(\theta)-\ell_{i,t}^{\mathrm{old}}\right)\right].
$$

它不是 token ratio 的算术平均。长度归一化使不同回答长度的比率处于可比较的数值范围。一个更新批次有 $`N`$ 条回答时，最小化

$$
L(\theta)=-\frac{1}{N}\sum_{i=1}^{N}\min\left[s_i(\theta)A_i,
\mathrm{clip}\left(s_i(\theta),1-3\times10^{-4},1+4\times10^{-4}\right)A_i\right].
$$

每回答等权，回答内部的 token 通过 $`1/T_i`$ 平均贡献梯度。正优势且比率过高时裁剪；负优势且比率过低时裁剪，不能把所有越界序列都视为梯度被截断。零优势回答也计入 $`N`$。

一个 rollout batch 按完整题组等分成四个 mini-batch，顺序更新四次，每条回答只使用一次。四次更新共用 rollout 时的旧概率。第一次更新前当前策略与旧策略一致，ratio 接近 1；后续批次面对已更新的策略，序列裁剪才有机会发挥作用。micro-batch 只是显存管理与梯度累积单位，不改变这四次更新。

本章实验使用 Qwen3.5-4B LoRA、DAPO-Math 与 AIME25；论文使用 Qwen3-30B-A3B-Base 微调得到的冷启动模型，评测 AIME24、LiveCodeBench 和 CodeForces。本章不复现论文的 MoE 规模指标。参考仓库的 GSPO 每轮仅更新一次，本章采用论文的四批更新安排。

## 关键算法的代码实现解释

`task.py` 集中实现任务与计算，`train.py` 串联以下流程：

1. **数据与 rollout**：检查 train/dev 题面无交集、训练题不重复；每轮固定取一批问题，按完整题组分配给各进程。`sample_group` 用模型 chat template 构建 prompt，关闭 thinking，要求最终答案放入 `\boxed{}`。`generate` 用温度 1、top-p 1、top-k 0 采样，生成 micro-batch 可分批生成同题的回答。
2. **奖励与优势**：`score_answer` 提取最后一个完整 boxed 答案，使用 `math_verify` 判断等价；`group_advantages` 一次处理该题全部奖励。优势随回答保存，此后不重新计算。
3. **旧概率缓存**：`cache_old` 在任何 optimizer 更新前计算整批回答概率并 detach。`batch_completion_logprobs` 输入 prompt 加回答去掉末 token，取 prompt 最后位置起的 logits 预测回答 token。右侧 padding 通过 attention mask 屏蔽；只返回有效回答概率。生成保留首个 EOS，删除其后的 padding；达到长度上限的回答保留所有已生成 token。
4. **序列 loss**：`sequence_objective` 在 float32 中计算平均 log-ratio、指数和 clipped surrogate，返回单个回答的 loss、ratio 和实际裁剪标记。所有概率计算关闭 dropout，避免把随机性误认为策略变化。
5. **策略更新**：本卡题组等分成四段，对应四个全局 mini-batch。每段先清空梯度，按 micro-batch 前向和反向，最后同步并裁剪梯度，再执行一次 optimizer step。DDP 默认平均梯度，所以本卡 loss sum 乘以 `world_size / 全局回答数`；零优势组仍在分母中。仅全局全零优势时共同跳过更新，某一进程全零仍参与同步。
6. **下一轮与保存**：四段结束后用新策略采样下一批；TensorBoard 分别记录 rollout 轮数与 mini-batch 更新位置。定期与结束时通过 PEFT 保存 adapter 和 tokenizer。

`prepare_data.py` 固定 DAPO-Math 和 AIME25 的数据版本，剥离 DAPO 原答案模板、按规范化题面去重、丢弃答案冲突题组，固定 seed 切出 50 题 dev。`evaluate.py` 对 JSONL dev 或本地 AIME25 评测，复用训练提示词、采样和答案判定，输出逐回答记录与指标。Average@12 是所有回答正确率，Pass@12 是每题 12 次中至少一次正确的题目比例；同时报告格式率、平均长度与截断率。

## 运行方法

在仓库根目录执行。Python 环境使用根目录 `requirements.txt`；按本机 CUDA 环境安装匹配的 PyTorch wheel 后安装其余依赖：

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python experiments/gspo/prepare_data.py
```

默认模型为 `Qwen/Qwen3.5-4B`，也可用 `--model .cache/models/Qwen3.5-4B` 指向本地缓存。

本机从原任务子集跑通流程：

```bash
.venv/bin/python experiments/gspo/train.py \
  --model .cache/models/Qwen3.5-4B \
  --train-limit 32 --steps 2 --group-size 2 \
  --max-tokens 256 --output experiments/gspo/runs/local
```

较短的生成上限可能导致回答没有最终答案、整组零优势；流程跑通不等于学到了数学能力。正式实验保留足够的回答长度：

```bash
.venv/bin/python experiments/gspo/train.py \
  --steps 100 --rollout-batch-size 8 --group-size 8 \
  --num-mini-batches 4 --max-prompt-tokens 2048 --max-tokens 4096 \
  --output experiments/gspo/runs/math-4b
```

参数含义与默认值：

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `rollout-batch-size` | 8 | 每轮全局题数 |
| `group-size` | 8 | 每题回答数，至少 2 |
| `num-mini-batches` | 4 | 每轮依次执行的更新批次数 |
| `micro-batch-size` | 1 | 每次训练前向的回答数 |
| `generation-micro-batch-size` | 1 | 同题一次生成的回答数 |
| `steps` / `epochs` | 100 / 1 | rollout 轮数上限 / 数据遍历上限，先达到者结束；steps=0 表示仅按 epochs |
| `train-limit` | 0 | 训练题数上限；0 使用全部 |
| `learning-rate` | 4e-5 | AdamW 学习率，betas=(0.9, 0.95)，weight decay=0 |
| `lora-rank` / `lora-alpha` | 32 / 64 | all-linear LoRA，dropout=0 |
| `max-grad-norm` | 1 | 每次更新的全局梯度范数上限 |
| `precision` | bf16 | 可选择 fp32 |
| `save-every` | 20 | 每多少轮保存 adapter；0 仅保存最终 adapter |

四卡通过相同入口启动，并调整题数使每个 mini-batch 能均匀分配到四卡：

```bash
.venv/bin/torchrun --standalone --nproc_per_node=4 experiments/gspo/train.py \
  --rollout-batch-size 16 --num-mini-batches 4 \
  --output experiments/gspo/runs/math-4gpu
```

DDP 每卡持有完整模型；全局题数必须能被 `num-mini-batches × 进程数` 整除。训练丢弃不足完整 rollout batch 的尾部，数据不足一批时报错。评测不丢弃尾部。模型、上下文长度和 batch 参数须结合实际硬件安排。

分别评测基础模型和 adapter，保持数据、seed、长度和重复数一致：

```bash
.venv/bin/python experiments/gspo/evaluate.py \
  --model .cache/models/Qwen3.5-4B --repeats 12 \
  --output experiments/gspo/runs/eval-base.json
.venv/bin/python experiments/gspo/evaluate.py \
  --model .cache/models/Qwen3.5-4B --repeats 12 \
  --adapter experiments/gspo/runs/math-4b/adapter \
  --output experiments/gspo/runs/eval-adapter.json
.venv/bin/tensorboard --logdir experiments/gspo/runs
```

评测默认 AIME25、最大回答 8192 token。可用 `--data experiments/gspo/datasets/dev.jsonl --limit 4 --repeats 1` 做小规模评测。adapter 必须搭配训练时的基础模型。运行目录应使用新路径，训练拒绝覆盖非空目录。

## 实验结果与分析

TODO
