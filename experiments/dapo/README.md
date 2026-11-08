# DAPO：数学推理的组相对策略优化

## 算法原理

本章在 DAPO-Math-17K 上对 Qwen3.5-4B 做 LoRA 强化学习，以 AIME25 的平均回答正确率评测。DAPO 不训练 critic，也不使用 reference KL；它在组相对策略优化中引入非对称裁剪、动态采样、token 等权损失和长度奖励。方法来源为 [DAPO 论文](https://dapo-sia.github.io/static/pdf/dapo_paper.pdf)，实验设置参考 [agentic-rl-lab](https://github.com/datawhalechina/agentic-rl-lab/tree/main/experiments/dapo)。

### 奖励与组内优势

给定题目 $`q`$，当前采样策略 $`\pi_{\mathrm{old}}`$ 独立生成 $`G`$ 条回答。每条回答的最后 300 个字符中，提取最后一个完整的 $`\mathrm{boxed}`$ 答案，使用 math-verify 判断数学等价；答对得 +1，否则得 −1。

训练与评测都在题目后追加：

```text
Please reason step by step, and put your final answer within \boxed{}.
```

使用模型 chat template，设置 `enable_thinking=False`；这是参考实验的提示模式，输出仍可包含逐步推理。

令回答长度为 $`T_i`$、生成上限为 $`L`$、惩罚窗口为 $`C`$，Soft Overlong Punishment 为：

$$
p(T_i)=-\min\left(1,\max\left(0,\frac{T_i-(L-C)}{C}\right)\right),\qquad
r_i=2c_i-1+p(T_i).
$$

其中 $`c_i\in\lbrace0,1\rbrace`$ 表示正确性。长度不超过 $`L-C`$ 时没有惩罚，之后线性下降，在上限处为 −1。截断回答仍参与训练，不额外屏蔽其 loss。

同一道题内计算奖励均值与样本标准差：

$$
\bar r=\frac1G\sum_i r_i,\quad
s=\sqrt{\frac1{G-1}\sum_i(r_i-\bar r)^2},\quad
A_i=\frac{r_i-\bar r}{s+10^{-8}}.
$$

一个回答的所有有效 token 共用该优势。这里用 $`G-1`$ 分母，与参考 DAPO 实现一致。

### 动态采样

仅保留满足以下条件的完整题组：

$$
0\lt\sum_{i=1}^{G}c_i\lt G.
$$

全对或全错的组被过滤，继续采样新题，直到凑齐目标全局题组数。筛选依据是正确性，优势依据是加上长度惩罚后的奖励；即使全错组的长度奖励不同，也不保留。

本章设置候选预算，默认最多采样目标组数的两倍。预算耗尽仍凑不满时整批跳过，模型不更新；参考实现允许不足目标组数时更新，本章按完整全局 batch 约定调整。训练尾部不足一批候选题时舍弃，下一 epoch 重新打乱。

### 非对称裁剪与 token 归约

令 $`o_{i,t}`$ 为回答 token，逐 token 概率比为：

$$
\rho_{i,t}=\exp\left(\log\pi_\theta(o_{i,t}\mid q_i,o_{i,\lt t})
-\log\pi_{\mathrm{old}}(o_{i,t}\mid q_i,o_{i,\lt t})\right).
$$

采用裁剪区间 [0.8, 1.28]，提高正优势动作的上限：

$$
\ell_{i,t}=-\min\left(\rho_{i,t}A_i,
\mathrm{clip}(\rho_{i,t},0.8,1.28)A_i\right),\qquad
\mathcal L=\frac{\sum_i\sum_{t=1}^{T_i}\ell_{i,t}}{\sum_i T_i}.
$$

所有回答 token 等权，长回答包含更多梯度项。不能先对每条回答平均，再对回答平均，也不能对每个 micro-batch 的均值等权累加。prompt 和 padding 不计入分母，生成出的首个 EOS 计入。

## 关键算法的代码实现解释

`prepare_data.py` 按规范化题面去重，删除答案冲突组，seed 42 划出 50 题 dev，并下载固定版本的 30 题 AIME25。训练入口再次检查训练题重复与 train/dev 交集。训练与 dev JSONL 的核心字段为 `question`、`answer`，源 `id` 用于对应输出。

一次训练尝试按以下顺序执行：

1. `train.py` 从打乱后的数据中取完整全局候选批，按完整题组分配给各进程。整个补采过程模型参数保持不变。
2. `task.sample_group` 编码 prompt，将同题的 G 条回答分成生成 micro-batch。生成使用 temperature=1、top-p=1、禁用 top-k，保持采样分布与训练 logprob 一致。
3. 每条回答解码、计算正确性和长度惩罚，再由 `group_advantages` 在完整组内标准化。各卡收集 CPU 记录，按候选位置排序；`select_batch` 过滤并选取目标数量的组，不够则继续补采。
4. 凑齐后将题组均匀分配到各卡。`cache_old` 在无梯度模式下缓存旧策略的逐 token logprob，更新期间固定这些值。
5. `batch_completion_logprobs` 用 prompt + completion[:-1] 前向，从 prompt 的最后一个位置开始预测回答。不同长度输入右侧补齐，仅抽取有效回答位置；`token_loss` 返回逐 token 裁剪损失。
6. `update_policy` 先汇总全局有效 token 数，再逐 micro-batch 累积 loss sum。DDP 会平均梯度，因此本卡损失乘以进程数再除以全局 token 数。只有最后一个 micro-batch 同步梯度，随后统一裁剪、执行一次 AdamW 更新。

Fabric 管理设备、精度、DDP、梯度裁剪和 TensorBoard。LoRA dropout 为零，基础模型 dropout 关闭，旧/新概率比较不会受到 dropout 随机性影响。默认每批只更新一次；第一次前向概率比为 1，但仍具有策略梯度。

`evaluate.py` 使用同一提示词和正确性判定，不加长度惩罚。dev 每题采样一次；正式 AIME25 每题采样 12 次，对所有回答计算平均正确率，不是“12 次中至少一次答对”的 pass@12。评测保留尾部数据。

## 运行方法

从仓库根目录执行。Python 3.12，PyTorch CUDA wheel 按本机 CUDA 环境安装，其余依赖：

```bash
.venv/bin/pip install -r requirements.txt
.venv/bin/python experiments/dapo/prepare_data.py
```

模型默认读取 `.cache/models/Qwen3.5-4B`，也可通过 `--model` 指定模型 ID 或本地路径。模型准备示例：

```bash
.venv/bin/python -c 'from modelscope import snapshot_download; snapshot_download("Qwen/Qwen3.5-4B", local_dir=".cache/models/Qwen3.5-4B")'
.venv/bin/python -c 'from modelscope import snapshot_download; snapshot_download("Qwen/Qwen3.5-0.8B", local_dir=".cache/models/Qwen3.5-0.8B")'
```

本机先用同系列小模型和原数据子集：

```bash
.venv/bin/python experiments/dapo/train.py \
  --model .cache/models/Qwen3.5-0.8B \
  --output experiments/dapo/runs/local \
  --steps 1 --train-limit 8 --batch-size 1 --group-size 4 \
  --max-candidate-multiplier 8 --max-tokens 512 --overlong-cache 128 \
  --eval-limit 2 --eval-max-tokens 512
```

小模型、较短生成上限可能使所有回答都错误，动态筛选后没有更新。这时查看日志中的有效组比例，扩大原任务候选预算或生成长度；不取消筛选条件。

正式配置：

```bash
.venv/bin/python experiments/dapo/train.py \
  --output experiments/dapo/runs/qwen4b
```

两卡使用同一入口，保持全局每批 4 题：

```bash
.venv/bin/torchrun --standalone --nproc_per_node=2 experiments/dapo/train.py \
  --batch-size 4 --output experiments/dapo/runs/qwen4b-ddp
```

| 参数 | 默认值与含义 |
| --- | --- |
| `--batch-size` | 4，全局有效题组数，必须能被进程数整除 |
| `--group-size` | 8，每题回答数，至少 2 |
| `--generation-micro-batch-size` | 1，一次生成的同题回答数 |
| `--micro-batch-size` | 1，每进程概率计算的回答数 |
| `--max-candidate-multiplier` | 2，每次尝试的候选题预算倍数 |
| `--max-prompt-tokens` | 512，训练 prompt 上限；超限报错，不截断题面 |
| `--max-tokens / --overlong-cache` | 4096 / 1024，回答上限 / 长度惩罚窗口 |
| `--epochs / --steps` | 1 / 0，遍历次数 / 总训练尝试上限，0 不额外限制；跳过也计一次尝试 |
| `--train-limit` | 0，0 使用全部训练题 |
| `--learning-rate` | 4e-5；AdamW betas=(0.9,0.95)，weight decay=0 |
| `--lora-rank / --lora-alpha` | 32 / 64；all-linear，dropout=0 |
| `--max-grad-norm` | 1，梯度范数裁剪 |
| `--eval-every / --eval-limit` | 25 / 50，验证间隔 / 验证题数；0 分别表示仅训练前后验证 / 全部 dev |
| `--eval-max-tokens` | 4096，dev 回答上限 |
| `--precision` | bf16，可选 fp32 |

训练保留完整全局候选 batch，数据少于一批直接报错。各 micro-batch 参数只控制显存与计算批量，不改变损失权重。采样温度、top-p 和裁剪边界固定为本章配置。

独立评测基础模型和训练后 adapter：

```bash
.venv/bin/python experiments/dapo/evaluate.py \
  --output experiments/dapo/runs/aime-base.json
.venv/bin/python experiments/dapo/evaluate.py \
  --adapter experiments/dapo/runs/qwen4b/adapter \
  --output experiments/dapo/runs/aime-dapo.json
.venv/bin/tensorboard --logdir experiments/dapo/runs
```

评测默认每题 12 次、8192 token 上限；可用 `--data` 指定 dev JSONL，以 `--repeats`、`--limit`、`--max-tokens` 调整规模。比较模型时使用相同设置。训练目录保存 TensorBoard 配置与曲线、逐条 rollout、验证结果、更新次数汇总和 adapter/tokenizer。每次训练指定新的输出目录。

本章用本地 Transformers + Fabric 替换参考实验的远程训练服务；LoRA alpha 明确设为 64。小模型运行用于流程验证，正式实验使用 4B；均不等同于论文 32B 模型的指标复现。

## 实验结果与分析

TODO
