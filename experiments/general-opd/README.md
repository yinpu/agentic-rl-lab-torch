# General OPD：DeepMath 上的在线蒸馏

## 算法原理

本章用 Qwen3.5-4B 作为学生、Qwen3.6-27B 作为冻结教师，在 DeepMath-103K 的数学题目上训练学生 LoRA。训练只读取题目，标准答案用于独立评测。

参考 [General OPD 同步实验](https://github.com/datawhalechina/agentic-rl-lab/tree/238bc688ca7d5c994c580f732ee9c2c92483bd20/experiments/general-opd)。本章对齐参考同步脚本的模型和显式训练参数，使用本地 Transformers、PEFT 和 Lightning Fabric，并增加留出集评测。该配置来自参考教程的实验脚本，不将其视为原论文的完整复现配置。

学生先从当前策略采样回答 $y$。教师读取完全相同的 prompt 和回答 token，只打分，不另生成训练答案。在固定的学生轨迹上计算逐 token 信号：

$$
A_t=\beta\left[\log\pi_T(y_t\mid x,y_{<t})-\log\pi_{\mathrm{old}}(y_t\mid x,y_{<t})\right].
$$

教师更偏好的 token 得到正优势；学生过度偏好的 token 得到负优势。它不同于 GRPO 的组内奖励：每个 token 有自己的优势，同题多个回答之间不计算均值或标准差。

令 $\rho_t=\exp(\log\pi_\theta-\log\pi_{\mathrm{old}})$，全局 batch 有 $N$ 个有效回答 token，更新目标为：

$$
L=-\frac{1}{N}\sum_{i,t}\rho_{i,t}\operatorname{stopgrad}(A_{i,t}).
$$

本章明确使用有效回答 token 均值，没有 PPO clip。参考脚本只指定服务端 `importance_sampling`，未在该脚本展开归约实现，因此不声称与其数值等价。逐 token 的即时优势遵循参考 demo，不将其描述为完整序列 KL 的无偏策略梯度。

缓存的旧概率和教师概率不求导；当前概率保留梯度。每批只更新一次，更新前概率比数值为 1，但其梯度不为零。每次更新后重新采样。

日志 `reverse_kl` 是学生轨迹上 $\log\pi_{old}-\log\pi_T$ 的 token 平均，可以为负，也不能代替数学正确率。

## 关键算法的代码实现

### 从题目到同轨迹概率

[task.py](task.py) 的 `load_deepmath` 读取 ModelScope parquet，`make_split` 按规范化题目去重、固定 seed 打乱、先留出验证题再截取训练子集。训练记录中的答案不进入 prompt 或 loss。

`prompt_ids` 使用学生原生 chat template、关闭 thinking，并在题目后添加：

```text
Please solve the problem step by step and put the final answer in \boxed{}.
```

`check_tokenizers` 要求两模型共享词表 ID 和特殊 token。教师直接读取学生编码的序列，不重新套用模板。

`rollout` 用当前学生为每题生成 `group_size` 条回答。生成时左补齐，按补齐后的输入宽度截取回答，保留首个 EOS，移除之后的 padding。temperature=1、top-p=1、top-k=0，避免采样变换改变概率分布。模型 dropout 关闭。

随后按 `micro_batch_size` 分块，无梯度计算学生和教师概率：

```python
old = batch_completion_logprobs(student, chunk)
target = batch_completion_logprobs(teacher, chunk)
```

返回记录包含 `prompt`、`tokens`、`old`、`teacher` 和截断标记；概率缓存转到 CPU，下一阶段只将当前 micro-batch 搬回设备。

### token 对齐、优势与更新

`batch_completion_logprobs` 右补齐 `prompt + completion[:-1]`，取回答预测位置的词表 logits。例如：

```text
输入：p0 p1 p2 c0 c1
目标：      c0 c1 EOS
```

仅投影回答预测位置的并集，再逐条选取真实位置，以交叉熵取负得到 logprob。prompt 和 padding 不进入损失；截断回答的已有 token 仍参与训练。

[train.py](train.py) 每批调用 `rollout`，然后调用 `update_policy`。`opd_loss` 将上述公式直接写为：

```python
advantage = beta * (teacher.detach() - old.detach())
return -torch.exp(current - old.detach()) * advantage
```

`update_policy` 先清空梯度，汇总所有进程有效 token 数 `count`，按 micro-batch 计算 token 损失之和并反传：

```python
fabric.backward(loss * fabric.world_size / count)
```

DDP 会平均各卡梯度，乘进程数后得到全局 token 均值。这样不同回答长度、不同卡的 token 数不会改变归约含义。只有最后一次反传同步梯度；整批累积结束后裁剪梯度、更新 LoRA，再进入新一轮采样。教师始终冻结。

### 独立评测

[evaluate.py](evaluate.py) 对固定留出集贪心生成，保留不足一批的尾部。训练入口在训练前评测基础学生和教师，训练后评测学生。每条记录保存题目、答案、回答、格式、长度和截断标记。

提取最后一个完整 `\boxed{...}`（支持嵌套花括号），去除空白、外层美元符号和 `\left`/`\right` 后做精确字符串匹配。`exact_match` 是这一规则下的数学答案匹配率，不进行代数等价判定。三种模型必须使用同一题集和生成预算比较；KL 降低不必然带来匹配率提高。

## 运行方法

从仓库根目录运行，使用 Python 3.12 和根目录 `.venv`：

```bash
python -m venv .venv
.venv/bin/python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -c 'from modelscope import snapshot_download; snapshot_download("Qwen/Qwen3.5-4B", local_dir=".cache/models/Qwen3.5-4B"); snapshot_download("Qwen/Qwen3.6-27B", local_dir=".cache/models/Qwen3.6-27B")'
```

数据首次运行自动下载，默认读取 DeepMath 的第一个分片。按参考同步脚本的默认训练规模运行：

```bash
.venv/bin/python experiments/general-opd/train.py \
  --output experiments/general-opd/runs/main \
  --num-shards 1 --train-limit 1000 --steps 10 \
  --batch-size 2 --group-size 2 --max-tokens 512 \
  --lora-rank 32 --learning-rate 4e-5 --beta 1 --seed 42
```

每步全局 2 道题、每题 2 条回答，共 4 条回答。采样 temperature=1、top-p=1、禁用 top-k，关闭 thinking；每步更新后刷新采样策略。优化器 betas=(0.9, 0.95)，与参考脚本一致。参考 PyTRIO 的 `top_k=-1` 与本章 Transformers 的 `top_k=0` 均表示禁用 top-k。

多卡使用同一入口，每卡各自加载完整教师副本，学生使用 DDP。下面两卡启动仍保持全局 2 题、4 条回答：

```bash
.venv/bin/torchrun --standalone --nproc_per_node=2 experiments/general-opd/train.py \
  --output experiments/general-opd/runs/ddp \
  --num-shards 1 --train-limit 1000 --steps 10 \
  --batch-size 2 --group-size 2 --max-tokens 512 \
  --micro-batch-size 1 --lora-rank 32 --learning-rate 4e-5 --beta 1 --seed 42
```

增加进程数时，全局 batch 仍需被进程数整除；增加全局 batch 会改变参考训练规模。DDP 不会将教师权重切分到多张卡。

| 参数 | 默认值与含义 |
| --- | --- |
| `--student` / `--teacher` | `.cache/models/Qwen3.5-4B` / `.cache/models/Qwen3.6-27B`；也接受模型 ID |
| `--batch-size` | 2；全局题数，必须被进程数整除 |
| `--group-size` | 2；每题回答数，不计算组内相对奖励 |
| `--micro-batch-size` | 1；每进程缓存概率和训练前向的回答数，不控制生成批量 |
| `--max-tokens` | 512；训练回答上限 |
| `--steps` / `--train-limit` | 10 / 1000；步数上限 / 训练题上限，0 表示遍历全部 |
| `--validation-size` / `--eval-limit` | 64 / 64；留出题数 / 实际评测题数，eval-limit=0 使用全部 |
| `--eval-max-tokens` / `--eval-batch-size` | 512 / 1；贪心评测生成上限 / 批量 |
| `--beta` / `--learning-rate` | 1 / 4e-5；优势强度 / AdamW 学习率 |
| `--lora-rank` / `--lora-alpha` | 32 / 64；all-linear，dropout=0 |
| `--max-grad-norm` / `--precision` | 1 / bf16；梯度裁剪 / 可选 fp32 |
| `--seed` / `--num-shards` | 42 / 1；随机种子 / 数据分片数 1–10 |

LoRA alpha=64、all-linear、dropout=0、weight decay=0、梯度裁剪=1 和 bf16 是本地实现的显式设置；参考脚本没有完整给出这些服务端设置，不宣称它们与服务端默认值相同。留出集去重与划分也属于本章新增设置，因此训练题目不与参考的直接抽样逐条一致。

单卡和多卡都舍弃训练尾部不足一个全局 batch 的题目；数据不足一批直接报错。`steps` 不回绕数据。实际训练规模由模型、回答长度和可用显存决定。

独立加载 adapter 重评固定留出集：

```bash
.venv/bin/python experiments/general-opd/evaluate.py \
  --adapter experiments/general-opd/runs/main/adapter \
  --split-file experiments/general-opd/runs/main/splits.json \
  --max-tokens 512 --output experiments/general-opd/runs/main/reloaded.json
.venv/bin/tensorboard --logdir experiments/general-opd/runs
```

不传 `--adapter` 评测基础学生；设置 `--model .cache/models/Qwen3.6-27B` 评测教师。保持相同 `--limit` 和 `--max-tokens`。留出集文件保存实际问答内容，独立评测无需再次下载或重新划分数据。

产物包括 LoRA adapter 和 tokenizer、固定验证集、训练前后及教师评测 JSON；运行参数与训练曲线由 TensorBoard 保存。输出目录必须为空。

## 实验结果与分析

TODO
