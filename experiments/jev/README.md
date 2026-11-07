# Jev 决策模型

## 算法原理

Jev 将语言模型用于固定候选决策：输入状态和若干问题，一次前向计算各字段的候选概率。例如，同一张工单同时决定负责团队、优先级和是否退款。本章使用 Qwen3.5-0.8B，通过 LoRA masked SFT 学习决策。

### 候选编码与占位骨架

一条数据包含 `state`、`questions` 和训练标签 `targets`：

```json
{
  "id": "ticket-1",
  "state": {"request": "I was charged twice. Please refund the extra charge."},
  "questions": {
    "team": {
      "type": "choice",
      "instructions": "Which team should handle this ticket?",
      "criteria": {"billing": "Payments and refunds", "technical": "Technical support"}
    },
    "refund": {"type": "noul", "instructions": "Is a refund explicitly requested?"}
  },
  "targets": {"team": {"label": "billing"}, "refund": {"label": "yes"}}
}
```

各字段的候选依次映射到 `A-Z a-z 0-9`，最多 62 个。`choice` 保持 criteria 的插入顺序；`score` 列表按下标排列，字典保持给定顺序；`noul` 固定 `no → A, yes → B`。

system 提示词要求用符号作答，user 提示词包含状态和决策 schema，assistant 输入保留固定骨架：

```json
{
    "team": "decision",
    "refund": "decision"
}
```

`decision` 是 tokenizer 已有的单 token，占据各字段的决策位置。普通文本分段编码，marker 直接以 token ID 插入，不与字段名或引号一起分词。训练和推理共用这一编码方式及 [固定模板](templates/qwen3_5_jev.jinja)，assistant 前包含空的 `<think>` 块。输入始终保留占位符，后面的字段看到的是前面字段的占位符。

### 训练：marker 位置的全词表交叉熵

设第 $f$ 个字段的 marker 下标为 $m_f$，正确候选符号为 $y_f$。因果语言模型使用位置 $m_f-1$ 的 logits 预测它：

$$
\ell_f=-\log\frac{\exp z_{m_f-1,y_f}}{\sum_{v\in V}\exp z_{m_f-1,v}},
\qquad
L=\frac{1}{\sum_i|F_i|}\sum_i\sum_{f\in F_i}\ell_{i,f}.
$$

训练 softmax 的分母覆盖完整词表，只在 marker 的预测位置计算监督损失。全局 batch 内每个字段等权，因此五字段案例贡献五项损失，单字段案例贡献一项。

### 推理：候选内归一化

推理使用同一骨架，一次前向提取所有字段的候选 logits。对于候选集合 $C_f$ 和温度 $T>0$：

$$
p_f(c;T)=\frac{\exp(z_{m_f-1,c}/T)}{\sum_{c'\in C_f}\exp(z_{m_f-1,c'}/T)}.
$$

| 题型 | 输出 |
| --- | --- |
| `choice` | 候选概率分布、最大概率选项 |
| `noul` | no/yes 概率、`p_yes`、二选一结果；平局选 yes |
| `score` | 等级概率、最大概率等级、期望分 $\sum_c\mathrm{value}(c)p_f(c)$ |

温度校准在 calibration 划分上最小化字段平均 NLL：

$$
T^*=\arg\min_{T>0}\frac{1}{M}\sum_{f=1}^{M}-\log p_f(y_f;T).
$$

正温度保持 argmax，调整预测置信度。

任务、切分、提示词和决策机制参考 [PyTrio-Jev 章节](https://github.com/datawhalechina/agentic-rl-lab/tree/8493321013c02eef6cf79ca2594811f53c6fdac1/experiments/pytrio-jev)。参考章使用 Qwen3.5-4B，本章默认使用 0.8B，并直接从本地模型取得全部候选 logits；参考服务通过 top-k logprobs 近似候选概率。改编代码与模板的许可证见 [LICENSE.upstream](LICENSE.upstream)。

## 关键算法的代码实现

### 从案例到决策的流程

本章先进行监督训练，再对固定 checkpoint 做校准和评测：

1. `Encoder.build_example` 将状态、问题和占位骨架编码为输入 token，输出各字段的预测位置与正确符号 token ID。
2. `train.py::main` 根据单卡 batch、梯度累积次数和卡数计算全局 batch，每轮打乱案例并取完整批次；`update` 分配各卡案例，每次按单卡 batch 前向，计算 marker 位置的全词表交叉熵并累积梯度，最后更新 LoRA。
3. 训练后比较 checkpoint 的 dev 指标，选定 adapter。
4. `evaluate.py::calibrate` 在 calibration 案例上收集各字段的候选 logits 和标签，交给 `fit_temperature` 拟合一个温度。
5. `evaluate.py::evaluate` 用固定 adapter 和温度处理测试案例，经 `candidate_logits`、`build_decisions` 得到决策，再汇总准确率与校准指标。

训练批次直接使用编码好的案例，核心调用为：

```python
examples = [encoder.build_example(r, args.max_length) for r in records]
```

```python
global_batch_size = (args.per_device_train_batch_size
                     * args.gradient_accumulation_steps * fabric.world_size)
for epoch in range(args.epochs):
    random.Random(args.seed + epoch).shuffle(examples)
    for start in range(0, train_size, global_batch_size):
        batch = examples[start:start + global_batch_size]
        metrics = update(model, optimizer, batch, fabric,
                         args.max_grad_norm, args.per_device_train_batch_size)
```

### 编码与 marker 定位

[jev.py](jev.py) 的 `compile_record` 将候选映射为符号，并生成以各字段 marker 为边界的 JSON 骨架片段；字段名由 `json.dumps` 转义。`Encoder` 验证候选符号均为不同的单 token，先编码提示词，再依次编码骨架片段、插入 `marker_id`，最后编码骨架结尾与模板后缀。插入 marker 时，当前 token 序列长度就是它的位置：

```python
for field, part in zip(compiled["fields"], parts[:-1], strict=True):
    ids.extend(self.tokenizer.encode(part, add_special_tokens=False))
    markers[field] = len(ids)
    ids.append(self.marker_id)
```

`build_example` 将各字段的预测位置设为 `marker_index - 1`，标签设为正确符号的 token ID。推理也调用同一个 `Encoder.encode`，但不读取训练标签。相较参考章对整段文本一次分词，这里显式固定 marker 的 token 边界；分段拼接后的 token 序列可能不同，解码后仍是同一份模板文本。

`decision_logits` 复用 `batch_decision_logits`：批量右侧补齐，用 attention mask 标记有效输入，将各案例 marker 前一位置合并为 `selected`：

```python
logits = model(input_ids=ids, attention_mask=mask, use_cache=False,
               logits_to_keep=selected).logits
positions = torch.tensor(example["positions"], device=device)
columns = torch.searchsorted(selected, positions)
field_logits.append(logits[i, columns].float())
```

`logits_to_keep` 只在所需位置执行词表投影，返回每个字段的完整词表 logits。

### 字段等权与梯度累积

[train.py](train.py) 的 `update` 先统计全局 batch 的字段总数 `count`，再按进程分配案例。每个案例将全部字段的交叉熵求和：

```python
loss = F.cross_entropy(values, targets, reduction="sum")
losses.append(loss * fabric.world_size / count)
fabric.backward(torch.stack(losses).sum())
```

DDP 会平均各卡梯度，乘 `world_size / count` 后得到全局字段平均损失。各卡字段数可以不同。每个更新批次开始先清空梯度，各卡每次处理 `per_device_train_batch_size` 条案例，执行 `gradient_accumulation_steps` 次前向和反向。`fabric.no_backward_sync` 使梯度只在最后一次反传时同步。累积完成后，统一处理梯度范数并执行一次 `optimizer.step()`，然后进入下一批。每次反传都按整个更新批次的字段总数归一，保证字段等权。

### 决策概率与温度拟合

推理时，`jev.py::candidate_logits` 用 `include_targets=False` 编码案例，从 marker 前一位置的全词表 logits 中选出各字段候选符号，返回“字段名 → 候选 logits 向量”。`build_decisions` 对每个字段计算：

```python
probabilities = torch.softmax(logits[field] / temperature, dim=0).tolist()
```

再将符号概率映射回原选项值，输出最大概率选项，并为 `noul` 计算 `p_yes`、为 `score` 计算期望分。评测中的核心调用为：

```python
logits = candidate_logits(model, encoder, record, args.device, args.max_length)
decisions = build_decisions(record, logits, temperature)
```

校准复用同一个 `candidate_logits`，将每个字段的 `(候选 logits, 正确候选下标)` 收集到 `instances`，随后调用 `fit_temperature(instances)`。模型权重保持固定，只拟合供 `build_decisions` 使用的温度。

[evaluate.py](evaluate.py) 的 `fit_temperature` 使用逆温度 $\beta=1/T$。NLL 对 $\beta$ 凸，导数为 $\mathbb{E}_{p_\beta}[z]-z_{\mathrm{gold}}$，对应：

```python
def derivative(beta):
    return sum((torch.softmax(z * beta, dim=0).dot(z) - z[y]).item()
               for z, y in instances) / len(instances)
```

函数在 $\beta\in[0.01,100]$ 内二分求导数零点，最优点超出区间时取边界。校准结果绑定基础模型和 adapter，评测时检查匹配关系。

## 运行方法

以下命令从仓库根目录执行，使用 Python 3.12 和项目 `.venv`。首次创建环境及安装 PyTorch 见 [GRPO 的运行方法](../grpo/README.md#运行方法)。

### 模型与数据准备

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
export HF_HOME="$PWD/.cache/huggingface"
export HF_DATASETS_CACHE="$PWD/.cache/datasets"

python - <<'MODEL'
from modelscope import snapshot_download
snapshot_download("Qwen/Qwen3.5-0.8B", local_dir=".cache/models/Qwen3.5-0.8B")
MODEL
export JEV_MODEL="$PWD/.cache/models/Qwen3.5-0.8B"

python experiments/jev/prepare_data.py
```

[prepare_data.py](prepare_data.py) 生成以下数据，保存在 `experiments/jev/datasets/`：

| 划分 | 来源与规模 | 用途 |
| --- | --- | --- |
| train | Typed Decisions 960 案例/4800 字段 + ToolACE 1200 案例/1200 字段 | masked SFT |
| dev | Typed Decisions 120 案例/600 字段 | 选择 checkpoint |
| calibration | Typed Decisions 120 案例/600 字段 | 拟合温度 |
| test | Jevbench Easy 48、Original 72、Hard 111；Typed Decisions 400 案例/2000 字段；ToolACE 310 | 最终评测 |

Typed Decisions 使用 `LocalLLaMA/typed-decisions` 的 `all` 配置。每个 workflow 的 300 案例按 `sha256("pytrio-jev:" + id)` 排序，切为 240/30/30，训练使用硬标签。

ToolACE 从 `Team-ACE/ToolACE` 提取只调用一个工具的记录；请求文本按小写和空白归一，排除与评测请求重叠的样本，再用 seed 42 打乱采样 1200 条。五项测试数据来自 [Intern-Decision](https://github.com/InternLM/Intern-Decision/tree/565350244af860e1c276768cd80bdd9b8be93913/benchmarks/accuracy-v1) 的固定 commit。

### 子集运行与完整训练

先用原训练集的小子集运行训练和推理：

```bash
python experiments/jev/train.py --model "$JEV_MODEL" \
  --max-samples 16 --per-device-train-batch-size 2 --gradient-accumulation-steps 2 \
  --epochs 1 --max-steps 2 \
  --save-every 0 --output experiments/jev/runs/jev-smoke

python experiments/jev/jev.py --model "$JEV_MODEL" \
  --adapter experiments/jev/runs/jev-smoke/final \
  --input experiments/jev/datasets/dev.jsonl --limit 2
```

全局 batch = 单卡 batch × 梯度累积次数 × 卡数。完整训练在单卡默认配置下使用 2160 条案例、2 个 epoch、单卡 batch 1、累积 16 次，全局 batch 为 16，共 270 次更新：

```bash
python experiments/jev/train.py --model "$JEV_MODEL" \
  --epochs 2 --per-device-train-batch-size 1 --gradient-accumulation-steps 16 \
  --output experiments/jev/runs/jev-full
```

多卡使用同一脚本。下面两卡配置将累积次数改为 8，保持全局 batch 为 16：

```bash
torchrun --standalone --nproc_per_node=2 experiments/jev/train.py \
  --model "$JEV_MODEL" --epochs 2 \
  --per-device-train-batch-size 1 --gradient-accumulation-steps 8 \
  --output experiments/jev/runs/jev-ddp
```

| 参数 | 默认值与含义 |
| --- | --- |
| `--model` | `Qwen/Qwen3.5-0.8B`；接受模型 ID 或本地目录 |
| `--per-device-train-batch-size` | 1；每张卡一次前向、反向处理的案例数 |
| `--gradient-accumulation-steps` | 16；每张卡累积多少次反向传播后更新一次参数；卡数增加而参数不变时，全局 batch 相应增大 |
| `--epochs` | 2；每轮按 seed+epoch 打乱 |
| `--learning-rate` | 1e-4；恒定学习率，AdamW，betas=(0.9, 0.999)，eps=1e-8，weight decay=0 |
| `--lora-rank` / `--lora-alpha` | 32 / 32；文本骨干 all-linear，dropout=0 |
| `--max-grad-norm` | 0；关闭梯度裁剪，大于 0 时对整批梯度统一裁剪 |
| `--max-length` | 8192；输入 token 上限，超长报错，0 取消限制 |
| `--max-samples` / `--max-steps` | 0 / 0；案例数 / 更新次数上限，0 使用全部 |
| `--save-every` | 60；每隔若干步保存，0 关闭步骤保存；每轮结束和 final 仍保存 |
| `--precision` | bf16；可设为 fp32 |

训练冻结基础模型，仅更新 LoRA，启用梯度检查点。单卡和多卡都舍弃不足一个全局 batch 的训练尾部，数据至少需要一批。评测和校准使用所选的全部案例。

每次训练使用新的输出目录。训练配置与指标保存在 `tensorboard/`，`step-*`、`epoch-*`、`final` 中保存 LoRA 权重与 tokenizer；默认输出目录为 `experiments/jev/runs/jev/`。

### 选优、校准与评测

先比较各 checkpoint 的 dev 指标，选定 adapter，再用 calibration 拟合温度，最后评测 test。以下命令以 `final` 为例：

```bash
python experiments/jev/evaluate.py eval --model "$JEV_MODEL" --suite dev \
  --output experiments/jev/runs/jev-full/dev-base.json
python experiments/jev/evaluate.py eval --model "$JEV_MODEL" \
  --adapter experiments/jev/runs/jev-full/final --suite dev \
  --output experiments/jev/runs/jev-full/dev-final.json

python experiments/jev/evaluate.py calibrate --model "$JEV_MODEL" \
  --adapter experiments/jev/runs/jev-full/final \
  --output experiments/jev/runs/jev-full/calibration.json

python experiments/jev/evaluate.py eval --model "$JEV_MODEL" --suite all \
  --output experiments/jev/runs/jev-full/test-base.json
python experiments/jev/evaluate.py eval --model "$JEV_MODEL" \
  --adapter experiments/jev/runs/jev-full/final --suite all \
  --output experiments/jev/runs/jev-full/test-raw.json
python experiments/jev/evaluate.py eval --model "$JEV_MODEL" \
  --adapter experiments/jev/runs/jev-full/final --suite all \
  --calibration experiments/jev/runs/jev-full/calibration.json \
  --output experiments/jev/runs/jev-full/test-calibrated.json
```

评测保存逐字段预测和以下指标：

- accuracy：字段命中率，score 使用最大概率等级。
- ECE：以最大候选概率作为置信度，使用 10 个等宽分桶。
- Brier：每字段对全部候选求 $(p-\mathrm{one\_hot}_{\mathrm{gold}})^2$ 之和，再跨字段平均。
- NLL：正确候选的负对数概率。
- 五项宏平均准确率：各基准 accuracy 的算术平均。

比较基础模型与训练后各基准的 accuracy，以及校准前后的 ECE、Brier 和 NLL。训练参数与曲线可通过 TensorBoard 查看，横轴为实际优化器更新次数：

```bash
.venv/bin/tensorboard --logdir experiments/jev/runs
```

浏览器打开 `http://localhost:6006`。

## 实验结果与分析

TODO
