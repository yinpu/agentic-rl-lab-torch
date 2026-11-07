# 第二节：Jev 决策模型

本章把语言模型用于固定选项的决策：输入一段状态和若干问题，一次前向输出每个字段的候选概率，不生成答案文本。例如同一张工单同时决定负责团队、优先级和是否退款。

本章对应参考仓库的 [PyTrio-Jev](https://github.com/datawhalechina/agentic-rl-lab/tree/8493321013c02eef6cf79ca2594811f53c6fdac1/experiments/pytrio-jev)。这里的训练是 **LoRA masked SFT**，没有奖励、优势计算或策略梯度；它不是 GRPO。

默认使用较小的 [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B)，参考章使用 Qwen3.5-4B。数据任务、切分、提示词和决策机制对齐参考章，结果不能直接称为参考章 4B 实验复现。

## 1. 模型怎样做决策

一条数据采用 canonical JSON，`questions` 中每个字段包含 `type`、`instructions` 和可选的 `criteria`；训练时另有 `targets`，推理不需要标签。

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

[`data.py`](data.py) 的 `compile_record` 将选项依次映射到 `A-Z a-z 0-9`，最多 62 个选项。`choice` 保持 criteria 插入顺序；`score` 的列表按下标 0、1、2…，字典保持给定顺序；`noul` 固定 `no → A, yes → B`。`score` 的字典键须能转换为数值，以便计算期望分。

system 提示词要求用符号作答，user 提示词包含 state 和决策 schema，assistant 内容却始终是如下骨架：

```json
{
    "team": "decision",
    "refund": "decision"
}
```

`decision` 是已有的单 token，不扩展词表。训练和推理共用参考章的 [固定模板](templates/qwen3_5_jev.jinja)，assistant 前包含空的 `<think>` 块，不执行思维链生成。`Encoder` 验证所有符号是不同的单 token，用 JSON value 的字符范围定位 marker；state 或字段名里也出现 `decision` 不会混淆。

### 训练：只监督 marker 前一位置

设第 f 个字段的 marker 位于 token 下标 m_f，正确符号是 y_f。因果语言模型在位置 m_f−1 输出下一个 token 的 logits：

\[
\ell_f=-\log\frac{\exp z_{m_f-1,y_f}}{\sum_{v\in V}\exp z_{m_f-1,v}},\qquad
L=\frac{1}{\sum_i |F_i|}\sum_i\sum_{f\in F_i}\ell_{i,f}.
\]

分母是**完整词表**。训练输入保留 marker，不将其替换成正确答案，所以后面的字段看不到前面字段的金标准答案。普通语言建模的“答案填入输入，再 mask labels”会改变这里的条件信息。

[`modeling.py`](modeling.py) 的 `decision_logits` 用 `logits_to_keep` 仅在这些位置计算 lm_head，降低长序列的 logits 显存占用；[`train.py`](train.py) 的 `update` 对所选位置执行全词表交叉熵。一个全局 batch 内逐案例累积，以字段总数归一。它没有把每条记录的平均 loss 再平均，否则 ToolACE 的单字段记录与 Typed Decisions 的五字段记录会被错误加权。

### 推理：只在候选之间归一化

[`inference.py`](inference.py) 使用同一个骨架，在一次前向中提取所有字段的候选 logits。对于字段候选集合 C_f：

\[
p_f(c;T)=\frac{\exp(z_{m_f-1,c}/T)}{\sum_{c'\in C_f}\exp(z_{m_f-1,c'}/T)}.
\]

| 题型 | 输出 |
| --- | --- |
| `choice` | 原始选项值的概率分布、最大概率选项 |
| `noul` | `no/yes` 分布、`p_yes`、二选一结果；平局选 yes，与参考代码一致 |
| `score` | 等级概率分布、最大概率等级、`expected_score = Σ value × probability` |

各字段共享 state。后面的字段会看到前面字段的占位符，但不会看到前面字段的预测结果。受限 softmax 与训练的全词表 CE 各有用途，不应混用。

参考服务只返回词表 top-k logprobs，候选不在 top-k 时用一个很小的数近似。本地直接取得全部候选的真实 logits，不会丢候选。对同一候选集合，logits 与完整 logprobs 做受限 softmax 数学等价；它和参考服务的 top-k 截断近似可能产生差异。

## 2. 数据与防止泄漏

[`prepare_data.py`](prepare_data.py) 延续参考章的数据转换与采样：

| 划分 | 来源与规模 | 用途 |
| --- | --- | --- |
| train | Typed Decisions 960 案例/4800 字段 + ToolACE 1200 案例/1200 字段 | masked SFT |
| dev | Typed Decisions 120 案例/600 字段 | 选择 checkpoint |
| calibration | Typed Decisions 120 案例/600 字段 | 拟合单个温度 |
| test | Jevbench Easy 48、Original 72、Hard 111；Typed Decisions 400 案例/2000 字段；ToolACE 310 | 最终报告 |

Typed Decisions 使用 `LocalLLaMA/typed-decisions` 的 `all` 配置。每个 workflow 的 300 案例按 `sha256("pytrio-jev:" + id)` 排序，切为 240/30/30；训练使用硬标签，忽略原数据的软分布。

ToolACE 从 `Team-ACE/ToolACE` 的 train 提取工具目录和可转换的用户请求，按参考正则只保留一个工具名被调用的记录；请求文本按小写和空白归一，先排除与 310 条评测请求重叠的样本，再以 seed=42 打乱采样 1200 条。这里没有额外引入新的训练去重规则，以保留参考采样方式。超出 62 候选、重复工具名或不合法 JSON 会被拒绝。

五项测试数据取自 [Intern-Decision](https://github.com/InternLM/Intern-Decision/tree/565350244af860e1c276768cd80bdd9b8be93913/benchmarks/accuracy-v1) 的固定 commit。Jevbench 的 choice 按 `labels` 数组恢复顺序，不能使用文件中 criteria 的字母排序。数据脚本记录 Hugging Face 数据版本及所有产物的 SHA256 到 `datasets/manifest.json`。可用 `--typed-revision`、`--toolace-revision` 传入已记录版本重跑。

训练只读 train；先用 dev 选 checkpoint，再用 calibration 拟合 T，最后评 test。测试数据不会用于更新参数、选 checkpoint 或拟合温度。

## 3. 环境与本机试跑

以下命令均从仓库根目录执行。沿用项目 Python 3.12、`.venv` 与根目录 [requirements.txt](../../requirements.txt)，其中包含 TensorBoard，无需 PyTRIO 或远程训练账户。如果尚未建环境，按 [第一节的环境说明](../grpo/README.md) 安装适合本机 CUDA 的 PyTorch，再安装锁定依赖；已有环境也需执行 `.venv/bin/python -m pip install -r requirements.txt` 更新依赖。

```bash
source .venv/bin/activate
export HF_HOME="$PWD/.cache/huggingface"
export HF_DATASETS_CACHE="$PWD/.cache/datasets"

# 下载较小的模型；也可直接用 Hugging Face 模型 ID。
python - <<'PY'
from modelscope import snapshot_download
snapshot_download("Qwen/Qwen3.5-0.8B", local_dir=".cache/models/Qwen3.5-0.8B")
PY
export JEV_MODEL="$PWD/.cache/models/Qwen3.5-0.8B"

# 原任务全量数据准备；模型变小不改变数据集。
python experiments/jev/prepare_data.py

# 从原训练集取小子集，验证更新、保存与重载。
python experiments/jev/train.py --model "$JEV_MODEL" \
  --max-samples 16 --batch-size 4 --epochs 1 --max-steps 2 \
  --save-every 0 --output experiments/jev/runs/jev-smoke

python experiments/jev/inference.py --model "$JEV_MODEL" \
  --adapter experiments/jev/runs/jev-smoke/final \
  --input experiments/jev/datasets/dev.jsonl --limit 2
```

训练先完整编码所选案例；超过 `--max-length` 默认 8192 会报出案例 ID 与 token 数，**不会静默截掉决策字段**。可增大长度或以 0 取消限制，实际可运行性取决于硬件与序列长度。Qwen3.5 的线性注意力未安装可选内核时会使用正确但较慢的 PyTorch 实现；本章不强制安装这些编译扩展。

`--precision bf16` 是默认设置；CPU 或不支持 BF16 的硬件可传 `--precision fp32`。推理、校准与评测可用 `--device cpu`。训练设备由 Accelerate 决定。输出目录必须为空，重复实验换一个目录名。

## 4. 完整训练、多卡与参数

```bash
# 单卡完整实验：2160 条 × 2 epoch / 16 = 270 次参数更新。
python experiments/jev/train.py --model "$JEV_MODEL" \
  --epochs 2 --batch-size 16 --output experiments/jev/runs/jev-full

# 多卡使用同一脚本；16 仍是所有进程合计的案例数。
torchrun --standalone --nproc_per_node=2 experiments/jev/train.py \
  --model "$JEV_MODEL" --epochs 2 --batch-size 16 --output experiments/jev/runs/jev-ddp
```

| 参数 | 默认值与含义 |
| --- | --- |
| `--model` | `Qwen/Qwen3.5-0.8B`；可传本地目录，切换回参考 4B 时也使用此参数 |
| `--batch-size` | 16 个全局案例；每卡每次只前向一个案例，通过梯度累积完成整批更新 |
| `--epochs` | 2；每轮按 seed+epoch 打乱当前顺序 |
| `--learning-rate` | 1e-4；恒定学习率，没有额外 warmup 或 scheduler |
| `--lora-rank` / `--lora-alpha` | 32 / 32；文本骨干 `all-linear`，dropout=0、无 bias 训练 |
| `--max-grad-norm` | 0，关闭裁剪；大于 0 时对整批累积梯度统一裁剪 |
| `--max-samples` / `--max-steps` | 0，全部；用于子集与步数限制 |
| `--save-every` | 每 60 步保存；0 关闭步骤保存，各轮结束和 final 仍保存 |

优化器是 AdamW，weight_decay=0（等价于无权重衰减的 Adam），betas=(0.9,0.999)、eps=1e-8。只训练 LoRA，冻结基础模型与 lm_head，启用非 reentrant 梯度检查点。Transformers 加载 Qwen3.5 的文本骨干，不加载视觉塔。

DDP 会平均各卡梯度，所以 `update` 对本卡每个字段的 CE 乘 `world_size / 全局字段数`。尾批保留，卡间案例数不等时用零权重占位，让各卡执行相同次数的 backward，避免重复样本进入目标函数。相对于参考脚本丢弃不足一批的尾部，这是显式差异；默认 2160/16 整除时没有影响。

产物包括训练配置与选中案例 ID、数据文件指纹、逐步 `train.jsonl` 日志、`tensorboard/` 事件日志、`step-*`/`epoch-*`/`final` LoRA 权重和 tokenizer。checkpoint 是供推理/评测使用的 adapter，没有保存优化器状态，不支持精确断点续训。

运行产物放在本章的 `experiments/jev/runs/`；不传 `--output` 时默认使用本章目录下的 `runs/jev/`，与启动时的工作目录无关。显式传入的相对路径按当前工作目录解析。TensorBoard 以实际优化器更新步数为横轴，记录全局字段平均 loss、裁剪前梯度范数、监督字段数、案例数、epoch 和学习率；多卡仅主进程写入。从仓库根目录启动：

```bash
.venv/bin/tensorboard --logdir experiments/jev/runs
```

在浏览器打开 `http://localhost:6006` 查看训练曲线。校准与独立评测结果保存为 JSON。

参考服务没有在章节代码中暴露 LoRA target modules、alpha、Adam betas/eps、梯度裁剪及远端数值实现。本章明确给出本地选择，不将这些选择表述为已经核实的服务端设置。也没有保留异步请求与 SwanLab；训练同步运行并写本地 JSONL 和 TensorBoard。

## 5. 选优、温度校准与评测

先比较 `epoch-*` 或 `step-*` 的 dev 指标，固定所选 adapter。以下以 final 为命令示例，不代表 final 必定最好。

```bash
# 基础模型与训练后都可评 dev；不传 --adapter 即基础模型。
python experiments/jev/evaluate.py --model "$JEV_MODEL" --suite dev \
  --output experiments/jev/runs/jev-full/dev-base.json
python experiments/jev/evaluate.py --model "$JEV_MODEL" \
  --adapter experiments/jev/runs/jev-full/final --suite dev --output experiments/jev/runs/jev-full/dev-final.json

# 只在 calibration 划分拟合所选 checkpoint 的 T。
python experiments/jev/calibrate.py --model "$JEV_MODEL" \
  --adapter experiments/jev/runs/jev-full/final --output experiments/jev/runs/jev-full/calibration.json

# 固定模型和温度后跑五项测试；同时保留校准前结果。
python experiments/jev/evaluate.py --model "$JEV_MODEL" \
  --adapter experiments/jev/runs/jev-full/final --suite all --output experiments/jev/runs/jev-full/test-raw.json
python experiments/jev/evaluate.py --model "$JEV_MODEL" \
  --adapter experiments/jev/runs/jev-full/final --suite all --calibration experiments/jev/runs/jev-full/calibration.json \
  --output experiments/jev/runs/jev-full/test-calibrated.json
```

[`calibrate.py`](calibrate.py) 最小化字段平均 NLL。令逆温度 β=1/T，NLL 对 β 凸，导数是 `E_p[z] − z_gold`；在 β∈[0.01,100] 二分找零点，最优点超出范围时取边界。T>0 不改变 argmax，只改变置信度。校准文件保存原始候选 logits、标签下标、NLL、模型位置与 adapter 权重指纹，评测会拒绝模型或 adapter 不匹配的温度文件。

[`evaluate.py`](evaluate.py) 保存逐字段预测和以下汇总：

- accuracy：按字段计算命中率；score 用最大概率等级，不用期望值取整。
- ECE：最大候选概率作为置信度，10 个等宽分桶。
- Brier：每字段各候选 `(p − one_hot_gold)²` 求和，再跨字段平均，不除以候选数。
- NLL：正确候选的负对数概率，使用 logsumexp 稳定计算。
- 五项宏平均准确率：各基准 accuracy 的算术平均，不按字段数混合加权。

校准和评测的 `--limit` 仅供检查流程；限量输出不可充当全量基准。完整训练前后的基础模型对照、不同随机种子和 Hard 分布泛化，需要实际实验后再下结论。本章已完成 0.8B 的少量单卡更新、重载、限量校准与评测，尚未完成全量训练与基准对照，真实多 GPU 训练尚未验证。

`data.py`、`prepare_data.py` 和固定模板改编或复制自上述 Datawhale 参考 commit，许可证保留在 [LICENSE.upstream](LICENSE.upstream)。数据各自遵循原数据源的许可证。模型 API 依据 [Transformers Qwen3.5 文档](https://huggingface.co/docs/transformers/en/model_doc/qwen3_5)，实际依赖版本以根目录锁定文件为准。
