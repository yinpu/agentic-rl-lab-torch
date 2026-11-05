# GRPO 验证记录

验证日期：2026-11-05。已完成原版模型的完整 256 题验证集评测、两步真实 GRPO 训练，以及四种实验配置的 GPU 正确性检查。全量训练已按用户要求取消，完整对照实验留待服务器执行。

## 当前实验：Qwen2.5-0.5B Base 原版基线

本章使用 `Qwen/Qwen2.5-0.5B` Base。模型从 ModelScope 下载到 `.cache/models/Qwen2.5-0.5B`；下载文件的 SHA256 与提供方声明一致，文件版本和哈希记录在该目录的 `source.json`。

训练与评测共用固定的无示例提示词：单条 user 消息包含格式指令和当前问题，无 few-shot、无预填 think 标签。使用 tokenizer 自带的 chat template，完整输入保存到逐题结果的 `prompt` 字段。

完整验证集从 GSM8K train 中按 seed 42 留出 256 题，与训练部分无交集。沿用小规模基线的划分、提示词、BF16 精度和贪心解码，每题最多生成 1024 个 token。

完整评测实际命令（使用已经保存的划分）：

```bash
OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 .venv/bin/python experiments/grpo/evaluate.py \
  --model .cache/models/Qwen2.5-0.5B --split validation \
  --split-file runs/qwen05-validation-full/splits.json \
  --limit 0 --max-tokens 1024 --precision bf16 \
  --output runs/qwen05-validation-full/baseline.json
```

| 指标 | 实测结果 |
| --- | ---: |
| 已评测验证题数 | 256 / 256 |
| 正确题数 | 69 |
| boxed 判分正确率 | 26.95% |
| 有 boxed 答案的题数 | 139（54.30%） |
| 严格 think 格式合规题数 | 0 |
| 包含 opening / closing think 标签的题数 | 1 / 1 |
| 平均生成 token 数（含 EOS） | 301.0625 |
| 达到生成上限而截断的题数 | 13（5.08%） |

奖励沿用 boxed 答案的 0/1 判定，没有格式奖励。117 题未提取到 boxed 答案，另外 70 题的 boxed 答案错误；严格 think 格式仅用于监控，不影响正确性判分。不能把缺失 boxed 的题目直接解释为算术能力不足。该结果是本章固定提示词和判分规则下的完整留出验证集成绩，不是官方 GSM8K test 成绩。

结果保存在 `runs/qwen05-validation-full/`：`baseline.json` 为汇总和完整逐题输出，`baseline.rank-0.jsonl` 为逐题记录，`splits.json`、`baseline.config.json`、`manifest.json` 和 `model-source.json` 保存划分、参数、运行环境与来源。`audit.json` 记录复核结果：256 题无重复且完整覆盖验证划分，原始问题与答案、全部实际提示词、逐题判分和汇总一致，代码与划分哈希一致。前 16 题与此前小规模基线逐题一致；此前为 5/16（31.25%），只用于流程检查。

上述基线评测时没有对 0.5B 模型执行 SFT 或 GRPO 训练，也没有调整提示词。此前 15 项离线检查通过，包括新模型架构的 token 对齐、LoRA 更新/冻结/重载，以及训练评测提示词一致、格式监控不改变奖励、逐题文件与汇总一致；双进程 CPU 的尾批归约检查也通过。

当前评测逐题保存 `baseline.rank-0.jsonl`，结束后生成完整 JSON。代码增加了中止时保留已完成结果的能力，尚未实现自动断点续评。

## GRPO 训练流程检查

用户授权训练后，从原版模型独立启动两步短训练，没有沿用其他适配器。实际命令：

```bash
OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 .venv/bin/python experiments/grpo/train.py \
  --model .cache/models/Qwen2.5-0.5B --output runs/qwen05-grpo-smoke/train \
  --steps 2 --train-limit 8 --batch-size 4 --group-size 8 \
  --max-tokens 1024 --eval-limit 2 --eval-every 0 --eval-max-tokens 1024 --precision bf16
```

实际采样 8 道训练题、64 条回答，执行了 2 次优化器更新。两批 reward 分别为 5/32、1/32，非零优势回答分别含 6,605、3,798 个 token；梯度范数分别约 0.2290、0.0516，初始 ratio 均为 1。峰值 CUDA allocated 为 2.999 GiB，reserved 为 3.961 GiB，这些数值只对应本次实际配置。

训练及训练前后共 4 条贪心验证回答耗时约 109 秒。仅检查的两道验证题从 0/2 变为 2/2，不能据此判断训练收益。保存后重新加载 adapter，回答、token 数和指标与内存中训练后模型完全一致；LoRA B 矩阵存在非零更新。原版与初始 LoRA 在这两题的输出也完全一致，验证索引与完整原版基线一致，训练与验证无交集。产物和复核见 `runs/qwen05-grpo-smoke/` 下的 `train/summary.json`、`reloaded.json`、`audit.json`。

## 全量 GRPO 主实验：已取消

原计划从同一原版模型重新开始，不使用短训练的 adapter。运行目录为 `runs/qwen05-grpo-main/`，产物位于其中的 `train/`。完整启动参数和源码哈希保存在 `launch.json`；`status.json` 已标记取消，`console.log` 保留实际输出。此次后台启动脚本仅保留在运行目录中，不作为章节训练框架。

启动配置为：seed 42，训练集 7,217 题一遍（1,805 个采样批次），单卡每批 4 题，每题 8 条回答，上限 1024 token；每批更新一次，优势只减组内均值，beta=0、clip=0.2；AdamW 学习率 4e-5，LoRA rank 32、alpha 64，BF16。

用户要求停止时，训练前验证仅完成 26 题，尚未进入第一个训练批次，未产生主实验的训练日志或 adapter。已终止整个进程组，包含训练进程、后台管理进程及后续评测队列；没有主实验训练后准确率。

## 新增实验实现

同一训练入口提供当前 clipped GRPO、加 reference KL、有无格式奖励及二者组合：`--beta` 控制 KL，`--format-reward-weight` 控制额外格式奖励。默认均为 0，保持原版基线与两步短训练的算法设置。

格式奖励按严格 think + boxed 结构给分，总奖励为 `correct + λ * think_format`，先相加再计算优势；正确性与格式奖励分别保存到逐题 rollout 和训练日志。独立评测仍只用 boxed 答案的 0/1 正确性判分。实验设置与命令见 README，新增配置尚未完成完整训练对照，没有相应收益结论。

补实验实现时，在隐藏 GPU 的 CPU 环境下运行离线检查，30 项全部通过。覆盖正确性/格式奖励的四种输出情况、格式奖励关闭与开启、与 KL 的四种参数组合、全错组因格式差异获得优势、旧/参考概率冻结，以及非法权重拒绝。`train.py --help` 已确认显示新增参数；随后按用户授权进行了下面的少量 GPU 检查。

## 四种配置的 GPU 正确性检查

使用本机 RTX 4070 Ti SUPER 和真实 Qwen2.5-0.5B Base，BF16、LoRA rank 32 / alpha 64、学习率 4e-5。四组分别从原版模型初始化，参数为 `(beta, λ) = (0, 0)、(0.04, 0)、(0, 0.5)、(0.04, 0.5)`，不沿用前组权重。以下只是运行与数值检查，不是训练效果对照。

### 真实采样与完整入口

各组使用相同的 GSM8K 训练题（原始 index 0，不在验证集中），实际调用 `train.py`：`--steps 1 --train-limit 1 --batch-size 1 --group-size 8 --max-tokens 1024 --updates-per-batch 2 --eval-limit 1 --eval-every 1 --eval-max-tokens 1024 --precision bf16`，再分别设置 beta 与格式奖励权重。

每组均完成 8 条自由采样、2 次有效优化器更新、训练前/定期/训练后的单题贪心验证、定期检查点与最终 adapter 保存，以及独立启动 `evaluate.py --measure-kl` 重载评测。日志中总奖励等于正确性均值加上加权格式奖励；所有 loss、梯度和概率比指标有限，初始 ratio 为 1。

| 配置 | 第二次更新 clip fraction | 第二次更新 KL | 独立重载回答一致 |
| --- | ---: | ---: | --- |
| baseline | 3.93% | 未计算 | 是 |
| +KL | 4.06% | 0.005766 | 是 |
| +格式奖励 | 3.94% | 未计算 | 是 |
| +KL +格式奖励 | 4.10% | 0.006026 | 是 |

两组 KL 配置的初始 KL 为 0，第二次更新为正；定期检查点与最终 adapter 的权重文件逐字节一致。每组真实采样检查耗时约 37–41 秒，峰值 CUDA allocated 约 2.997 GiB，仅代表这个单题配置。

这些自由采样的严格格式率均为 0，格式奖励实际未提供额外信号；因此另外使用受控回答检查非零格式分支。单道验证题的正确与否不作为训练收益证据。

### 非零格式奖励、参数冻结与重载

新增 `test_grpo_gpu.py`，在真实模型上对四种配置分别构造“正确/错误 × 格式合规/不合规”四条回答。这些是基于同一 GSM8K 题目的固定检查输入，不声称由模型自由采样得到。使用生产代码的 `rollout`、`update_policy` 执行两次更新，4 项 GPU 检查全部通过：

- λ=0 时总奖励 `[1, 1, 0, 0]`，优势 `[0.5, 0.5, -0.5, -0.5]`。
- λ=0.5 时总奖励 `[1.5, 1, 0.5, 0]`，优势 `[0.75, 0.25, -0.25, -0.75]`。
- 只有 LoRA 更新；全部冻结基础参数的字节哈希不变。旧 logprob 缓存、固定参考策略的 logprob 均不变。
- 梯度有限，实际梯度总范数满足裁剪上限；初始 ratio/KL 和更新后 KL 符合预期。
- 重新加载后，四条回答逐 token 的 logprob 完全一致，32 token 贪心生成的 ID 序列和结束标记处理也完全一致。

此次还重跑了双进程 CPU 检查，3 题不均匀分配和 1 题含空进程时的更新均与集中式全局均值一致。单卡 GPU 与 CPU 多进程检查不代替真实多 GPU 验证；服务器上仍需先试跑目标进程数的配置。

产物保存在 `runs/qwen05-gpu-validation/`：`audit.json` 汇总真实采样和受控检查结果、实际命令、源码哈希；各组目录保留日志、adapter、逐题验证与重载结果；`controlled-final/` 保留最终 GPU 检查报告和 adapter。可复跑的 GPU 检查命令见 README。

## 环境与来源

- Python 3.12.4，使用根目录 `.venv`。
- PyTorch 2.10.0 / CUDA 12.8，Transformers 5.22.1，PEFT 0.21.2，Accelerate 1.15.0，Datasets 5.1.0。
- 单张 NVIDIA GeForce RTX 4070 Ti SUPER，16GB 显存。
- GSM8K revision：`740312add88f781978c0658806c59bc2815b9866`。
- 原始 train 内容 SHA256：`4d18d6dc73e1c092de041f6763f77ace65207030c62b00bff6f80068028a7cf9`。

## 当前验证边界

已完成原版模型的完整 256 题验证集评测、两步真实 GRPO 训练与保存重载检查、四组配置的单卡 GPU 正确性检查，以及离线算法与 CPU 双进程尾批归约检查。全量主实验已取消，尚未完成完整训练对照、官方完整测试集评测或真实多 GPU 实验，没有训练收益或论文指标复现结论。

## 代码清理后的回归检查

完整评测前，15 项离线检查通过。现有 16 题基线的实际输入提示词逐字一致；单题 GPU 回归的回答、token 数与判分也与原结果完全一致，产物保存在 `runs/qwen05-cleanup-check/`。随后完成了上文记录的完整验证集评测。


## 公式阅读与训练入口整理后的检查

本次仅整理代码与教程：优势、token logprob、clipping、KL 及逐回答更新的计算保持原样；尾批补齐、全局更新判断、日志和 adapter 保存提取到 `train.py` 的本章辅助函数。README 增加公式到实际函数的阅读顺序及代码摘录，多卡、精度和保存说明集中到后文。命令行参数、默认值、输出字段和既有实验设置保持不变。

本次使用 `CUDA_VISIBLE_DEVICES=''`、`OMP_NUM_THREADS=2`、`HF_HUB_OFFLINE=1`，执行以下有限检查：

```bash
.venv/bin/python -m pytest experiments/grpo/test_grpo.py -q \
  -k 'group_advantages or positive_and_negative or clip_uses or kl_zero or eos_is or autoregressive_alignment or rollout_combines'
.venv/bin/torchrun --standalone --nproc_per_node=2 experiments/grpo/test_grpo.py
```

- 相关离线测试 **13 项通过，17 项未选中**，覆盖四种奖励/KL 组合的 rollout、组内优势、正负优势梯度、clipping、KL、EOS 与自回归对齐。
- CPU 双进程检查通过：全局 3 题时各进程题数不齐、全局 1 题时一个进程无真实回答，使用实际 `pad_samples` 和 `update_policy` 得到的参数更新均与集中式全局均值一致。另检查各进程共同更新、全零优势且无 KL 共同跳过、全零优势但启用 KL 共同更新。
- 临时目录中的日志检查通过：保留原有 13 个汇总字段及逐回答记录结构；两条真实回答加一条占位项时，均值、token 数和实际更新计数正确，占位项与概率张量不写入记录。
- README 的 10 段 Python 摘录均按忽略空白和注释后的 token 序列与实际源码核对一致；`git diff --check` 通过。

本次没有运行全量测试、GPU 检查、模型下载、GSM8K 训练或评测；不增加训练收益或真实多 GPU 已验证的结论。原有实验结果及验证边界仍按上文记录。
