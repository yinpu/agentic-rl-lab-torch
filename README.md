# agentic-rl-lab-torch

参考 [datawhalechina/agentic-rl-lab](https://github.com/datawhalechina/agentic-rl-lab)，使用 PyTorch 生态建设以中文教程、算法讲解和可运行实验为核心的 LLM / Agent 强化学习实验项目。

每章作为独立的小项目，结合实际任务解释公式与代码，通过参数调整本机试跑和服务器实验规模。逐章选择适合的开源工具，不依赖 PyTRIO 的远程训练服务。

## 当前章节

| 章节 | 任务与模型 | 内容 |
| --- | --- | --- |
| [GRPO](experiments/grpo/README.md) | GSM8K / Qwen2.5-0.5B Base | 显式采样、组内优势、clipped loss、LoRA、可选 KL 与格式奖励 |
| [General OPD](experiments/general-opd/README.md) | DeepMath-103K / Qwen3.5-4B ← Qwen3.6-27B | 同步在线蒸馏、token 优势、LoRA、独立数学评测 |
| [Jev 决策模型](experiments/jev/README.md) | Typed Decisions + ToolACE / Qwen3.5-0.8B | marker masked SFT、一次前向决策、温度校准、五项基准评测 |

GRPO 使用 Transformers、PEFT 和 Lightning Fabric，核心算法直接写在本章代码中。训练与评测共用提示词和答案判定，支持单卡运行与 `torchrun` DDP 启动。

GRPO 使用 `--micro-batch-size` 调整每进程训练前向的样本数，默认 1。JEV 使用 `--per-device-train-batch-size`（默认 1）与 `--gradient-accumulation-steps`（默认 16），全局 batch 为单卡 batch × 梯度累积次数 × 卡数。

Jev 使用较小的 Qwen3.5-0.8B，通过 LoRA masked SFT 学习有限候选决策；保留参考章的原任务与数据切分，训练、推理、校准和评测均在本地运行。完整流程见 [第二节教程](experiments/jev/README.md)。

General OPD 使用 Qwen3.5-4B 学生和 Qwen3.6-27B 教师，在 DeepMath 上进行逐 token 在线蒸馏，并对固定留出集评测。

## 开始使用

```bash
git clone https://github.com/yinpu/agentic-rl-lab-torch.git
cd agentic-rl-lab-torch
```

各章的模型下载、算法说明、实验参数及启动命令见对应教程。项目使用 Python 3.12、根目录 `.venv` 与 [requirements.txt](requirements.txt)；首次环境准备见 [GRPO 教程](experiments/grpo/README.md)。模型、数据缓存和运行产物保存在本地，不提交到仓库。运行产物分别放在各章的 `experiments/<章节>/runs/`，每次训练在输出目录的 `tensorboard/` 下记录 TensorBoard 日志。

## 测试

测试集中在根目录 `tests/`，按 `grpo/`、`jev/`、`general-opd/` 分组；章节目录保留教程与实验代码。从仓库根目录运行：

```bash
.venv/bin/python -m pytest -q
```

真实模型 GPU 检查需要 CUDA 和本地缓存权重，否则自动跳过。两进程 CPU DDP 检查单独运行：

```bash
.venv/bin/torchrun --standalone --nproc_per_node=2 tests/grpo/check_ddp.py
.venv/bin/torchrun --standalone --nproc_per_node=2 tests/jev/check_ddp.py
.venv/bin/torchrun --standalone --nproc_per_node=2 tests/general-opd/check_ddp.py
```
