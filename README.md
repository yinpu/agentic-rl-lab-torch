# agentic-rl-lab-torch

参考 [datawhalechina/agentic-rl-lab](https://github.com/datawhalechina/agentic-rl-lab)，使用 PyTorch 生态建设以中文教程、算法讲解和可运行实验为核心的 LLM / Agent 强化学习实验项目。

每章作为独立的小项目，结合实际任务解释公式与代码，通过参数调整本机试跑和服务器实验规模。逐章选择适合的开源工具，不依赖 PyTRIO 的远程训练服务。

## 当前章节

| 章节 | 任务与模型 | 内容 |
| --- | --- | --- |
| [GRPO](experiments/grpo/README.md) | GSM8K / Qwen2.5-0.5B Base | 显式采样、组内优势、clipped loss、LoRA、可选 KL 与格式奖励 |

GRPO 使用 Transformers、PEFT 和 Accelerate，核心算法直接写在本章代码中。训练与评测共用提示词和答案判定，支持单卡运行与 `torchrun` DDP 启动。

## 开始使用

```bash
git clone https://github.com/yinpu/agentic-rl-lab-torch.git
cd agentic-rl-lab-torch
```

环境准备、模型下载、算法说明、实验参数及启动命令见 [GRPO 教程](experiments/grpo/README.md)。项目使用 Python 3.12、根目录 `.venv` 与 [requirements.txt](requirements.txt)。模型、数据缓存和运行产物保存在本地，不提交到仓库。

## 已执行的验证

- 原版 Qwen2.5-0.5B Base：固定留出的 256 道 GSM8K 验证题答对 69 题，准确率 26.95%；这是本章提示词和判分规则下的结果。
- 当前 loss、加 KL、加格式奖励及二者组合，均完成单卡 BF16 GPU 的少量采样、更新、保存与重载检查。
- 已通过离线算法检查和 CPU 双进程尾批归约检查；尚未完成完整训练对照和真实多 GPU 验证。

详细设置、实际结果和验证边界见 [GRPO 验证记录](experiments/grpo/validation.md)。
