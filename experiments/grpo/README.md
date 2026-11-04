# GRPO：从公式到 GSM8K 上的 LoRA 训练

这一章研究一个具体问题：给模型同一道数学题的多个回答打分，能否改善它在未参与训练的题目上的表现？先使用最终答案正确性奖励，再分别对照加入 KL 约束和格式奖励的影响。

任务与判分规则参考 [agentic-rl-lab 的 GRPO 章节](https://github.com/datawhalechina/agentic-rl-lab/tree/8493321013c02eef6cf79ca2594811f53c6fdac1/experiments/grpo)。本章使用 Qwen2.5-0.5B Base 和无示例的 think + boxed 提示词。Transformers 加载和生成，PEFT 管理 LoRA，Accelerate 处理设备和 DDP；优势、概率比、clip、KL 和更新循环都直接放在本章代码里。

从项目根目录执行下文命令。文件之间的关系是：

- `task.py`：提示词、奖励、数据划分、模型加载和 GRPO 公式。
- `train.py`：采样、缓存旧概率、逐回答反向传播、评测与保存。
- `evaluate.py`：用同一任务定义独立评测基础模型或保存的 LoRA。
- `test_grpo.py`：不下载权重的算法与接口检查。
- `test_grpo_gpu.py`：使用缓存的真实模型，在单张 GPU 上检查四种配置的更新与保存重载。

## 环境和模型

整个项目使用根目录的 `.venv`。Python 3.12，依赖版本固定在根目录 `requirements.txt`。以下 CUDA 12.8 安装命令针对本机 NVIDIA 环境；服务器按其驱动选择官方 PyTorch wheel。

```bash
python -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
.venv/bin/python -m pip install -r requirements.txt
```

从 ModelScope 下载本章当前选定的基础模型：

```bash
.venv/bin/python -c 'from modelscope import snapshot_download; snapshot_download("Qwen/Qwen2.5-0.5B", local_dir=".cache/models/Qwen2.5-0.5B")'
```

默认从这个本地路径加载。[Qwen2.5-0.5B 官方说明](https://huggingface.co/Qwen/Qwen2.5-0.5B)将它标为约 0.49B 参数、仅预训练的基础模型，不是 Instruct 或推理蒸馏版。通过 `AutoModelForCausalLM` 加载，实际模型类为 `Qwen2ForCausalLM`。`--model` 也接受本地目录或 Hugging Face 模型名，方便在服务器上指定模型位置。

GSM8K 首次运行由 Datasets 自动下载到 `.cache/datasets`。缓存、`.venv` 和 `runs` 不进入版本控制。本章不需要 PyTRIO、训练 API、vLLM 或额外训练服务。

## 从一道题开始

GSM8K 的每条数据包含 `question` 和带解题过程的 `answer`，标准答案在 `####` 后。训练只把 question 交给模型，不把标准解答放进 prompt。

训练与评测都调用 `task.py::prompt_ids`，使用同一份固定指令。只发送一个包含当前题目的 user 消息，没有示范解答：

```text
Solve the following math problem.
Write your reasoning inside <think>...</think>.
Then give the final numerical answer inside \boxed{}.

Problem:
{question}
```

使用该 tokenizer 自带的 chat template；它会自动添加默认的 system 文本与 assistant 起始标记。没有手工预填 `<think>`，也没有给出答案或解题过程。对 Qwen2.5 Base，标签是输出协议，不是原生 thinking 开关；需要实际检查模型能否遵循。

正确性奖励取回答最后一个 `\boxed{...}`，去掉逗号、首尾空白、末尾句点，与标准答案比较：正确为 1，否则为 0。不做数学等价判断，例如 `1000.0` 与 `1000` 仍可能判成不同答案。即使没有 think 标签，只要 boxed 答案正确，正确性奖励仍为 1。默认不加格式奖励。

`think_format_rate` 监控严格格式：一个非空 `<think>...</think>` 块，后面紧跟 boxed 答案，允许首尾空白。标签正确不代表答案正确。训练可通过 `--format-reward-weight` 给格式合规额外加分，独立评测的 accuracy 始终只按正确性判定。

该格式借鉴 [DeepSeek-R1-Zero 的训练模板](https://arxiv.org/html/2501.12948v1#S2.SS2)，但最终答案保留 boxed 判分，而非其模板的 answer 标签。本章的格式判定和奖励权重是本地实验选择，属于简化 GRPO 实验，不宣称完整复现 R1-Zero。

## 可选格式奖励

`task.py::training_rewards` 将奖励分项写在一起。令正确性为 $c_i\in\{0,1\}$，严格格式合规为 $f_i\in\{0,1\}$：

\[
r_i=c_i+\lambda f_i.
\]

`--format-reward-weight 0` 对应当前基线；设为 `0.5` 时，格式合规额外得 0.5 分，总奖励范围为 0 到 1.5。这只是对照实验的起始权重，尚未训练验证。

| 输出情况 | 正确性 $c_i$ | 格式 $f_i$ | 总奖励（λ=0.5） |
| --- | ---: | ---: | ---: |
| 答案正确、格式合规 | 1 | 1 | 1.5 |
| 答案正确、格式不合规 | 1 | 0 | 1 |
| 答案错误、格式合规 | 0 | 1 | 0.5 |
| 答案错误、格式不合规 | 0 | 0 | 0 |

格式奖励只检查输出结构，不判断推理质量，也不以答案正确为前提。先把两项相加，再用总奖励计算组内优势；KL 仍独立加入 loss。例如同组两条答案都错，只有第一条格式合规，总奖励为 `[0.5, 0]`，仅减均值的优势为 `[0.25, -0.25]`。

若同组所有回答都不合规，格式奖励全为 0，就没有额外的格式学习信号；应查看实际 rollout 和格式率，不能仅根据开启了奖励推断训练有效。

## 不训练 critic：用同题回答建立 baseline

对题目 $q$，用当前策略生成 $G$ 个回答 $o_1,\ldots,o_G$，奖励为 $r_i$。默认与参考代码一致，只减去本组均值：

\[
\bar r=\frac{1}{G}\sum_{j=1}^{G}r_j,
\qquad A_i=r_i-\bar r.
\]

在 `task.py::group_advantages` 中就是：

```python
advantages = rewards - rewards.mean()
```

例如 `[1, 0, 1, 0]` 得到 `[0.5, -0.5, 0.5, -0.5]`。同一回答的每个有效 token 共用一个 $A_i$。不额外训练 value model。

论文常见的组内标准化形式为：

\[
A_i=\frac{r_i-\bar r}{\operatorname{std}(r_1,\ldots,r_G)+10^{-8}}.
\]

用 `--standardize-advantages` 启用，代码采用总体标准差 `correction=0`，上例得到 `[1, -1, 1, -1]`。这是显式算法选择，不是依赖训练库默认值。所有奖励相同时，两种方式都得到零优势。

## 采样后，固定旧策略的概率

一个采样批次先完成所有题目的 rollout，再开始更新。`train.py::rollout` 保存每条回答的 token、reward、advantage 和无梯度的旧 logprob。旧概率在本批的所有更新中保持不变。

训练采样固定 temperature=1、top-p=1、top-k=0、repetition penalty=1。新建 `GenerationConfig`，避免模型仓库默认配置改变采样分布。模型与 LoRA 的 dropout 均为 0；采样、概率计算和独立评测均使用相同精度的 autocast。

概率比为：

\[
\rho_{i,t}=\exp\bigl(\log\pi_\theta(o_{i,t}\mid q,o_{i,<t})
-\log\pi_{\mathrm{old}}(o_{i,t}\mid q,o_{i,<t})\bigr).
\]

这里的 old policy 是本批采样时的策略。它与后文的固定 reference 不同。

## 逐 token logprob 如何对齐

假设 prompt 是 `[p0, p1, p2]`，completion 是 `[c0, c1, EOS]`。自回归模型使用：

```text
输入：p0 p1 p2 c0 c1
取值：      ↓  ↓  ↓
目标：      c0 c1 EOS
```

`task.py::completion_logprobs` 把 `prompt + completion[:-1]` 输入模型，取最后 `len(completion)` 个输出位置，再取目标 token 的 logprob。`logits_to_keep` 只保留所需位置，减少大词表 logits 的占用。

每次 forward 只处理一条未 padding 的回答，因此不需要在 loss 里构造大批次的 prompt/padding mask：返回的整个向量正好对应有效 completion。`trim_completion` 保留首个 EOS，删除其后的 padding；达到生成上限但未遇到 EOS 的回答仍保留，另行记录截断率。

## Clipped 策略目标直接写出来

本章实现 PPO clipped surrogate，用组内优势替代 critic 优势：

\[
\ell_{i,t}^{\mathrm{policy}}=-\min\left(
\rho_{i,t} A_i,
\operatorname{clip}(\rho_{i,t},1-\epsilon,1+\epsilon) A_i
\right),\qquad \epsilon=0.2.
\]

`task.py::grpo_loss` 的对应代码是：

```python
ratio = torch.exp(logprobs - old_logprobs)
unclipped = ratio * advantage
clipped = ratio.clamp(1 - clip, 1 + clip) * advantage
policy = -torch.minimum(unclipped, clipped)
```

正优势鼓励提高回答的概率；负优势鼓励降低。`min` 根据优势符号选择受限制的方向，不能简单理解成“任何超过阈值的 ratio 都没有梯度”。日志中的 `clip_fraction` 统计 clipped 分支严格改变目标的 token 比例。

本章先对一条回答的 token 求均值，再对全局真实回答求均值：

\[
L=\frac{1}{N}\sum_{i=1}^{N}\frac{1}{T_i}\sum_{t=1}^{T_i}\ell_{i,t}.
\]

逐回答 backward 是显存上的拆分，所有回答累积完才执行 optimizer step，算法上仍是这一个全局均值。默认 AdamW 的学习率为 `4e-5`、betas 为 `(0.9, 0.95)`、weight decay 为 0；梯度范数上限 1。

默认每批只更新一次。更新前 current 与 old 相同，ratio 应接近 1，clip 通常尚未触发。`--updates-per-batch` 可让同一批数据重复更新以观察 ratio 变化，但这改变了参考实验的训练预算。不要根据单次更新实验宣称已验证 clipping 的收益。

初始 ratio=1 且组内优势均值为 0 时，记录的平均 policy loss 可能也是 0；不同回答的 logprob 梯度却不相同，参数仍可获得非零梯度。结合 `grad_norm` 和实际优化步数判断，不能只看 loss 数值。

参考还比较 IS 与 CISPO：IS 使用 $-\rho A$；CISPO 使用 $-\operatorname{detach}(\operatorname{clip}(\rho))\log\pi_\theta A$。本章暂不增加这两套训练分支，先把 GRPO 主线讲清楚。

## 可选 reference KL

主实验 `--beta 0`，不计算、不记录未测得的 KL。小型对照可以使用 `--beta 0.04`：

\[
\Delta_{i,t}=\log\pi_{\mathrm{ref}}-\log\pi_\theta,
\quad \widehat D_{i,t}=\exp(\Delta_{i,t})-\Delta_{i,t}-1,
\quad \ell_{i,t}=\ell_{i,t}^{\mathrm{policy}}+\beta\widehat D_{i,t}.
\]

基础模型冻结，LoRA 默认零增量初始化。关闭适配器计算 reference logprob，就得到训练初始策略，不需要再放一整份模型。reference 始终固定，old 则每批刷新。

KL 直接加入 loss，不混进奖励再计算组内优势。它是采样 token 上的估计量，不是对整个词表精确求和的 KL。

总奖励全相同的组保留在损失分母中；它们没有奖励优势，但在 beta>0 时仍可产生 KL 约束。仅正确性奖励时，全对或全错的组都是这种情况；开启格式奖励后，格式差异也可能产生优势。beta=0 且整个全局批次没有优势信号时，跳过 optimizer step，避免 Adam 历史动量继续改变参数。采样批次数与实际优化步数分别记录。

## 数据划分与运行

从 GSM8K train 按 seed 42 固定抽取 256 题作为验证集，剩余题保持原始顺序用于训练。保存真实索引和整个原训练集的内容 SHA256。训练前、训练后及每 50 个采样批次，固定贪心解码评测同一验证子集；默认取 64 题。test 保留给最终评测。

先检查原版模型的小规模基线，不需要先运行训练：

```bash
.venv/bin/python experiments/grpo/evaluate.py \
  --limit 16 --max-tokens 1024 \
  --output runs/qwen05-base/baseline.json
```

省略 `--split-file` 时，按 seed 42 和 validation size 256 创建划分并保存为 `baseline.splits.json`；后续可用 `--split-file` 复用。训练使用相同 seed 和 validation size 时生成相同验证索引。`--limit 0` 才是全部 256 题。

先尝试很小的真实任务流程检查：

```bash
.venv/bin/python experiments/grpo/train.py \
  --output runs/qwen05-smoke \
  --steps 1 --train-limit 2 --batch-size 1 --group-size 2 \
  --max-tokens 128 --eval-limit 2 --eval-max-tokens 128
```

短生成上限容易让正确解答来不及输出 boxed 答案，因此这个设置只检查链路。若全零优势跳过了更新，不能把它当作已经完成有效训练。

完整训练配置的起点如下，实际显存与耗时需在目标硬件验证：

```bash
.venv/bin/python experiments/grpo/train.py \
  --output runs/qwen05-main \
  --batch-size 4 --group-size 8 \
  --max-tokens 1024 --eval-limit 0
```

`--steps 0`（默认）遍历排除验证题后的训练集一遍。`--train-limit N` 取其中前 N 题；`--steps N` 限制采样批次数，不会循环重复数据凑步数。`--batch-size` 是**每进程题数**，不是回答数；一次全局完整批次有 `进程数 × batch-size × group-size` 条回答。

多卡使用同一入口：

```bash
.venv/bin/torchrun --standalone --nproc_per_node=2 experiments/grpo/train.py \
  --output runs/qwen05-ddp --batch-size 2 --group-size 8 --eval-limit 0
```

这是 DDP，每卡各有完整基础模型，不会把模型显存分摊到多张卡。完整采样组留在同一进程；最后不足整批时，只用真实回答数归一化。空槽做零权重计算以维持 DDP 同步，不重复计入样本或评测。梯度乘 `进程数 / 全局真实回答数`，抵消 DDP 的进程平均。

训练产物保存在 `--output`：配置与实际任务指令、依赖版本、划分索引、训练 JSONL、实际 rollout、训练前后逐题验证结果，以及最终的 `adapter/`。每次定期验证前还会保存 `adapter-batch-N/`，可用 `evaluate.py --adapter` 单独评测；检查点仅含模型适配器与 tokenizer，不含优化器或随机数状态，不支持完整恢复训练。评测逐题写入 `*.rank-N.jsonl`，完整结束后汇总为 JSON；中止时已完成的题目仍保留，但未实现自动断点续评。使用新目录启动，避免覆盖旧实验。

重新加载保存的 LoRA，验证保存链路：

```bash
.venv/bin/python experiments/grpo/evaluate.py \
  --adapter runs/qwen05-smoke/adapter \
  --split validation --split-file runs/qwen05-smoke/splits.json \
  --limit 2 --max-tokens 128 --output runs/qwen05-smoke/reloaded.json
```

最终在测试集评测时，对基础模型和 LoRA 使用相同生成上限：

```bash
.venv/bin/python experiments/grpo/evaluate.py \
  --split test --limit 0 --output runs/test-base.json
.venv/bin/python experiments/grpo/evaluate.py \
  --adapter runs/qwen05-main/adapter \
  --split test --limit 0 --output runs/test-grpo.json
```

## 三类实验与组合对照

所有设置共用 `train.py`，通过两个独立参数切换；不另建训练实现。前三组对应当前 loss、加 KL、加格式奖励，第四组用于检查二者组合。

| 组别 | `--beta` | `--format-reward-weight` | 训练目标 |
| --- | ---: | ---: | --- |
| baseline | 0 | 0 | 正确性奖励的 clipped GRPO |
| kl | 0.04 | 0 | baseline + reference KL |
| format | 0 | 0.5 | 正确性 + 格式奖励的 clipped GRPO |
| kl-format | 0.04 | 0.5 | 格式奖励 + reference KL |

以下是完整实验的待执行命令；本机只进行少量 GPU 正确性检查，不启动全量训练。四组都从原版 Qwen2.5-0.5B Base 初始化，默认使用相同 seed、数据顺序、batch/group size、学习率、LoRA 配置和生成长度，训练集一遍，验证使用全部 256 题。每组使用独立输出目录，不接着另一组的 adapter 训练。

```bash
# （1）当前 loss：只有正确性奖励
.venv/bin/python experiments/grpo/train.py \
  --output runs/qwen05-baseline --eval-limit 0 \
  --beta 0 --format-reward-weight 0

# （2）加 KL：与 baseline 比较 KL 的影响
.venv/bin/python experiments/grpo/train.py \
  --output runs/qwen05-kl --eval-limit 0 \
  --beta 0.04 --format-reward-weight 0

# （3）加格式奖励：与 baseline 比较格式奖励的影响
.venv/bin/python experiments/grpo/train.py \
  --output runs/qwen05-format --eval-limit 0 \
  --beta 0 --format-reward-weight 0.5

# 组合对照：与 kl 比较格式奖励；与 format 比较 KL
.venv/bin/python experiments/grpo/train.py \
  --output runs/qwen05-kl-format --eval-limit 0 \
  --beta 0.04 --format-reward-weight 0.5
```

本机用子集时，对四组添加相同的 `--train-limit N`、`--steps N` 等参数；多卡也使用相同进程数，保证全局采样题数一致。相同种子不保证不同策略训练中会采出相同回答，应该控制数据和采样预算。

结果分别读取各目录的 `validation-after.json` 和 `summary.json`，报告 accuracy、think_format_rate、boxed_rate、生成长度与截断率，并确认 `splits.json` 一致。各组评测始终只用正确性判分，不把训练的总 reward 当作准确率。比较 KL 时固定格式奖励权重，比较格式奖励时固定 beta。

`evaluate.py --measure-kl` 可以诊断相同设置的贪心回答相对 reference 的偏离；这个量不能视为策略分布下的无偏 KL。

## 如何解读结果

独立评测报告正确率、boxed 格式成功率、think 格式成功率、平均生成长度与截断率；实际输入提示词和每题回答也保存下来。训练日志分别记录 `reward`（总奖励）、`accuracy`（采样回答正确率）、`format_reward`（加权格式奖励均值）、`think_format_rate`，以及退化组比例、有效奖励信号 token、ratio、clip 和实际更新次数。格式奖励开启后，总 reward 可以大于 1，各组总 reward 的数值不能直接作为性能比较；训练采样正确率也不能代替留出验证集准确率。若正确率改善主要来自 boxed 格式改善，应按实际原因解释，不能自动归因为数学推理能力增强。

本实现与参考的明确差异：采用 Qwen2.5-0.5B 预训练基础模型和无示例的 think 格式指令；本地文本模型训练替代远程服务；明确选择 LoRA alpha=64、all-linear、dropout=0；只实现 clipped 策略目标；按回答等权归约；保留退化组；增加验证划分、独立贪心评测、可选 KL、可选格式奖励和梯度裁剪。参考远程服务的 LoRA 层选择、内置 loss 归约和梯度缩放未在客户端完整公开，因此不宣称数值等价。

这里要区分三件事：流程能够运行、参考任务上的本地实验、原论文指标复现。前者不能替代后两者。实际验证状态见本目录 `validation.md`；未执行的配置不写成已验证。

## 算法检查

```bash
.venv/bin/python -m pytest experiments/grpo/test_grpo.py -q
.venv/bin/torchrun --standalone --nproc_per_node=2 experiments/grpo/test_grpo.py
```

第一条检查答案判定、正确性/格式奖励分离、四种参数组合的 rollout 与固定 reference、优势、正负梯度方向、clip、KL、EOS、token 对齐、LoRA 基础权重冻结和适配器重新加载。第二条在两个 CPU 进程检查不足整批及空进程时，训练函数的更新是否等于集中式全局均值；这不能代替真实多 GPU 验证。随机小网络只用于这些离线检查，不替代 GSM8K 主实验。

准备好模型缓存后，可在本机或服务器的一张 BF16 CUDA GPU 上运行真实模型检查：

```bash
OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 .venv/bin/python -m pytest \
  experiments/grpo/test_grpo_gpu.py -q -s
```

四组分别从原版模型加载，使用真实 GSM8K 题目，并构造“正确/错误 × 格式合规/不合规”四种检查回答，确保格式奖励的非零分支也被覆盖。它们是明确的检查输入，不是模型自由采样结果。每组执行两次更新，检查优势、初始概率比、KL、LoRA 更新、全部基础权重的字节哈希、旧/参考概率冻结，以及保存重载后逐 token 概率和贪心生成完全一致。检查报告与适配器保存在 pytest 的临时目录；可用 `--basetemp` 指定本次检查的输出目录。缺少 CUDA 或模型缓存时会跳过这四项检查。

完整训练入口也可先用一题验证采样、日志、定期保存和独立评测链路，例如：

```bash
.venv/bin/python experiments/grpo/train.py \
  --output runs/qwen05-gpu-smoke --steps 1 --train-limit 1 \
  --batch-size 1 --group-size 8 --max-tokens 1024 --updates-per-batch 2 \
  --eval-limit 1 --eval-every 1 --eval-max-tokens 1024 \
  --beta 0.04 --format-reward-weight 0.5
```

这里显式更新两次，是为了让第二次更新检查变化后的 ratio、clipping 和 KL；完整实验的默认值仍为一次。按上面的实验表替换两个系数即可检查其他三组。小规模正确性检查不用于判断训练收益；真实多 GPU 配置仍需在目标服务器验证。

进一步阅读：[DeepSeekMath 论文](https://arxiv.org/abs/2402.03300)、[PEFT LoRA](https://huggingface.co/docs/peft/package_reference/lora)、[Accelerate 梯度同步](https://huggingface.co/docs/accelerate/concept_guides/gradient_synchronization)。
