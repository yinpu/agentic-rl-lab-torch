# GRPO：在 GSM8K 上训练数学解题模型

## 目标与当前状态

参考本地 `~/Projects/agentic-rl-lab/experiments/grpo/readme.md` 与 `01-demo-sync.py`，建设一节独立的中文教程和可运行实验。主线是观察 **GRPO 训练前后，模型在未参与训练的 GSM8K 数学题上的表现是否改善**，并提供有无 reference KL、有无格式奖励的对照。

用户已授权实现本章。当前采用 Transformers、PEFT、Accelerate，核心公式保留在章节代码中，不使用 TRL Trainer。实现入口和说明见 `README.md`，实际验证状态单独记录在 `validation.md`。本文件保存实验安排，不作为实验结果。

本章使用 Qwen2.5-0.5B Base 和一套固定的无示例 think + boxed 格式指令。原版模型完整 256 题验证集基线为 69/256（26.95%），已完成两步真实训练流程检查。全量任务已按用户要求取消；当前只进行授权的少量 GPU 正确性检查，完整实验留待服务器运行，不自动启动全量训练。实验命令与算法说明见 `README.md`，实际验证记录见 `validation.md`。

## 实现方式与组织

- 使用 Transformers 加载模型，PEFT 管理 LoRA，Accelerate 处理设备、梯度累积与多卡；自己编写采样、优势、逐 token logprob 和 loss，使公式能直接对应代码。
- 不建设公共训练框架、后端适配层或独立采样服务；使用模型自身的 `generate`。暂不引入 vLLM、FSDP、断点恢复等额外工程功能。
- 训练和评测入口分开，共用本章的提示词、答案判定及必要的模型处理函数。只保留本章需要的少量函数，不为其他章节预先抽象。
- **整个项目共用根目录的 `.venv` 和依赖清单**，不为每章建立环境；具体依赖版本在实施并验证后记录。
- 同一套代码用于本机与多卡服务器，通过启动参数调整规模。Accelerate 初版采用 DDP；每道题的完整采样组留在同一进程计算优势。不承诺模型一定能在尚未确认的硬件上运行。
- 优先从 ModelScope 下载 `Qwen/Qwen2.5-0.5B` 到项目的 `.cache/models/Qwen2.5-0.5B`；训练、评测接受模型名称或本地目录参数。缓存和运行输出不进入版本控制。

## 教程结构

1. **实验问题与样例。** 展示一道 GSM8K 题目、模型回答和奖励判定，说明为何只有最终答案标签也能进行强化学习。
2. **从 PPO 到 GRPO。** 解释同题多次采样、组内奖励 baseline，以及不需要额外 value model 的原因。用 `[1, 0, 1, 0]` 对照仅减均值与标准化优势，解释全对、全错时的零优势。
3. **准备真实任务。** 结合代码解释 GSM8K 划分、zero-shot 格式指令、chat template、think 标签、boxed 答案与 0/1 奖励；区分格式监控和格式奖励。
4. **展开训练循环。** 依次讲采样、缓存旧策略 logprob、计算优势、计算当前 logprob、completion mask、loss 与 LoRA 更新。重点讲自回归错位、EOS、padding 和损失归约。
5. **试跑与主实验。** 解释参数含义、启动方法、训练曲线和训练前后评测表，结果只填写实际运行的数据。
6. **对照实验。** 区分 reference 与 old policy、正确性与格式奖励，固定其他设置比较有无 KL、有无格式奖励。
7. **差异与限制。** 说明相对参考教程的变化，区分流程跑通、参考实验迁移和原论文指标复现。

原教程的 importance sampling、PPO clipping、CISPO 保留简短原理对照。本节只实现 PPO clipped 策略目标，不将三种 loss 的训练对比作为实验主线。

## 任务、算法与默认参数

### 模型、提示词和奖励

- 默认模型为 `Qwen/Qwen2.5-0.5B` 预训练基础版，使用 Transformers 的 `Qwen2ForCausalLM`，仅训练 LoRA。
- 数据采用 `openai/gsm8k` 的 `main` 配置；提示词改为无示例的当前问题与输出格式指令，要求推理位于 `<think>...</think>`，最终数值放入 boxed。训练与评测调用同一构造函数，不预填 opening think 标签。这是相对参考 one-shot 提示词的明确调整。
- 使用 tokenizer 自带的 chat template。Qwen2.5 Base 没有原生 thinking 开关；是否遵循标签要通过真实输出检查。严格格式率单独记录，可作为可选的训练奖励，不改变评测正确性，不把这一简化设置称为完整 R1-Zero 复现。
- 提取模型回答中最后一个 `\\boxed{...}`。去掉逗号、首尾空白和末尾句点后，与标准答案比较；正确性奖励为 1 或 0。
- 可选格式奖励：总奖励 `R = correct + λ * think_format`，通过 `--format-reward-weight` 设置 λ，默认 0；对照起点 0.5 尚未训练验证。严格格式指一个非空 think 块后紧跟 boxed 答案，奖励不要求答案正确。两项先相加再计算组内优势；评测始终只判答案正确性。
- GSM8K 标准答案从 `####` 后提取。保持参考实现的轻量字符串比较，不擅自替换为更宽松的数学等价判定。
- LoRA rank 32、alpha 64、dropout 0；作用于模型线性投影层，排除输出头。参考远程服务未公开的 LoRA 细节作为本地选择明确记录。

### 组内优势

按本次讨论，默认先保留参考代码的 `reward - mean_reward`。通过 `--standardize-advantages` 才启用下面的标准化公式；两者的选择写入运行配置。

对同一题的 G 个回答计算：

\[
A_i = \frac{r_i-\operatorname{mean}(r_1,\ldots,r_G)}
{\operatorname{std}(r_1,\ldots,r_G)+10^{-8}}.
\]

启用标准化时，标准差采用整组总体标准差（`correction=0`），不进行跨题标准化。该选项与参考代码仅减去均值的实现不同，教程中必须说明。

### 策略目标与更新

\[
\rho_{i,t}=\exp(\log\pi_\theta(o_{i,t}\mid q,o_{i,<t})
-\log\pi_{\mathrm{old}}(o_{i,t}\mid q,o_{i,<t})),
\]

\[
L_{\mathrm{policy},i,t}=-\min\left(\rho_{i,t}A_i,
\operatorname{clip}(\rho_{i,t},1-\epsilon,1+\epsilon)A_i\right).
\]

- clip 默认 `0.2`。仅 completion token 参与 loss；EOS 参与，EOS 后 padding 和 prompt 不参与。达到生成上限的回答保留，并记录截断率。
- 每条回答先按有效 completion token 求均值，再按回答求均值。多卡和梯度累积保持相同的全局归约含义；最后不足整批的真实样本不能因补齐而重复计数。
- 每批用当前策略完成全部 rollout，更新前以无梯度方式计算并缓存 old logprob，然后默认更新一次。old logprob 在本批更新期间固定。
- 采样及训练概率计算均关闭 dropout。训练采样 temperature、top-p 为 1，关闭 top-k 等额外分布变换，避免采样分布与训练概率口径不一致。
- 默认仅更新一次时，更新前 ratio 应接近 1，clipping 可能很少触发；不能宣称本实验已经展示了 clipping 的稳定性收益。
- 默认学习率 `4e-5`，Adam betas `(0.9, 0.95)`，weight decay 0；每题生成 8 个回答，生成上限 1024 token。
- 单进程默认每次更新累计 4 道题，可逐题采样、逐回答反向传播来限制峰值占用。题数、梯度累积、group size、步数和生成长度通过参数调整，并明确有效全局题数如何随进程数变化。
- 区分采样批次数与实际优化步数；默认训练集一遍。有限步数的试跑仍从原任务取题，不使用其他玩具任务替代。

### KL 小实验

主实验采用 `beta=0`；附加对照从 `beta=0.04` 开始，这只是待验证的起始系数。

\[
\Delta_{i,t}=\log\pi_{\mathrm{ref}}(o_{i,t}\mid q,o_{i,<t})
-\log\pi_\theta(o_{i,t}\mid q,o_{i,<t}),
\qquad
\widehat D_{i,t}=\exp(\Delta_{i,t})-\Delta_{i,t}-1,
\]

\[
L=\operatorname{mean}_i\left[
\operatorname{mean}_{t\in\mathrm{completion}_i}
\left(L_{\mathrm{policy},i,t}+\beta\widehat D_{i,t}\right)\right].
\]

- KL 直接加到 loss，不加入奖励后再进行组内标准化。它是采样 token 上的估计量，不当作全词表精确 KL。
- reference 固定为训练初始策略。基础模型冻结，LoRA 默认零增量初始化；关闭适配器读取 reference，无须另存一整份模型。检查关闭适配器前后的初始概率一致性。
- reference 不随训练更新；old policy 是本次 rollout 的采样策略，两者用途不同。
- 各组都保留总奖励全相同的退化组，保持相同损失归约。它们没有奖励优势信号，但有 KL 时仍参与 KL 项。无 KL 且整批优势全零时跳过优化器更新。加入格式奖励后，全对或全错的组仍可能因格式差异而产生优势。

## 数据划分、评测与实验安排

### 固定划分

- 从 GSM8K train 中按 seed 42 固定抽取 256 题作为验证集，其余为训练集；保存真实索引及数据版本或内容摘要，检查训练、验证不重叠。
- 保留原始训练顺序，排除验证题后取数。本机子集从该训练部分选取，并记录实际使用的索引。
- test 仅用于最终评测，不用于选择训练参数或检查点。

### 流程试跑

使用少量原任务题目和少量步数，验证模型加载、生成、奖励、梯度更新、LoRA 保存与重新加载。group size 至少为 2。若短生成上限导致全部奖励为零，只能说明链路运行或没有学习信号，不能据此判断算法效果。

### 主实验：GRPO 是否改善解题表现

- 比较初始模型与 GRPO 模型，默认训练集一遍；初始模型也必须使用相同提示词和评测方式。
- 固定贪心解码、提示词、答案判定和生成上限，报告单次回答正确率、boxed 格式成功率、平均生成长度和截断率。
- 训练前后及每 50 个采样批次评测固定验证子集；本机默认 64 题，完整实验使用全部验证集。子集题目保持不变。
- 训练日志分别记录总 reward、采样正确率、加权格式奖励、严格格式率，并记录退化组比例、生成长度、截断率、策略 loss、KL、clip fraction、奖励信号 token 数及实际优化步数。无 KL 时不能把未计算的 KL 记录成测得的零值。
- 保存固定题目的训练前后回答，以及训练中的实际 rollout 样例。训练 reward 仅作为过程指标，不代替独立评测。
- 保存 LoRA、配置、依赖版本、数据划分、指标和逐题评测结果；初版使用本地 JSONL，暂不接入 SwanLab。

### 对照实验：KL 与格式奖励

- 三组主要设置是 `(beta=0, λ=0)` 当前 loss、`(beta=0.04, λ=0)` 加 KL、`(beta=0, λ=0.5)` 加格式奖励；额外提供 `(beta=0.04, λ=0.5)` 组合对照。只使用同一训练入口和两个独立参数，命令见 README。
- 各组从同一初始模型独立训练，固定模型、数据顺序、随机种子、采样参数和题目训练预算。比较 KL 时固定 λ，比较格式奖励时固定 beta，不把两种变化混为同一个因素。
- 比较验证正确率、输出长度、截断率、训练曲线；各组均可在固定评测回答上测量相对 reference 的偏离。贪心回答上的 KL 估计只作为统一诊断，不能视为策略分布下的无偏 KL。
- 格式奖励启用后，总 reward 的范围变化，不能拿不同配置的总 reward 直接比较性能；使用同一评测规则下的 accuracy、think_format_rate 和 boxed_rate。格式奖励全零时不会新增格式优势信号。
- 第一轮作为探索性结果；出现值得追踪的差异后补多个种子，不根据一次结果宣称 KL 普遍有效或无效。

## 必要验证与验收

- 单独验证 boxed 提取、标准答案解析、归一化判分、正确性与格式奖励的独立性、标准化优势与退化组行为，并覆盖四种参数组合。
- 用小张量验证自回归 token 对齐、EOS/padding mask、策略 loss 的正负梯度方向、clip 行为、KL 在相同分布时为零及其梯度。
- 实际模型试跑检查初始 ratio 接近 1、reference KL 接近 0，只有预期 LoRA 参数获得梯度，基础模型与 reference 不变化。
- 单卡 GPU 分别检查四组实际采样流程，并使用显式构造的真实任务回答覆盖非零格式奖励；检查全部冻结参数哈希、梯度裁剪、旧/参考概率不变，以及保存重载后的概率与生成一致。这些受控回答不作为自由采样结果或训练收益证据。
- 验证退化组保留、无信号批次跳过、最后不足整批时的损失归约，以及保存后重新加载可以评测。
- 多卡检查完整 group 留在同一进程、梯度的全局平均和评测去重；没有执行过的配置明确标为未验证。
- 不预填训练收益、显存需求、耗时或原论文指标。若硬件不足需要调整模型，记录模型、数据、训练或评测设置的变化。

## 参考资料

- 原教程：`~/Projects/agentic-rl-lab/experiments/grpo/readme.md`。
- 原同步实现：`~/Projects/agentic-rl-lab/experiments/grpo/01-demo-sync.py`；`03-grpo.py` 仅支持两个 loss，不能混用说明。
- [参考仓库 GRPO 目录](https://github.com/datawhalechina/agentic-rl-lab/tree/main/experiments/grpo)
- [DeepSeekMath 论文](https://arxiv.org/abs/2402.03300)
- [Qwen2.5-0.5B 模型说明](https://huggingface.co/Qwen/Qwen2.5-0.5B)
- [DeepSeek-R1 训练模板与评测说明](https://arxiv.org/html/2501.12948v1)
- [Accelerate Quicktour](https://huggingface.co/docs/accelerate/quicktour)
- [TRL GRPO 说明，用于核对算法差异](https://huggingface.co/docs/trl/grpo_trainer)
