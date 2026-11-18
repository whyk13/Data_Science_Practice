# -*- coding: utf-8 -*-
"""Derive the student worksheet from the reviewer (answers-filled) lab notebook.

Reads   notebook/pbl-{slug}-lab.ipynb            (reviewer copy, answers filled)
Writes  notebook/pbl-{slug}-lab-student.ipynb    (blanks to be completed by students)

The reviewer copy is never modified. Re-run this after editing reviewer prose so
the two stay in sync. Unlike build_review_notebook.py, this script does not
regenerate the lab from scratch - it copies and blanks, so hand-edited wording
in the reviewer notebook is preserved.

What it does:
  * replaces every "示范作答" block with a fillable prompt
  * removes teacher-facing material (objective-evidence map, reviewer notes)
  * unchecks all decision checkboxes
  * turns the transfer-homework demo code into a TODO scaffold
  * sets REVIEW_MODE = False and clears every cell output
"""
from __future__ import annotations

import json
from pathlib import Path

COURSE_DIR = Path(__file__).resolve().parents[2]
SLUG = "machine-learning-basics-naive-bayes"
SOURCE = COURSE_DIR / "notebook" / f"pbl-{SLUG}-lab.ipynb"
TARGET = COURSE_DIR / "notebook" / f"pbl-{SLUG}-lab-student.ipynb"

BLANK = "＿＿＿＿"
WRITE = "*在这里作答*"

# --------------------------------------------------------------------------
# Markdown cells replaced wholesale in the student version.
# Cells not listed here are carried over from the reviewer notebook unchanged.
# --------------------------------------------------------------------------
MD = {}

MD[0] = r"""# 第 7 讲：机器学习基础与朴素贝叶斯
## Interactive PBL Worksheet + Reproducible Lab（学生版）

> **用途边界**：这份 Notebook 只用于机器学习教学。它使用的是医学专家按规则构造出来的公开教学数据，不能拿去支持临床诊断、患者照护、模型部署或任何因果结论。

| 项目 | 内容 |
|---|---|
| 学习者 | 生命科学专业、即将进入大四的直博生 |
| 主数据 | UCI Acute Inflammations，120 条由医学专家构造的症状—标签组合，6 个原始特征 |
| 主标签 | 是否存在膀胱炎症（59 阳性，61 阴性） |
| 核心决策 | 现有证据支持把这个模型用到哪一级：教学基线、进一步验证，还是前瞻性筛查试点？ |
| 学习结构 | 问题触发 → ML 基础复习 → 课间 Packet A 预习 → Section 2 课堂 Lab → Packet C/D 综合 → 迁移 |
| 完成方式 | 个人独立完成 |

> **怎么填：** 带 ✍️ 的地方都要你写下答案，表格里的 `＿＿＿＿` 和 `*在这里作答*` 同理——双击单元格就能编辑，写完按 `Shift+Enter` 渲染。答案没有标准模板：评分看的是每个结论能不能指出对应的证据，以及边界有没有说清楚。"""

MD[3] = r"""## 学习目标（由课程提纲压缩为 6 项）

学完这份 Notebook，你应该能够：

1. **LO1｜定义预测任务：** 说清 observation unit、feature、label 分别是什么，以及模型的输入和输出各是什么。
2. **LO2｜解释学习流程：** 区分 training、prediction、generalization 和 loss，并说明 train/validation/test 各自负责什么。
3. **LO3｜解释朴素贝叶斯：** 用 prior、likelihood、posterior、条件独立和 Laplace smoothing 讲清它的机制，并手算一次预测。
4. **LO4｜评价分类模型：** 从混淆矩阵算出并解释 accuracy、sensitivity/recall、specificity、precision 和 F1，同时讲清 ROC-AUC、PR-AUC 与 calibration 的分工。
5. **LO5｜审查模型假设：** 借助特征依赖、重复证据和概率质量，判断 NB 的长处与局限各在哪里。
6. **LO6｜形成有边界的结论：** 综合 locked test、不确定性和错误代价，说明医学模型为什么不能只看 accuracy。"""

MD[5] = r"""## STEP 1｜问题触发：100% 是否足够？（6 min）

有个研究小组用 6 项症状特征训练了一个 Bernoulli 朴素贝叶斯模型，用来判断是否存在膀胱炎症。现在他们把结果送来评审，而摘要里关于性能只有一句话：

> “模型在锁定测试集上的 accuracy 是 100%。”

你要给出的判断是：这些证据只够把模型当作教学/方法基线，还是已经可以进入进一步验证，甚至前瞻性筛查试点？

> ⚠️ 先不要往下运行任何数据分析。这一步要的就是你在证据不足时的判断。

### ✍️ 初始决定

勾选一项（把 `[ ]` 改成 `[x]`）：

- [ ] A. 证据足以进入前瞻性筛查试点
- [ ] B. 目前只支持教学/方法基线，并应继续收集验证证据
- [ ] C. 现有证据足以直接否定该模型
- [ ] D. 不先定义用途和错误代价，无法评价

**✍️ 理由**（2–3 句：你依据什么做出这个选择？）

*在这里作答*

**✍️ 当前置信度：** ＿＿＿＿%

**✍️ 向模型团队追问**（写 3 条你最想问的问题）

1. ＿＿＿＿
2. ＿＿＿＿
3. ＿＿＿＿"""

MD[7] = r"""## STEP 2｜机器学习基础快速复习：一次完整预测流程（14 min）

> 这一段不是让你背定义，而是要能说出每个概念在整个研究流程里负责哪一段。

**本课工作流：** 数据输入 → 定义 sample / feature / label → 划分数据 → 用 train 拟合模型 → 用 validation 选方案与阈值 → 冻结方案 → 用 locked test 评价 → 说明泛化边界。

### 2.1 数据模型：从输入到输出

监督学习模型可以写成 $f_\theta:X\rightarrow\hat{Y}$：参数 $\theta$ 刻画的是输入到输出的映射关系。分类模型通常先给出概率 $P(Y=1\mid X=x)$，再用一个阈值把它转成类别。

**✍️ 把最后一列填完**——每个概念在本案例里具体对应什么？

| 概念 | 工作含义 | 本案例中的对应物 |
|---|---|---|
| sample / observation unit | 一个独立观察对象，即数据表的一行 | ＿＿＿＿ |
| feature $X$ | 提供给模型的输入变量 | ＿＿＿＿ |
| label $Y$ | 希望模型预测的已知结果 | ＿＿＿＿ |
| model $f_\theta$ | 把特征映射成分数、概率或类别的函数 | ＿＿＿＿ |
| training | 用训练数据估计参数 $\theta$ | ＿＿＿＿ |
| prediction | 参数固定之后，对新样本产生输出 | ＿＿＿＿ |
| generalization | 模型在没参与建模的新样本上仍然有效 | ＿＿＿＿ |

### 2.2 为什么要分 train / validation / test？

| 数据部分 | 主要职责 | 不应做什么 |
|---|---|---|
| Train | 拟合模型参数 | 把训练表现当成泛化表现 |
| Validation | 选模型、阈值和超参数 | 一边看结果一边无限次调整 |
| Locked test | 方案冻结后做一次最终评价 | 看完结果再回头改方案 |

训练表现与新样本表现之间的落差就是**泛化差距**。之所以要拆分数据，是为了在手头这批有限的数据里，人为造出一批“模型还没见过”的样本。

### 2.3 Loss：模型错得有多严重？

Loss 把“一次预测离真实标签有多远”折算成一个数。很多模型就是靠最小化平均 loss 训练出来的。它和最后写进报告的评价指标有关系，但两者不是一回事。

- 二分类概率常用 cross-entropy：
  $L=-[y\log p+(1-y)\log(1-p)]$。预测错了还很自信，罚得特别重。
- 连续结局的回归常用 mean squared error：
  $L=(y-\hat y)^2$，误差越大，平方之后被放得越大。
- 本课的 NB 靠数频次来估概率，并不最小化上面任何一个 loss；但我们后面照样可以用 log loss 检查它给出的概率好不好。

### 2.4 评价不是只报一个分数

阈值一旦定下来，每个样本就落进 TP、FP、FN、TN 四个格子里。下面这些分类指标，无非是对这四个数的不同概括方式。

| 层次 | 指标 | 它主要回答什么？ |
|---|---|---|
| 固定阈值后的分类 | Accuracy | 全部样本中分对了多少？ |
|  | Sensitivity = Recall | 真阳性中找出了多少？ |
|  | Specificity | 真阴性中排除了多少？ |
|  | Precision | 判为阳性的样本里有多少是真的？ |
|  | F1 | Precision 与 recall 的调和平均 |
| 不固定单一阈值的排序 | ROC-AUC | 阳性是否总体上排在阴性前面？ |
|  | PR-AUC | 针对阳性类别，precision–recall 怎么权衡？ |
| 概率质量 | Calibration | 预测为 0.8 的那批样本，是否真有约 80% 为阳性？ |

Accuracy 这一类指标依赖你选的那个阈值，AUC 概括的是跨阈值的排序能力，calibration 看的是概率数值本身可不可信。三者互相替代不了，解释时还要一并考虑患病率、两类错误的代价和估计的不确定性。

### ✍️ 三个快速判断

1. 降低阳性阈值，通常会让 sensitivity ＿＿＿＿（上升/下降）、specificity ＿＿＿＿（上升/下降）。
2. Validation 和 locked test 的职责有什么不同？如果 test 参与了调参，会失去什么？

   *在这里作答*

3. 如果首要目标是少漏诊，应该优先看哪个指标？只报这一个够不够，为什么？

   *在这里作答*

### 2.5 Loss 证据活动：相同 accuracy，loss 可以不同

下面两个概率模型在同样 4 个样本上都只错 1 个，accuracy 完全一样。区别只在于：模型 B 在错的那个样本上给出了 0.99 的高置信度。

**✍️ 先预测再运行：** 哪个模型的 cross-entropy 会更大？为什么？（写完再运行下一个单元验证）

*在这里作答*"""

MD[9] = r"""## STEP 3｜把案例变成预测问题：3W 与竞争解释（12 min）

### ✍️ 统计检验还是预测？

1. 腰痛与膀胱炎症之间是否存在关联？→ ＿＿＿＿
2. 根据一个新就诊者的症状判断他是否可能患病？→ ＿＿＿＿
3. 这两个问题各自关心什么？用的是同一批变量，为什么回答的不是同一个问题？

   *在这里作答*

### ✍️ 3W

| What we know | What we need to know | What we will do |
|---|---|---|
| ＿＿＿＿ | ＿＿＿＿ | ＿＿＿＿ |
| ＿＿＿＿ | ＿＿＿＿ | ＿＿＿＿ |
| ＿＿＿＿ | ＿＿＿＿ | ＿＿＿＿ |

### ✍️ 竞争解释

“模型在测试集上 100% 正确”这件事，可以有好几种互相竞争的解释。至少写出 4 种，并说明各自该看什么证据。

| 解释/决策 | 如果它成立，应看到什么证据？ | 什么证据会推翻它？ | 错误选择的后果 |
|---|---|---|---|
| H1：＿＿＿＿ | ＿＿＿＿ | ＿＿＿＿ | ＿＿＿＿ |
| H2：＿＿＿＿ | ＿＿＿＿ | ＿＿＿＿ | ＿＿＿＿ |
| H3：＿＿＿＿ | ＿＿＿＿ | ＿＿＿＿ | ＿＿＿＿ |
| H4：＿＿＿＿ | ＿＿＿＿ | ＿＿＿＿ | ＿＿＿＿ |"""

MD[10] = r"""## STEP 4｜参考阅读、个人学习契约与 Evidence Card（13 min，含 Exit ticket）

### 本 Notebook 内的参考阅读路线

1. Bayes theorem：prior → likelihood → posterior。
2. Bernoulli NB：二元特征、条件独立、Laplace smoothing。
3. 混淆矩阵：错误类型及其代价。
4. 三类评价：分类决策、排序/区分能力、概率可信度。
5. 泛化边界：小样本、数据来源、测量误差和分布变化。

### ✍️ 个人学习契约

- **要解决的问题：** ＿＿＿＿
- **要查的来源：** ＿＿＿＿
- **完成判据：**（怎样算学会了？写成可检验的标准）＿＿＿＿
- **来源核验方法：**（你打算怎么确认这些来源是可信的？）＿＿＿＿

### ✍️ Evidence Card

挑一条你在阅读或讨论中遇到的主张，把它审一遍。

| 字段 | 个人记录 |
|---|---|
| 来源/工具 | ＿＿＿＿ |
| 原始主张 | ＿＿＿＿ |
| 支持证据 | ＿＿＿＿ |
| 反证/限制 | ＿＿＿＿ |
| 最终可信度 | ＿＿＿＿ |

### ✍️ Section 1 Exit ticket

现在最缺的是哪一项证据？拿到它之后，你会怎么改变决定？

*在这里作答*"""

MD[11] = r"""# 课间自学｜Packet A + Bayes/NB 准备

> Section 2 之前先做完 Packet A，并把 Bayes/NB 的公式预习一遍；但先别急着拟合模型，也不要提前运行 Packet B–D。这几个 Packet 会在课堂上按 checkpoint 逐步释放。"""

MD[14] = r"""### ✍️ Packet A 审计四问

1. **可以得出什么？**

   *在这里作答*

2. **还不能得出什么？**

   *在这里作答*

3. **如何修订当前决定？**

   *在这里作答*

4. **什么证据会改变决定？**

   *在这里作答*"""

MD[20] = r"""### ✍️ 后验手算任务

对下一单元给定的症状组合，用训练集的先验乘上六个条件概率。数值直接从上一单元打印出来的 prior / likelihood 表里读。

- $score_0 = P(Y=0)\times\prod_j P(x_j\mid Y=0) \approx$ ＿＿＿＿
- $score_1 = P(Y=1)\times\prod_j P(x_j\mid Y=1) \approx$ ＿＿＿＿
- 归一化后 $P(Y=1\mid x)=\dfrac{score_1}{score_0+score_1} \approx$ ＿＿＿＿

所以模型给出的患病概率约为 ＿＿＿＿，在 0.50 的阈值下预测为 ＿＿＿＿（阳性 / 阴性）。

**✍️ 这一步真正要看懂的是什么？**（提示：六个症状项是靠什么假设被合到一起的？）

*在这里作答*

> 算完再运行下一个单元。如果你的手算结果和 `sklearn` 的输出对不上，先回头检查：条件概率有没有读错行？取值为 0 的症状用的是 $P(x_j=1\mid Y=c)$ 还是 $1-P(x_j=1\mid Y=c)$？"""

MD[22] = r"""> ### ✍️ CHECKPOINT 2
> 模型到底从训练集里学到了什么？（提示：先验、条件概率，外加 `alpha=1` 在防的那件事）
>
> *在这里作答*"""

MD[25] = r"""### ✍️ 从混淆矩阵到指标

$$
Sensitivity=\frac{TP}{TP+FN},\quad Specificity=\frac{TN}{TN+FP}
$$

$$
Precision=\frac{TP}{TP+FP},\quad Accuracy=\frac{TP+TN}{N}
$$

1. 从上面的输出读出 validation 混淆矩阵：TP=＿＿＿＿、TN=＿＿＿＿、FP=＿＿＿＿、FN=＿＿＿＿。
2. 手写算式并算出结果：sensitivity = ＿＿＿＿；specificity = ＿＿＿＿。
3. Prior-only baseline 把所有样本判成了哪一类？它的 specificity 和 sensitivity 各是多少？这说明用单一指标评价筛查模型有什么问题？

   *在这里作答*

4. Sensitivity 和 precision 分别在问什么问题？两者有什么区别？

   *在这里作答*"""

MD[27] = r"""### ✍️ Packet B 审计四问

1. **可以得出什么？**

   *在这里作答*

2. **还不能得出什么？**

   *在这里作答*

3. **如何修订当前决定？**

   *在这里作答*

4. **什么证据会改变决定？**

   *在这里作答*

### ✍️ 指标主张 Evidence Card

**待审查主张：** “医学筛查应该优先看 sensitivity。”教材、官方指标定义、论文或 AI 都可以作为来源，但都要拿本数据的混淆矩阵独立核对一遍。

**✍️ 核验结论：** 这句话在什么条件下成立？什么情况下会失效？如果要把它改写得更准确，你会怎么写？

*在这里作答*"""

MD[30] = r"""### ✍️ 先解释，再下结论

看完上面的热图和依赖表再回答，不要凭印象：

1. 类别内相关系数等于 1 意味着什么？这两个症状还算得上两份独立的证据吗？

   *在这里作答*

2. 热图里的 `NaN` 是“没有相关”的意思吗？它更可能说明什么？

   *在这里作答*

3. 条件独立不成立，是不是就等于 NB 一定会分错？

   *在这里作答*

4. 如果类别判对了，是不是就说明 posterior 的数值同样可信？

   *在这里作答*"""

MD[33] = r"""### 概念辨析

| 概念 | 回答的问题 | 本例风险 |
|---|---|---|
| Accuracy | 当前阈值下总体分对了多少？ | 受类别比例和错误代价影响 |
| Sensitivity/Recall | 真阳性里找出了多少？ | 不说明阳性预测中有多少是真的 |
| Precision | 阳性预测里有多少是真的？ | 随患病率变化 |
| ROC-AUC | 随机一个阳性排在随机一个阴性前面的能力 | 不代表阈值选对了，也不代表校准好 |
| PR-AUC | 阳性类别上的 precision–recall 权衡 | 基线本身取决于阳性率 |
| Calibration | 预测 0.8 的那批样本真有约 80% 阳性吗？ | 小样本上曲线非常不稳 |
| Log loss/Brier | 预测概率离真实标签有多远？ | 极端而错误的概率会被重罚 |

### ✍️ Packet C 决策矩阵

面对“症状之间并不条件独立”这个事实，下面四种做法都有人主张。把后四列填完。

| 可选行动 | 理由/假设 | 期待证据 | 验证方式 | 停止条件或副作用 |
|---|---|---|---|---|
| 保留全部特征 | ＿＿＿＿ | ＿＿＿＿ | ＿＿＿＿ | ＿＿＿＿ |
| 删掉重复或高度相关的症状 | ＿＿＿＿ | ＿＿＿＿ | ＿＿＿＿ | ＿＿＿＿ |
| 保留模型，但不解释原始 posterior | ＿＿＿＿ | ＿＿＿＿ | ＿＿＿＿ | ＿＿＿＿ |
| 先收集外部数据再决定 | ＿＿＿＿ | ＿＿＿＿ | ＿＿＿＿ | ＿＿＿＿ |

### ✍️ Packet C 审计四问

1. **可以得出什么？**

   *在这里作答*

2. **还不能得出什么？**

   *在这里作答*

3. **如何修订当前决定？**

   *在这里作答*

4. **什么证据会改变决定？**

   *在这里作答*"""

MD[34] = r"""> ### ✍️ CHECKPOINT 3
> 到这里为止，支持“模型表现好”的证据有哪些？要说它“可信”，还差什么？
>
> *在这里作答*"""

MD[35] = r"""## Packet D｜课堂核心：锁定测试与统计不确定性

这个 Packet 要等模型、特征和阈值全部冻结之后才释放——教师在课堂上宣布之后，把下一个单元里的 `UNLOCK_PACKET_D` 改成 `True` 再运行。

课堂上只做三件事：locked test、混淆矩阵和 Wilson 区间。后面的测量误差压力测试是课后挑战题。

> ⚠️ 在此之前请确认：你的阈值是在 validation 上选定的，而且没有因为看了 test 结果再回头改过。"""

MD[40] = r"""### ✍️ Packet D 审计四问

1. **可以得出什么？**

   *在这里作答*

2. **还不能得出什么？**

   *在这里作答*

3. **如何修订当前决定？**（把 Wilson 区间的下限读出来再回答）

   *在这里作答*

4. **什么证据会改变决定？**

   *在这里作答*

**✍️ 挑战题结论**（课后完成压力测试后再写）：症状录入错误会让性能怎么变？这个模拟能替代外部验证吗？

*在这里作答*

**别忘了患病率：** $PPV=\frac{Sensitivity\times Prevalence}{Sensitivity\times Prevalence+(1-Specificity)(1-Prevalence)}$。同样的 sensitivity 和 specificity，患病率一变，precision 就会跟着动——后面的迁移作业会用到这条式子。

### ✍️ 当前决定

勾选一项（把 `[ ]` 改成 `[x]`）：

- [ ] 批准进入临床试点
- [ ] 仅批准继续教学/方法研究
- [ ] 要求真实外部队列和预先注册的评价方案
- [ ] 拒绝继续使用朴素贝叶斯

**✍️ 依据：**

*在这里作答*"""

MD[41] = r"""# 课后迁移作业｜近迁移 + 新情境迁移

> **课堂导航：** 课上做完 Packet D 之后，请跳过本节和下一个代码单元，直接翻到后面的“Section 2 收束记录”。这一节课后再做。

把目标换成 `nephritis`（肾盂肾炎），完成以下任务：

1. 在不动 locked test 的前提下重新训练模型。
2. 报告 validation 的 confusion matrix、sensitivity、specificity、precision 和 F1。
3. 权衡漏诊与误报的代价，选定一个阈值，并说明依据。
4. 找出至少一组违反条件独立的特征。
5. 回答：**为什么医学诊断模型不能只看 accuracy？**

下一个代码单元给了骨架，把里面的 `TODO` 补完。

**✍️ 迁移任务小结**（第 3 点的阈值依据 + 第 4 点找到的特征对 + 第 5 点的回答）

*在这里作答*

### ✍️ 新情境迁移：低患病率筛查

假设有一项遗传病筛查，目标人群的患病率只有 1%，模型的 sensitivity 和 specificity 都是 95%。两个指标看着都很高，代入上面的 PPV 公式：

$$
Precision=\frac{0.95\times0.01}{0.95\times0.01+0.05\times0.99}
$$

- 算出来 precision = ＿＿＿＿
- 也就是说，被判为阳性的人里只有约 ＿＿＿＿ 真的患病。
- **✍️ 这说明了什么？** 为什么 sensitivity 和 specificity 都是 95%，结果还是这样？

  *在这里作答*"""

MD[43] = r"""# ✍️ CERB 模型审计 Memo

这是本讲的主要提交物。四个部分都要写，每条 Evidence 都要能指回 Notebook 里的具体输出（单元编号、表格或图）。

## Claim

一句话说清你的结论：这个模型可以被用到哪一级？

*在这里作答*

## Evidence

至少三条，分别来自不同类型的证据（分类结果 / 概率与假设 / 不确定性与稳健性）：

1. **分类证据：** ＿＿＿＿
2. **概率/假设证据：** ＿＿＿＿
3. **不确定性/稳健性证据：** ＿＿＿＿

## Reasoning

这些证据为什么支持你的 Claim？把推理链写出来，不要只罗列数字。

*在这里作答*

## Boundary

这份数据和这次分析**不能**说明什么？下一步需要什么样的研究？

*在这里作答*"""

MD[44] = r"""# Section 2 收束记录｜证据综合与观点修订

> Section 2 的时间表在 Packet B 之前已经给过；这里集中记录最后 7 分钟的 re-vote、CERB 和 exit ticket。

### ✍️ Re-vote

初始选择：＿＿＿＿；最终选择：＿＿＿＿；置信度：＿＿＿＿ → ＿＿＿＿。

真正让你改变（或坚持）看法的是哪些证据？

*在这里作答*

### ✍️ Peer challenge

- **你听到的最强反对意见：** ＿＿＿＿
- **你的回应：** ＿＿＿＿
- **目前仍然回答不了的问题：** ＿＿＿＿

### ✍️ Exit ticket

1. **“朴素”朴素在哪？**

   *在这里作答*

2. **数据划分最容易被用错的地方？**

   *在这里作答*

3. **为什么 discrimination 和 calibration 不能互相替代？**

   *在这里作答*

4. **迁移到基因表达分类时先检查什么？**

   *在这里作答*"""

# --------------------------------------------------------------------------
# Code cells: targeted edits only.
# --------------------------------------------------------------------------
CODE_REPLACEMENTS = [
    (
        2,
        'REVIEW_MODE = True  # 审阅稿保持 True；正式分阶段发布时改为 False 并清除后续输出',
        'REVIEW_MODE = False  # 学生版保持 False，Packet D 由教师在课堂上宣布释放',
    ),
    (
        36,
        '# 正式学生分阶段版本应保持 False，教师到 Packet D 再改为 True。',
        '# 先保持 False。等教师在课堂上宣布释放 Packet D，再改成 True 并重新运行本单元。',
    ),
]

TRANSFER_SCAFFOLD = '''# 课后迁移作业：把目标换成 nephritis，自己补完下面的 TODO。
# 规则不变：参数只从 train 估计，阈值在 validation 上选，locked test 只用一次。
TRANSFER_TARGET = "nephritis"
y_transfer = data[TRANSFER_TARGET].copy()
print(f"Transfer target positive rate: {y_transfer.mean():.3f}")

# TODO 1：把 X 和 y_transfer 划分成 train / validation / locked test。
#         沿用主分析的比例（0.20 test，再从剩余里切 0.25 做 validation）、
#         random_state=RANDOM_STATE，并记得 stratify。参考 Packet B1 的写法。

# TODO 2：只用 train 拟合 BernoulliNB(alpha=ALPHA)。

# TODO 3：在 validation 上用 metric_row() 比较几个阈值
#         （例如 0.10 / 0.25 / 0.50 / 0.75 / 0.90），做成一张表。

# TODO 4：说明你选定阈值的依据，然后在 locked test 上评价一次，
#         并打印 confusion_matrix。选完阈值就不要再回头改了。

# TODO 5（可选）：把结果写进 ASSET_DIR，文件名自拟。
'''


def to_source(text: str) -> list[str]:
    lines = text.split("\n")
    return [line + "\n" for line in lines[:-1]] + [lines[-1]]


def main() -> None:
    nb = json.loads(SOURCE.read_text(encoding="utf-8"))

    # 1. markdown blanks
    for index, text in MD.items():
        cell = nb["cells"][index]
        assert cell["cell_type"] == "markdown", f"cell {index} is not markdown"
        cell["source"] = to_source(text)

    # 2. targeted code edits
    for index, old, new in CODE_REPLACEMENTS:
        cell = nb["cells"][index]
        assert cell["cell_type"] == "code", f"cell {index} is not code"
        joined = "".join(cell["source"])
        assert old in joined, f"anchor missing in code cell {index}: {old!r}"
        cell["source"] = to_source(joined.replace(old, new))

    # 3. transfer homework becomes a scaffold
    transfer = nb["cells"][42]
    assert transfer["cell_type"] == "code"
    assert "TRANSFER_TARGET" in "".join(transfer["source"])
    transfer["source"] = to_source(TRANSFER_SCAFFOLD.rstrip("\n"))

    # 4. clear every output so no answer leaks through a stored result
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            cell["outputs"] = []
            cell["execution_count"] = None

    # 5. metadata flags
    nb["metadata"]["pbl"].update({
        "review_answers_filled": False,
        "review_draft": False,
        "student_distribution_ready": True,
        "derived_from": SOURCE.name,
        "derived_by": "planning/scripts/build_student_notebook.py",
    })

    TARGET.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"Wrote {TARGET}")
    print(f"  markdown cells blanked : {sorted(MD)}")
    print(f"  code cells edited      : {[i for i, _, _ in CODE_REPLACEMENTS] + [42]}")
    print(f"  outputs cleared        : {sum(1 for c in nb['cells'] if c['cell_type'] == 'code')} code cells")


if __name__ == "__main__":
    main()
