from __future__ import annotations

import json
from pathlib import Path

import nbformat as nbf
import pandas as pd


COURSE_DIR = Path(__file__).resolve().parents[2]
PLANNING_DIR = COURSE_DIR / "planning"
RAW_DATA = (
    PLANNING_DIR
    / "source-data"
    / "acute-inflammations"
    / "diagnosis.data"
)
SLUG = "machine-learning-basics-naive-bayes"
ASSET_DIR = COURSE_DIR / "assets" / f"pbl-{SLUG}"
NOTEBOOK_DIR = COURSE_DIR / "notebook"
NOTEBOOK_PATH = NOTEBOOK_DIR / f"pbl-{SLUG}-lab.ipynb"


def markdown(text: str, tags: list[str] | None = None):
    cell = nbf.v4.new_markdown_cell(text.strip() + "\n")
    if tags:
        cell.metadata["tags"] = tags
    return cell


def code(text: str, tags: list[str] | None = None):
    cell = nbf.v4.new_code_cell(text.strip() + "\n")
    if tags:
        cell.metadata["tags"] = tags
    return cell


def prepare_dataset() -> None:
    columns = [
        "temperature_c",
        "nausea",
        "lumbar_pain",
        "urine_pushing",
        "micturition_pains",
        "burning_urethra",
        "bladder_inflammation",
        "nephritis",
    ]
    frame = pd.read_csv(
        RAW_DATA,
        sep="\t",
        header=None,
        names=columns,
        encoding="utf-16",
    )
    frame["temperature_c"] = (
        frame["temperature_c"].str.replace(",", ".", regex=False).astype(float)
    )
    for column in columns[1:]:
        frame[column] = frame[column].str.strip().str.lower()

    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(ASSET_DIR / "acute-inflammations.csv", index=False)

    summary = pd.DataFrame(
        [
            {
                "dataset": "UCI Acute Inflammations",
                "samples": len(frame),
                "original_features": 6,
                "targets": 2,
                "primary_target": "bladder_inflammation",
                "primary_positive_n": int(
                    (frame["bladder_inflammation"] == "yes").sum()
                ),
                "primary_negative_n": int(
                    (frame["bladder_inflammation"] == "no").sum()
                ),
                "missing_cells": int(frame.isna().sum().sum()),
                "license": "CC BY 4.0",
                "doi": "10.24432/C5V59S",
            }
        ]
    )
    summary.to_csv(ASSET_DIR / "dataset-summary.csv", index=False)

    analysis_design = {
        "topic": "机器学习基础与朴素贝叶斯",
        "status": "review_draft",
        "primary_target": "bladder_inflammation",
        "transfer_target": "nephritis",
        "features": [
            "fever_ge_38",
            "nausea",
            "lumbar_pain",
            "urine_pushing",
            "micturition_pains",
            "burning_urethra",
        ],
        "preprocessing": "temperature_c >= 38 is converted to fever_ge_38; other features are yes/no binary indicators",
        "split": {"train": 72, "validation": 24, "test": 24},
        "random_state": 7,
        "model": "BernoulliNB(alpha=1.0)",
        "baseline": "DummyClassifier(strategy='prior')",
        "review_mode": True,
        "boundary": "Teaching-only expert-rule dataset; not clinical, deployment, causal, or patient-care evidence.",
    }
    (ASSET_DIR / "analysis-design.json").write_text(
        json.dumps(analysis_design, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def build_notebook() -> None:
    notebook = nbf.v4.new_notebook()
    notebook.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3"},
        "pbl": {
            "topic": "Machine Learning Basics and Naive Bayes",
            "language": "zh-CN",
            "review_draft": True,
            "individual_worksheet": True,
        },
    }

    cells = [
        markdown(
            r"""
# 第 7 讲：机器学习基础与朴素贝叶斯
## Interactive PBL Worksheet + Reproducible Lab（审阅稿）

> **用途边界**：本 Notebook 仅用于机器学习教学。数据来自专家规则构造的公开教学数据集，不能作为临床诊断、患者照护、模型部署或因果推断的证据。

| 项目 | 内容 |
|---|---|
| 学习者 | 生命科学专业、即将进入大四的直博生 |
| 主数据 | UCI Acute Inflammations，120 个样本，6 个原始特征 |
| 主标签 | 是否存在膀胱炎症（59 阳性，61 阴性） |
| 核心决策 | 是否应仅凭“锁定测试集 accuracy = 100%”批准模型进入前瞻性筛查试点？ |
| 学习结构 | 问题触发 -> ML 基础复习 -> Packet A–D -> 迁移与综合 |
| 完成方式 | 每位学生独立填写；讨论可以合作，最终判断必须个人完成 |

<details>
<summary><strong>教师/审阅者说明（点击展开）</strong></summary>

这是审阅版，`REVIEW_MODE=True`，所以 Packet A–D 的代码和输出全部保留，便于审核。正式分阶段发布时可将 `REVIEW_MODE=False`、清除 Packet D 输出，或拆分为多个 Notebook。本文不提供解释性答案，只提供完成判断所需的证据。

</details>
"""
        ),
        markdown(
            r"""
## 如何使用

### 学习顺序

**真实决策触发 -> ML 基础快速复习 -> 明确预测问题与证据需求 -> 数据审计 -> Bayes/NB -> validation 与阈值 -> 假设、校准与稳健性 -> locked test -> 迁移与 CERB**

1. 先完成 **Section 1** 的初始判断，不要先运行数据分析。
2. 到达每个 `CHECKPOINT` 后，记录当前结论再继续。
3. 代码单元可以直接运行；标有“可修改”的参数用于交互实验。
4. 不要求记住全部 API，但必须能够解释输入、输出、假设和指标。
5. 最终提交 Claim–Evidence–Reasoning–Boundary（CERB）短文，而不是一张最高分截图。

建议在 VS Code 或 Jupyter 中选择课程 Python 环境后，从上到下执行一次。
"""
        ),
        code(
            r"""
from pathlib import Path
import json
import platform
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Markdown, display
from sklearn.calibration import calibration_curve
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB

warnings.filterwarnings("ignore", category=UserWarning)
pd.set_option("display.max_columns", 30)
pd.set_option("display.precision", 4)
plt.rcParams.update({"figure.dpi": 115, "axes.spines.top": False, "axes.spines.right": False})

SLUG = "machine-learning-basics-naive-bayes"
REVIEW_MODE = True  # 审阅稿保持 True；正式分阶段发布时改为 False 并清除后续输出
RANDOM_STATE = 7

def locate_course_dir():
    start = Path.cwd().resolve()
    for candidate in [start, *start.parents]:
        if candidate.name == "7. Naive-Bayes":
            return candidate
        nested = candidate / "class2026" / "materials" / "7. Naive-Bayes"
        if nested.exists():
            return nested
    raise FileNotFoundError("找不到 class2026/materials/7. Naive-Bayes；请从仓库内运行本 Notebook。")

COURSE_DIR = locate_course_dir()
ASSET_DIR = COURSE_DIR / "assets" / f"pbl-{SLUG}"
DATA_PATH = ASSET_DIR / "acute-inflammations.csv"
ASSET_DIR.mkdir(parents=True, exist_ok=True)

print("Course directory:", COURSE_DIR)
print("Dataset:", DATA_PATH)
print("Review mode:", REVIEW_MODE)
""",
            tags=["setup"],
        ),
        markdown(
            r"""
## 学习目标

完成本 Notebook 后，你应能：

1. 将一个生物医学问题写成监督学习任务，明确样本、特征和标签。
2. 用“输入 -> 数学模型 -> 概率/类别输出”描述一个预测模型。
3. 区分 training、prediction 和 generalization，并说明 train/validation/test 的不同职责。
4. 说明 loss 的作用，并区分 classification 常用的 cross-entropy 与 regression 常用的 MSE。
5. 用 Bayes 定理、条件概率和 Laplace smoothing 解释 Bernoulli 朴素贝叶斯。
6. 从混淆矩阵解释 accuracy、sensitivity/recall、specificity、precision 和 F1，并说明 ROC-AUC、PR-AUC 与 calibration 各回答什么问题。
7. 用阈值、不确定区间、条件独立性和压力测试审计一个漂亮的模型结果。
"""
        ),
        markdown(
            r"""
# Section 1：课堂问题启动（总计 45 分钟）

| 时间 | 个人任务 | 必须留下的证据 |
|---:|---|---|
| 0–6 min | 阅读问题触发，作出初始决定 | 选择、理由、置信度 |
| 6–20 min | 快速复习预测流程、数据划分、损失与评价 | 案例映射与三个判断 |
| 20–32 min | 区分推断/预测，完成 3W 与竞争解释 | 至少三种解释/决策 |
| 32–40 min | 制定个人学习契约并填写 Evidence Card | 问题、来源与核验标准 |
| 40–45 min | Exit ticket | 当前最缺的一项证据 |
"""
        ),
        markdown(
            r"""
## STEP 1｜问题触发：100% 是否足够？（6 min）

某模型团队使用 6 项症状特征训练 Bernoulli 朴素贝叶斯，预测是否存在膀胱炎症。团队报告：

> “模型在锁定测试集上的 accuracy 是 100%，因此建议进入前瞻性筛查试点。”

### 初始决定（运行任何模型代码之前填写）

- [ ] A. 批准进入试点
- [ ] B. 暂不批准，要求补充证据
- [ ] C. 拒绝该模型方案
- [ ] D. 当前信息不足，无法判断

**你的理由（2–4 句）：**  
_在这里填写……_

**当前置信度（0–100%）：** ___ %

**你最想向模型团队追问的三个问题：**

1. _……_
2. _……_
3. _……_
"""
        ),
        markdown(
            r"""
> ### CHECKPOINT 1
> 在继续前，确认你已经留下初始决定。后面允许改变观点；改变观点不是失分，而是 PBL 的目标。
""",
            tags=["checkpoint"],
        ),
        markdown(
            r"""
## STEP 2｜机器学习基础快速复习：一次完整预测流程（14 min）

> 目标不是背术语，而是知道每个概念在研究流程中负责什么。

**本课工作流：** 数据输入 -> 定义 sample / feature / label -> 划分数据 -> 用 train 拟合模型 -> 用 validation 选方案与阈值 -> 冻结方案 -> 用 locked test 评价 -> 说明泛化边界。

### 2.1 数据模型：从输入到输出

一个监督学习模型可写成 $f_\theta:X\rightarrow\hat{Y}$。它用参数 $\theta$ 描述输入与输出之间的数学关系；分类模型常先输出 $P(Y=1\mid X=x)$，再用阈值转成类别。

| 概念 | 工作含义 | 本案例中的对应物 |
|---|---|---|
| sample | 一个独立观察对象，即数据表的一行 | _填写_ |
| feature $X$ | 提供给模型的输入变量 | _填写_ |
| label $Y$ | 希望模型预测的已知结果 | _填写_ |
| model $f_\theta$ | 从特征映射到分数、概率或类别的函数 | _填写_ |
| training | 用训练数据估计参数 $\theta$ | _填写_ |
| prediction | 固定参数后，对新样本产生输出 | _填写_ |
| generalization | 模型在未参与建模的新样本上仍有效 | _填写_ |

### 2.2 为什么要分 train / validation / test？

| 数据部分 | 主要职责 | 不应做什么 |
|---|---|---|
| Train | 拟合模型参数 | 把训练表现当作泛化表现 |
| Validation | 选模型、阈值和超参数 | 一边看结果一边无限次调整 |
| Locked test | 方案冻结后做一次最终评价 | 看完结果再回头改方案 |

**泛化差距**是训练表现与新样本表现的差异；拆分数据的目的，是模拟“未来样本尚未见过”。

### 2.3 Loss：模型错得有多严重？

Loss 把单个预测与真实标签的差距变成一个数。许多模型通过最小化平均 loss 来训练；它与最终报告的评价指标有关，但不是同一个概念。

- 二分类概率常用 cross-entropy：
  \(L=-[y\log p+(1-y)\log(1-p)]\)。对“很自信但预测错”的情况惩罚很重。
- 连续结局回归常用 mean squared error：
  \(L=(y-\hat y)^2\)。较大的数值误差会被平方放大。
- 本课的 NB 主要通过计数估计概率；稍后仍可用 log loss 检查其概率预测。

### 2.4 评价不是只报一个分数

固定阈值后先得到 TP、FP、FN、TN；下面的分类指标都是对这四个计数的不同概括。

| 层次 | 指标 | 它主要回答什么？ |
|---|---|---|
| 固定阈值后的分类 | Accuracy | 全部样本中分对多少？ |
|  | Sensitivity = Recall | 真阳性中找出了多少？ |
|  | Specificity | 真阴性中排除了多少？ |
|  | Precision | 判为阳性的样本中多少是真的？ |
|  | F1 | Precision 与 recall 的调和平均 |
| 不固定单一阈值的排序 | ROC-AUC | 阳性总体是否排在阴性之前？ |
|  | PR-AUC | 对阳性类别，precision–recall 权衡如何？ |
| 概率质量 | Calibration | 预测为 0.8 的样本是否约有 80% 为阳性？ |

Accuracy 等依赖选定阈值；AUC 概括跨阈值的排序；calibration 检查概率数值。三者不能互相替代，还要结合患病率、错误代价与不确定性。

### 三个快速判断

1. 降低阳性阈值，sensitivity 和 specificity 通常分别怎样变化？ _填写_  
2. 为什么 validation 可以反复比较，而 locked test 应尽量只看一次？ _填写_  
3. 若筛查首先要避免漏诊，你会优先观察什么指标？还必须同时报告什么？ _填写_
"""
        ),
        markdown(
            r"""
## STEP 3｜把案例变成预测问题：3W 与竞争解释（12 min）

### 统计检验还是预测？

1. 腰痛与膀胱炎症是否有关联？ _填写_  
2. 根据一个新就诊者的症状判断其是否可能患病？ _填写_  
3. 二者可以使用同一批变量，但结论为何不同？ _填写_

### 3W

| What we know | What we need to know | What we will do |
|---|---|---|
| 团队声称 test accuracy=100% | _至少写三项_ | _对应的核验方法_ |
| 数据包含症状和疾病标签 | _填写_ | _填写_ |
| _填写_ | _填写_ | _填写_ |

### 至少保留三种竞争解释

| 解释/决策 | 如果它成立，应看到什么证据？ | 什么证据会推翻它？ | 错误选择的后果 |
|---|---|---|---|
| H1：模型学到了可泛化的症状模式 | _填写_ | _填写_ | _填写_ |
| H2：数据过度干净或由规则构造 | _填写_ | _填写_ | _填写_ |
| H3：测试集太小，100% 具有偶然性 | _填写_ | _填写_ | _填写_ |
| H4：相关症状被重复计作独立证据 | _填写_ | _填写_ | _填写_ |
"""
        ),
        markdown(
            r"""
## STEP 4｜参考阅读、个人学习契约与 Evidence Card（13 min，含 Exit ticket）

### 本 Notebook 内的参考阅读路线

1. Bayes theorem：prior -> likelihood -> posterior。
2. Bernoulli NB：二元特征、条件独立、Laplace smoothing。
3. 混淆矩阵：错误类型及其代价。
4. 三类评价：分类决策、排序/区分能力、概率可信度。
5. 泛化边界：小样本、数据来源、测量误差和分布变化。

### 个人学习契约

- 我需要解决的两个问题：_……_
- 我准备检查的两个来源：_……_
- 我认为“完成”的判据：_……_
- 如果 AI 给出答案，我将如何独立核验：_……_

### Evidence Card

| 字段 | 个人记录 |
|---|---|
| 来源/工具 | _教材、论文、官方文档或 AI 输出_ |
| 原始主张 | _准确复述，不先评价_ |
| 支持证据 | _数据、公式或来源_ |
| 反证/限制 | _填写_ |
| 最终可信度 | _高/中/低及理由_ |

### Section 1 Exit ticket

目前最缺少的一项证据是：_……_  
如果得到这项证据，我可能如何改变决定：_……_
"""
        ),
        markdown(
            r"""
# Homework｜Packet A–D 渐进式证据审计

> 每完成一个 Packet，先写当前结论再继续。审阅版保留全部输出；正式教学时应按 checkpoint 分阶段释放。
"""
        ),
        markdown(
            r"""
## Packet A｜数据与“漂亮结论”

先核验最基本的问题：数据有多少样本？每行和每列代表什么？目标是否平衡？有没有缺失值？

注意：**公开数据集**不等于**真实世界代表性数据集**，而“没有缺失值”也不一定是优点。
""",
            tags=["packet-a"],
        ),
        code(
            r"""
raw = pd.read_csv(DATA_PATH)
binary_columns = [
    "nausea", "lumbar_pain", "urine_pushing",
    "micturition_pains", "burning_urethra",
    "bladder_inflammation", "nephritis",
]

data = raw.copy()
for column in binary_columns:
    data[column] = data[column].map({"no": 0, "yes": 1})
data["fever_ge_38"] = (data["temperature_c"] >= 38.0).astype(int)

summary = pd.DataFrame({
    "item": ["samples", "original features", "missing cells", "bladder inflammation positive", "nephritis positive"],
    "value": [len(data), 6, int(data.isna().sum().sum()), int(data["bladder_inflammation"].sum()), int(data["nephritis"].sum())],
})
display(raw.head(8))
display(summary)
display(pd.crosstab(raw["bladder_inflammation"], columns="count").rename_axis("target"))
""",
            tags=["packet-a", "data-loading"],
        ),
        markdown(
            r"""
### Packet A 审计记录

- 现在可以得出什么？ _……_
- 还不能得出什么？ _……_
- 120 个样本是否足以支持临床试点？需要什么额外信息？ _……_
- 数据“整齐、无缺失”可能有哪些解释？ _……_
"""
        ),
        markdown(
            r"""
## 必要概念｜Bayes theorem 与 Bernoulli Naive Bayes

前面的通用流程说明一个预测模型要完成什么；这里进一步说明 Bernoulli NB **如何**从训练数据得到类别概率。

对于类别 (Y=c) 和特征向量 (x)：

\[
P(Y=c\mid X=x) \propto P(Y=c)\prod_{j=1}^{p}P(X_j=x_j\mid Y=c)
\]

- (P(Y=c))：先验概率（prior）。
- (P(X_j=x_j\mid Y=c))：给定类别后的特征概率（likelihood term）。
- (P(Y=c\mid X=x))：观察症状后的后验概率（posterior）。
- “朴素”假设：给定 (Y) 后，各特征相互独立。

对二元特征，Laplace smoothing（此处 α=1）为：

\[
P(X_j=1\mid Y=c)=\frac{N_{jc}+\alpha}{N_c+2\alpha}
\]

**重要区别**：这里的“训练”主要是估计先验和条件概率，并不意味着所有模型都通过梯度下降最小化同一个损失。Cross-entropy/log loss 可用于评价概率预测，但 Bernoulli NB 的参数估计来自其生成式概率模型。
"""
        ),
        code(
            r"""
FEATURES = [
    "fever_ge_38", "nausea", "lumbar_pain", "urine_pushing",
    "micturition_pains", "burning_urethra",
]
TARGET = "bladder_inflammation"
X = data[FEATURES].copy()
y = data[TARGET].copy()

# 先锁定 20% test；再把剩余数据分成 train/validation。
X_train_validation, X_test, y_train_validation, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)
X_train, X_validation, y_train, y_validation = train_test_split(
    X_train_validation,
    y_train_validation,
    test_size=0.25,
    random_state=RANDOM_STATE,
    stratify=y_train_validation,
)

split_audit = pd.DataFrame({
    "split": ["train", "validation", "locked test"],
    "n": [len(y_train), len(y_validation), len(y_test)],
    "positive_n": [int(y_train.sum()), int(y_validation.sum()), int(y_test.sum())],
    "positive_rate": [y_train.mean(), y_validation.mean(), y_test.mean()],
})
display(split_audit)
print("Index overlap train/validation:", len(set(X_train.index) & set(X_validation.index)))
print("Index overlap train/test:", len(set(X_train.index) & set(X_test.index)))
print("Index overlap validation/test:", len(set(X_validation.index) & set(X_test.index)))
""",
            tags=["split-design"],
        ),
        markdown(
            r"""
### 数据划分职责

- **Train**：估计模型参数。
- **Validation**：比较方案、选择阈值、发现问题。
- **Locked test**：方案锁定后只使用一次，估计最终泛化表现。

在本例中，把体温转为 `fever_ge_38` 使用了预先声明的固定阈值。如果阈值是看完整数据后挑出的，就可能引入信息泄露。
"""
        ),
        code(
            r"""
ALPHA = 1.0
model = BernoulliNB(alpha=ALPHA).fit(X_train, y_train)

prior_table = pd.DataFrame({
    "class": model.classes_,
    "train_count": [int((y_train == c).sum()) for c in model.classes_],
    "estimated_prior": np.exp(model.class_log_prior_),
})

probability_rows = []
for class_position, disease_class in enumerate(model.classes_):
    for feature_position, feature in enumerate(FEATURES):
        probability_rows.append({
            "class": int(disease_class),
            "feature": feature,
            "P(feature=1 | class)": np.exp(model.feature_log_prob_[class_position, feature_position]),
        })
likelihood_table = pd.DataFrame(probability_rows)

display(prior_table)
display(likelihood_table.pivot(index="feature", columns="class", values="P(feature=1 | class)"))
""",
            tags=["packet-b", "naive-bayes-mechanics"],
        ),
        markdown(
            r"""
### 后验手算任务

选择一个症状组合，先用上表手工写出两个未归一化得分：

\[
score_c=P(Y=c)\prod_j P(X_j=x_j\mid Y=c)
\]

然后归一化：

\[
P(Y=1\mid x)=\frac{score_1}{score_0+score_1}
\]

**手算记录：** _……_  
**你预计模型输出：** _……_  
运行下一单元核对，但不要只抄结果。
"""
        ),
        code(
            r"""
# 可修改交互 1：改变一个病例的症状组合，再运行本单元。
PATIENT = {
    "fever_ge_38": 0,
    "nausea": 1,
    "lumbar_pain": 1,
    "urine_pushing": 0,
    "micturition_pains": 1,
    "burning_urethra": 1,
}

def manual_nb_audit(model, profile):
    x_row = np.array([profile[name] for name in FEATURES], dtype=int)
    log_scores = model.class_log_prior_.copy()
    contribution_rows = []
    for class_position, disease_class in enumerate(model.classes_):
        for feature_position, feature in enumerate(FEATURES):
            p_one = np.exp(model.feature_log_prob_[class_position, feature_position])
            term = np.log(p_one) if x_row[feature_position] == 1 else np.log(1 - p_one)
            log_scores[class_position] += term
            contribution_rows.append({
                "class": int(disease_class), "feature": feature,
                "observed": int(x_row[feature_position]), "log contribution": term,
            })
    shifted = np.exp(log_scores - log_scores.max())
    posterior = shifted / shifted.sum()
    return posterior, pd.DataFrame(contribution_rows)

posterior, contributions = manual_nb_audit(model, PATIENT)
display(pd.DataFrame([PATIENT]))
display(contributions.pivot(index="feature", columns="class", values="log contribution"))
print(f"Manual P(disease=1 | symptoms) = {posterior[1]:.4f}")
print(f"sklearn P(disease=1 | symptoms) = {model.predict_proba(pd.DataFrame([PATIENT]))[0,1]:.4f}")
""",
            tags=["interactive", "manual-calculation"],
        ),
        markdown(
            r"""
> ### CHECKPOINT 2
> 用一句话说明：本模型“学习”的到底是什么？如果某个训练组合从未出现，Laplace smoothing 起什么作用？

_你的回答：……_
""",
            tags=["checkpoint"],
        ),
        markdown(
            r"""
## Packet B｜第一次模型评价：先看 validation，不碰 locked test

先验基线只知道训练集的患病比例；朴素贝叶斯额外使用症状。比较模型时不能只看一个总分。
""",
            tags=["packet-b"],
        ),
        code(
            r"""
def metric_row(y_true, probability, threshold=0.5):
    prediction = (np.asarray(probability) >= threshold).astype(int)
    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, prediction),
        "sensitivity_recall": recall_score(y_true, prediction, zero_division=0),
        "specificity": recall_score(y_true, prediction, pos_label=0, zero_division=0),
        "precision": precision_score(y_true, prediction, zero_division=0),
        "f1": f1_score(y_true, prediction, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probability),
        "pr_auc": average_precision_score(y_true, probability),
        "brier": brier_score_loss(y_true, probability),
        "log_loss": log_loss(y_true, probability),
    }

baseline = DummyClassifier(strategy="prior").fit(X_train, y_train)
validation_probability_nb = model.predict_proba(X_validation)[:, 1]
validation_probability_baseline = baseline.predict_proba(X_validation)[:, 1]

validation_metrics = pd.DataFrame([
    {"model": "prior-only baseline", **metric_row(y_validation, validation_probability_baseline)},
    {"model": "Bernoulli naive Bayes", **metric_row(y_validation, validation_probability_nb)},
])
display(validation_metrics)
validation_metrics.to_csv(ASSET_DIR / "packet-b-validation-metrics.csv", index=False)

cm_validation = confusion_matrix(y_validation, validation_probability_nb >= 0.5)
fig, ax = plt.subplots(figsize=(5.3, 4.4))
image = ax.imshow(cm_validation, cmap="Blues", vmin=0)
for row in range(2):
    for col in range(2):
        ax.text(col, row, cm_validation[row, col], ha="center", va="center", fontsize=18, fontweight="bold")
ax.set(xticks=[0,1], yticks=[0,1], xticklabels=["Predicted no", "Predicted yes"],
       yticklabels=["Actual no", "Actual yes"], xlabel="Prediction", ylabel="Reference label",
       title="Packet B: validation confusion matrix (threshold=0.50)")
fig.colorbar(image, ax=ax, shrink=0.8)
fig.tight_layout()
fig.savefig(ASSET_DIR / "packet-b-validation-confusion-matrix.png", dpi=180, bbox_inches="tight")
plt.show()
""",
            tags=["packet-b", "model-evaluation"],
        ),
        markdown(
            r"""
### 从混淆矩阵到指标

\[
Sensitivity=\frac{TP}{TP+FN},\quad Specificity=\frac{TN}{TN+FP}
\]

\[
Precision=\frac{TP}{TP+FP},\quad Accuracy=\frac{TP+TN}{N}
\]

1. 请从图中读出 TP、TN、FP、FN。 _……_  
2. 手算 sensitivity 和 specificity。 _……_  
3. 为什么 baseline 的 specificity 很高却毫无筛查价值？ _……_  
4. sensitivity 与 precision 分别回答什么问题？ _……_
"""
        ),
        code(
            r"""
# 可修改交互 2：调整阈值，观察错误类型如何变化。
THRESHOLD = 0.50  # 尝试 0.10、0.25、0.50、0.75、0.90

threshold_grid = np.linspace(0.02, 0.98, 49)
threshold_results = pd.DataFrame([
    metric_row(y_validation, validation_probability_nb, threshold=t)
    for t in threshold_grid
])
selected_result = pd.DataFrame([
    metric_row(y_validation, validation_probability_nb, threshold=THRESHOLD)
])
display(selected_result)
threshold_results.to_csv(ASSET_DIR / "packet-b-threshold-path.csv", index=False)

fig, ax = plt.subplots(figsize=(8.5, 4.6))
for metric, color in [("sensitivity_recall", "#dc2626"), ("specificity", "#059669"), ("precision", "#7c3aed"), ("accuracy", "#2563eb")]:
    ax.plot(threshold_results["threshold"], threshold_results[metric], label=metric, color=color, linewidth=2)
ax.axvline(THRESHOLD, color="#111827", linestyle="--", label=f"selected={THRESHOLD:.2f}")
ax.set(xlabel="Decision threshold", ylabel="Metric value", ylim=(-0.03, 1.05),
       title="Packet B: threshold changes the error trade-off")
ax.legend(ncol=3, frameon=False)
fig.tight_layout()
fig.savefig(ASSET_DIR / "packet-b-threshold-tradeoff.png", dpi=180, bbox_inches="tight")
plt.show()
""",
            tags=["packet-b", "interactive", "threshold"],
        ),
        markdown(
            r"""
### Packet B 审计记录

- 如果漏诊代价更高，阈值应倾向怎样调整？为什么？ _……_
- ROC-AUC 和 PR-AUC 不依赖当前单一阈值；它们是否因此足够？ _……_
- Validation 表现很好时，为什么仍不能报告 test 结论？ _……_

### AI Evidence Card

让 AI 解释“医学分类模型应该优先看 sensitivity 而不是 accuracy”。然后核验：

1. AI 是否说明了任务和错误代价？
2. AI 是否混淆 sensitivity 与 precision？
3. AI 是否把一个指标说成适用于所有医学任务？
4. 你依据什么独立证据接受或修改它的答案？
"""
        ),
        markdown(
            r"""
## Packet C｜“朴素”假设审计：相关证据会被重复计算吗？

条件独立不是说特征在总体上互不相关，而是要求：**在给定疾病类别后，知道一个症状不再改变另一个症状的概率**。

真实症状常由共同机制产生，因此该假设很可能只是近似。
""",
            tags=["packet-c"],
        ),
        code(
            r"""
dependence_rows = []
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
for axis, disease_status in zip(axes, [0, 1]):
    corr = X[y == disease_status].corr()
    shown = np.ma.masked_invalid(corr.to_numpy())
    image = axis.imshow(shown, cmap="coolwarm", vmin=-1, vmax=1)
    axis.set_xticks(range(len(FEATURES)), FEATURES, rotation=55, ha="right")
    axis.set_yticks(range(len(FEATURES)), FEATURES)
    axis.set_title(f"Within class Y={disease_status}")
    for row in range(len(FEATURES)):
        for col in range(len(FEATURES)):
            value = corr.iloc[row, col]
            if not np.isnan(value):
                axis.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=7)
    for row in range(len(FEATURES)):
        for col in range(row + 1, len(FEATURES)):
            dependence_rows.append({
                "target_status": disease_status,
                "feature_1": FEATURES[row],
                "feature_2": FEATURES[col],
                "phi_correlation": corr.iloc[row, col],
            })
fig.colorbar(image, ax=axes, shrink=0.82, label="Phi correlation")
fig.suptitle("Packet C: feature dependence conditional on the target", fontweight="bold")
fig.savefig(ASSET_DIR / "packet-c-conditional-dependence.png", dpi=180, bbox_inches="tight")
plt.show()

dependence_table = pd.DataFrame(dependence_rows)
dependence_table.to_csv(ASSET_DIR / "packet-c-conditional-dependence.csv", index=False)
display(dependence_table.assign(abs_phi=lambda d: d["phi_correlation"].abs()).sort_values("abs_phi", ascending=False).head(10))
""",
            tags=["packet-c", "assumption-audit"],
        ),
        markdown(
            r"""
### 先解释，再下结论

- 相关系数为 1 的症状对说明什么？ _……_
- 图中的 `NaN` 是否等于“没有相关”？检查对应特征在该类别内是否为常数。 _……_
- 条件独立不成立是否意味着朴素贝叶斯一定分类失败？ _……_
- 即使分类仍正确，概率可能出现什么问题？ _……_
"""
        ),
        code(
            r"""
# 可修改交互 3：模拟同一症状被重复录入，观察 posterior 是否不合理地变化。
DUPLICATED_FEATURE = "nausea"

X_train_duplicate = X_train.copy()
X_train_duplicate[f"{DUPLICATED_FEATURE}_duplicate"] = X_train_duplicate[DUPLICATED_FEATURE]
duplicate_model = BernoulliNB(alpha=ALPHA).fit(X_train_duplicate, y_train)

all_profiles = pd.DataFrame(
    [[(profile >> bit) & 1 for bit in range(len(FEATURES))] for profile in range(2 ** len(FEATURES))],
    columns=FEATURES,
)
all_profiles_duplicate = all_profiles.copy()
all_profiles_duplicate[f"{DUPLICATED_FEATURE}_duplicate"] = all_profiles_duplicate[DUPLICATED_FEATURE]

duplicate_audit = all_profiles.copy()
duplicate_audit["original_probability"] = model.predict_proba(all_profiles)[:, 1]
duplicate_audit["duplicate_probability"] = duplicate_model.predict_proba(all_profiles_duplicate)[:, 1]
duplicate_audit["probability_change"] = duplicate_audit["duplicate_probability"] - duplicate_audit["original_probability"]
duplicate_audit["decision_changed_at_0.5"] = (
    (duplicate_audit["original_probability"] >= 0.5)
    != (duplicate_audit["duplicate_probability"] >= 0.5)
)
duplicate_audit.to_csv(ASSET_DIR / "packet-c-duplicate-feature-audit.csv", index=False)

display(duplicate_audit.reindex(duplicate_audit["probability_change"].abs().sort_values(ascending=False).index).head(10))
print("Decisions changed:", int(duplicate_audit["decision_changed_at_0.5"].sum()), "/", len(duplicate_audit))
print("Largest absolute probability shift:", duplicate_audit["probability_change"].abs().max())

fig, ax = plt.subplots(figsize=(6.3, 5.4))
colors = np.where(duplicate_audit["decision_changed_at_0.5"], "#dc2626", "#2563eb")
ax.scatter(duplicate_audit["original_probability"], duplicate_audit["duplicate_probability"],
           c=colors, alpha=0.78, edgecolor="white", linewidth=0.4)
ax.plot([0,1], [0,1], "--", color="#111827", linewidth=1)
ax.axvline(0.5, color="#6b7280", linestyle=":")
ax.axhline(0.5, color="#6b7280", linestyle=":")
ax.set(xlim=(-0.03,1.03), ylim=(-0.03,1.03),
       xlabel="P(disease) with original features",
       ylabel=f"P(disease) after duplicating {DUPLICATED_FEATURE}",
       title="Packet C: redundant evidence can change a decision")
fig.tight_layout()
fig.savefig(ASSET_DIR / "packet-c-duplicate-feature-shift.png", dpi=180, bbox_inches="tight")
plt.show()
""",
            tags=["packet-c", "interactive", "assumption-audit"],
        ),
        code(
            r"""
validation_duplicate = X_validation.copy()
validation_duplicate[f"{DUPLICATED_FEATURE}_duplicate"] = validation_duplicate[DUPLICATED_FEATURE]
validation_probability_duplicate = duplicate_model.predict_proba(validation_duplicate)[:, 1]

probability_quality = pd.DataFrame([
    {
        "model": "original",
        "accuracy": accuracy_score(y_validation, validation_probability_nb >= 0.5),
        "brier": brier_score_loss(y_validation, validation_probability_nb),
        "log_loss": log_loss(y_validation, validation_probability_nb),
    },
    {
        "model": f"duplicate {DUPLICATED_FEATURE}",
        "accuracy": accuracy_score(y_validation, validation_probability_duplicate >= 0.5),
        "brier": brier_score_loss(y_validation, validation_probability_duplicate),
        "log_loss": log_loss(y_validation, validation_probability_duplicate),
    },
])
display(probability_quality)

fig, ax = plt.subplots(figsize=(5.8, 5.0))
ax.plot([0,1], [0,1], "--", color="#6b7280", label="perfect calibration")
for label, probability, color in [
    ("original", validation_probability_nb, "#2563eb"),
    ("duplicate symptom", validation_probability_duplicate, "#dc2626"),
]:
    observed, predicted = calibration_curve(y_validation, probability, n_bins=4, strategy="quantile")
    ax.plot(predicted, observed, marker="o", linewidth=2, color=color, label=label)
ax.set(xlabel="Mean predicted probability", ylabel="Observed positive fraction",
       xlim=(-0.03,1.03), ylim=(-0.03,1.03),
       title="Packet C: calibration audit (validation n=24)")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(ASSET_DIR / "packet-c-calibration-audit.png", dpi=180, bbox_inches="tight")
plt.show()
print("Caution: with only 24 validation samples, this curve is diagnostic evidence, not a stable calibration claim.")
""",
            tags=["packet-c", "calibration"],
        ),
        markdown(
            r"""
### 概念辨析

| 概念 | 回答的问题 | 本例风险 |
|---|---|---|
| Accuracy | 当前阈值下总体分对多少？ | 受类别比例和错误代价影响 |
| Sensitivity/Recall | 真阳性中找出多少？ | 不说明阳性预测有多少是真的 |
| Precision | 阳性预测中多少是真的？ | 随患病率改变 |
| ROC-AUC | 随机阳性排序高于随机阴性的能力 | 不等于选好了阈值，也不等于校准 |
| PR-AUC | 阳性类别的 precision–recall 权衡 | 基线依赖阳性率 |
| Calibration | 预测 0.8 的样本是否约 80% 阳性？ | 小样本曲线很不稳定 |
| Log loss/Brier | 概率预测离标签有多远？ | 极端错误概率会被惩罚 |

### Packet C 决策矩阵

| 可选行动 | 理由/假设 | 期待证据 | 验证方式 | 停止条件或副作用 |
|---|---|---|---|---|
| 保留所有特征 | _填写_ | _填写_ | _填写_ | _填写_ |
| 删除重复/高度相关症状 | _填写_ | _填写_ | _填写_ | _填写_ |
| 保留模型但不解释原始 posterior | _填写_ | _填写_ | _填写_ | _填写_ |
| 收集外部数据后再决定 | _填写_ | _填写_ | _填写_ | _填写_ |
"""
        ),
        markdown(
            r"""
> ### CHECKPOINT 3
> 在看到 locked test 前重新写一句判断：模型目前“表现好”的证据是什么？“可信”的证据还缺什么？

_你的回答：……_
""",
            tags=["checkpoint"],
        ),
        markdown(
            r"""
## Packet D｜锁定测试、统计不确定性与压力测试

正式教学中，本 Packet 应在模型、特征和阈值冻结后才释放。审阅版默认解锁。
""",
            tags=["packet-d"],
        ),
        code(
            r"""
# 正式学生分阶段版本应保持 False，教师到 Packet D 再改为 True。
UNLOCK_PACKET_D = False
PACKET_D_AVAILABLE = REVIEW_MODE or UNLOCK_PACKET_D
print("Packet D available:", PACKET_D_AVAILABLE)
""",
            tags=["packet-d", "release-control"],
        ),
        code(
            r"""
def wilson_interval(successes, total, z=1.959963984540054):
    proportion = successes / total
    denominator = 1 + z**2 / total
    center = (proportion + z**2 / (2 * total)) / denominator
    half = z * np.sqrt(proportion * (1 - proportion) / total + z**2 / (4 * total**2)) / denominator
    return center - half, center + half

if not PACKET_D_AVAILABLE:
    display(Markdown("🔒 **Packet D 尚未释放。请先完成 CHECKPOINT 3。**"))
else:
    test_probability = model.predict_proba(X_test)[:, 1]
    test_prediction = (test_probability >= THRESHOLD).astype(int)
    test_metrics = pd.DataFrame([metric_row(y_test, test_probability, THRESHOLD)])
    display(test_metrics)
    test_metrics.to_csv(ASSET_DIR / "packet-d-locked-test-metrics.csv", index=False)

    cm_test = confusion_matrix(y_test, test_prediction)
    tn, fp, fn, tp = cm_test.ravel()
    interval_rows = []
    for name, successes, total in [
        ("accuracy", tp + tn, len(y_test)),
        ("sensitivity", tp, tp + fn),
        ("specificity", tn, tn + fp),
    ]:
        low, high = wilson_interval(successes, total)
        interval_rows.append({"metric": name, "estimate": successes/total, "ci_low": low, "ci_high": high, "denominator": total})
    interval_table = pd.DataFrame(interval_rows)
    display(interval_table)
    interval_table.to_csv(ASSET_DIR / "packet-d-wilson-ci.csv", index=False)

    fig, ax = plt.subplots(figsize=(5.3, 4.4))
    image = ax.imshow(cm_test, cmap="Blues", vmin=0)
    for row in range(2):
        for col in range(2):
            ax.text(col, row, cm_test[row, col], ha="center", va="center", fontsize=18, fontweight="bold")
    ax.set(xticks=[0,1], yticks=[0,1], xticklabels=["Predicted no", "Predicted yes"],
           yticklabels=["Actual no", "Actual yes"], xlabel="Prediction", ylabel="Reference label",
           title=f"Packet D: locked test (threshold={THRESHOLD:.2f}, n={len(y_test)})")
    fig.colorbar(image, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(ASSET_DIR / "packet-d-locked-test-confusion-matrix.png", dpi=180, bbox_inches="tight")
    plt.show()
""",
            tags=["packet-d", "locked-test", "uncertainty"],
        ),
        code(
            r"""
# 可修改交互 4：症状测量/录入错误压力测试。
FLIP_RATES = [0.00, 0.02, 0.05, 0.10, 0.15, 0.20]
N_SIMULATIONS = 500
STRESS_THRESHOLD = THRESHOLD

if not PACKET_D_AVAILABLE:
    display(Markdown("🔒 **Packet D 尚未释放。**"))
else:
    rng = np.random.default_rng(20260903)
    x_base = X_test.to_numpy(dtype=bool)
    y_base = y_test.to_numpy(dtype=bool)
    stress_rows = []
    for flip_rate in FLIP_RATES:
        masks = rng.random((N_SIMULATIONS, *x_base.shape)) < flip_rate
        stressed = np.logical_xor(x_base[None, :, :], masks).astype(int)
        flat = pd.DataFrame(stressed.reshape(-1, len(FEATURES)), columns=FEATURES)
        probability = model.predict_proba(flat)[:, 1].reshape(N_SIMULATIONS, len(y_base))
        prediction = probability >= STRESS_THRESHOLD
        truth = y_base[None, :]
        tp_sim = (prediction & truth).sum(axis=1)
        tn_sim = ((~prediction) & (~truth)).sum(axis=1)
        fp_sim = (prediction & (~truth)).sum(axis=1)
        fn_sim = ((~prediction) & truth).sum(axis=1)
        values = {
            "accuracy": (tp_sim + tn_sim) / len(y_base),
            "sensitivity": tp_sim / (tp_sim + fn_sim),
            "specificity": tn_sim / (tn_sim + fp_sim),
        }
        for metric_name, metric_values in values.items():
            stress_rows.append({
                "flip_rate": flip_rate,
                "metric": metric_name,
                "mean": metric_values.mean(),
                "q025": np.quantile(metric_values, 0.025),
                "q975": np.quantile(metric_values, 0.975),
            })
    stress_table = pd.DataFrame(stress_rows)
    display(stress_table.pivot(index="flip_rate", columns="metric", values="mean"))
    stress_table.to_csv(ASSET_DIR / "packet-d-measurement-error-stress.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    colors = {"accuracy": "#2563eb", "sensitivity": "#dc2626", "specificity": "#059669"}
    for metric_name, color in colors.items():
        subset = stress_table[stress_table["metric"] == metric_name]
        x_percent = subset["flip_rate"].to_numpy() * 100
        ax.plot(x_percent, subset["mean"], marker="o", linewidth=2, color=color, label=metric_name)
        ax.fill_between(x_percent, subset["q025"], subset["q975"], color=color, alpha=0.10)
    ax.set(xlabel="Independent symptom-entry flip rate (%)", ylabel="Metric value",
           ylim=(0.45, 1.03), title=f"Packet D: measurement-error stress test ({N_SIMULATIONS} simulations)")
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(ASSET_DIR / "packet-d-measurement-error-stress.png", dpi=180, bbox_inches="tight")
    plt.show()

print("This is a transparent synthetic stress test, not external clinical validation.")
""",
            tags=["packet-d", "interactive", "robustness"],
        ),
        code(
            r"""
if PACKET_D_AVAILABLE:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)

    # A: locked test confusion matrix
    axis = axes[0, 0]
    image = axis.imshow(cm_test, cmap="Blues", vmin=0)
    for row in range(2):
        for col in range(2):
            axis.text(col, row, cm_test[row, col], ha="center", va="center", fontsize=17, fontweight="bold")
    axis.set(xticks=[0,1], yticks=[0,1], xticklabels=["Pred no", "Pred yes"],
             yticklabels=["Actual no", "Actual yes"], title="A. Locked test looks perfect (n=24)")

    # B: baseline versus NB
    axis = axes[0, 1]
    names = ["accuracy", "sensitivity_recall", "specificity", "precision"]
    positions = np.arange(len(names)); width = 0.36
    baseline_values = [validation_metrics.iloc[0][name] for name in names]
    nb_values = [validation_metrics.iloc[1][name] for name in names]
    axis.bar(positions-width/2, baseline_values, width, color="#9aa4b2", label="prior baseline")
    axis.bar(positions+width/2, nb_values, width, color="#2563eb", label="naive Bayes")
    axis.set_xticks(positions, [name.replace("_recall", "") for name in names], rotation=18)
    axis.set_ylim(0, 1.08); axis.set_ylabel("Metric value"); axis.legend(frameon=False)
    axis.set_title("B. One metric cannot describe the task")

    # C: duplicate-feature shift
    axis = axes[1, 0]
    colors = np.where(duplicate_audit["decision_changed_at_0.5"], "#dc2626", "#2563eb")
    axis.scatter(duplicate_audit["original_probability"], duplicate_audit["duplicate_probability"],
                 c=colors, alpha=0.75, edgecolor="white", linewidth=0.3)
    axis.plot([0,1], [0,1], "--", color="#111827", linewidth=1)
    axis.axvline(0.5, color="#6b7280", linestyle=":"); axis.axhline(0.5, color="#6b7280", linestyle=":")
    axis.set(xlabel="Original P(disease)", ylabel="P(disease) after duplicate symptom",
             title="C. Redundant evidence can change decisions")

    # D: stress path
    axis = axes[1, 1]
    for metric_name, color in {"accuracy":"#2563eb", "sensitivity":"#dc2626", "specificity":"#059669"}.items():
        subset = stress_table[stress_table["metric"] == metric_name]
        axis.plot(subset["flip_rate"]*100, subset["mean"], marker="o", linewidth=2, color=color, label=metric_name)
    axis.set(xlabel="Symptom-entry flip rate (%)", ylabel="Mean metric", ylim=(0.65,1.03),
             title="D. Measurement error exposes brittleness")
    axis.legend(frameon=False)

    fig.suptitle("Packet D synthesis: why a 100% score is not enough", fontsize=16, fontweight="bold")
    fig.savefig(ASSET_DIR / "main-result-clean-vs-audit.png", dpi=180, bbox_inches="tight")
    plt.show()
""",
            tags=["packet-d", "main-result"],
        ),
        markdown(
            r"""
### Packet D 最终审计

1. 24/24 正确与“真实 sensitivity=100%”为什么不是同一句话？ _……_
2. Wilson 区间揭示了什么不确定性？ _……_
3. 压力测试可以支持什么结论？不能替代什么证据？ _……_
4. 条件独立明显不成立时，为什么模型仍可能分类正确？ _……_
5. 如果真实筛查人群患病率更低，precision 为什么可能下降？用 Bayes 定理解释。 _……_

### 当前决定

- [ ] 批准进入临床试点
- [ ] 仅批准继续教学/方法研究
- [ ] 要求真实外部队列和预先注册的评价方案
- [ ] 拒绝继续使用朴素贝叶斯

**你的选择及证据：** _……_
"""
        ),
        markdown(
            r"""
# 迁移作业｜同一数据，不同疾病标签

将目标改为 `nephritis`（肾盂肾炎），完成以下任务：

1. 在不改变 locked test 的前提下重新训练模型。
2. 报告 validation confusion matrix、sensitivity、specificity、precision 和 F1。
3. 根据漏诊与误报代价选择一个阈值，并说明依据。
4. 找出至少一组违反条件独立的特征。
5. 回答：**为什么医学诊断模型不能只看 accuracy？**

下面提供起点，不直接给出解释性答案。
"""
        ),
        code(
            r"""
# 迁移任务起点：请复制并修改，而不是直接覆盖主分析。
TRANSFER_TARGET = "nephritis"
y_transfer = data[TRANSFER_TARGET].copy()

# TODO 1: 使用相同 RANDOM_STATE 创建 train/validation/locked-test。
# TODO 2: 只在 train 上拟合 BernoulliNB。
# TODO 3: 在 validation 上比较阈值。
# TODO 4: 最后一次性查看 locked test，并报告边界。

print("Transfer target positive rate:", y_transfer.mean())
""",
            tags=["assignment", "transfer"],
        ),
        markdown(
            r"""
# CERB 模型审计 Memo（个人提交）

## Claim

我对“是否批准该模型”的最终主张是：_……_

## Evidence

至少引用三项不同类型证据：

1. 分类错误或混淆矩阵证据：_……_
2. 概率/阈值/校准证据：_……_
3. 假设、不确定性或稳健性证据：_……_

## Reasoning

这些证据为什么支持你的主张？错误代价如何影响选择？_……_

## Boundary

本数据和分析不能支持哪些结论？下一步需要什么数据或验证？_……_
"""
        ),
        markdown(
            r"""
# Section 2：证据综合与观点修订（45 分钟）

| 时间 | 活动 | 个人输出 |
|---:|---|---|
| 0–5 min | 重新投票 | 新选择与置信度 |
| 5–18 min | 比较最有力证据 | 一项支持、一项反证 |
| 18–30 min | 同伴质询 | 回答一个指标或假设问题 |
| 30–40 min | 教师综合 | 修订概念图 |
| 40–45 min | 修订 CERB 与 exit ticket | 最终边界陈述 |

### Re-vote

初始选择：___　最终选择：___　置信度变化：___  
真正改变我观点的证据是：_……_

### Peer challenge 记录

- 对方最强的反对意见：_……_
- 我的回应及证据：_……_
- 我仍不能回答的问题：_……_

### Exit ticket

1. 朴素贝叶斯“朴素”在哪里？ _……_
2. 训练、验证和测试最容易被误用的地方是什么？ _……_
3. 为什么 discrimination 和 calibration 不能相互替代？ _……_
4. 把本课推理迁移到基因表达分类时，最先检查什么？ _……_
"""
        ),
        markdown(
            r"""
# 工作词汇表

| English | 本课工作含义 |
|---|---|
| sample | 一行观测对象；本数据中是一个潜在患者样本 |
| feature | 用于预测的输入变量 |
| label/target | 希望预测的已知结果 |
| model | 用参数表示输入与输出关系的数学函数 |
| training | 从训练数据估计模型参数 |
| prediction | 用已固定的模型对新输入产生输出 |
| validation | 选择方案和阈值、发现问题 |
| locked test | 方案冻结后用于最终评价的数据 |
| generalization | 对未用于建模的新样本保持表现 |
| loss | 衡量单个预测与真实标签差距的函数 |
| cross-entropy / log loss | 常用于评价分类概率；自信的错误会受较重惩罚 |
| mean squared error | 回归中常用的平方误差损失 |
| sensitivity / recall | 真阳性中被模型找出的比例 |
| specificity | 真阴性中被模型排除的比例 |
| precision | 阳性预测中真正为阳性的比例 |
| ROC-AUC / PR-AUC | 跨阈值评价排序；PR-AUC 更聚焦阳性类别 |
| prior | 观察当前特征前的类别概率 |
| likelihood | 给定类别时看到这些特征的概率项 |
| posterior | 观察特征后的类别概率 |
| conditional independence | 给定类别后，各特征被模型视为相互独立 |
| calibration | 预测概率与实际发生频率的一致程度 |

## 数据来源

UCI Machine Learning Repository, **Acute Inflammations**, DOI: 10.24432/C5V59S，许可：CC BY 4.0。原始数据由医学专家为泌尿系统疾病专家系统测试而构造，并非真实临床队列。
"""
        ),
        code(
            r"""
import sklearn

session_info = {
    "python": platform.python_version(),
    "numpy": np.__version__,
    "pandas": pd.__version__,
    "scikit_learn": sklearn.__version__,
    "review_mode": REVIEW_MODE,
    "random_state": RANDOM_STATE,
    "dataset_exists": DATA_PATH.exists(),
    "generated_assets": sorted(path.name for path in ASSET_DIR.glob("packet-*")),
}
display(session_info)
print("Notebook execution complete. Interpretive answers remain for the student/reviewer.")
""",
            tags=["reproducibility"],
        ),
    ]

    notebook["cells"] = cells
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, NOTEBOOK_PATH)
    print(f"Created {NOTEBOOK_PATH}")
    print(f"Cells: {len(cells)}")


if __name__ == "__main__":
    prepare_dataset()
    build_notebook()
