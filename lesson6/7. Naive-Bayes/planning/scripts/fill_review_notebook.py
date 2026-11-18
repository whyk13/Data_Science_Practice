from pathlib import Path

import nbformat


COURSE_DIR = Path(__file__).resolve().parents[2]
NOTEBOOK_PATH = (
    COURSE_DIR
    / "notebook"
    / "pbl-machine-learning-basics-naive-bayes-lab.ipynb"
)


def md(text: str) -> str:
    return text.strip() + "\n"


notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)

notebook.cells[0].source = md(
    r"""
# 第 7 讲：机器学习基础与朴素贝叶斯
## Interactive PBL Worksheet + Reproducible Lab（教师/审阅者已填写稿）

> **用途边界**：本 Notebook 仅用于机器学习教学。数据来自专家规则构造的公开教学数据集，不能作为临床诊断、患者照护、模型部署或因果推断的证据。

> **分发提醒**：本版本已经填写全部示范答案，供课程设计审核使用，**不可直接作为学生空白 worksheet 发布**。空白版可由 `planning/scripts/build_review_notebook.py` 重新生成。

| 项目 | 内容 |
|---|---|
| 学习者 | 生命科学专业、即将进入大四的直博生 |
| 主数据 | UCI Acute Inflammations，120 个样本，6 个原始特征 |
| 主标签 | 是否存在膀胱炎症（59 阳性，61 阴性） |
| 核心决策 | 是否应仅凭“锁定测试集 accuracy = 100%”批准模型进入前瞻性筛查试点？ |
| 学习结构 | 问题触发 -> ML 基础复习 -> Packet A–D -> 迁移与综合 |
| 完成方式 | 学生版要求个人完成；本稿展示一种简洁、证据导向的参考作答 |

<details>
<summary><strong>教师/审阅者说明（点击展开）</strong></summary>

本稿设置 `REVIEW_MODE=True`，因此 Packet A–D 的代码和输出全部保留。示范答案不是唯一正确表达，重点是让每个结论都能对应证据，并明确不能外推的边界。

</details>
"""
)

notebook.cells[5].source = md(
    r"""
## STEP 1｜问题触发：100% 是否足够？（6 min）

某模型团队使用 6 项症状特征训练 Bernoulli 朴素贝叶斯，预测是否存在膀胱炎症。团队报告：

> “模型在锁定测试集上的 accuracy 是 100%，因此建议进入前瞻性筛查试点。”

### 初始决定（示范作答）

- [ ] A. 批准进入试点
- [x] B. 暂不批准，要求补充证据
- [ ] C. 拒绝该模型方案
- [ ] D. 当前信息不足，无法判断

**理由：** 100% 只是一个点估计，还不知道测试样本量、错误类型、概率校准和数据来源。进入临床试点前，至少要核查完整混淆矩阵、不确定区间和外部泛化证据。

**当前置信度：** 85%

**向模型团队追问：**

1. 测试集有多少人、阳性多少，并且是否真正只使用了一次？
2. 症状是否满足条件独立，预测概率是否校准？
3. 数据来自真实临床队列还是规则构造，是否做过外部验证？
"""
)

notebook.cells[7].source = md(
    r"""
## STEP 2｜机器学习基础快速复习：一次完整预测流程（14 min）

> 目标不是背术语，而是知道每个概念在研究流程中负责什么。

**本课工作流：** 数据输入 -> 定义 sample / feature / label -> 划分数据 -> 用 train 拟合模型 -> 用 validation 选方案与阈值 -> 冻结方案 -> 用 locked test 评价 -> 说明泛化边界。

### 2.1 数据模型：从输入到输出

一个监督学习模型可写成 $f_\theta:X\rightarrow\hat{Y}$。它用参数 $\theta$ 描述输入与输出之间的数学关系；分类模型常先输出 $P(Y=1\mid X=x)$，再用阈值转成类别。

| 概念 | 工作含义 | 本案例中的对应物（示范作答） |
|---|---|---|
| sample | 一个独立观察对象，即数据表的一行 | 一位潜在患者的一行记录 |
| feature $X$ | 提供给模型的输入变量 | 体温和五项症状 |
| label $Y$ | 希望模型预测的已知结果 | 是否存在膀胱炎症 |
| model $f_\theta$ | 从特征映射到分数、概率或类别的函数 | 症状到患病后验概率的映射 |
| training | 用训练数据估计参数 $\theta$ | 估计两类先验和各症状的条件概率 |
| prediction | 固定参数后，对新样本产生输出 | 输出患病概率及阈值后的类别 |
| generalization | 模型在未参与建模的新样本上仍有效 | 在目标人群的新病例上保持表现 |

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

### 三个快速判断（示范作答）

1. 降低阳性阈值通常会使 **sensitivity 上升、specificity 下降**。  
2. Validation 承担选择作用，因此可以比较候选方案；test 若参与调整，就不再是对冻结方案的独立评价。  
3. 避免漏诊时优先看 **sensitivity**，但仍应同时报告 specificity、precision、阈值、校准和不确定性。
"""
)

notebook.cells[8].source = md(
    r"""
## STEP 3｜把案例变成预测问题：3W 与竞争解释（12 min）

### 统计检验还是预测？（示范作答）

1. 腰痛与膀胱炎症是否有关联？**统计推断。**  
2. 根据一个新就诊者的症状判断其是否可能患病？**预测。**  
3. 前者关注总体关联及其不确定性；后者关注未见样本上的预测误差。使用相同变量不代表回答相同问题。

### 3W（示范作答）

| What we know | What we need to know | What we will do |
|---|---|---|
| 团队声称 test accuracy=100% | 测试集大小、混淆矩阵、阈值和置信区间 | 查看 Packet B/D 并手算指标 |
| 数据包含症状和疾病标签 | 数据来源、标签生成方式、类别比例和特征依赖 | 查 UCI 元数据并做分布/相关审计 |
| 医学应用中漏诊和误报代价不同 | 哪个指标应优先、概率是否校准、是否抗测量误差 | 比较阈值、Brier/log loss 和压力测试 |

### 竞争解释（示范作答）

| 解释/决策 | 如果它成立，应看到什么证据？ | 什么证据会推翻它？ | 错误选择的后果 |
|---|---|---|---|
| H1：模型学到了可泛化的症状模式 | 外部数据仍有稳定 sensitivity、specificity 和校准 | 外部表现显著下降 | 过早否定一个有用基线 |
| H2：数据过度干净或由规则构造 | 几乎完美分离、无缺失、症状组合高度规则化 | 真实队列中仍保持相同表现 | 把教学规律误当临床规律 |
| H3：测试集太小，100% 具有偶然性 | 指标区间很宽、不同抽样结果波动 | 大型预注册测试仍精确接近 100% | 夸大证据强度 |
| H4：相关症状被重复计作独立证据 | 重复一个症状就明显改变 posterior 或类别 | 删除重复项后概率完全稳定 | 产生过度自信和错误阈值决策 |
"""
)

notebook.cells[9].source = md(
    r"""
## STEP 4｜参考阅读、个人学习契约与 Evidence Card（13 min，含 Exit ticket）

### 本 Notebook 内的参考阅读路线

1. Bayes theorem：prior -> likelihood -> posterior。
2. Bernoulli NB：二元特征、条件独立、Laplace smoothing。
3. 混淆矩阵：错误类型及其代价。
4. 三类评价：分类决策、排序/区分能力、概率可信度。
5. 泛化边界：小样本、数据来源、测量误差和分布变化。

### 个人学习契约（示范作答）

- **要解决的问题：** 为什么 100% accuracy 仍不足；条件依赖如何影响 posterior。
- **检查的来源：** UCI 数据说明；scikit-learn 的 Naive Bayes 与 metrics 文档。
- **完成判据：** 能手算一次 posterior、从混淆矩阵计算指标，并写出不能部署的理由。
- **AI 核验方法：** 运行代码复算数值，再与官方定义和原始数据说明对照。

### Evidence Card（示范作答）

| 字段 | 个人记录 |
|---|---|
| 来源/工具 | AI 解释 + UCI 元数据 + scikit-learn 官方定义 |
| 原始主张 | “医学分类模型应优先看 sensitivity，而不是 accuracy。” |
| 支持证据 | prior-only baseline 的 specificity=1 但 sensitivity=0；阈值会改变漏诊数 |
| 反证/限制 | 指标优先级取决于用途和错误代价；还必须报告 specificity、precision、校准和区间 |
| 最终可信度 | 中等：方向合理，但不能把 sensitivity 当成唯一指标 |

### Section 1 Exit ticket（示范作答）

最缺少的是**真实外部队列上的预注册评价**。若外部 sensitivity、specificity 和校准仍稳定，我会考虑受控研究；若明显下降，则拒绝临床试点。
"""
)

notebook.cells[13].source = md(
    r"""
### Packet A 审计记录（示范作答）

- **可以得出：** 数据有 120 行、6 个原始特征、无缺失；主标签接近平衡，可以定义一个二分类教学任务。
- **不能得出：** 数据是否代表真实患者、模型是否校准、是否能在其他医院或人群泛化。
- **是否足以支持临床试点：** 不足。需要更大且独立的前瞻性外部队列、可靠标签和预先规定的阈值与评价指标。
- **“整齐、无缺失”的可能解释：** 数据经过强筛选，或本来就是专家规则构造的教学数据，缺少真实世界测量误差。
"""
)

notebook.cells[18].source = md(
    r"""
### 后验手算任务（示范作答）

对下一单元给定的症状组合，使用训练集先验和六个条件概率连乘：

\[
score_0\approx 0.000742,\qquad score_1\approx 0.000446
\]

归一化后：

\[
P(Y=1\mid x)=\frac{0.000446}{0.000742+0.000446}\approx 0.3755
\]

因此我预计模型输出患病概率约 **0.38**；在阈值 0.50 下预测为阴性。重点不是小数本身，而是每个症状项都被按“给定类别后独立”相乘。
"""
)

notebook.cells[20].source = md(
    r"""
> ### CHECKPOINT 2（示范作答）
> 模型学习的是训练集中两类的先验概率，以及每个二元症状在各类别中的条件概率。Laplace smoothing 给未出现的组合保留小的非零概率，避免一个零项把整类 posterior 直接乘成零。
"""
)

notebook.cells[23].source = md(
    r"""
### 从混淆矩阵到指标（示范作答）

\[
Sensitivity=\frac{TP}{TP+FN},\quad Specificity=\frac{TN}{TN+FP}
\]

\[
Precision=\frac{TP}{TP+FP},\quad Accuracy=\frac{TP+TN}{N}
\]

1. Validation 混淆矩阵：**TP=12、TN=12、FP=0、FN=0**。  
2. Sensitivity (=12/(12+0)=1)；specificity (=12/(12+0)=1)。  
3. Prior-only baseline 把所有样本判为阴性，所以 specificity=1，却漏掉全部阳性，sensitivity=0，不能用于筛查。  
4. Sensitivity 问“真实患者中找出多少”；precision 问“判为阳性的人中多少确实阳性”。
"""
)

notebook.cells[25].source = md(
    r"""
### Packet B 审计记录（示范作答）

- **漏诊代价更高时：** 通常降低阈值，优先减少 FN、提高 sensitivity；代价是 FP 增多，specificity 和 precision 可能下降。
- **ROC-AUC/PR-AUC 是否足够：** 不够。它们不说明实际阈值、概率校准、置信区间和目标人群中的错误代价。
- **为何不能报告 test：** Validation 已用于看结果和选方案；继续用 test 调整模型会让 test 也变成 validation，造成乐观偏差。

### AI Evidence Card（示范核验）

AI 的说法只在“漏诊代价显著更高”的筛查情境下方向合理。它若把 sensitivity 当作所有医学任务的唯一指标，就是过度概括。应根据混淆矩阵、阈值曲线和指标定义，修正为：**预先规定主要指标，同时报告 sensitivity、specificity、precision、校准和不确定性。**
"""
)

notebook.cells[28].source = md(
    r"""
### 先解释，再下结论（示范作答）

- 类别内相关系数为 1 表明两个症状在该类别中完全同步，明显不是独立证据。
- `NaN` 不等于“无相关”；它通常表示某个特征在该类别内是常数、方差为零，因此相关系数无法定义。
- 条件独立不成立不代表 NB 必然分类失败；即使假设不真实，决策边界仍可能恰好有效。
- 分类仍正确时，posterior 也可能过度自信、校准不良，甚至因重复录入一个症状而跨过阈值。
"""
)

notebook.cells[31].source = md(
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

### Packet C 决策矩阵（示范作答）

| 可选行动 | 理由/假设 | 期待证据 | 验证方式 | 停止条件或副作用 |
|---|---|---|---|---|
| 保留所有特征 | 简单模型可能仍有较好分类能力 | 指标和概率在新数据中稳定 | 外部测试、校准和误差分析 | 若依赖单一症状或概率失真则停止 |
| 删除重复/高度相关症状 | 减少重复计算同一证据 | posterior 更稳定且性能不明显下降 | 特征消融和重复验证 | 可能删掉真正有用的信号 |
| 保留模型但不解释原始 posterior | 分类可能可用，但概率含义可疑 | 阈值决策稳定 | 比较 Brier/log loss 与校准 | 需要绝对风险时不可采用 |
| 收集外部数据后再决定 | 内部规则数据不足以支持应用 | 外部指标、区间和校准可接受 | 预注册前瞻性外部验证 | 成本较高，但临床试点前必要 |
"""
)

notebook.cells[32].source = md(
    r"""
> ### CHECKPOINT 3（示范作答）
> “表现好”的证据是 validation 上所有样本分类正确，且 0.25–0.50 阈值范围内结果稳定；“可信”仍缺真实外部队列、可靠校准、对条件依赖的处理，以及更精确的不确定区间。
"""
)

notebook.cells[38].source = md(
    r"""
### Packet D 最终审计（示范作答）

1. 24/24 正确只是这个小样本的点估计；它不证明总体中的真实 sensitivity=100%，也不解决数据代表性问题。
2. 95% Wilson 区间显示：accuracy 下限约 0.862，sensitivity 和 specificity 下限约 0.758，说明“零错误”仍有明显不确定性。
3. 压力测试说明模型对症状误录可能较脆弱；它不能替代真实外部验证，也不能告诉我们真实误差率是多少。
4. NB 即使假设近似错误，也可能在规则化数据上正确分类；受影响更明显的常是 posterior 的大小和校准。
5. \(PPV=\frac{Sensitivity\times Prevalence}{Sensitivity\times Prevalence+(1-Specificity)(1-Prevalence)}\)。患病率下降时，假阳性更容易在阳性预测中占较大比例，因此 precision 下降。

### 当前决定（示范作答）

- [ ] 批准进入临床试点
- [ ] 仅批准继续教学/方法研究
- [x] 要求真实外部队列和预先注册的评价方案
- [ ] 拒绝继续使用朴素贝叶斯

**依据：** 内部测试虽为 24/24，但区间仍宽；数据由专家规则构造；条件依赖和模拟测量误差都会影响可信度。模型可以作为教学或研究基线，不能据此进入临床试点。
"""
)

notebook.cells[40].source = r'''# 迁移任务示范：使用同样流程预测 nephritis，不覆盖主分析。
TRANSFER_TARGET = "nephritis"
y_transfer = data[TRANSFER_TARGET].copy()

Xt_train_validation, Xt_test, yt_train_validation, yt_test = train_test_split(
    X, y_transfer, test_size=0.20, random_state=RANDOM_STATE, stratify=y_transfer
)
Xt_train, Xt_validation, yt_train, yt_validation = train_test_split(
    Xt_train_validation,
    yt_train_validation,
    test_size=0.25,
    random_state=RANDOM_STATE,
    stratify=yt_train_validation,
)

transfer_model = BernoulliNB(alpha=ALPHA).fit(Xt_train, yt_train)
transfer_validation_probability = transfer_model.predict_proba(Xt_validation)[:, 1]

transfer_thresholds = pd.DataFrame([
    metric_row(yt_validation, transfer_validation_probability, threshold=t)
    for t in [0.10, 0.25, 0.50, 0.75, 0.90]
])
display(transfer_thresholds)

# Validation 在 0.50 阈值已无错误，因此保留预设阈值，不根据 test 调参。
TRANSFER_THRESHOLD = 0.50
transfer_test_probability = transfer_model.predict_proba(Xt_test)[:, 1]
transfer_test_metrics = pd.DataFrame([
    metric_row(yt_test, transfer_test_probability, threshold=TRANSFER_THRESHOLD)
])
display(transfer_test_metrics)
display(pd.DataFrame(
    confusion_matrix(yt_test, transfer_test_probability >= TRANSFER_THRESHOLD),
    index=["actual no", "actual yes"],
    columns=["predicted no", "predicted yes"],
))

transfer_thresholds.to_csv(ASSET_DIR / "transfer-nephritis-validation-thresholds.csv", index=False)
transfer_test_metrics.to_csv(ASSET_DIR / "transfer-nephritis-locked-test-metrics.csv", index=False)

print(f"Transfer target positive rate: {y_transfer.mean():.3f}")
print("Boundary: this is still the same small expert-rule dataset; perfect internal metrics do not establish clinical utility.")
'''

notebook.cells[41].source = md(
    r"""
# CERB 模型审计 Memo（示范作答）

## Claim

不批准该模型进入临床试点；可以保留为教学或方法研究基线，并在预注册的真实外部验证后重新评估。

## Evidence

1. **分类证据：** 锁定测试为 24/24，但只有 12 个阳性和 12 个阴性。
2. **概率/假设证据：** 重复录入“恶心”使 64 种可能症状组合中 5 种跨过 0.5 阈值，最大概率变化约 0.221。
3. **不确定性/稳健性证据：** sensitivity 和 specificity 的 Wilson 下限约 0.758；10% 症状翻转时平均 sensitivity 约 0.897。

## Reasoning

高内部准确率更可能说明模型复现了规则化数据中的模式，而不是证明真实临床泛化。筛查尤其需要控制漏诊，因此不能用小样本中的零错误代替外部 sensitivity、校准和错误代价评估。

## Boundary

数据只有 120 个专家构造样本，没有医院、人口学、真实诊断过程、缺失和测量噪声。下一步应使用独立多中心前瞻性数据，预先规定标签、阈值、主要指标、缺失处理和亚组分析。
"""
)

notebook.cells[42].source = md(
    r"""
# Section 2：证据综合与观点修订（45 分钟）

| 时间 | 活动 | 个人输出 |
|---:|---|---|
| 0–5 min | 重新投票 | 新选择与置信度 |
| 5–18 min | 比较最有力证据 | 一项支持、一项反证 |
| 18–30 min | 同伴质询 | 回答一个指标或假设问题 |
| 30–40 min | 教师综合 | 修订概念图 |
| 40–45 min | 修订 CERB 与 exit ticket | 最终边界陈述 |

### Re-vote（示范作答）

初始选择：**暂不批准、补充证据**；最终选择：**要求真实外部验证**；置信度：**85% -> 95%**。  
真正改变我观点的是数据由专家规则构造，以及小测试集区间、重复特征和误差压力测试共同揭示的边界。

### Peer challenge（示范作答）

- **最强反对意见：** 模型简单、可解释且内部测试完全正确，低成本试点可能值得尝试。
- **回应：** 内部样本共享同一构造规则，24 例无法代表临床风险；患者参与前应先完成独立、预注册的外部研究。
- **仍不能回答：** 真实人群中的症状误报率、标签误差、患病率和跨医院分布变化有多大。

### Exit ticket（示范作答）

1. **“朴素”在哪里？** 给定类别后，把各特征近似视为条件独立。
2. **数据划分最易误用之处？** 用 validation/test 反复调参，或在划分前使用全数据做特征处理。
3. **为何 discrimination 与 calibration 不能替代？** 前者评价排序，后者评价概率数值是否可信；排序正确不保证 0.8 真代表约 80%。
4. **迁移到基因表达分类先检查什么？** 先按患者划分数据，再在训练集内做标准化和特征选择，并检查高维、小样本、批次效应和泄漏。
"""
)

notebook.metadata.setdefault("pbl", {})["review_answers_filled"] = True
notebook.metadata["pbl"]["student_distribution_ready"] = False
nbformat.write(notebook, NOTEBOOK_PATH)

print(f"Filled reviewer answers in {NOTEBOOK_PATH}")
