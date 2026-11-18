from pathlib import Path
from docx import Document
from docx.shared import RGBColor

root = Path(__file__).resolve().parents[1]
sections = [
('任务四 主要结果参考答案', [
'分析计划与缺失处理：将 thalach（最大运动心率，单位次/分钟）作为主要终点，比较 disease=1 与 disease=0 两个独立组。定义均值差 Δ=μ₁−μ₀；H0：Δ=0；H1：Δ≠0。使用双侧 Welch t 检验，α=0.05。原始数据共920人，疾病组509人、无疾病组411人；排除 thalach 缺失的55人后，分析865人，疾病组474人、无疾病组391人。分别缺失35人和20人，不能把全体组人数写成检验样本量。',
'描述统计：疾病组 n=474，均值128.26，标准差24.02，中位数128；无疾病组 n=391，均值148.80，标准差23.61，中位数151，单位均为次/分钟。',
'方法依据：结局是连续变量，两组来自不同受试者。Welch t 检验比较总体均值，不要求两组方差相等。分析前应查看分布图和离群值，并检查记录独立性；较大的样本量支持均值的近似推断，但不能消除中心差异和选择偏倚。这里给出的结果是未调整的课堂主分析。',
'效应与不确定性：均值差（疾病组减无疾病组）为−20.54次/分钟，95%置信区间为[−23.73, −17.35]。该原始单位均值差就是本题的效应量。Welch t=−12.633，自由度约837.18，双侧 p≈1.30×10⁻³³；因此在α=0.05下拒绝H0。',
'结果解释：有疾病证据的受试者最大运动心率平均较低，样本支持两组总体均值存在差异。p值是在H0和检验假设成立时，得到当前或更极端检验统计量的概率，不是H0为真的概率。95%置信区间描述估计的不确定性；其频率学解释是重复抽样构造的区间约有95%覆盖真实参数。',
'实际意义与因果边界：约20.5次/分钟的差异值得进一步研究，但本题没有预先给出临床重要差异阈值，不能仅凭小p值认定具有临床价值。年龄、用药、运动能力和检测方案等可能影响结果。该数据不是随机治疗试验，不能说疾病导致心率降低、低心率导致疾病或某项治疗有效，也不能由组均值差证明个体诊断能力。'
]),
('任务五 风险与退出卡参考答案', [
'可能混杂一：年龄。年龄可能同时与疾病状态和最大运动心率有关。可预先指定年龄为协变量，在连续结局模型中调整；必要时检查非线性，或按预先规定的年龄层报告组间差异。不能仅根据某个协变量的p值决定是否调整。',
'可能混杂二：性别。性别可能与疾病构成、运动能力和心率分布有关。可调整sex或报告按sex分层的结果；如讨论效应是否因性别不同，交互作用分析应预先说明，且不能把两个分层p值一个显著、一个不显著直接当作组间效应不同。',
'中心与未测量因素：site可能反映转诊人群、检测流程和病例构成的不同。可比较各中心结果，并拟合 thalach ~ disease + age + sex + C(site)，将site作为分类变量。这里的回归用于比较调整前后的关联，不能自动消除未测量混杂。用药和运动能力若未提供，不能声称已控制；应列为限制，并在未来研究中收集。',
'敏感性分析计划：报告每组thalach缺失比例，比较有无缺失者的已观测特征；比较未调整、协变量调整及各中心的估计方向和区间。完整病例分析可能有选择偏倚，必要时在明确缺失机制假设后讨论多重插补。Mann–Whitney U可作为分布或秩关联的补充分析，但不应自动解释为均值差或中位数差检验。这里列出的是计划，未声称已得到调整模型或插补结果。',
'多重比较风险：若20个原假设均为真、检验彼此独立且每个α=0.05，至少出现一次假阳性的概率为1−0.95²⁰≈64.2%。指标相关时这一数值不再精确，但只挑显著结果仍会造成选择性报告和效应高估；它也不是某个显著结果为假的概率。',
'控制方案：在分析前固定thalach为主要终点、双侧方向、α及缺失处理；其余结果明确标为次要或探索性并完整报告。若20项共同作为确认性检验，可用Holm控制家族错误率，或用Bonferroni阈值0.05/20=0.0025。大规模探索可用Benjamini–Hochberg控制FDR，需说明其控制目标和适用假设。事后改写主要终点不能变成真正的预先指定，探索性发现应独立验证。',
'退出卡示例——主张：在本数据thalach完整的受试者中，疾病组的最大运动心率平均低于无疾病组，双侧Welch检验拒绝均值相同的H0。',
'退出卡示例——证据：两组样本量为474和391，均值分别为128.26和148.80次/分钟；均值差−20.54，95% CI为[−23.73, −17.35]，p≈1.30×10⁻³³。',
'退出卡示例——推理：区间未包含零且均值差为负，支持两组均值存在统计差异；效应量及区间展示差异幅度和估计精度。',
'退出卡示例——边界：这是未调整的观察性完整病例比较，年龄、性别、中心、用药及缺失可能影响估计，尚不能确立临床重要性、个体预测能力或因果作用。'
])]
md = root/'teacher-reference.md'
text = md.read_text(encoding='utf-8').split('## 结果解读与风险')[0].rstrip()
for title, paras in sections:
    text += '\n\n## '+title+'\n\n'+'\n\n'.join(paras)
md.write_text(text+'\n', encoding='utf-8')
path=root/'pbl-hypothesis-teacher-reference.docx'
doc=Document(path)
start=next(i for i,p in enumerate(doc.paragraphs) if p.text=='结果解读与风险')
for p in list(doc.paragraphs)[start:]:
    p._element.getparent().remove(p._element)
for title, paras in sections:
    heading=doc.add_heading(title,1)
    heading.paragraph_format.page_break_before=True
    for para in paras: doc.add_paragraph(para)
for style in ['Title','Heading 1','Heading 2']:
    doc.styles[style].font.color.rgb=RGBColor(0,0,0)
for t in doc.tables:
    for c in t.rows[0].cells:
        for p in c.paragraphs:
            for r in p.runs: r.font.color.rgb=RGBColor(255,255,255)
doc.save(path)
check=Document(path)
alltext='\n'.join(p.text for p in check.paragraphs)
assert '任务四 主要结果参考答案' in alltext and '任务五 风险与退出卡参考答案' in alltext
assert '\ufffd' not in alltext and '???' not in alltext
print('Updated Word and Markdown; Unicode checks passed.')
