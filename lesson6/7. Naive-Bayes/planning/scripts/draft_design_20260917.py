"""Approval-stage design only. Does not author final course documents or assets."""
from pathlib import Path
import json
import importlib.util

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'planning/review-20260917'
OUT.mkdir(exist_ok=True)
objectives = [
('LO1','什么是数据模型？；用数学函数描述输入和输出之间的关系','为六症状输入、类别概率、阈值输出分别写出数学对象，并说明关联检验与新记录预测的区别。','Q1'),
('LO2','数据输入；特征；标签','从八列数据中选出六个输入和主标签，说明第二个疾病标签为何不自动成为可用输入。','Q1'),
('LO3','什么是训练、预测和泛化？','给训练计数、固定参数预测、验证选择及最终测试排序，并定义每一步可读的数据。','Q2'),
('LO4','训练集；验证集；测试集；模型训练；性能评估','用72/24/24的划分表指定拟合、阈值选择和评价职责，区分行编号隔离与新输入组合。','Q2'),
('LO5','什么是损失函数？；模型预测与真实标签之间的差距','对四条相同标签、两组概率计算逐样本损失和平均损失，解释相同accuracy为何不足以比较概率。','Q4'),
('LO6','常见损失：cross-entropy, mean squared error','计算二分类cross-entropy与连续体温预测的MSE，写出两者预测对象和单位差异。','Q4'),
('LO7','accuracy；sensitivity；specificity；precision；recall；F1-score','从非完美混淆矩阵算出六个命名指标，辨别recall与sensitivity同义、precision分母不同。','Q5'),
('LO8','ROC-AUC；PR-AUC','在给定ROC/PR图上定位两阈值，解释曲线面积不能给出阈值或概率校准；识别梯形PR面积与AP的命名。','Q5'),
('LO9','calibration','比较小样本分箱的平均预测概率与实际阳性频率，结合每箱样本数写出概率解释边界。','Q5'),
('LO10','基于 Bayes theorem','从训练计数估计先验与条件概率，包含症状为0的项，并归一化两个类别分数得到后验。','Q3'),
('LO11','“朴素”假设：特征之间条件独立','在疾病类别内比较联合频率与边缘频率乘积，并解释人工复制症状后概率变化。','Q6'),
('LO12','适合文本分类、简单医学分类任务','对六症状和生物医学摘要关键词是否出现两种输入说明Bernoulli表示适配；区分出现指示与次数。','Q3'),
('LO13','简单、快速、可解释；条件独立假设常常不完全成立','用训练计数和重复症状证据各解释一项NB优点和一项局限，避免把可追溯计算说成因果解释。','Q6'),
('LO14','课堂练习用一个小型症状数据集训练朴素贝叶斯模型：输入：症状特征；输出：是否患病；计算 sensitivity 和 specificity','在Phase 3课堂用本数据的训练计数拟合二症状NB，计算新组合预测，并由给定逐例结果汇总sensitivity和specificity。','Q3'),
('LO15','作业 解释为什么医学诊断模型不能只看 accuracy。','利用Q1—Q6形成一份含任务、训练、损失、指标、假设、边界的研究方案；用明确标为假设的1%患病率场景作数值反例。','Q6'),
]
questions = [
('Q1','表 A1 的六症状如何成为模型输入，两个疾病标签中哪一个是本次要预测的结果？','A',12,['表 A1','表 A2'],'任务与输入输出','变量、数据模型、统计检验与预测','给出X、Y、f的案例映射及两个研究问题的比较。','界定预测对象、可用输入及结论对象。'),
('Q2','表 A3 的72/24/24条记录应怎样分配拟合、阈值选择与最终评价职责？','A',14,['表 A3'],'训练与评价流程','训练、预测、泛化、三种数据集','排序五张流程卡并给每步分配数据；指出一次看test后改阈值的错误。','规定可复现流程与测试解封条件。'),
('Q3','用表 B1 的训练计数，怎样得到表 B2 指定症状组合的膀胱炎症预测？','B',24,['表 B1','表 B2','表 B3'],'朴素贝叶斯模型','Bayes、先验、似然、后验、条件独立、二元特征、加一平滑','计算两个缺失参数和两个类别分数，归一化后验；给二元关键词迁移一条适配理由。','解释模型从什么数据学到什么及使用假设。'),
('Q4','表 B4 中两模型判对同样多时，概率损失为何不同；表 B5 的体温误差该如何计量？','B',16,['表 B4','表 B5'],'损失与预测对象','cross-entropy、MSE、单例损失与平均损失、训练目标与评价','比较两组四样本概率损失，手算一项log loss和两个体温误差的MSE。','选择适当的概率评价量并说明NB参数估计方式。'),
('Q5','表 C1 的阈值变化、图 C2 的排序和表 C3 的概率频率，分别允许对这个模型说什么？','C',28,['表 C1','图 C2','表 C3'],'指标、阈值与概率','accuracy、sensitivity/recall、specificity、precision、F1、ROC-AUC、PR-AUC、calibration','用0.10阈值计算混淆矩阵指标，比较0.50/0.90阈值；读ROC/PR图和分箱计数。','提出有依据的阈值和分类、排序、概率三类报告方案。'),
('Q6','表 D1—D3 的类别内依赖、重复证据和测试组合覆盖，如何改变你交给研究组的方案？','D',26,['表 D1','表 D2','表 D3','表 D4'],'假设、泛化与最终用途','条件独立、简单快速可解释、泛化边界、患病率与precision','核对一项联合概率；解释重复症状前后概率；用10000人假设场景计算阳性预测组成；综合六模块形成250—400字方案。','从追踪表全部模块作最终方案选择，完成原大纲accuracy作业。'),
]
evidence_specs = [
('表 A1','A','原始症状记录与变量说明；每行一个专家设计的潜在病例描述，120×8的来源表抽取6条；六输入、两标签；体温℃；各yes/no列非缺失n=120。','planned-a1-records.csv','Q1'),
('表 A2','A','来源审计；120条、59/61标签、0缺失；21条全行重复余量；编码UTF-16对比文档ASCII；不将一个记录等同独立患者。','planned-a2-provenance.csv','Q1'),
('表 A3','A','固定random_state=7分层划分；每行一集合：72/24/24，阳性35/12/12；行编号交集0，暂不报告症状组合交集。','planned-a3-split.csv','Q2'),
('表 B1','B','每行一个症状；训练72条中35阳性、37阴性的症状阳性计数；体温按预设38℃二值化；用于参数估计，保留少数计算格。','planned-b1-counts.csv','Q3'),
('表 B2','B','给定两条明确的六症状二元向量；列明0/1、两类先验与参考乘积；手算目标后验留作答题，参考核对值在教师材料。','planned-b2-queries.csv','Q3'),
('表 B3','B','Phase 3课堂数据：同源训练72条的二症状计数和独立验证24条的二症状逐例聚合；每组含真实标签、人数、症状；学生拟合和计数。','planned-b3-classroom.csv','Q3'),
('表 B4','B','明确标为教学合成的四条标签与两组概率；阈值0.5，全部输入给出；不预写哪一模型更好。','planned-b4-loss.csv','Q4'),
('表 B5','B','教学合成连续结局示例；实测体温[37,39]℃，预测[38,37]℃；要求算平方误差均值，单位℃²。','planned-b5-mse.csv','Q4'),
('表 C1','C','验证24条，每行一阈值；0.10/0.50/0.90的TP/FP/FN/TN及先验基线；无阳性预测时precision显示未定义，另说明软件约定。','planned-c1-threshold.csv','Q5'),
('图 C2','C','同一24条验证记录的ROC和PR曲线；标出阈值点；梯形PR面积和AP分别标注；附全输入的小合成排序对照，非新增临床样本。','planned-c2-ranking.png','Q5'),
('表 C3','C','验证24条概率的分箱；每行一个非空箱，给边界、n、阳性数、平均预测概率与观测比例；不凭小样本画平滑线宣称校准。','planned-c3-calibration.csv','Q5'),
('表 D1','D','仅训练72条的类别内双症状2×2计数；要求学生比较P(A,B|Y)和P(A|Y)P(B|Y)，不先给判断。','planned-d1-dependence.csv','Q6'),
('表 D2','D','同一训练72条，原六特征模型与人为复制恶心的七列模型；穷举64种二元向量不冒充64患者；列三条具体向量及概率。','planned-d2-redundancy.csv','Q6'),
('表 D3','D','固定教学测试24条：TN12、FP0、FN0、TP12；测试8种二值组合，24/24条的组合在训练出现；原始输入相同的6/24条。','planned-d3-test-and-overlap.csv','Q6'),
('表 D4','D','纯假设筛查10000人、患病率1%、sensitivity/specifity均95%；给输入不预填precision，学生计算四格。','planned-d4-transfer.csv','Q6'),
]
objlist=[dict(id=i,source_excerpt=s,observable_outcome=o) for i,s,o,q in objectives]
qs=[]
clusters=[]
analyses=[]
for i,(qid,question,packet,mins,eids,row,learning,action,contrib) in enumerate(questions,1):
    ids=[oid for oid,_,_,q in objectives if q==qid]
    cid=f'C{i}'
    clusters.append(dict(id=cid,name=row,objective_ids=ids,concepts=learning.split('、')))
    qs.append(dict(id=qid,question=question,objective_ids=ids,concept_ids=[cid],required_learning=learning.split('、'),student_action=action,evidence_ids=eids,release_after=packet,minutes=mins,tracker_row=f'第{i+1}行：{row}',student_output=action,contribution_to_synthesis=contrib))
    analyses.append(dict(id=f'AN{i}',objective_ids=ids,inquiry_question_ids=[qid],question=question,domain='model_behavior' if i in [3,4,5] else 'validation_robustness' if i in [2,6] else 'data_structure',method=action,applicability=contrib,artifacts=[f for _,_,_,f,q in evidence_specs if q==qid],packet=packet,interpretation_boundary='教学内部证据；无临床、因果或外部效度声明。',notebook_ref=qid))
coverage=[]
for oid,source,outcome,qid in objectives:
    q=next(q for q in qs if q['id']==qid);idx=qs.index(q)
    coverage.append(dict(objective_id=oid,source_excerpt=source,observable_outcome=outcome,concept_cluster_ids=q['concept_ids'],inquiry_question_ids=[qid],analysis_ids=[f'AN{idx+1}'],required_learning=q['required_learning'],student_action=q['student_action'],analysis_method=analyses[idx]['method'],evidence_ids=q['evidence_ids'],packet=q['release_after'],student_output=outcome,worksheet_location='Phase 3 / 课堂必做训练与指标练习' if oid=='LO14' else f'Phase 2 / {qid}',synthesis_use=q['contribution_to_synthesis'],assessment_evidence=outcome,applicability='第7讲大纲明确要求',interpretation_boundary='只针对提供的来源和数据，不外推因果或临床效度。'))
options=['A：把六症状NB和0.50阈值作为下一轮方法比较的冻结基线；不新增病例特征或先验校正。','B：先修改症状表示、模型设置或验证阈值中的至少一项，再冻结新基线；须逐项给出修改依据。','C：暂缓提交任何模型，先补齐一项会改变A/B选择的数据或评价证据。']
problem='研究组要把六症状到膀胱炎症标签的模型用于下一轮离线方法比较；你需提交完整、可复核的训练与评价方案，并在A保留冻结基线、B修改后冻结、C证据不足暂缓中作决定，说明哪些数据来源、假设或评价发现会使方案停止或改变。'
design=dict(status='approval draft, not released',study_goal='建立生命科学研究所需的预测建模共同语言，并用可计算的朴素贝叶斯案例整合模型、训练、损失、评价与假设。',learning_objectives=objlist,concept_clusters=clusters,decision_problem=problem,driving_question=problem,entry_type='direct_problem',problem_architecture=dict(overarching_problem=problem,downstream_use='由研究组用于下一轮离线方法比较和研究方案评审；不进入患者照护。',decision_options=options,inquiry_questions=qs,stop_conditions=['不能说明训练参数来自哪些记录时，停止报告测试性能并重建流程。','只有行编号隔离而输入组合全部见过时，撤回新症状组合泛化声明。','若将相关或重复输入当新证据，暂停个体风险解释并重新核验概率。','若阈值只按test选择，停止独立测试声明并另收未参与选择的数据。']),course_coverage=coverage,data_context=dict(dataset_name='UCI Acute Inflammations',source='https://doi.org/10.24432/C5V59S',license='CC BY 4.0',task='教学二分类',observation_unit='医学专家为测试专家系统设计的潜在病例症状与决策记录；无法从元数据确认是独立患者抽样。',sample_count=120,feature_count=6,missing_cells=0,variable_roles=dict(features=['temperature','nausea','lumbar pain','urine pushing','micturition pains','burning urethra'],labels=['bladder inflammation','nephritis']),target=dict(name='bladder inflammation',positive=59,negative=61),grouping_fields=[]),analyses=analyses,evidence_items=[dict(id=i,packet=p,context=c,artifacts=[f]) for i,p,c,f,q in evidence_specs],split_plan=dict(required=True,unit='row',rationale='示范训练、验证、测试职责；不宣称行隔离等于组合或患者独立。',train=72,validation=24,test=24,random_state=7,author_test_exposure='作者在前次探索中已见测试结果；当前固定方案是课堂模拟冻结，不能称新盲测。'),packets={p:dict(purpose=t,reveals=r,withholds=w) for p,t,r,w in [('A','定义任务与工作流',['变量、来源、原始重复、划分'],['参数、性能、条件依赖、test']),('B','训练及损失',['训练计数、查询症状、两种损失'],['验证性能、条件依赖、test']),('C','多维评价',['阈值、ROC/PR、校准箱'],['条件依赖和最终test']),('D','假设及综合',['联合频率、重复特征、测试组合覆盖、低患病率迁移'],[])]},syllabus_tasks=[dict(source_excerpt=objectives[13][1],worksheet_location='Phase 3前15分钟课堂必做；数据表 B3；独立纸笔训练二症状NB及计算sensitivity/specificity；可用辅助notebook核对。'),dict(source_excerpt=objectives[14][1],worksheet_location='Q6最终综合：明确回答原题并引用表 C1、表 D4与Q1—Q6追踪记录。')],boundaries=['教学用途，不用于临床诊断或患者照护','不作因果推断','不把64种穷举向量计作64患者','不把规则化教学样本的Wilson区间解释成真实患者总体性能保证','分箱校准的小样本噪声不能凭单图量化总体误差'],workload=dict(phase1_minutes=45,phase2_minutes=120,phase3_minutes=45,phase2_includes='阅读、计算、记录、证据卡、250—400字综合和accuracy作业；不另加必做迁移。',optional_extensions=['Wilson区间推导','测量误差模拟','按症状组合分组验证','肾盂肾炎标签重训','Gaussian NB和其他模型比较']))
(OUT/'analysis-design-draft.json').write_text(json.dumps(design,ensure_ascii=False,indent=2),encoding='utf-8')
spec=importlib.util.spec_from_file_location('qa',ROOT.parents[2]/'.codex/skills/pbl-ml-package/scripts/worksheet_check.py')
qa=importlib.util.module_from_spec(spec);spec.loader.exec_module(qa)
report,bad=qa.design(OUT/'analysis-design-draft.json',120)
(OUT/'draft-design-check.md').write_text(report,encoding='utf-8')
print(report)
assert not bad
lines=['# 第7讲：课程覆盖链（审批草案）','','本文件对应新版设计草案，不是正式课件。每项都需要学生操作与答题；仅出现在定义或教师总结不计覆盖。','']
for c in coverage:
    lines += [f"## {c['objective_id']}｜{c['source_excerpt']}",'',f"- 可观察结果：{c['observable_outcome']}",f"- 模块/概念：{', '.join(c['inquiry_question_ids'])}；{'、'.join(c['required_learning'])}",f"- 学习与分析：{c['student_action']}",f"- 证据：{'、'.join(c['evidence_ids'])}；Packet {c['packet']}",f"- 个体答题位置：{c['worksheet_location']}",f"- 最终综合用途：{c['synthesis_use']}",f"- 考核凭据：{c['assessment_evidence']}",f"- 边界：{c['interpretation_boundary']}",'']
(OUT/'course-coverage-draft.md').write_text('\n'.join(lines),encoding='utf-8')
