"""Course-specific Lesson 7 PBL package, approved 2026-09-17.

Numerics: D:/Python/python.exe scripts/lesson7_analysis.py
Documents: bundled runtime Python this_file.py --docs
All reads/writes for the generated package are local to this lesson.
The skill supplies a checker, never a content generator.
"""
from pathlib import Path
import argparse, json, shutil, sys, subprocess, hashlib
ROOT=Path(__file__).resolve().parents[1]
SLUG='pbl-machine-learning-basics-naive-bayes'
ASSET=ROOT/'assets'/SLUG
REVIEW=ROOT/'review'
TITLE='第7讲 机器学习基础与朴素贝叶斯'

def load_evidence():
    return json.loads((ASSET/'computed-evidence.json').read_text(encoding='utf-8'))

def prepare_design():
    draft=json.loads((ROOT/'planning/review-20260917/analysis-design-draft.json').read_text(encoding='utf-8'))
    names={'planned-a1-records.csv':'packet-a1.csv','planned-a2-provenance.csv':'packet-a2.csv','planned-a3-split.csv':'packet-a3.csv','planned-b1-counts.csv':'packet-b1.csv','planned-b2-queries.csv':'packet-b2.csv','planned-b3-classroom.csv':'packet-b3.csv','planned-b4-loss.csv':'packet-b4.csv','planned-b5-mse.csv':'packet-b5.csv','planned-c1-threshold.csv':'packet-c1.csv','planned-c2-ranking.png':'packet-c2-ranking.png','planned-c3-calibration.csv':'packet-c3.csv','planned-d1-dependence.csv':'packet-d1.csv','planned-d2-redundancy.csv':'packet-d2.csv','planned-d3-test-and-overlap.csv':'packet-d3.csv','planned-d4-transfer.csv':'packet-d4.csv'}
    text=json.dumps(draft,ensure_ascii=False)
    for a,b in names.items(): text=text.replace(a,b)
    design=json.loads(text); design['status']='approved and implemented 2026-09-17'
    for cov in design['course_coverage']:
        if cov['objective_id']=='LO8':cov['observable_outcome']=cov['observable_outcome'].replace('定位两阈值','定位0.10阈值的ROC及PR坐标')
    design['implementation_notes']=['表 B1只给原始训练计数；待求参数写在独立答题区，避免证据与答案混排。','表 B2两条查询只必算查询1；第二条为选做。','概率单调变换给出数值汇总；AP与梯形PR面积不混名。','课堂二症状练习使用同一72/24划分；不是在验证集重新估计条件概率。']
    (ASSET/'analysis-design.json').write_text(json.dumps(design,ensure_ascii=False,indent=2),encoding='utf-8')
    return design

def build_docs():
    from docx import Document
    from docx.shared import Inches, Pt, Cm, RGBColor
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
    E=load_evidence(); T=E['tables']; M=E['metrics']; design=prepare_design()
    class Writer:
        def __init__(self,kind):
            self.d=Document(); self.kind=kind; self.page=0
            s=self.d.sections[0]; s.page_width=Cm(21);s.page_height=Cm(29.7)
            s.top_margin=s.bottom_margin=s.left_margin=s.right_margin=Inches(.65)
            s.header_distance=s.footer_distance=Inches(.28)
            for name,size in [('Normal',10.5),('Title',23),('Subtitle',13),('Heading 1',16),('Heading 2',12),('Heading 3',11),('Caption',9)]:
                st=self.d.styles[name];st.font.name='Arial';st.font.size=Pt(size);st.font.color.rgb=RGBColor(0,0,0)
                st.element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'),'微软雅黑')
                st.paragraph_format.space_after=Pt(5);st.paragraph_format.line_spacing=1.12
                if name.startswith('Heading'):st.paragraph_format.space_before=Pt(9)
                for border in list(st.element.iter(qn('w:pBdr'))): border.getparent().remove(border)
            n=self.d.styles['Normal'];n.paragraph_format.widow_control=True
            hp=s.header.paragraphs[0];hp.text='AI for Biomedical Data Science     第7讲     '+kind;hp.style='Caption';hp.runs[0].font.size=Pt(8)
            fp=s.footer.paragraphs[0];fp.alignment=WD_ALIGN_PARAGRAPH.RIGHT
            fp.add_run('教学研究用途  •  ')
            fld=OxmlElement('w:fldSimple');fld.set(qn('w:instr'),'PAGE');fp._p.append(fld)
        def p(self,text,style=None):
            return self.d.add_paragraph(text,style)
        def h(self,text,level=2):return self.d.add_heading(text,level)
        def new(self,heading):
            self.page+=1;h=self.h(heading,1);h.paragraph_format.page_break_before=True
        def table(self,headers,rows,widths=None,answer=False):
            table=self.d.add_table(rows=1,cols=len(headers));table.style='Table Grid';table.alignment=WD_TABLE_ALIGNMENT.CENTER;table.autofit=False
            if widths is None: widths=[6.96/len(headers)]*len(headers)
            for col,w in zip(table.columns,widths):col.width=Inches(w)
            for i,text in enumerate(headers):table.rows[0].cells[i].text=str(text)
            trpr=table.rows[0]._tr.get_or_add_trPr();repeat=OxmlElement('w:tblHeader');trpr.append(repeat)
            for row in rows:
                cells=table.add_row().cells
                for i,value in enumerate(row):cells[i].text=str(value)
            for ri,row in enumerate(table.rows):
                no=OxmlElement('w:cantSplit');row._tr.get_or_add_trPr().append(no)
                if answer and ri>0:row.height=Inches(.50)
                for ci,cell in enumerate(row.cells):
                    cell.width=Inches(widths[ci]);cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
                    tcpr=cell._tc.get_or_add_tcPr();marg=OxmlElement('w:tcMar')
                    for side in ['top','left','bottom','right']:
                        v=OxmlElement('w:'+side);v.set(qn('w:w'),'45');v.set(qn('w:type'),'dxa');marg.append(v)
                    tcpr.append(marg)
                    if ri==0:
                        shd=OxmlElement('w:shd');shd.set(qn('w:fill'),'E8EEF5');tcpr.append(shd)
                    for p in cell.paragraphs:
                        p.paragraph_format.space_after=Pt(0);p.paragraph_format.space_before=Pt(0);p.paragraph_format.line_spacing=1.04
                        p.paragraph_format.keep_with_next=ri<len(table.rows)-1
                        for run in p.runs:run.font.size=Pt(9 if len(headers)>6 else 10);run.bold=(ri==0)
            borders=OxmlElement('w:tblBorders')
            for edge in ['top','left','bottom','right','insideH','insideV']:
                b=OxmlElement('w:'+edge);b.set(qn('w:val'),'single');b.set(qn('w:sz'),'4');b.set(qn('w:color'),'D9D9D9');borders.append(b)
            table._tbl.tblPr.append(borders)
            spacer=self.d.add_paragraph();spacer.paragraph_format.space_after=Pt(0);spacer.paragraph_format.space_before=Pt(0);spacer.paragraph_format.line_spacing=Pt(3);spacer.add_run().font.size=Pt(1)
            return table
        def answer(self,prompt,label='个人作答',lines=3):
            self.p(prompt).paragraph_format.keep_with_next=True
            t=self.table([label],[['']],answer=True)
            t.rows[1].height=Inches(.20*lines+.08)
            return t
        def evidence(self,key):
            v=T[key]; self.p(v['caption'],'Caption').paragraph_format.keep_with_next=True
            widths={'A2':[1.6,5.36],'D3':[2,4.96],'D4':[3,3.96]}.get(key)
            self.table(v['headers'],v['rows'],widths)
        def eq(self,name):
            self.d.add_picture(str(ASSET/f'equation-{name}.png'),width=Inches({'mse':3.4,'loss':6.0,'bernoulli':6.2,'bayes':6.6}[name]))
            self.d.paragraphs[-1].alignment=WD_ALIGN_PARAGRAPH.CENTER
        def title(self,subtitle):
            p=self.p(TITLE,'Title');p.alignment=WD_ALIGN_PARAGRAPH.CENTER
            p=self.p(subtitle,'Subtitle');p.alignment=WD_ALIGN_PARAGRAPH.CENTER
        def save(self,suffix):
            path=ROOT/f'{SLUG}-{suffix}.docx';self.d.save(path);print(path.name)
    options=[['A 保留当前基线','把六症状NB与0.50阈值作为下一轮离线比较的冻结基线，配置不变。'],['B 修改后冻结','先改变症状表示、模型设置或验证阈值中的至少一项，再冻结新基线；指出具体改动。'],['C 暂缓提交','先不提交任何模型；补齐一项足以改变A或B选择的证据。']]
    qmap=[['Q1','表 A1、表 A2：八列中谁是输入，谁是要预测的结果？','任务与输入输出','12'],['Q2','表 A3：72/24/24条如何负责拟合、选择与最终评价？','训练与泛化','14'],['Q3','表 B1至表 B3：症状计数如何变成新组合的预测？','Bayes与NB训练','24'],['Q4','表 B4、表 B5：判对同样多，为什么损失仍不同？','交叉熵与MSE','16'],['Q5','表 C1、图 C2、表 C3：分类、排序、概率分别怎样评价？','模型评价','28'],['Q6','表 D1至表 D4：假设与数据范围如何改变研究方案？','假设与综合','26']]
    roadmap=[['启动课堂','45分钟','只发启动页；不展示参数与性能。'],['课外自学','120分钟','按A→B→C→D顺序释放；完成Q1至Q6及初稿。'],['翻转课堂','45分钟','个人训练练习；随机1至2人白板说明；同伴提问与修订。']]
    phase1=[['7','读案例与表 S1，初选A/B/C','总答案追踪表第1行'],['8','澄清输入、标签与预测对象','启动澄清'],['8','排序流程卡并说明不确定处','启动流程'],['10','按三个概念簇写3W','3W表'],['8','定向阅读并制订个人学习计划','个人学习计划'],['4','向研究组提出一个影响选择的问题','启动退出条']]
    readings=[['R1','Q1/Q2','本worksheet的A概念读页；scikit-learn User Guide 3.1开头的训练与测试，以及3.1.1的validation说明。'],['R2','Q3/Q6','本worksheet的B概念读页；User Guide 1.9开头Bayes公式与1.9.4 Bernoulli Naive Bayes。'],['R3','Q4/Q5','本worksheet的损失及评价读页；User Guide 3.4中confusion matrix、precision/recall/F、ROC、PR、log loss与mean squared error条目。'],['R4','Q5/Q6','本worksheet的校准读页；User Guide 1.16.1 Calibration curves中reliability diagram及样本量的解释。']]
    w=Writer('学生学习与答题册')
    w.page=1;w.title('Problem Based Learning Student Worksheet')
    w.p('姓名与学号可写在页眉空白处。每人独立提交本册；可讨论、查资料或使用AI，但计算、证据核对与判断由本人负责。仅用于教学研究，不用于临床诊断、部署或患者照护。')
    w.h('研究组委托')
    w.p('研究组要把一个“根据六项症状预测膀胱炎症标签”的模型纳入下一轮离线方法比较。请提交输入、训练、验证、评价及适用范围明确的研究方案，并决定怎样处理当前的朴素贝叶斯模型。')
    w.table(['下一步选项','研究组实际采取的动作'],options,[1.35,5.61])
    w.p('选项互斥：A不改当前配置，B先改再冻结，C先补证据且不提交模型。三项均不授权临床使用；每项都可得到高分，取决于是否完成六个模块并提供充分理由。')
    w.p('表 S1  启动事实｜Q1/Q2。每行是一项已知条件；同一个UCI数据集共120条、8列，来源是医学专家为测试专家系统设计的潜在病例描述。')
    w.table(['已知条件','当前事实'],[['数据','体温及五个有无症状，共六项候选输入；另外两列是膀胱炎症与肾炎决策。'],['已有方案','六项输入全部二值化；体温≥38℃记为发热；拟用NB和0.50阈值。'],['资料缺口','未提供独立患者身份、抽样过程或外部人群验证；尚未公布模型结果。'],['用途','研究组内部比较方法的基线，不是诊疗系统。']],[1.3,5.66])
    w.p('综合研究方案：总答案追踪表与一份250至400字的研究备忘录。备忘录必须使用Q1至Q6的结果，不能仅回答“不能用于临床”。')
    w.p('停止条件：无法追溯记录时暂停计算；测试数据参与调阈值时撤回独立测试声明。修改条件：若发现重复输入改变概率，检查概率解释；若只覆盖已见组合，收缩对新组合的主张。')
    w.new('学习路线与探究问题')
    w.table(['阶段','预算','你将完成什么'],roadmap,[1.1,.8,5.06])
    w.table(['模块','案例中的问题及证据','学习内容','分钟'],qmap,[.5,3.75,2.1,.61])
    w.p('Q是探究问题编号，不是独立的额外作业。Q3的24分钟不含第二次课堂的15分钟训练练习；Q6的26分钟已包含最终备忘录，所有课外必做合计120分钟。需要时用计算器；不要求从零编程。')
    w.h('本册过程用语')
    w.p('总答案追踪表：唯一持续修改的研究判断表。第1行记录选项A/B/C；第2至7行分别存放六个模块的结论。保留原判断可用修订或不同颜色，不另抄一份。')
    w.p('3W：按“我已知道什么、还不知道什么、下一步做什么”组织自学。证据卡：只核查一个具体主张，并写明来源与本案例的检验。退出条：每阶段结束时的短答，用来暴露仍未解决的问题。白板说明：随机抽到的学生解释自己的证据与推理，不背固定讲稿。')
    w.h('分阶段发放')
    w.p('启动页至“启动学习计划与退出条”在第一次课堂使用；Packet A完成Q1/Q2后交检查点A，再领取B；B交后领取C；C提交冻结草案后领取D。完整册由教师保管或在课程平台分段发布，不要提前查看D。翻转课堂页在课外自学结束后使用。')
    w.new('总答案追踪表')
    w.p('为向研究组留下可核对的决策轨迹，启动时只根据表 S1在第1行选择方案A、B或C并写理由；此时未获证据写“待领取”，不猜测性能。完成Q1至Q6后依次更新第2至7行和第1行；各单元用关键词记录判断、证据编号、反向证据、改判条件及最后更新模块。')
    t=w.table(['行号及主题','当前判断','支持证据编号','反向证据','什么发现会改变判断','最后更新模块'],[[r,'','','','',''] for r in ['1 当前选项及主要理由','2 任务与输入输出','3 训练与评价流程','4 NB参数与预测','5 损失选择','6 评价与阈值','7 假设与用途']],[1.15,1.4,1.0,1.15,1.5,.76],answer=True)
    for row in t.rows[1:]:row.height=Inches(.83)
    w.p('第1行的当前判断始终包含选项字母。反向证据既可支持另一选项，也可限制你自己的结论；尚无反向证据时写明“目前未发现”，不要写成“已证明不存在”。')
    w.new('启动课堂 45分钟')
    w.table(['分钟','每人的动作','写答位置'],phase1,[.55,4.55,1.86])
    w.answer('启动澄清：为判断表 S1的六症状模型能回答什么，请用本案例各举一个输入和标签，再写明“预测一条新记录”与“检验症状和标签是否相关”在目标上的区别；用三句话作答。','启动澄清',3)
    w.p('启动流程卡：甲，冻结配置；乙，计算未参与选择的最终评价；丙，用训练记录估计参数；丁，用验证预测比较阈值；戊，用已估计参数计算验证记录的概率。')
    w.answer('启动流程：为安排表 S1模型的研究步骤，请把甲乙丙丁戊排成先后顺序，并说明其中哪一步目前最需要学习；写一个顺序和一个具体疑问，不要求此时答对。','启动流程',2)
    w.new('启动3W与定向自学')
    w.p('3W格式例（另一情境，不属于本课数据）：比较两种培养基时，“已知两组平均生长量；不知方差与样本量；下一步查实验记录，画每组散点以判断均值是否有代表性”。')
    w.p('为让自学直接推进表 S1对应的方案A/B/C，请完成3W表三个概念簇的各一行：写已有知识、阻碍方案判断的知识缺口，以及具体学习或分析动作。资源可写R1至R4并注明要找的概念；编号在本页“定向阅读指南”中定义。')
    w.table(['概念簇','关于这个问题，我已经知道什么？','要正确回答问题，我还不知道什么？','下一步我要学习什么、查找什么或分析什么？'],[[s,'','',''] for s in ['任务与流程 Q1/Q2','概率与损失 Q3/Q4','评价与边界 Q5/Q6']],[1.15,1.7,1.75,2.36],answer=True)
    w.h('定向阅读指南')
    w.table(['资源','模块','只读这些部分'],readings,[.5,.8,5.66])
    w.p('本册概念读页包含完成必做题所需知识，在线原文用于追溯与核查，不要求整章通读。英文术语第一次出现时附中文含义；完整链接在末页“数据来源与数据获取”。')
    w.new('启动学习计划与退出条')
    w.answer('个人学习计划：为补齐3W表中最影响方案A/B/C选择的一个缺口，请指定Q1至Q6中的一个问题、R1至R4中的具体资源与要做的计算或比较；写明一个可检验的完成标准和预计分钟数。标准示例是“能不用笔记解释某张表中的分母”，不能只写“看懂”。','个人学习计划',6)
    w.answer('启动退出条：依据表 S1仍缺少的资料，写给研究组一个会影响方案A/B/C选择的具体问题，并说清两种可能回答分别会让你保留还是改变哪一选项；用两至三句话说明。','启动退出条',5)
    w.p('提交检查点S：追踪表第1行、启动澄清、启动流程、3W表、个人学习计划与退出条。此时只检查问题是否具体，不按猜中模型成绩给分。')
    w.new('Packet A 任务与数据')
    w.h('Q1 八列数据怎样形成一个预测任务')
    w.p('12分钟，含阅读与写答。学习目标：将数学模型对应到实际输入、标签及预测对象；区分关联检验和预测。必读本页概念与表 A1、表 A2；来源R1。')
    w.p('概念读页：数据模型用函数把输入映射到输出。X是一条记录的特征向量，Y是希望预测的标签；本课记阳性为1、阴性为0。模型fθ(X)输出阳性概率p，θ是从训练数据估计的参数；阈值t把p变成类别，p≥t时预测1。统计关联不等于能准确预测另一条记录，也不证明因果。')
    w.evidence('A1'); w.evidence('A2')
    w.new('Packet A 定义与流程')
    w.answer('Q1建模：为界定交给研究组的输入输出，请根据表 A1写出X的六个分量、主标签Y及fθ(X)的含义；说明为何本题不把另一疾病决策列当作输入。再把“恶心与膀胱炎症有关吗”和“新记录有多大阳性概率”分别对应到关联检验或预测任务。','Q1 任务定义',5)
    w.answer('Q1解释范围：结合表 A2的来源文字和重复记录，写明你暂时允许把模型结果用于什么对象，以及仍需确认哪项来源信息；用两句话给出结论与不确定性。将结论写入总答案追踪表第2行，并据表 A2保留或修改第1行的方案A/B/C。','Q1 结论与未知',3)
    w.h('Q2 怎样分工才算评价新预测')
    w.p('14分钟，含阅读、流程作答与追踪更新。训练是用已知标签估计参数；预测是固定参数后计算输出。验证集用于选择配置或阈值；测试集用于配置冻结后的评价，不能再据其结果改选择。泛化是对未参与拟合或选择、且与目标使用人群相关的数据仍有预测能力；行号不同本身不保证这一点。来源R1。')
    w.evidence('A3')
    w.new('Packet A 工作流与待查证据')
    w.answer('Q2数据分工：为了让表 A3的最终评价有独立职责，请重排启动流程卡甲乙丙丁戊，并为每步写“训练72条”“验证24条”“测试24条”或“无需新数据”；注明在哪一步冻结阈值。再判断“看完测试错误再把阈值改为0.10”为什么改变了这次测试的用途。','Q2 流程及数据职责',5)
    w.p('Q2缺口清单：为安排A之后的自学，请把每项缺失证据对应到一个模块和一个Packet，并写该证据将检验总答案追踪表哪一行的判断；用本册探究问题地图定位，不预填未发布的数值。')
    w.table(['待查证据','模块与Packet','将检验的追踪表行号及判断'],[['训练参数与预测','',''],['分类与概率表现','',''],['假设及测试范围','','']],[2,1.5,3.46],answer=True)
    w.answer('Q2暂定协议：依据表 A3，写明研究组应在何时、用哪部分数据选阈值，以及哪项有关新记录来源的问题仍未解决；写两至三句话。更新总答案追踪表第3行与第1行的方案A/B/C，避免把记录编号隔离写成患者独立的证明。','Q2 协议与未知',3)
    w.p('提交检查点A：Q1/Q2答题与追踪表第1至3行；领取Packet B。')
    w.new('Packet B 从计数训练朴素贝叶斯')
    w.h('Q3 症状计数怎样变成预测')
    w.p('24分钟，含本页、下一页的手算与迁移。学习目标：估计先验与条件概率，应用Bayes归一化，理解0值症状也提供信息。资源R2；只必算查询1。')
    w.p('概念读页：Bayes定理把“该类别下见到症状的可能性”与类别先验结合，得到见到症状后的类别后验。先验P(Y=c)由训练集中类别比例估计。“朴素”指给定类别Y后，模型把各症状当作条件独立，因而把各症状的条件概率相乘；不是说所有症状在总体中互不相关。')
    w.eq('bayes')
    w.p('c是类别0或1，j是症状编号；S是未归一化的类别分数，分母把两类分数加总。此处只看这两个类别。')
    w.eq('bernoulli')
    w.p('n_c是类别c中的训练记录数；n_jc是其中症状j为1的条数。θ是“症状为1”的条件概率。加一平滑把每个0/1计数各加1，因此分母加2，避免一个零计数把整个乘积变成零。症状为0时须取1−θ，不能漏掉。')
    w.evidence('B1')
    w.p('训练的产物是两类先验以及每类的六个θ。它不记住一个查询的答案；预测时使用已经估计好的参数。这里由计数直接估计参数，不需要学生编写优化算法。')
    w.new('Packet B 参数与后验手算')
    w.evidence('B2')
    w.answer('Q3计算：为检查模型是否能被复核，请从表 B1先算P(Y=1)、P(尿意频繁=1|Y=1)和P(恶心=0|Y=0)，写出各自分母。再对表 B2的查询1列出S1与S0的六个条件因子、计算两分数并归一化为阳性后验，最后按0.50阈值预测类别。','Q3先验、条件概率与后验',6)
    w.answer('Q3表示迁移：若研究组改用六个医学摘要关键词的“出现/未出现”预测文献类别，请对照表 B2的0/1表示说明Bernoulli NB是否可直接使用；再说明把输入改成词频计数时为何不能照抄本题的二元条件概率式。用两句话指出适配条件，而非声称文本任务都相同。','Q3 文本迁移',3)
    w.answer('Q3暂定结论：依据表 B1、表 B2，说明参数来自哪里，以及查询1的概率计算还依赖哪条未经验证的症状关系假设；把结论写入总答案追踪表第4行。然后在第1行保留或修改方案A/B/C，引用一次具体计数或后验作为理由。','Q3 计算可解释性与未知',3)
    w.p('表 B3保留在“翻转课堂的独立训练练习”页供课堂使用；课外Q3不要求提前完成该练习。查询2为选做，不计入24分钟。')
    w.new('Packet B 判对不等于概率预测相同')
    w.h('Q4 概率损失在衡量什么')
    w.p('16分钟，含本页与下一页。学习目标：把“预测与真实标签的差距”写成可计算的量；比较交叉熵和连续结局的均方误差。来源R3。')
    w.p('概念读页：损失函数为一条或一组预测赋予误差大小。二分类交叉熵也称log loss：若真实标签为1，只取−ln(p)；若标签为0，只取−ln(1−p)。对错误的高置信概率惩罚很大；这里ln是自然对数，损失越小越好。')
    w.eq('loss');w.evidence('B4')
    w.answer('Q4概率比较：为了决定研究组是否只报告分类正确数，请对表 B4两模型分别写出0.50阈值下的四个类别、accuracy与平均交叉熵；至少展开记录3的损失，再说明概率错得很自信如何影响平均值。保留三位小数，可使用计算器。','Q4 分类与概率损失',5)
    w.p('理解训练与评价：本课NB通过类别计数和加一平滑估计生成模型参数，并没有直接数值最优化“验证集条件交叉熵”。这不妨碍我们用交叉熵评价其预测，也不意味着NB与似然完全无关。联合似然的推导为选读。')
    w.new('Packet B 连续误差与证据核查')
    w.p('概念读页：均方误差MSE用于比较连续数值预测与真实数值，先逐条平方再平均。它随目标单位变化；体温的MSE单位是℃²。二元概率也可用平方误差评价，称Brier score；但不要把它等同于纯校准误差。')
    w.eq('mse');w.evidence('B5')
    w.answer('Q4连续误差：为区分预测目标，请用表 B5写出两项平方误差、MSE与单位，再说明为什么不能把这个数与表 B4的交叉熵直接比较大小。把损失选择及仍不确定的错误代价写入总答案追踪表第5行，并用表 B4保留或修改第1行的方案A/B/C。','Q4 MSE及报告选择',3)
    w.answer('证据卡：为核查你对表 B1或表 B4的一项解释，请从R2、R3或你使用的AI输出中摘录一个明确主张；注明具体段落或链接，引用本案例一个计数或损失结果检验它，并写出成立范围。只核查一个主张，按“原话—来源—案例核对—限定”四项作答；使用AI时必须另核对到R2或R3。','个人证据卡',4)
    w.p('提交检查点B：Q3/Q4、证据卡及追踪表第4至5行；领取Packet C。')
    w.new('Packet C 分类错误与阈值')
    w.h('Q5 怎样评价研究组的基线')
    w.p('28分钟，含分类、排序、校准三页。学习目标：区分六种分类指标的分母、两类排序曲线与概率校准；将评价与研究用途相连。来源R3/R4。')
    w.p('概念读页：TP/FP是预测阳性中的真/假；TN/FN是预测阴性中的真/假。Sensitivity（敏感度）=Recall（召回率）=TP/(TP+FN)；Specificity（特异度）=TN/(TN+FP)；Precision（精确率，阳性预测值）=TP/(TP+FP)。Accuracy（准确率）=(TP+TN)/总数；F1=2TP/(2TP+FP+FN)。分母为0的量未定义，不能随意当作0。')
    w.evidence('C1')
    w.answer('Q5分类计算：为追踪表第6行确定主要错误，请使用表 C1阈值0.10这一行，分别写出accuracy、sensitivity、specificity、precision、recall和F1的分子、分母及结果。再说明先验基线的precision为何不能按通常分式得到一个数；将答案按六个指标加基线说明排列。','Q5 指标计算',5)
    w.answer('Q5阈值比较：比较表 C1中0.10、0.50和0.90三行，写两句话指出阈值变化使哪类错误增加，并引用两组计数；再写研究组还应告诉你哪一种错误代价，才能判断是否应改阈值。','Q5 阈值与代价',3)
    w.new('Packet C 排序能力')
    w.p('概念读页：ROC横轴是假阳性率1−specificity，纵轴是sensitivity；PR横轴是recall，纵轴是precision。改变阈值可沿曲线移动。ROC-AUC概括排序：随机抽一阳性一阴性，阳性得分更高的频率，平局计一半；它不是某个阈值的准确率。PR特别显示预测阳性的组成，受阳性比例影响。')
    w.p('PR-AUC需说明积分方式。本课称“梯形PR面积”的量按相邻点线性连接积分；AP（average precision）按召回增量加权precision，不等于一般的梯形面积。本例两者相等不表示处处相等；不要求手推插值算法。')
    w.p('图 C2  验证集排序曲线｜Q5 ROC与PR。每个点对应同24条验证记录的阈值；曲线来自六症状NB，阳性比例12/24。图中标出0.10、0.50、0.90；面积无单位。')
    w.d.add_picture(str(ASSET/'packet-c2-ranking.png'),width=Inches(6.9))
    w.table(['概率表示','ROC-AUC','梯形PR面积','AP','平均交叉熵'],[['原概率p','1.000','1.000','1.000',f"{M['val_log_loss']:.4f}"],['变换0.25+0.50p','1.000','1.000','1.000',f"{M['transformed_log_loss']:.4f}"]],[2.1,1,1.2,1,1.66])
    w.p('图 C2附表：同一24条验证记录；第二行仅将每个p替换为0.25+0.50p，不重新训练，不是另一批患者。变换保持概率的大小顺序及平局关系；给定变换全部输入，损失由该变换重算。')
    w.answer('Q5排序判断：用表 C1的0.10行计算图 C2的ROC点和PR点坐标，再解释这两个点各回答什么问题。对照图 C2附表写明：相同AUC是否足以保证相同的概率质量，以及还需读哪一项证据；不得把AUC解释成0.50阈值的accuracy。','Q5 坐标与AUC的边界',4)
    w.new('Packet C 概率能否按字面理解')
    w.p('概念读页：校准（calibration）问的是：收到相近预测概率的记录中，实际阳性频率是否接近该概率？例如足够多同类样本都预测0.7时，若约70%阳性才与该概率吻合。分箱图或表给出局部观察；每箱人数决定观察有多不稳定。良好排序、某阈值下全对，都不自动保证校准。')
    w.evidence('C3')
    w.answer('Q5校准观察：为说明表 C3中的概率能否直接作为可靠风险值，请选记录数最少的非空箱，抄出其人数、阳性数、平均预测概率与实际阳性比例，并计算二者差值。用两句话说明这种差值能提示什么、为何不能凭这个小箱认定总体已校准或失校准。','Q5 概率与频率',4)
    w.answer('Q5报告方案：结合表 C1、图 C2、表 C3，各选一项分类、排序及概率评价证据，并说明三项不能互相替代的原因；用三句话形成研究组的报告清单。把报告选择、阈值及尚缺的错误代价写入总答案追踪表第6行，再在第1行保留或修改方案A/B/C并注明引用的证据编号。','Q5分类、排序与概率的分工',4)
    w.p('提交检查点C：先交Q5和追踪表第1行的冻结草案，再领取D。D的测试结果只供评价已经写下的配置；若据此修改模型，必须声明需要新的独立测试，不可回称本次测试仍然盲测。')
    w.new('Packet D 假设与重复信息')
    w.h('Q6 哪些结论需要修改或补证据')
    w.p('26分钟，包含D两页的分析及下一页备忘录。学习目标：检查条件独立、说明计算简单的优点与限制、限定泛化并迁移到低患病率场景。来源R2/R4。')
    w.evidence('D1')
    w.answer('Q6条件独立：为核查表 B1乘法规则，请只用表 D1真实类别为0的37条记录，计算P(尿意频繁=1且排尿痛=1|Y=0)与两项边缘条件概率的乘积。写出两值是否相同，并说明这种经验比较为何不等于已经证明总体独立或不独立。','Q6 类别内关系',3)
    w.evidence('D2')
    w.answer('Q6复制症状：对表 D2概率差最大的向量，写出复制前后概率和变化方向；结合表 B1的参数估计方式，各说明NB一项“简单快速可复核”的优点及一项局限。必须交代为什么没有增加生物学信息却可能改变概率，而不要把64种向量称为64名患者。','Q6 算法优点与重复信息',3)
    w.new('Packet D 测试范围与低患病率')
    w.evidence('D3');w.evidence('D4')
    w.answer('Q6迁移计算：为回答“解释为什么医学诊断模型不能只看 accuracy”，请由表 D4推算TP、FN、FP、TN与precision，并计算“所有人均预测阴性”的accuracy。将计算与表 C1的错误类型联系，指出单一accuracy可能掩盖什么。','Q6 低患病率反例',4)
    w.answer('Q6泛化边界：根据表 D3，分别写出“测试24/24正确”可支持的内部描述和不能支持的新组合或新患者主张；注明还缺哪项能改变方案A/B/C选择的证据。把本页及表 D1、表 D2的结论合并更新总答案追踪表第7行及第1行，勿把组合重复直接判作泄漏。','Q6 测试范围与未知',3)
    w.new('课外综合 交给研究组的备忘录')
    w.p('以下写作已计入Q6的26分钟，其中预留约12分钟，不额外增加作业。')
    w.p('决策检验矩阵：为说明选择有反证条件，请依据表 C1及表 D1至表 D4，为方案A、B、C各写一项支持它的发现、一项会使它不再合适的发现，以及若选它研究组下一步怎样核查；每格用关键词，不重写整篇理由。')
    w.table(['选项','支持它的发现','反向或否定发现','下一步核查及涉及模块'],[['A 保留当前基线','','',''],['B 修改后冻结','','',''],['C 暂缓提交','','','']],[1.25,1.85,1.85,2.01],answer=True)
    w.answer('初稿备忘录：请从总答案追踪表第1至7行写250至400字，明确方案A/B/C及理由；依次包含Q1的输入输出、Q2的数据职责、Q3的参数与假设、Q4的损失选择、Q5的分类/排序/概率评价，以及Q6的边界与下一步。必须引用至少三个证据编号，并用表 C1或表 D4的数字直接回答原题“解释为什么医学诊断模型不能只看 accuracy”；写明什么新发现会改变你的方案。','个人研究备忘录初稿',10)
    w.p('提交检查点D：完整个人worksheet至本页。保留计算过程与证据卡；不要上传教师材料或试填答案。准备在翻转课堂解释自己的学习路径、一次判断修订和完整解题逻辑。')
    w.new('翻转课堂的独立训练练习')
    w.p('课堂45分钟：前15分钟每人独立完成本页与下一页计算；随后随机1至2人白板说明12分钟，同伴提问6分钟，教师归纳5分钟，个人修订5分钟，退出条2分钟。白板内容由你的完整方案组织，不规定固定讲稿或组内角色。')
    w.h('只用两项症状重新训练')
    w.p('这是真正的参数估计与预测练习，不是照抄六症状模型的结果。输入改为尿意频繁与排尿痛，标签仍为膀胱炎症；沿用表 B1的训练72条、先验35/72与37/72、加一平滑。只取这两项症状，所有其他条件因子都不进入乘积；类别阈值0.50。')
    w.evidence('B3')
    w.answer('课堂训练：为亲自建立二症状基线，请用表 B1尿意频繁与排尿痛两行，估计两类各两个条件概率；写出四个带分母的值。再对两症状为(1,1)的记录计算S1、S0和后验；此时只用两因子相乘，不得用表 B3的验证标签估计参数。','课堂参数及一个后验',5)
    w.p('其余组合的计算方法相同：两症状的0值均取补概率，先验不变。同一输入组合的不同真实标签记录得到同一预测概率。可以用计算器，也可在计算后用辅助Notebook核对。')
    w.new('翻转课堂 验证与个人修订')
    w.answer('课堂评价：依据刚训练的二症状参数，为表 B3每一行写阳性后验与0.50阈值类别，用该行记录数加权汇总TP、FN、FP、TN；最后计算sensitivity和specificity。请保留各组合的概率，避免把聚合表的行数误当成24条记录数。','课堂逐组合预测与评价',5)
    w.answer('同伴提问与教师归纳：在白板说明后，针对表 B1、表 C1或表 D3的一项具体推理写一个可检验的问题，并记录回答或教师纠正的一个概念；用两至三句话说明它影响总答案追踪表哪一行。','课堂同伴问题与概念修正',3)
    w.answer('课堂修订：请重读初稿备忘录，依据课堂二症状结果或同伴提出的表 B1、表 C1、表 D3证据，写100至150字的修订段落；点明保留还是修改方案A/B/C及具体理由。将变化同步到总答案追踪表第1行和受影响的模块行；若不改变选项，也必须补写一项限定。','个人研究备忘录修订段落',5)
    w.answer('课堂退出条：从表 C1、图 C2、表 C3、表 D3中选择一项最改变你理解的证据，用一句话说明它纠正了什么概念，再写一个你下一次会主动检查的问题。若课堂超时，本框作为课后2分钟补记，不另加新分析。','课堂退出条',2)
    w.new('数据来源与数据获取')
    w.p('完成必做题不需要编程或上网：所有必要数字均已给出。辅助Notebook按Q1至Q6顺序组织，可离线运行；请在相应Packet发放后才执行对应单元，完整执行会提前显示D。')
    w.table(['资料','访问与核查信息'],[['数据','UCI Acute Inflammations；DOI 10.24432/C5V59S；https://archive.ics.uci.edu/dataset/184/acute%2Binflammations'],['许可及出处','CC BY 4.0；Jacek Czerniak，2009入库。保留来源、许可及本课重编码说明。'],['文件与读取','原始diagnosis.data为UTF-16、制表符分隔、小数逗号；说明为diagnosis.names。8列直接对应，无合并。未观察到缺失；若重读发现空值应停止核查，不能默填0。'],['本地副本','assets/pbl-machine-learning-basics-naive-bayes/acute-inflammations.csv；yes/no表示有无。Notebook在notebook/pbl-machine-learning-basics-naive-bayes-lab.ipynb。'],['预处理','固定体温≥38℃为发热，五个症状yes/no转1/0；膀胱炎症为主标签，肾炎不作为输入。切点是教学预设，不从测试搜索。'],['R1','https://scikit-learn.org/stable/modules/cross_validation.html  3.1与3.1.1'],['R2','https://scikit-learn.org/stable/modules/naive_bayes.html  1.9与1.9.4'],['R3','https://scikit-learn.org/stable/modules/model_evaluation.html  3.4中相关指标条目'],['R4','https://scikit-learn.org/stable/modules/calibration.html  1.16.1']],[1.05,5.91])
    w.h('选做延伸 不计入必做时间与评分')
    w.p('可在完成主任务后重算表 B2查询2，或按症状组合而非行号划分数据，观察结论如何改变。按组验证、跨医院验证、置信区间推导、Gaussian NB及其他算法比较属于后续学习；本课不要求从零实现。')
    w.p('自查提交：有六模块计算与判断；每次追踪更新可找到证据；备忘录回答原题并明确使用范围；课堂训练中实际估计了参数；引用AI主张时已经核对。')
    w.save('student-worksheet')

    # Evidence pack is student-safe; answer logic lives only in the teaching plan.
    e=Writer('分阶段证据包');e.page=1
    e.d.styles['Normal'].font.size=Pt(10);e.d.styles['Normal'].paragraph_format.line_spacing=1.02;e.d.styles['Normal'].paragraph_format.space_after=Pt(3)
    e.d.styles['Caption'].paragraph_format.line_spacing=1.0;e.d.styles['Caption'].paragraph_format.space_after=Pt(3)
    e.title('Packet A D Evidence Pack')
    e.p('发放规则：本包不含学生填答答案。不要一次发放全包；启动只用worksheet启动页，A/B/C/D分别在相应检查点后发布。所有证据也嵌入学生worksheet，Notebook不是找数字的唯一入口。')
    e.p('研究组要选择下一轮离线方法比较的基线。学生须交完整输入、训练、损失、评价与边界方案；A保持六症状NB与0.50，B修改后冻结，C补证据前暂不提交。')
    e.table(['模块','问题与证据','概念','分钟'],qmap,[.5,3.75,2.1,.61])
    e.p('A待查训练参数与损失；B后待查验证评价；C后仍待查类别内假设与固定测试；这些缺口分别由B、C、D提供。')
    for packet,keys in [('A',['A1','A2','A3']),('B',['B1','B2','B3','B4','B5']),('C',['C1','C3']),('D',['D1','D2','D3','D4'])]:
        e.new('Packet '+packet+' 证据与核查')
        for key in keys:
            if key=='D3': e.new('Packet D 测试与迁移')
            e.evidence(key)
        if packet=='C':
            e.p('图 C2  同24条验证记录；ROC与PR曲线，三处阈值同表 C1。')
            e.d.add_picture(str(ASSET/'packet-c2-ranking.png'),width=Inches(6.9))
            e.table(['概率表示','ROC-AUC','梯形PR面积','AP','平均交叉熵'],[['p',1,1,1,round(M['val_log_loss'],4)],['0.25+0.50p',1,1,1,round(M['transformed_log_loss'],4)]])
        prompts={'A':'核查：表 A2来源与表 A3分割分别允许你如何界定记录和评价？回到追踪表第2/3行说明A/B/C的理由还缺什么。','B':'核查：表 B1的分母与表 B4的概率损失如何改变你的模型理解？更新追踪表第4/5行及第1行，不以计算透明推断因果解释。','C':'核查：表 C1的错误数量、图 C2的排序与表 C3的概率频率，分别支持哪一种评价？把选定阈值写入第6行，冻结第1行配置后才看D。','D':'核查：表 D1、表 D2、表 D3是否要求改变第7行的假设或使用范围？用表 D4补充accuracy作业，再整合第1至7行，而非只写临床不可用。'}
        e.p(prompts[packet])
    e.new('证据的来源与边界')
    e.p('原始来源：UCI Acute Inflammations，https://doi.org/10.24432/C5V59S，CC BY 4.0。记录为医学专家设计的潜在病例；不能从公开资料确认独立临床队列。原文件UTF-16与说明ASCII不一致；本课显式读取并保留来源，没有删除重复记录。')
    e.p('B4、B5、D4是明确给定输入的教学构造；D2是对输入列的干预与64种组合穷举。它们不是额外患者资料。所有数值由固定划分及教学输入重算；作者曾看过固定测试，课堂只模拟冻结后解封。')
    e.p('本包支持学习参数估计、损失、分类/排序/概率评价与假设审查，不支持因果结论、真实患者风险保证或临床部署。表 D3的组合重复不单独证明泄漏；应结合来源和抽样过程判断。')
    e.save('evidence-pack')

    t=Writer('教师用书 含参考答案');t.page=1;t.title('PBL Flipped Classroom Teacher Plan')
    t.table(['项目','安排'],[['学生','生命科学高年级直博预备生；标准微积分、数理统计与前六讲；不预设CS训练。'],['组织','45分钟启动＋120分钟个人自学＋45分钟翻转课堂；教师讲授以追问和5分钟归纳为主。'],['数据','同一UCI数据120×8；六输入，两标签；主标签59阳性/61阴性。'],['总问题','向研究组交完整预测研究方案，选择保留、修改或暂缓基线。'],['学习产物','worksheet中的六模块、追踪表、证据卡、研究备忘录及课堂训练；Notebook仅辅助。']],[1.1,5.86])
    t.p('教师专用：本文件含参考计算及判卷逻辑，不向学生提前发放。所有个体均应准备完整解释，随机抽1至2人；不设置固定小组角色。')
    t.new('按源大纲验收学习行为')
    for cov in design['course_coverage']:
        t.p(cov['objective_id']+'  '+cov['observable_outcome']+'  对应'+','.join(cov['inquiry_question_ids']))
    t.new('三阶段组织与发放')
    t.table(['启动分钟','动作','学生写答位置'],phase1,[.6,4.3,2.06])
    t.p('启动讨论允许错误。用表 S1的来源与用途让初选有理由；不要透露最终100%成绩。先问“你缺什么资料”而非解释全部算法。流程正确顺序丙→戊→丁→甲→乙，启动时只收初想，Q2再修。')
    t.table(['自学模块','证据及检查点','预算'],[[q[0],q[1],q[3]+'分钟'] for q in qmap],[.7,5.35,.91])
    t.p('平台依次释放A/B/C/D：A交Q1/Q2和追踪更新，B交Q3/Q4及证据卡，C交Q5及冻结配置，D交综合初稿。R1至R4均给到具体章节，worksheet概念读页足够完成核心题；不要求通读英文文档。AI可用于解释或核对，证据卡必须把一个主张与源资料及案例数字核验。')
    t.table(['翻转分钟','组织','可观察产出'],[['15','每人训练二症状NB并汇总24条验证','参数、各组合后验、敏感度和特异度'],['12','随机1至2人白板说明','自己的证据链与一次修订'],['6','同伴针对证据提问','具体问题及回应'],['5','教师归纳相邻概念差异','分母、条件独立与泛化的纠偏'],['5','个人修订100至150字','备忘录修订和追踪表更新'],['2','退出条','一个概念变化与下一次检查']],[.65,3.4,2.91])
    t.new('Q1与Q2 任务和流程参考')
    t.p('Q1：输入为体温二值化的发热及五项症状；主标签是膀胱炎症。另一个疾病决策不属于本课指定六症状任务，且其预测时可用性未确立；不要一概说任何其他疾病标签都必然泄漏。fθ给出模型阳性概率，阈值给类别。关联检验针对总体关系；预测要评估新输入的表现，两者不等价。')
    t.p('Q2：训练72条拟合先验与条件概率；固定参数预测验证24条；验证比较阈值；冻结配置不需新样本；测试24条只评价被冻结配置。看test后改阈值把test用作选择，后续需另一个未使用测试集。行号不交叉不等于输入组合、患者或来源独立。')
    t.p('合理追问：若方案A是教学比较基线，为何仍可保留它？若选C，到底哪个未决证据会让你改选A或B？仅答“样本小，不能临床”而不给研究流程不能获得综合题高分。')
    t.h('来源核对')
    t.p('官方说明是专家设计潜在病例；不添加未公开生成机制。UTF-16与文档ASCII矛盾、体温范围、yes/no层级、零缺失及重复均写入表 A2。保留全部行，避免暗中改变与已有分析对应的样本。二值化切点38℃仅教学预设，不说由此发现医学最佳阈值。')
    t.new('Q3 参数估计参考')
    t.evidence('B1')
    t.p('先验35/72与37/72。阳性类每个θ分母37；阴性类分母39。P(尿意频繁=1|Y=1)=(35+1)/37=36/37；P(恶心=0|Y=0)=1−(5+1)/39=33/39。所有数字以表 B1为准。')
    # Derive the hand solution directly rather than relying on hardcoded counts.
    countrows=T['B1']['rows'];fac=[]
    x=[0,0,0,1,1,0]
    for c,n,index in [(1,35,1),(0,37,2)]:
        nums=[(int(row[index])+1 if xx else n-int(row[index])+1) for row,xx in zip(countrows,x)]
        score=(n/72)
        for num in nums:score*=num/(n+2)
        t.p(f'查询1：S{c} = {n}/72 × '+' × '.join(f'{v}/{n+2}' for v in nums)+f' = {score:.8f}。')
    t.p(f"归一化查询1阳性后验={M['query_probabilities'][0]:.6f}；查询2={M['query_probabilities'][1]:.6f}（选做）。查询1按0.50预测阳性。若学生漏掉症状0的补概率，要求回到二元概率定义，不直接给乘积答案。")
    t.p('六个关键词的出现/未出现适配Bernoulli表示；词频是非负计数，不符合每列只有0/1的这一似然，应改表示或选择与计数匹配的模型。可解释的是参数与计算路径，不是因果机制。零计数问题通过加一平滑处理；不要声称它证明独立。')
    t.new('Q4 损失参考与概念边界')
    t.p(f"表 B4两模型0.50阈值均预测[1,0,0,0]，accuracy=3/4。模型甲损失={M['loss_A']:.6f}，模型乙={M['loss_B']:.6f}；记录3为阳性却预测0.4或0.01，对应−ln0.4=0.916291与−ln0.01=4.605170。三次很小损失不能抵消一次极自信错误。")
    t.p('表 B5平方误差为1和4，MSE=2.5℃²；不是RMSE，也不是2.5℃。不能把不同预测对象、不同单位或评分规则的原始损失数值直接比高低。')
    t.p('训练细节：Bernoulli NB可用生成模型的类别条件计数估计；这里加一平滑。不能说“NB没有任何损失/似然”，也不能说该实现直接最小化了验证条件log loss。推导联合似然不在本课考核范围。')
    t.p('证据卡接受一个短而可检验的文档/AI主张，如“二元NB把未出现的信息纳入模型”。学生需引用本例0症状的补概率，给来源小节，再限定二元表示与条件独立。只有聊天截图或泛泛“AI说正确”不足。')
    t.new('Q5 分类 排序 校准参考')
    t.p('阈值0.10：accuracy=19/24=0.7917；sensitivity=recall=12/12=1；specificity=7/12=0.5833；precision=12/17=0.7059；F1=24/29=0.8276。先验基线无阳性预测，precision分母0，未定义；软件zero_division约定不改变数学含义。F1按2TP/(2TP+FP+FN)可得0。')
    t.p('阈值从0.10提高到0.50，FP由5到0且FN仍0；再到0.90，FP仍0而FN变2。已给候选中0.50在此验证集表现最好，但这不证明任意新分布或成本下最佳。成本未知应明确，不必人为为了高敏感度一律降阈值。')
    t.p(f"图 C2阈值0.10：ROC点=(5/12,1)；PR点=(1,12/17)。ROC-AUC、梯形PR面积与AP均1。严格递增变换0.25+0.50p保留排序，平均log loss却从{M['val_log_loss']:.4f}变为{M['transformed_log_loss']:.4f}；说明排序与概率评分不是同一个量。不能仅由这项损失差把改善全部归因于校准。")
    t.evidence('C3')
    smallest=min(T['C3']['rows'],key=lambda r:r[1]);t.p('最小非空箱：'+str(smallest)+f'；平均预测概率减实际阳性比例={smallest[3]-smallest[4]:.4f}。小箱频率随机性很大；观察差异不能直接证明总体校准/失校准，亦不能由分类全对推出概率精确。')
    t.p('可接受报告：混淆矩阵与敏感度/特异度/precision，加明确阈值；ROC与PR及阳性比例、积分方式；log loss和带人数的校准表。Brier同时受多个概率预测性质影响，不是纯校准误差。AP与梯形PR面积相等只是此例，不能混名。')
    t.new('Q6 假设 泛化 与综合参考')
    pos=[r for r in T['D1']['rows'] if r[0]==0];joint=sum(r[3] for r in pos if r[1:3]==[1,1]);a=sum(r[3] for r in pos if r[1]==1);b=sum(r[3] for r in pos if r[2]==1)
    t.p(f'阴性类：联合={joint}/37={joint/37:.4f}；边缘乘积=({a}/37)×({b}/37)={a*b/37**2:.4f}。经验频率不等，提示乘积假设不符合该有限表的联合分布；不是总体独立性检验。若对照阳性类，会发现尿意频繁恒为1导致经验相等；也不能由此证明所有症状总体条件独立。')
    t.p(f"复制恶心不增加信息，但乘积模型对该条件概率再计一次权重。64种向量有{M['duplicate_class_changes']}种跨0.50类别边界，最大概率变化{M['duplicate_max_change']:.4f}；它们不是观测患者。朴素模型训练仅需计数，成本随记录×特征近似线性；小模型参数可追溯，但依赖可能影响概率，解释不等于因果。")
    t.p('表 D3仅支持固定教学分割上24条全部正确；这些二值组合都在训练出现，不证明新组合、真实新患者、跨机构泛化。重复组合本身不是泄漏的充分证据；来源、重复行、无患者身份共同限制独立性主张。作者曾看test，不包装成全新盲测。')
    t.p('表 D4：患病100、无病9900；TP95、FN5、FP495、TN9405；precision=95/590≈16.10%。此假设模型accuracy=9500/10000=95%；全阴性基线accuracy=99%，却漏掉全部100名患者。说明错误结构、阳性组成及使用代价不能被一个accuracy取代；这些是假设外推，不是本课数据性能。')
    t.p('综合备忘录评分不限定A/B/C。A可保留为方法学基线且明确适用范围；B须指出实际修改及仅用训练/验证重新选配置，不能声称未测的新模型更好；C须指定待补证据及其改判作用。三者均需纳入Q1至Q6，不允许靠“不能临床”跳过学习。')
    t.new('课堂训练答案与评分')
    t.p('二症状模型先验仍35/72与37/72。两症状θ按表 B1各自计数加一，不能使用验证计数。删除四个症状改变了模型，必须重新计算乘积；不能把六症状概率当作答案。')
    t.table(['尿意频繁','排尿痛','真实标签','记录数','阳性后验','预测类别'],[[*r[:4],f'{r[4]:.6f}',r[5]] for r in M['two_results']])
    tn,fp=M['two_confusion'][0];fn,tp=M['two_confusion'][1]
    t.p(f'合计TN={tn}、FP={fp}、FN={fn}、TP={tp}；sensitivity={tp}/{tp+fn}={tp/(tp+fn):.4f}；specificity={tn}/{tn+fp}={tn/(tn+fp):.4f}。聚合表按记录数加权，不按表格行计数。')
    t.table(['评分项','权重','达标证据及部分分'],[['任务与流程','20%','Q1 8分：变量与预测对象；Q2 12分：数据职责、冻结与测试用途。'],['NB训练与预测','25%','Q3 15分：先验、平滑、0值因子、归一化与文本迁移；课堂10分：亲自拟合与加权评价。'],['损失理解','15%','Q4交叉熵比较8分、MSE与单位4分、证据卡3分。'],['分类 排序 概率','25%','Q5分类分母10分、阈值理由5分、AUC/PR5分、校准及人数5分。'],['假设与综合方案','15%','Q6条件独立与重复信息4分、迁移反例4分、六模块整合及边界7分。']],[1.35,.65,4.96])
    t.p('算术失误但分母/方法正确给方法分；不因选C自动加分，不因100%分数加分。优秀答案能主动引用反向证据，并说明改变判断的条件。抄概念而没有表中数字不能满足对应案例分析项。')
    t.new('教师归纳与实施风险')
    t.p('5分钟归纳只抓三组混淆：训练与预测及验证选择；P(症状|疾病)与P(疾病|症状)及条件独立；分类、排序、概率与目标人群。先用学生的具体错误追问，再给最小必要解释。')
    t.p('工作量：120分钟是强生物背景学生、允许计算器并只阅读指定短段的设计预算，不是实测保证。若英文阅读卡住，先读本册概念页；若算乘积慢，让学生先保留分数因子再用计算器。不要添加肾炎重训、置信区间推导等必做任务。')
    t.p('分阶段技术风险：完整Notebook从头执行会泄露D；课程平台按Packet分发worksheet页面并注明只运行对应单元。教师持有全册，学生最终保存合并个人作答版。质量检查及红字试填文件仅教师可见，不上传给学生。')
    t.p('若真实课堂出现大面积超时，应记录实际用时并减少重复抄写或把同伴回应移至课后，不静默删掉源大纲目标。真实学生试教仍是后续应做的检验；AI模拟不代替这一证据。')
    t.save('teaching-plan-flipped-classroom')

    r=Writer('教学设计说明');r.page=1;r.title('Purpose Design Rationale Expected Outcomes')
    r.h('为什么重构这一课')
    r.p('这批学生有生物学、微积分和数理统计基础，但没有稳定的算法背景。直接讲大量指标公式会让学习退化为记名词；只让Notebook跑出高分又掩盖了输入、参数、阈值和使用范围之间的区别。本课要求第一次形成能被另一位研究者复核的预测研究方案。')
    r.h('为什么采用这组问题')
    r.p('主问题是下一轮离线方法比较的基线决策，允许保留、修改或暂缓；它不能被一个分数或“不能临床”的结论完成。六模块将模型、训练流程、Bayes计算、损失、评价及假设连接到同一张追踪表。难度来自分母、计算与证据推断，不依赖工程技能。')
    r.table(['模块','覆盖簇','最终方案中的责任'],[['Q1','数据模型 特征 标签','明确研究对象与输入输出'],['Q2','训练 预测 泛化 划分','给出可复核的数据职责'],['Q3','Bayes Bernoulli 文本迁移','解释参数从何而来及预测过程'],['Q4','损失 交叉熵 MSE','说明概率与连续误差怎样衡量'],['Q5','分类 排序 校准','形成不混淆的评价报告'],['Q6','条件独立 优缺点 迁移','限定主张并给出下一步']],[.6,3,3.36])
    r.p('原大纲课堂训练与sensitivity/specificity计算保留在翻转课堂前15分钟；原题“解释为什么医学诊断模型不能只看 accuracy”保留为Q6数值迁移与备忘录的一部分。前六讲仅作为数据/统计基础，不借本课加入后续验证课程的完整内容。')
    r.new('数据和分析选择')
    r.p('保留用户已选UCI Acute Inflammations，便于六项症状的手算与可解释训练。主标签为膀胱炎症，不把另一疾病决策偷当输入。其来源是专家设计潜在病例，非可确认的独立患者队列；这使它适合算法教学，却不能承担临床效果证明。')
    r.p('120条保留原始重复；固定72/24/24分层与种子7，不选择漂亮划分。预设38℃把六输入统一为Bernoulli，原始99种输入变为9种组合，代价在D公开审查。测试作者已见，透明称固定教学测试；课堂冻结是教学流程，不是假装全新盲测。')
    r.p('为了让完整大纲在一个数据背景中可做：B4用四条合成概率对照呈现accuracy与交叉熵差异；B5以两条连续体温目标使MSE实际可算；C使用验证阈值0.10/0.50/0.90制造的是同数据上的真实取舍，不改标签。严格递增概率变换说明排序不等于概率质量。D4的1%患病率反例明确是假设场景，不当作模型外部验证。')
    r.p('不把有限经验频率的比较当作总体独立性证明：D1必做计算选阴性类，联合频率与边缘乘积不同；阳性类存在常量症状，可供教师说明相等也不是普遍证明。D2复制一列展示朴素乘积对冗余信息的敏感性；64种向量是设计空间，不是患者样本量。')
    r.h('预期学习压力与支架')
    r.p('不给推导优化算法的负担，但保留先验、平滑、0值因子与后验归一化；给指标公式却要求解释具体分母；给AUC结果却要求算点坐标与说明边界；给原始频数而不把判断写进证据。可用AI，必须用证据卡核查一个案例主张。')
    r.new('覆盖链与考核')
    for cov_i,cov in enumerate(design['course_coverage']):
        if cov_i in [5,10]:r.new('覆盖链与考核续页')
        r.h(cov['objective_id']+' '+','.join(cov['inquiry_question_ids']),3)
        r.p('来源要求：'+cov['source_excerpt'])
        r.p('可观察产出：'+cov['observable_outcome']+'；证据：'+'、'.join(cov['evidence_ids'])+'；最终使用：'+cov['synthesis_use'])
    r.new('阅读设计与质量验收')
    r.p('沿用课程已有worksheet的浅蓝表头、Arial文字与0.65英寸页边距；按本次批准改为A4，中文用微软雅黑10.5pt，表格10pt，极宽原始数据表9pt。短题卡直接置于可编辑答案表之前；追踪表仅一份，行名与模块引用一致。独立分页允许教师按阶段发放，避免Notebook变成唯一教材。')
    r.p('质量门：分析设计逐目标覆盖检查与worksheet lint应为0错误；冷读检查每个指代与选项；新读者只看学生文件逐格试填，记录估计时间与阻塞，修订后复查。四份DOCX均转PDF/页面图检查；Notebook离线执行验证核心数字。实际检查报告存review，不能以模板存在代替通过。')
    r.p('本课不声称：真实患者诊断有效、跨医院泛化、概率风险已校准、观察性关联具有因果意义、AI试填能替代真实学生试教。精确的内部成绩和数字重算只证明固定教学流程可复核。')
    r.p('迁移时，学生应先换对象与标签、重审预测时可用输入、明确数据职责，再选择损失与报告指标，最后核查模型假设和新环境的证据。不能把本课38℃切点、0.50阈值或50%验证阳性比例原样移植。')
    r.p('原大纲来源：D:/XMU/AI4BiomedicalDS/plan.docx第7讲；课程专属批准设计及逐目标覆盖保存在本课planning和assets。未修改pbl-ml-package技能文件。')
    r.save('design-rationale')
    write_notebook()
    manifest={'topic':TITLE,'slug':'machine-learning-basics-naive-bayes','mode':'PBL + flipped classroom','dataset':{'name':'UCI Acute Inflammations','source':'https://doi.org/10.24432/C5V59S','license':'CC BY 4.0','task':'binary teaching classification','target':'bladder_inflammation'},'outputs':{k:f'{SLUG}-{k}.docx' for k in ['student-worksheet','evidence-pack','teaching-plan-flipped-classroom','design-rationale']},'boundary':'Teaching only; no clinical, deployment, causal or patient-care claim.','core_question':design['problem_architecture']['overarching_problem'],'problem_architecture':design['problem_architecture'],'course_coverage':design['course_coverage'],'customized':True,'course_brief':'../../../../plan.docx','analysis_design':f'assets/{SLUG}/analysis-design.json','build_script':f'scripts/{SLUG}-build.py','analysis_script':'scripts/lesson7_analysis.py','assets':[str(p.relative_to(ROOT)).replace('\\','/') for p in ASSET.iterdir() if p.is_file()], 'qa':{'lint':f'review/{SLUG}-lint.md','design':'review/design-check.md','pilot_fill':f'review/{SLUG}-pilot-fill.docx','pilot_report':f'review/{SLUG}-pilot-report.md','release_status':'pending final checks'}}
    manifest['outputs']['notebook']=f'notebook/{SLUG}-lab.ipynb'
    active_assets={'acute-inflammations.csv','analysis-design.json','computed-evidence.json','dataset-summary.csv','split-indices.csv','validation-predictions.csv','redundancy-all64.csv','packet-c2-ranking.png',*[f'packet-{label.lower()}.csv' for label in T],*[f'equation-{x}.png' for x in ['bayes','bernoulli','loss','mse']]}
    manifest['assets']=[f'assets/{SLUG}/{name}' for name in sorted(active_assets)]
    manifest['legacy_note']='Older unlisted assets are retained for recoverability, not part of this release. Prior notebooks and generator are preserved under planning/archive-before-20260917.'
    (ROOT/'package-manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')

def write_notebook():
    import nbformat as nbf
    nb=nbf.v4.new_notebook();cells=[]
    def md(x):cells.append(nbf.v4.new_markdown_cell(x))
    def code(x):cells.append(nbf.v4.new_code_cell(x))
    md('# 第7讲 机器学习基础与朴素贝叶斯 辅助实验\n核心学习与答案写在Word worksheet。Notebook只供重现和核对；不需要从零编程。按已发Packet执行，不要提前运行D；完整执行仅供教师复现。全部数据离线可用，禁止临床应用。')
    code("from pathlib import Path\nimport sys, json, itertools\nimport numpy as np\nimport pandas as pd\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.naive_bayes import BernoulliNB\nfrom sklearn.metrics import confusion_matrix, log_loss, mean_squared_error, roc_auc_score, precision_recall_curve, average_precision_score, auc, brier_score_loss\nroot = Path.cwd().resolve()\nif root.name == 'notebook': root = root.parent\nassert (root/'assets').exists(), '请从本课目录或notebook目录运行'\nassets = root/'assets/pbl-machine-learning-basics-naive-bayes'\ndf = pd.read_csv(assets/'acute-inflammations.csv')\nprint('Python:', sys.executable)\nprint(df.shape)\ndisplay(df.head())")
    md('## Packet A Q1 记录与表示\n一行不是已确认独立患者。核对层级、缺失、重复与连续/二值表示；不要静默去重。')
    code("assert df.shape == (120,8)\nassert df.isna().sum().sum() == 0\ndisplay(df.describe(include='all'))\nprint('全行重复余量:', df.duplicated().sum(), '重复组总行:', df.duplicated(keep=False).sum())\nprint('原始输入组合:', df.iloc[:,:6].drop_duplicates().shape[0])\nX = np.column_stack([(df.iloc[:,0]>=38).astype(int), *(df[c].eq('yes').astype(int) for c in df.columns[1:6])])\ny = df['bladder_inflammation'].eq('yes').to_numpy(dtype=int)\nprint('二值组合:', np.unique(X,axis=0).shape[0], '类别计数:', np.bincount(y))")
    md('## Q2 固定数据职责\n先保留20%测试，再从剩余80%中取25%为验证，总比例60/20/20。不根据分数换种子。')
    code("dev, te = train_test_split(np.arange(len(y)), test_size=.2, random_state=7, stratify=y)\ntr, va = train_test_split(dev, test_size=.25, random_state=7, stratify=y[dev])\nassert not (set(tr)&set(va) or set(tr)&set(te) or set(va)&set(te))\nfor name, ix in [('训练',tr),('验证',va),('测试',te)]: print(name, len(ix), np.bincount(y[ix]))\nsaved=pd.read_csv(assets/'split-indices.csv')\nfor label,ix in [('tr',tr),('va',va),('te',te)]: assert saved.loc[saved.split==label,'source_row'].tolist()==(ix+1).tolist()")
    md('## Packet B Q3 用计数训练\n模型估计每类中各症状为1的概率，加一平滑；0症状使用补概率。先手算worksheet查询1，再运行核对。')
    code("counts=np.array([X[tr][y[tr]==c].sum(axis=0) for c in [0,1]])\nn=np.bincount(y[tr]); theta=(counts+1)/(n[:,None]+2); priors=n/n.sum()\nmodel=BernoulliNB(alpha=1).fit(X[tr],y[tr])\nassert np.allclose(theta,np.exp(model.feature_log_prob_))\nq=np.array([0,0,0,1,1,0])\nfactors=np.where(q,theta,1-theta)\nscores=priors*factors.prod(axis=1); post=scores/scores.sum()\nprint('counts',counts,'theta',theta,'priors',priors,sep='\\n')\nprint('因子',factors,'类别分数',scores,'后验',post,sep='\\n')\nassert np.allclose(post,model.predict_proba([q])[0])\nsaved_counts=pd.read_csv(assets/'packet-b1.csv'); assert np.array_equal(saved_counts.iloc[:,1:].to_numpy(),counts[[1,0]].T)")
    md('反思：为什么查询中0值的症状仍要参与乘积？若把关键词出现改成频次，这个式子改变了什么？')
    md('## Q4 两种损失\n以下是明确给定输入的合成教学例，不是额外患者。')
    code("toy_y=np.array([1,0,1,0]); pa=np.array([.8,.2,.4,.1]); pb=np.array([.99,.01,.01,.01])\nfor name,p in [('甲',pa),('乙',pb)]:\n    per=-(toy_y*np.log(p)+(1-toy_y)*np.log(1-p))\n    print(name,'逐例损失',per,'平均',per.mean(),'accuracy',((p>=.5)==toy_y).mean())\n    assert np.isclose(per.mean(),log_loss(toy_y,p))\nprint('MSE',mean_squared_error([37,39],[38,37]),'℃²')\nassert mean_squared_error([37,39],[38,37])==2.5")
    md('## Packet C Q5 分类、排序与概率\n所有选阈值动作只使用验证；先写冻结配置再运行D。')
    code("p=model.predict_proba(X[va])[:,1]\nrows=[]\nfor threshold in [.1,.5,.9]:\n    tn,fp,fn,tp=confusion_matrix(y[va],p>=threshold,labels=[0,1]).ravel()\n    rows.append([threshold,tn,fp,fn,tp,(tp+tn)/len(va),tp/(tp+fn),tn/(tn+fp),tp/(tp+fp),2*tp/(2*tp+fp+fn)])\nmetric_table=pd.DataFrame(rows,columns=['threshold','TN','FP','FN','TP','accuracy','sensitivity','specificity','precision','F1'])\ndisplay(metric_table)\nassert np.array_equal(metric_table.iloc[:3,1:5].to_numpy(),pd.read_csv(assets/'packet-c1.csv').iloc[:3,1:].to_numpy())\nprint('先验基线precision分母为0：未定义；recall=0。')")
    code("import matplotlib.pyplot as plt\nfrom sklearn.metrics import roc_curve\nfpr,tpr,_=roc_curve(y[va],p); precision,recall,_=precision_recall_curve(y[va],p)\nfig,ax=plt.subplots(1,2,figsize=(9,3))\nax[0].plot(fpr,tpr);ax[0].set(xlabel='False positive rate',ylabel='Sensitivity',title='Validation ROC')\nax[1].plot(recall,precision);ax[1].set(xlabel='Recall',ylabel='Precision',title='Validation PR');plt.show()\nfor name,prob in [('original',p),('0.25+0.50p',.25+.5*p)]:\n    pp,rr,_=precision_recall_curve(y[va],prob)\n    print(name, 'ROC',roc_auc_score(y[va],prob),'PR trapezoid',auc(rr,pp),'AP',average_precision_score(y[va],prob),'logloss',log_loss(y[va],prob))\n# 非完美排序支持例，说明AP与梯形PR面积一般不等\npp,rr,_=precision_recall_curve([1,0,1,0],[.9,.8,.7,.1])\nprint('Optional illustration: AP',average_precision_score([1,0,1,0],[.9,.8,.7,.1]),'PR trapezoid',auc(rr,pp))")
    code("cal=[]\nfor lo,hi in [(0,.1),(.1,.5),(.5,.9),(.9,1)]:\n    mask=(p>=lo)&((p<hi) if hi<1 else (p<=hi))\n    if mask.sum(): cal.append([lo,hi,int(mask.sum()),int(y[va][mask].sum()),p[mask].mean(),y[va][mask].mean()])\ncal=pd.DataFrame(cal,columns=['lower','upper','n','positive','mean_p','observed_rate']);display(cal)\nsaved_cal=pd.read_csv(assets/'packet-c3.csv')\nassert np.allclose(cal.iloc[:,2:].to_numpy(),saved_cal.iloc[:,1:].to_numpy(),atol=5e-5)\nprint('Brier',brier_score_loss(y[va],p),'（不是纯校准误差）')")
    md('反思：什么性质被单调变换保留？每箱人数怎样限制校准结论？')
    md('## Packet D Q6 条件独立与复制输入\n此处仅在D解封后运行。64种穷举向量不等于64患者。')
    code("joint=[]\nfor c in [1,0]:\n    for a,b in itertools.product([0,1],repeat=2):\n        joint.append([c,a,b,int(((y[tr]==c)&(X[tr,3]==a)&(X[tr,4]==b)).sum())])\ndisplay(pd.DataFrame(joint,columns=['class','urine','pain','n']))\nassert np.array_equal(np.array(joint),pd.read_csv(assets/'packet-d1.csv').values)\ngrid=np.array(list(itertools.product([0,1],repeat=6)))\ndup=BernoulliNB(alpha=1).fit(np.column_stack([X[tr],X[tr,1]]),y[tr])\npg=model.predict_proba(grid)[:,1];pdg=dup.predict_proba(np.column_stack([grid,grid[:,1]]))[:,1]\nprint('类别改变',((pg>=.5)!=(pdg>=.5)).sum(),'最大概率差',abs(pg-pdg).max())\nassert np.allclose(np.column_stack([grid,pg,pdg]),pd.read_csv(assets/'redundancy-all64.csv').to_numpy())")
    code("testp=model.predict_proba(X[te])[:,1]\nprint('固定测试',confusion_matrix(y[te],testp>=.5,labels=[0,1]),'log loss',log_loss(y[te],testp))\ntrain_patterns=set(map(tuple,X[tr]));raw=df.iloc[:,:6].astype(str).to_numpy();raw_train=set(map(tuple,raw[tr]))\nfor name,ix in [('validation',va),('test',te)]:\n    print(name,'binary patterns',len(set(map(tuple,X[ix]))),'seen',sum(tuple(v) in train_patterns for v in X[ix]),'raw seen',sum(tuple(v) in raw_train for v in raw[ix]))\nassert np.array_equal(confusion_matrix(y[te],testp>=.5),[[12,0],[0,12]])\nTP,FN,FP,TN=95,5,495,9405\nprint('假设低患病率precision',TP/(TP+FP),'accuracy',(TP+TN)/10000,'all negative accuracy',.99)")
    md('## 翻转课堂 二症状训练复现\n仍从训练72条拟合，仅保留尿意频繁与排尿痛；验证24条按人数加权。')
    code("two=BernoulliNB(alpha=1).fit(X[tr][:,[3,4]],y[tr])\nclassroom=pd.read_csv(assets/'packet-b3.csv')\nclassroom['posterior']=two.predict_proba(classroom.iloc[:,:2].to_numpy())[:,1]\nclassroom['prediction']=(classroom['posterior']>=.5).astype(int);display(classroom)\ncm=confusion_matrix(y[va],two.predict(X[va][:,[3,4]]),labels=[0,1]); print(cm)\nweighted=np.zeros((2,2),dtype=int)\nfor row in classroom.itertuples(index=False,name=None): weighted[int(row[2]),int(row[5])]+=int(row[3])\nassert np.array_equal(cm,weighted)\ntn,fp,fn,tp=cm.ravel();print('sensitivity',tp/(tp+fn),'specificity',tn/(tn+fp))\nprint('核心复现检查全部通过。')")
    md('最终反思：把任务、数据职责、训练、损失、分类/排序/校准和假设写进同一研究备忘录。选择A、B、C都需证据；不要把固定教学测试当成真实部署证据。')
    nb.cells=cells;nb.metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3'}}
    nbf.write(nb,ROOT/'notebook'/f'{SLUG}-lab.ipynb')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--docs',action='store_true');parser.add_argument('--all',action='store_true');args=parser.parse_args()
    REVIEW.mkdir(exist_ok=True)
    if args.all: subprocess.run([sys.executable,str(ROOT/'scripts/lesson7_analysis.py')],check=True)
    if args.docs or args.all: build_docs()
    else: parser.error('use --docs after numerical generation, or --all in a compatible environment')
