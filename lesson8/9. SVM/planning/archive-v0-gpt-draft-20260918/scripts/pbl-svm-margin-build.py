"""Lesson 9 only. Offline, deterministic analysis and four teaching documents.
Run from this lesson folder with Python containing numpy/pandas/sklearn/matplotlib/docx.
All outputs are confined to this course folder. Source data are bundled alongside.
"""
from pathlib import Path
import argparse, json, sys, subprocess, hashlib, math, platform, inspect
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
import sklearn
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT, WD_ROW_HEIGHT_RULE

ROOT=Path(__file__).resolve().parents[1]
AS=ROOT/'assets'/'pbl-svm-margin'
REVIEW=ROOT/'review'
SLUG='pbl-svm-margin'
SKILL=ROOT.parents[2]/'.codex'/'skills'/'pbl-ml-package'
FONT='Microsoft YaHei'
plt.rcParams.update({'font.sans-serif':[FONT,'DejaVu Sans'],'axes.unicode_minus':False,'font.size':10})
SOURCE='https://archive.ics.uci.edu/dataset/212/vertebral%2Bcolumn'
SVMREF='https://scikit-learn.org/stable/modules/svm.html'
RBFREF='https://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html'
CN=['骨盆入射角','骨盆倾斜角','腰椎前凸角','骶骨倾角','骨盆半径','滑脱程度']
FEATURES=['pelvic_incidence','pelvic_tilt','lumbar_lordosis_angle','sacral_slope','pelvic_radius','degree_spondylolisthesis']
OPTIONS=[('L','采用标准化线性核 SVM，冻结验证阶段选定的 C。'),('P','采用标准化二次多项式核 SVM，冻结 C 与核设置。'),('R','采用标准化 RBF 核 SVM，冻结 C 与 gamma。'),('D','暂缓将 SVM 纳入比较，先补足所指出的证据或修改方案。')]
PROBLEM='脊柱生物力学研究组要在下一轮离线算法比较中加入一个 SVM 基线，用六项测量预测已有的正常或异常标签，供研究人员比较方法。请在方案 L、P、R、D 中选择一个，用间隔几何、支持向量、软间隔、核选择和未见记录的证据说明选择；若验证表现不足或适用条件改变，指出应暂缓或修改的具体原因。'
OBJECTIVES=[
('LO1','找到一个分类超平面；什么是 margin？','计算得分与归一化距离，区分样本距离和规范化间隔带宽。','几何与泛化'),
('LO2','为什么最大化 margin 有助于泛化？margin 越大，模型对扰动越稳健','用固定边界的小扰动分析解释几何保障，并限制对未见人群的推断。','几何与泛化'),
('LO3','Support vectors：决定分类边界的关键样本','比较移除不同样本后的边界变化，解释支持向量的样本身份。','支持向量'),
('LO4','Soft margin：允许少量分类错误；通过参数 C 控制误差和 margin 的权衡','计算 hinge loss，比较 C 路径中的带宽、违例和训练错误。','软间隔'),
('LO5','kernel trick 是什么？linear kernel；polynomial kernel；RBF kernel','手算二次核内积并比较三种核的二维边界，区分 C 与 gamma。','核方法'),
('LO6','参数敏感','用尺度对照和验证网格冻结核与参数，识别调参依据。','参数选择'),
('LO7','适合中小样本；在高维数据中表现好；大数据上训练慢；可解释性有限','对 n、p、计算规模与解释需求不同的场景给出有条件的适用判断。','适用范围'),
('LO8','作业 用自己的话解释：为什么 SVM 不是简单地找一条“刚好分开两类”的线？','整合边界、间隔、支持向量、容错与验证证据完成原题。','综合解释')]
QS=[
dict(id='Q1',question='表 A2 的两条分界线都能分开六个模拟样本，哪条更能容忍同样大小的测量扰动？',objective_ids=['LO1','LO2'],concept_ids=['C1'],required_learning=['加权和、向量长度与垂直距离','带符号间隔与固定边界下的扰动界'],student_action='计算两种边界的最小距离，并比较重复缩放得分后的距离。',evidence_ids=['表 S1','图 S1','表 A1','表 A2'],release_after='A',minutes=17,tracker_row='T1 几何依据',student_output='Q1 距离计算表及 T1',contribution_to_synthesis='说明选择边界的几何依据及泛化边界'),
dict(id='Q2',question='表 B1 中移除远点与近点后，哪些变化说明训练样本在边界中的作用不同？',objective_ids=['LO3'],concept_ids=['C2'],required_learning=['支持向量是样本，不是特征','支持向量由拟合和参数共同决定'],student_action='对照移除记录与边界位置，区分有可能改变和必然改变。',evidence_ids=['表 A2','表 B1'],release_after='B',minutes=10,tracker_row='T2 关键样本',student_output='Q2 样本影响解释及 T2',contribution_to_synthesis='解释选择方案依赖哪些训练样本'),
dict(id='Q3',question='表 B2 中改变 C 会怎样改变二维模型的间隔与违例，宽带是否足以支持方案 L？',objective_ids=['LO4'],concept_ids=['C3'],required_learning=['软间隔目标与 hinge loss','违例与分类错误的区别'],student_action='计算四个 hinge loss，比较同一训练集的 C 路径。',evidence_ids=['图 S1','表 B2','图 B1'],release_after='B',minutes=15,tracker_row='T3 容错权衡',student_output='Q3 损失计算与 C 比较及 T3',contribution_to_synthesis='解释参数与容错选择'),
dict(id='Q4',question='图 C1 的三种核如何改变模拟细胞形态边界，二次核内积为什么不必显式升维？',objective_ids=['LO5'],concept_ids=['C4'],required_learning=['内积、显式映射与核函数','RBF 相似度与 gamma'],student_action='核验二次核与三维映射内积一致，比较三核边界。',evidence_ids=['图 S1','图 C1','表 C1'],release_after='C',minutes=20,tracker_row='T4 核的理由',student_output='Q4 核计算与三核比较及 T4',contribution_to_synthesis='用数学与图形解释核选择'),
dict(id='Q5',question='表 C2 和图 C2 的脊柱结果是否足以冻结方案 L、P 或 R，尺度变化会怎样改变判断？',objective_ids=['LO6'],concept_ids=['C5'],required_learning=['训练内标准化','验证选参、测试隔离和 balanced accuracy'],student_action='比较相同参数的尺度对照和不同参数的验证表现，冻结一个模型或暂缓。',evidence_ids=['表 A1','表 C2','图 C2'],release_after='C',minutes=13,tracker_row='T5 冻结方案',student_output='Q5 参数比较及 T5',contribution_to_synthesis='指定可复现的方案而非泛泛推荐 SVM'),
dict(id='Q6',question='表 D1 的新记录结果和表 D2 的研究规模，允许你保留已冻结方案并迁移到哪些任务？',objective_ids=['LO2','LO7','LO8'],concept_ids=['C1','C6'],required_learning=['内部泛化与几何稳健性不同','中小样本、高维、核计算和解释边界'],student_action='读测试分母、比较固定模型扰动，判断两种迁移场景并回答总纲作业。',evidence_ids=['表 D1','图 D1','表 D2'],release_after='D',minutes=22,tracker_row='T6 适用范围',student_output='Q6 迁移判断、原题作答及 T6',contribution_to_synthesis='限制适用范围并整合所有核心概念')]

def dump(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def savefig(name):
    plt.savefig(AS/name,dpi=190,bbox_inches='tight',facecolor='white'); plt.close()

def score(model,X,y):
    pr=model.predict(X); tn,fp,fn,tp=confusion_matrix(y,pr,labels=[0,1]).ravel()
    return dict(errors=int((pr!=y).sum()),ba=float(balanced_accuracy_score(y,pr)),tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp),sensitivity=float(tp/(tp+fn)),specificity=float(tn/(tn+fp)))

def fit_svm(X,y,kernel,C,gamma=1/6,scaled=True):
    s=SVC(kernel=kernel,C=C,gamma=gamma,degree=2,coef0=1,tol=1e-7)
    model=make_pipeline(StandardScaler(),s) if scaled else make_pipeline(s)
    model.fit(X,y); return model

def model_row(model,Xt,yt,Xv,yv,kernel,C,gamma,scaled):
    s=model[-1]; a=score(model,Xt,yt); b=score(model,Xv,yv)
    return dict(kernel=kernel,C=C,gamma=gamma,scaled=scaled,train_errors=a['errors'],train_ba=a['ba'],validation_ba=b['ba'],validation_sensitivity=b['sensitivity'],validation_specificity=b['specificity'],sv=int(s.n_support_.sum()),width=float(2/np.linalg.norm(s.coef_)) if kernel=='linear' else None)

def analyze():
    AS.mkdir(parents=True,exist_ok=True); REVIEW.mkdir(exist_ok=True)
    raw=pd.read_csv(AS/'source-data'/'column_2C.dat',sep=r'\s+',header=None,names=FEATURES+['class'])
    raw3=pd.read_csv(AS/'source-data'/'column_3C.dat',sep=r'\s+',header=None,names=FEATURES+['class'])
    X=raw[FEATURES].to_numpy(); y=(raw['class']=='AB').to_numpy(dtype=int)
    assert raw.shape==(310,7) and raw.isna().sum().sum()==0
    assert np.array_equal(X,raw3[FEATURES].to_numpy())
    assert np.array_equal(y,(raw3['class']!='NO').to_numpy(dtype=int))
    ids=np.arange(310); dev,test=train_test_split(ids,test_size=.2,stratify=y,random_state=9)
    train,val=train_test_split(dev,test_size=.25,stratify=y[dev],random_state=9)
    pd.DataFrame({'row_id':ids+1,'split':np.select([np.isin(ids,train),np.isin(ids,val)],['train','validation'],default='test')}).to_csv(AS/'split.csv',index=False)
    raw.to_csv(AS/'vertebral-column.csv',index=False)
    audit=[]
    for j,n in enumerate(CN):
        audit.append(dict(feature=n,n=310,missing=int(raw[FEATURES[j]].isna().sum()),min=float(X[:,j].min()),max=float(X[:,j].max()),unit='源页未注明单位',zero_count=int((X[:,j]==0).sum())))
    audit=pd.DataFrame(audit);audit.to_csv(AS/'packet-a-audit.csv',index=False)
    summary=dict(samples=310,features=6,normal=100,abnormal=210,missing=0,exact_duplicate_rows=int(raw.duplicated().sum()),duplicate_feature_rows=int(raw[FEATURES].duplicated().sum()),patient_identifier_available=False,train=186,validation=62,test=62,source=SOURCE,license='CC BY 4.0',source_sha256=hashlib.sha256((AS/'source-data'/'column_2C.dat').read_bytes()).hexdigest(),documentation_notes=['源页单位栏为空，不擅自添加单位。','三分类与二分类文件为同一批记录，不合并为620条。','存在负值和极端值；来源未提供合理范围，不把负值当缺失，不自动删除。','未见重复整行不等于已证实患者独立；发布数据没有患者标识可进一步核查。'],pelvic_identity_max_abs_error=float(np.max(abs(X[:,0]-X[:,1]-X[:,3]))))
    pd.DataFrame([dict(samples=310,features=6,normal=100,abnormal=210,missing=0)]).to_csv(AS/'dataset-summary.csv',index=False)
    dump(AS/'data-audit.json',summary)
    # Six completely specified geometric teaching points, not clinical records.
    toy=np.array([[-3,0],[-1,-1],[-1,1],[1,0],[3,-1],[3,1]],float); ty=np.array([0,0,0,1,1,1])
    pd.DataFrame({'id':['N1','N2','N3','P1','P2','P3'],'u':toy[:,0],'v':toy[:,1],'label':ty}).to_csv(AS/'packet-a-six-points.csv',index=False)
    deletions=[]
    for label,remove in [('全部保留',None),('移除 N1',0),('移除 P1',3)]:
        keep=np.ones(6,dtype=bool)
        if remove is not None:keep[remove]=False
        m=SVC(kernel='linear',C=1e4,tol=1e-9).fit(toy[keep],ty[keep]); w=m.coef_[0]; b=m.intercept_[0]
        deletions.append(dict(operation=label,remaining=int(keep.sum()),boundary_u=float(-b/w[0]),width=float(2/np.linalg.norm(w)),support_ids=','.join(np.array(['N1','N2','N3','P1','P2','P3'])[keep][m.support_])))
    pd.DataFrame(deletions).to_csv(AS/'packet-b-removal.csv',index=False)
    # Two-dimensional simulation for the mandated classroom experiment.
    mx,my=make_moons(n_samples=120,noise=.20,random_state=9)
    md,mt=train_test_split(np.arange(120),test_size=.2,stratify=my,random_state=9)
    mtr,mv=train_test_split(md,test_size=.25,stratify=my[md],random_state=9)
    sc=StandardScaler().fit(mx[mtr]); z=sc.transform(mx)
    pd.DataFrame({'id':np.arange(1,121),'marker_a':mx[:,0],'marker_b':mx[:,1],'label':my,'split':np.select([np.isin(np.arange(120),mtr),np.isin(np.arange(120),mv)],['train','validation'],default='test')}).to_csv(AS/'simulation.csv',index=False)
    smods={}; sr=[]
    for kernel,C,gamma in [('linear',.01,1),('linear',1,1),('linear',100,1),('poly',1,1),('rbf',1,1),('rbf',100,10)]:
        m=SVC(kernel=kernel,C=C,gamma=gamma,degree=2,coef0=1,tol=1e-8).fit(z[mtr],my[mtr]);name=f'{kernel}-{C}-{gamma}';smods[name]=m
        f=(2*my[mtr]-1)*m.decision_function(z[mtr]);sr.append(dict(name=name,kernel=kernel,C=C,gamma=gamma,errors=int((m.predict(z[mtr])!=my[mtr]).sum()),violations=int((f<1-1e-5).sum()),sv=int(m.n_support_.sum()),width=float(2/np.linalg.norm(m.coef_)) if kernel=='linear' else None,validation_ba=float(balanced_accuracy_score(my[mv],m.predict(z[mv])))))
    sim=pd.DataFrame(sr);sim.to_csv(AS/'packet-b-simulation.csv',index=False)
    gx,gy=np.meshgrid(np.linspace(-2.4,2.4,150),np.linspace(-2.6,2.5,150));grid=np.c_[gx.ravel(),gy.ravel()]
    def panel(ax,m,title,sv=False):
        zz=m.decision_function(grid).reshape(gx.shape)
        ax.contourf(gx,gy,zz>0,levels=[-.5,.5,1.5],colors=['#e3edf8','#f9e7db'])
        ax.contour(gx,gy,zz,levels=[-1,0,1],colors=['#6f8195','#101820','#6f8195'],linestyles=['--','-','--'],linewidths=[1,1.6,1])
        for c,mark,col in [(0,'o','#245b8f'),(1,'^','#ac542b')]:
            ix=mtr[my[mtr]==c];ax.scatter(z[ix,0],z[ix,1],s=13,c=col,marker=mark,label=f'模拟类 {c}')
        if sv:ax.scatter(m.support_vectors_[:,0],m.support_vectors_[:,1],s=48,facecolors='none',edgecolors='black',linewidths=.6)
        ax.set(xlabel='模拟形态指标 u（标准化，无量纲）',ylabel='模拟形态指标 v（标准化，无量纲）',title=title,xlim=(-2.4,2.4),ylim=(-2.6,2.5));ax.legend(fontsize=7,loc='upper right');ax.set_aspect('equal')
    fig,axs=plt.subplots(1,3,figsize=(10.8,3.8),constrained_layout=True)
    for ax,key,title in zip(axs,['linear-0.01-1','linear-100-1','rbf-1-1'],['线性核 C=0.01','线性核 C=100','RBF C=1，gamma=1']):panel(ax,smods[key],title)
    savefig('figure-s1.png')
    fig,axs=plt.subplots(1,3,figsize=(10.8,3.8),constrained_layout=True)
    for ax,key,title in zip(axs,['linear-0.01-1','linear-1-1','linear-100-1'],['C=0.01','C=1','C=100']):panel(ax,smods[key],title,True)
    savefig('figure-b1.png')
    fig,axs=plt.subplots(1,3,figsize=(10.8,3.8),constrained_layout=True)
    for ax,key,title in zip(axs,['linear-1-1','poly-1-1','rbf-1-1'],['线性核 C=1','二次多项式 C=1','RBF C=1，gamma=1']):panel(ax,smods[key],title,True)
    savefig('figure-c1.png')
    real=[]; models={}
    for kernel in ['linear','poly','rbf']:
        for C in [.01,.1,1,10,100]:
            for gamma in ([.01,.1,1,10] if kernel=='rbf' else [1/6]):
                m=fit_svm(X[train],y[train],kernel,C,gamma);real.append(model_row(m,X[train],y[train],X[val],y[val],kernel,C,gamma,True));models[(kernel,C,gamma)]=m
    un=fit_svm(X[train],y[train],'rbf',1,.1,False);real.append(model_row(un,X[train],y[train],X[val],y[val],'rbf',1,.1,False))
    results=pd.DataFrame(real);results.to_csv(AS/'packet-c-real-grid.csv',index=False)
    best=[];selected={}
    for kernel in ['linear','poly','rbf']:
        row=results[(results.kernel==kernel)&results.scaled].sort_values(['validation_ba','C','gamma'],ascending=[False,True,True]).iloc[0]
        best.append(row.to_dict());selected[kernel]=models[(kernel,row.C,row.gamma)]
    pd.DataFrame(best).to_csv(AS/'packet-c-selected.csv',index=False)
    piv=results[(results.kernel=='rbf')&results.scaled].pivot(index='gamma',columns='C',values='validation_ba')
    fig,ax=plt.subplots(figsize=(7,3.6),constrained_layout=True);im=ax.imshow(piv,vmin=.5,vmax=1,cmap='Blues',aspect='auto')
    ax.set(xticks=range(len(piv.columns)),xticklabels=[f'{v:g}' for v in piv.columns],yticks=range(len(piv.index)),yticklabels=[f'{v:g}' for v in piv.index],xlabel='C',ylabel='gamma',title='RBF 在 62 条验证记录上的平衡准确率')
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):ax.text(j,i,f'{piv.iloc[i,j]:.3f}',ha='center',va='center',color='black' if piv.iloc[i,j]<.8 else 'white')
    fig.colorbar(im,ax=ax,label='平衡准确率');savefig('figure-c2.png')
    # Test used once for a fixed, predeclared comparison; no re-fit or re-selection.
    tests=[]
    for kernel,m in selected.items():tests.append(dict(kernel=kernel,**score(m,X[test],y[test])))
    td=pd.DataFrame(tests);td.to_csv(AS/'packet-d-test.csv',index=False)
    # Change only pelvic radius; do not break PI = PT + SS by independent angular noise.
    stress=[];rng=np.random.default_rng(91);eps=rng.normal(size=(200,len(test)))
    sd=X[train,4].std(ddof=0)
    for kernel,m in selected.items():
        clean=m.predict(X[test])
        for size in [0,.1,.2]:
            vals=[]
            for e in eps:
                pert=X[test].copy();pert[:,4]+=size*sd*e;pr=m.predict(pert)
                vals.append([balanced_accuracy_score(y[test],pr),float(np.mean(pr!=clean))])
            a=np.array(vals);stress.append(dict(kernel=kernel,sd_fraction=size,mean_ba=float(a[:,0].mean()),sd_ba=float(a[:,0].std()),flip=float(a[:,1].mean()),repetitions=200))
    st=pd.DataFrame(stress);st.to_csv(AS/'packet-d-stress.csv',index=False)
    fig,axs=plt.subplots(1,2,figsize=(8,3.2),constrained_layout=True)
    for k,label,col in [('linear','线性','#245b8f'),('poly','二次多项式','#76518e'),('rbf','RBF','#ac542b')]:
        r=st[st.kernel==k];axs[0].plot(r.sd_fraction,r.mean_ba,'o-',label=label,color=col);axs[1].plot(r.sd_fraction,r.flip,'o-',label=label,color=col)
    for ax in axs:ax.set_xlabel('骨盆半径噪声标准差 / 训练标准差');ax.legend(fontsize=8);ax.grid(alpha=.2)
    axs[0].set(ylabel='平均平衡准确率',ylim=(.45,1),title='固定模型的分类表现');axs[1].set(ylabel='相对无扰动预测的翻转比例',ylim=(0,.3),title='固定模型的预测变化');savefig('figure-d1.png')
    pd.DataFrame([dict(scenario='脊柱主案例',n=310,p=6,full_gram_mb=310**2*8/1e6),dict(scenario='基因表达迁移',n=80,p=2000,full_gram_mb=80**2*8/1e6),dict(scenario='多中心大表迁移',n=100000,p=6,full_gram_mb=100000**2*8/1e6)]).to_csv(AS/'packet-d-scale.csv',index=False)
    dump(AS/'environment.json',dict(python=platform.python_version(),numpy=np.__version__,sklearn=sklearn.__version__,seed=9,measurement_seed=91))
    return dict(raw=raw,audit=audit,summary=summary,toy=toy,deletions=deletions,sim=sim,grid=results,best=best,test=td,stress=st)

def make_design():
    items=[('表 S1','S1','dataset-summary.csv','主案例 310 条记录，六项特征；启动阶段只有规模和任务事实。'),('图 S1','S1','figure-s1.png','模拟样本120个，图示72个训练点；固定模拟输入随 simulation.csv 发布。'),('表 A1','A','packet-a-audit.csv','六个变量，每变量310个非缺失值，源页单位未注明。'),('表 A2','A','packet-a-six-points.csv','六个完全指定的模拟点，无量纲。'),('表 B1','B','packet-b-removal.csv','三次固定参数的移除试验。'),('表 B2','B','packet-b-simulation.csv','同72个模拟训练点，不同C的误差、违例与带宽。'),('图 B1','B','figure-b1.png','三组线性C边界，圈标支持向量。'),('图 C1','C','figure-c1.png','同72个训练点，三种核。'),('表 C1','C','packet-b-simulation.csv','三核模型的24个模拟验证点平衡准确率。'),('表 C2','C','packet-c-real-grid.csv','186训练、62验证的预定参数扫描与尺度对照。'),('图 C2','C','figure-c2.png','脊柱RBF验证网格。'),('表 D1','D','packet-d-test.csv','62条测试记录，异常42，正常20；从未用于选参。'),('图 D1','D','figure-d1.png','只扰动骨盆半径，每强度200次，模型保持固定。'),('表 D2','D','packet-d-scale.csv','真实规模与两项设定迁移情境，完整核矩阵大小为算术示例。')]
    analyses=[]
    for q in QS:
        es=[i for i in items if i[0] in q['evidence_ids']]
        arts=sorted({i[2] for i in es})
        extra={'Q1':['dataset-summary.csv','data-audit.json','vertebral-column.csv','split.csv','packet-a-six-points.csv','simulation.csv'],'Q5':['packet-c-selected.csv','environment.json'],'Q6':['packet-d-stress.csv']}.get(q['id'],[])
        analyses.append(dict(id='A'+q['id'][1:],objective_ids=q['objective_ids'],inquiry_question_ids=[q['id']],question=q['question'],domain='similarity_distance' if q['id']=='Q1' else 'model_behavior' if q['id'] in ['Q2','Q3','Q4'] else 'validation_robustness',method=q['student_action'],applicability=q['contribution_to_synthesis'],artifacts=sorted(set(arts+extra)),packet=q['release_after'],interpretation_boundary='几何、模拟和历史内部比较不证明临床或因果结论。',notebook_ref=q['id']))
    coverage=[]
    for lo in OBJECTIVES:
        q=[q for q in QS if lo[0] in q['objective_ids']]
        coverage.append(dict(objective_id=lo[0],concept_cluster_ids=sorted({c for z in q for c in z['concept_ids']}),inquiry_question_ids=[z['id'] for z in q],analysis_ids=['A'+z['id'][1:] for z in q],required_learning=[s for z in q for s in z['required_learning']],student_output='；'.join(z['student_output'] for z in q),worksheet_location=' / '.join('Phase 2 '+z['id'] for z in q),synthesis_use='；'.join(z['contribution_to_synthesis'] for z in q),assessment_evidence=lo[2]))
    d=dict(study_goal='理解并审慎选择SVM边界，完成总纲二维课堂练习及解释作业。',authorization='2026-09-17用户明确要求按最新版技能复核后生成全新课件；直接完成制作，替代再次审批和临时目录流程。',learning_objectives=[dict(id=l[0],source_excerpt=l[1],observable_outcome=l[2]) for l in OBJECTIVES],concept_clusters=[dict(id='C'+str(i+1),name=n,objective_ids=[l[0] for l in OBJECTIVES if l[3]==n],concepts=[n]) for i,n in enumerate(['几何与泛化','支持向量','软间隔','核方法','参数选择','适用范围'])],decision_problem=PROBLEM,driving_question=PROBLEM,entry_type='direct_problem',problem_architecture=dict(overarching_problem=PROBLEM,downstream_use='研究人员下一轮离线算法比较的SVM基线',decision_options=[a+' '+b for a,b in OPTIONS],inquiry_questions=QS),course_coverage=coverage,data_context=dict(dataset_name='UCI Vertebral Column',source=SOURCE,task='binary classification',observation_unit='发布表中一条受试者测量记录，缺少可核查重复受试者的标识',variable_roles=dict(features=FEATURES,metadata=[]),grouping_fields=[],target='AB=1, NO=0'),analyses=analyses,split_plan=dict(required=True,unit='row',rationale='研究泛化与验证选参；186/62/62，分层且固定seed9；无患者ID，独立性不可再核查。'),packets={p:dict(purpose=pur,reveals=rev,withholds=hid) for p,pur,rev,hid in [('A','几何与数据语义',['六点输入','源文档和数据核验'],['移除试验','核参数验证','测试']),('B','支持向量与容错',['移除试验','C路径'],['核比较','真实测试']),('C','核选择和冻结',['三核图','尺度对照','验证网格'],['测试和扰动']),('D','综合和边界',['测试','扰动','规模迁移'],[])]},boundaries=['教学研究，不用于临床诊断、部署或患者护理。','模拟形态坐标不是真实细胞测量。','测试不能用于事后重新选择核。'],evidence_items=[dict(id=i,packet=p,artifacts=[a],context=c) for i,p,a,c in items],syllabus_tasks=[dict(source_excerpt=t,worksheet_location=loc) for t,loc in [('用二维模拟数据展示：不同 C 对分类边界的影响','Phase 1 启动观察，图 S1；Phase 2 Q3 深化'),('linear kernel 与 RBF kernel 的区别','Phase 1 启动观察，图 S1；Phase 2 Q4 深化'),('margin 如何变化','Phase 1 启动观察，图 S1；Phase 2 Q1/Q3'),('用自己的话解释：为什么 SVM 不是简单地找一条“刚好分开两类”的线？','Phase 2 Q6 原题作答；综合仅引用不重复')]],time_budget=dict(phase1=45,phase2=120,phase3=45,module_minutes=sum(q['minutes'] for q in QS),source_card_and_synthesis=23))
    d['concept_clusters'][-1]['objective_ids'].append('LO8')
    dump(AS/'analysis-design.json',d);return d

def font(run,size=10.5,bold=False,color='000000'):
    run.font.name='Arial';run.font.size=Pt(size);run.font.bold=bold;run.font.color.rgb=RGBColor.from_string(color)
    run._element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'),FONT)

def newdoc(kind):
    d=Document();s=d.sections[0];s.page_width=Inches(8.268);s.page_height=Inches(11.693)
    s.top_margin=s.bottom_margin=s.left_margin=s.right_margin=Inches(.65)
    for name,size in [('Normal',10.5),('Title',23),('Subtitle',11),('Heading 1',16),('Heading 2',12),('Heading 3',11),('Caption',9)]:
        st=d.styles[name];st.font.name='Arial';st.font.size=Pt(size);st.font.color.rgb=RGBColor(0,0,0)
        st.element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'),FONT)
        st.paragraph_format.space_after=Pt(5);st.paragraph_format.line_spacing=1.15
        if name.startswith('Heading'):st.paragraph_format.keep_with_next=True;st.paragraph_format.space_before=Pt(10)
    p=d.add_paragraph('第9讲 SVM 与 Margin',style='Title');p.alignment=1
    p=d.add_paragraph(kind,style='Subtitle');p.alignment=1
    p=d.sections[0].footer.paragraphs[0];p.alignment=2
    font(p.add_run('第 '),9);f=OxmlElement('w:fldSimple');f.set(qn('w:instr'),'PAGE');p._p.append(f);font(p.add_run(' 页'),9)
    return d

def para(d,text,style=None):
    p=d.add_paragraph(text,style);p.paragraph_format.keep_together=True;return p

def heading(d,text,level=1,new=False):
    p=d.add_heading(text,level)
    if new:p.paragraph_format.page_break_before=True
    return p

def table(d,headers,rows,widths=None,answer=False,height=.36):
    t=d.add_table(rows=1,cols=len(headers));t.style='Table Grid';t.alignment=WD_TABLE_ALIGNMENT.CENTER;t.autofit=False
    widths=widths or [6.968/len(headers)]*len(headers)
    for i,w in enumerate(widths):t.columns[i].width=Inches(w)
    for c,s in zip(t.rows[0].cells,headers):c.text=str(s)
    for row in rows:
        cells=t.add_row().cells
        for c,v in zip(cells,row):c.text='' if v is None else str(v)
    pr=t._tbl.tblPr;b=OxmlElement('w:tblBorders')
    for side in ['top','left','bottom','right','insideH','insideV']:
        x=OxmlElement('w:'+side);x.set(qn('w:val'),'single');x.set(qn('w:sz'),'4');x.set(qn('w:color'),'D9D9D9');b.append(x)
    pr.append(b)
    for ri,row in enumerate(t.rows):
        trpr=row._tr.get_or_add_trPr();ns=OxmlElement('w:cantSplit');trpr.append(ns)
        if ri==0:trpr.append(OxmlElement('w:tblHeader'))
        if answer and ri>0:row.height=Inches(height);row.height_rule=WD_ROW_HEIGHT_RULE.AT_LEAST
        for ci,c in enumerate(row.cells):
            c.width=Inches(widths[ci]);c.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
            cp=c._tc.get_or_add_tcPr();mar=OxmlElement('w:tcMar')
            for side in ['top','left','bottom','right']:
                el=OxmlElement('w:'+side);el.set(qn('w:w'),'85');el.set(qn('w:type'),'dxa');mar.append(el)
            cp.append(mar)
            if ri==0:
                sh=OxmlElement('w:shd');sh.set(qn('w:fill'),'DCE8F2');cp.append(sh)
            for p in c.paragraphs:
                p.paragraph_format.space_after=Pt(2);p.paragraph_format.space_before=Pt(2);p.paragraph_format.keep_together=True
                if ri==0 or (not answer and len(rows)<=4 and ri<len(rows)):p.paragraph_format.keep_with_next=True
                for run in p.runs:font(run,9.5 if len(headers)>4 else 10,ri==0)
    return t

def answer(d,prompt,rows,headers=('填写项目','个人作答'),widths=(1.3,5.668),height=.55):
    p=para(d,prompt);p.paragraph_format.keep_with_next=True
    return table(d,list(headers),rows,list(widths),True,height)

def caption(d,label,text):
    p=para(d,label+'  '+text,'Caption');p.paragraph_format.keep_with_next=True

def picture(d,name,label,text,width=6.8):
    caption(d,label,text);p=d.add_paragraph();p.add_run().add_picture(str(AS/name),width=Inches(width))

def eq(d,text):
    # Editable native Office math, linear notation for compact scalar formulas.
    p=d.add_paragraph();p.alignment=1;p.paragraph_format.keep_with_next=True
    om=OxmlElement('m:oMath');r=OxmlElement('m:r');t=OxmlElement('m:t');t.text=text;r.append(t);om.append(r);p._p.append(om)
    return p

def fmt(v):return f'{v:.3f}'

def validation_recalls(d,a):
    para(d,'表 C2 的分类别验证结果：异常召回率的分母为42，正常召回率的分母为20。以下三行与表 C2 前三行一一对应；用它们核查研究组预设的两类召回率条件。').paragraph_format.keep_with_next=True
    table(d,['候选核','异常召回率','正常召回率'],[[{'linear':'线性','poly':'二次多项式','rbf':'RBF'}[r['kernel']],f"{round(r['validation_sensitivity']*42)}/42 = {r['validation_sensitivity']:.3f}",f"{round(r['validation_specificity']*20)}/20 = {r['validation_specificity']:.3f}"] for r in a['best']],[2.2,2.384,2.384])

def tracker_update(d,q,focus):
    para(d,f"{q['id']} 追踪更新：依据{ '、'.join(q['evidence_ids'][-2:]) }，回到总答案追踪表“{q['tracker_row']}”行。“当前判断”写{focus}；“支持证据”填一个证据编号及数值；“反证或限制”填一项不足；“什么发现会改变判断”写一个可检验发现；“更新阶段”记 {q['id']}。结论只写在追踪表。")

def evidence(d,packet,a):
    if packet=='A':
        caption(d,'表 A1','来源与数值核对。每行一个变量，各有310个非缺失值；六项特征来自UCI脊柱表。源页未注明计量单位，表内保持原始单位，未删负值或极端值。服务 Q1、Q5。')
        table(d,['变量','非缺失数','最小值','最大值','缺失数'],[[r.feature,310,f'{r.min:.2f}',f'{r.max:.2f}',0] for r in a['audit'].itertuples()],widths=[2.2,1.1,1.2,1.2,1.268])
        para(d,f"官方称正常100、异常210；本地计数一致。异常合并椎间盘突出60例与脊椎滑脱150例。完整重复行 {a['summary']['exact_duplicate_rows']} 条；无患者ID，不能进一步核查重复受试者。两类版与三类版的六项测量逐行相同，不能当作两批样本。")
        caption(d,'表 A2','六点几何练习。每行一个人为构造的样本，输入为无量纲 u、v；负类用−1，正类用+1。它们只解释边界几何，与真实脊柱样本无对应关系。服务 Q1、Q2。')
        table(d,['样本','u','v','标签 y'],[['N1',-3,0,-1],['N2',-1,-1,-1],['N3',-1,1,-1],['P1',1,0,1],['P2',3,-1,1],['P3',3,1,1]])
        para(d,'候选线 a：u=0，得分 fₐ=u。候选线 b：u=0.8，得分 fᵦ=u−0.8。两线均将正分判为正类；这些系数为给定候选边界，并非都已按 SVM 的 ±1 约束规范化。')
    elif packet=='B':
        caption(d,'表 B1','固定线性 SVM 的移除试验。每行一次重拟合，起点为表 A2 的6点，C=10000、数值容差10⁻⁹、不缩放；v系数数值上为0。位置和带宽均以原始无量纲 u 坐标计。服务 Q2。')
        table(d,['操作','保留样本数','边界 u','带宽'],[[r['operation'],r['remaining'],f"{r['boundary_u']:.3f}",f"{r['width']:.3f}"] for r in a['deletions']])
        caption(d,'表 B2','二维模拟的 C 路径。每行一次拟合；同72个训练点，训练集标准化，线性核。违例计 y f(x)<1（容差10⁻⁵），可能包含判对的点；带宽是2/‖w‖。服务 Q3。')
        t=a['sim'][a['sim'].kernel=='linear']
        table(d,['C','训练错误数','间隔违例数','支持向量数','带宽'],[[r.C,r.errors,r.violations,r.sv,fmt(r.width)] for r in t.itertuples()])
        picture(d,'figure-b1.png','图 B1','同表 B2 的72个训练点。实线 f=0，虚线 f=±1，空圈为支持向量；各图坐标尺度相同。服务 Q3。')
    elif packet=='C':
        picture(d,'figure-c1.png','图 C1','三种核在同72个模拟训练点上的拟合。均 C=1；多项式次数2、gamma=1、常数项1；RBF gamma=1。实线 f=0，虚线 f=±1；RBF虚线不是原始平面等距离线。服务 Q4。')
        caption(d,'表 C1','每行一种核，输入与图 C1 相同；误差分母为训练72，平衡准确率来自验证24（两类各12）。不是脊柱临床指标。服务 Q4。')
        sel=a['sim'][(a['sim'].C==1)&(a['sim'].gamma==1)]
        table(d,['核','训练错误数','支持向量数','验证平衡准确率'],[[{'linear':'线性','poly':'二次多项式','rbf':'RBF'}[r.kernel],r.errors,r.sv,fmt(r.validation_ba)] for r in sel.itertuples()])
        caption(d,'表 C2','脊柱模型对照。每行一组设置；训练186（异常126、正常60），验证62（异常42、正常20），六特征。标准化只拟合训练集；前3行分别为预定网格中各核最高验证平衡准确率，同分按较小C再较小gamma选定。末2行为固定参数尺度对照。服务 Q5。')
        rows=[]
        for r in a['best']:
            rows.append([{'linear':'线性','poly':'二次多项式','rbf':'RBF'}[r['kernel']],f"{r['C']:g}",'不适用' if r['kernel']=='linear' else f"{r['gamma']:.3g}",int(r['train_errors']),fmt(r['validation_ba']),int(r['sv'])])
        for scaled in [False,True]:
            r=a['grid'][(a['grid'].kernel=='rbf')&(a['grid'].C==1)&(a['grid'].gamma==.1)&(a['grid'].scaled==scaled)].iloc[0]
            rows.append(['RBF 未缩放' if not scaled else 'RBF 已缩放','1','0.1',int(r.train_errors),fmt(r.validation_ba),int(r.sv)])
        table(d,['设置','C','gamma','训练错误','验证平衡准确率','支持向量'],rows,[1.65,.6,.8,1,1.55,1.368])
        validation_recalls(d,a)
        para(d,'预定网格：所有核 C 为0.01、0.1、1、10、100；二次核次数2、gamma=1/6、常数项1；RBF gamma为0.01、0.1、1、10。平衡准确率=(异常召回率+正常召回率)/2。网格最优只指这些候选中的最优。')
        picture(d,'figure-c2.png','图 C2','每格一组 RBF 参数在62条脊柱验证记录上的平衡准确率；训练数据和缩放相同。服务 Q5。',6.4)
    elif packet=='D':
        caption(d,'表 D1','固定方案的测试结果。每行一个核族的预先冻结模型；在186条训练记录上拟合，不用验证集重拟合，测试62条（异常42、正常20）。TP/FN指异常，TN/FP指正常；三模型比较为预先安排的教学展示，不能据此再次挑核。服务 Q6。')
        table(d,['模型','TP','FN','TN','FP','平衡准确率'],[[{'linear':'线性','poly':'二次多项式','rbf':'RBF'}[r.kernel],r.tp,r.fn,r.tn,r.fp,fmt(r.ba)] for r in a['test'].itertuples()],[1.8,.7,.7,.7,.7,2.368])
        picture(d,'figure-d1.png','图 D1','同62条测试记录，固定模型；只给骨盆半径加零均值Gaussian噪声，标准差为训练对应SD的0、10%、20%；每强度200次、三模型共用扰动。均为人为压力设定，非仪器误差估计。服务 Q6。',6.4)
        caption(d,'表 D2','每行一个研究规模，第一行是主数据，后两行是设定情境。n为样本数，p为特征数。完整双精度核矩阵存储按8n²字节估算，1 GB=10⁹字节；并非实际软件内存或耗时。服务 Q6。')
        table(d,['研究任务','n','p','完整核矩阵','解释需求'],[['脊柱方法比较',310,6,'0.769 MB','比较预测方法'],['基因表达小样本',80,2000,'0.051 MB','提出候选标志物'],['多中心大表',100000,6,'80 GB','快速训练和复现']],[1.65,.7,.7,1.5,2.418])

# Document authoring follows below; all student prompts are kept in this file.
def worksheet(a):
    d=newdoc('Problem Based Learning Student Worksheet')
    para(d,'这份学习单由每位学生独立完成。你可以查资料、讨论和使用 AI；判断必须指向本册证据。课堂和作业都能只用本册完成，辅助实验本用于核查计算。用途限于教学研究，不构成临床诊断、部署或患者护理建议。')
    heading(d,'研究组的委托',2);para(d,PROBLEM)
    table(d,['互斥方案','研究组实际采取的行动'],OPTIONS,[1,5.968])
    para(d,'先作暂定选择。收到 Packet C 后冻结一个方案及参数；Packet D 只允许保留冻结方案或改为 D 暂缓，不能看测试结果后换核。研究组希望验证平衡准确率至少0.75，且异常、正常召回率都不低于0.65；这是课堂离线比较的预设条件，不是临床使用标准。若不满足、数据单位无法统一或记录独立性不足，应修改或暂缓。')
    caption(d,'表 S1','启动事实。每行是研究组的一项已知条件；数据来源UCI Vertebral Column，观察单位为发布表中的一条受试者测量记录。服务 Q1、Q5。')
    table(d,['事实','数值或说明'],[['样本与变量','310条，6项连续测量；标签另列。'],['标签','正常100条、异常210条；异常含两种病理类别。'],['比较用途','给下一轮离线研究挑一个SVM基线，预测原数据标签。'],['候选模型','线性、二次多项式、RBF；先比较验证结果，再查看测试。']],[1.6,5.368])
    heading(d,'怎样使用本册',2)
    para(d,'总答案追踪表是保存判断变化的唯一位置：T0保存初选，T1–T6保存各探究问题的结论、证据和未知因素。最终“研究备忘录”是据此写成的行动主张、证据与推理、适用边界三段综合。每个模块的结论写一次，不在题后与追踪表重复抄写。')
    para(d,'Packet 指按阶段发放的证据包。3W 是“我已知什么、还不知道什么、下一步做什么”。证据卡用于核验一句具体说法。退出条是阶段结束时的一次简短回顾。停止或修改条件是能让你改选或暂缓的具体发现。白板讲解是说明自己的学习与分析过程，不要求背诵稿件。')
    para(d,'学习路线：Phase 1 课堂启动45分钟；Phase 2 课外自学120分钟（六个探究模块97分钟，证据卡与综合23分钟）；Phase 3 翻转课堂45分钟。Q1至Q6就是下面六个探究问题的编号。')
    heading(d,'六个探究问题',2)
    for q in QS:para(d,f"{q['id']}（Packet {q['release_after']}，{q['minutes']}分钟）：{q['question']} 完成后用于{q['contribution_to_synthesis']}。")

    heading(d,'Phase 1 课堂启动',new=True)
    para(d,'本阶段只使用研究组委托、表 S1、图 S1、启动概念读页和本阶段资源链接。暂不阅读 Packet A–D。老师安排短讨论，你仍需独立留存作答。')
    table(d,['分钟','要做什么','写在哪里'],[['0–6','阅读委托和方案 L、P、R、D，初选。','总答案追踪表的“T0 初选”行'],['6–10','阅读术语和资源导读。','术语澄清'],['10–25','完成总纲二维课堂练习。','启动观察表'],['25–34','整理影响选择的知识缺口。','3W表'],['34–41','制定一个可检验的自学计划。','个人学习计划'],['41–45','检查一项尚缺的边界证据。','启动退出条']],[.8,3.45,2.718])
    heading(d,'启动概念读页',2)
    para(d,'一个样本是一组测量值。二维图中把它写成 x=(u,v)；线性得分是 f(x)=w₁u+w₂v+b。w₁、w₂给出两个坐标的权重，b改变直线位置；f=0是分类边界，正负号给出类别。向量长度 ‖w‖=√(w₁²+w₂²)。')
    para(d,'Margin 讨论边界与样本之间的间隔。图中实线是 f=0，虚线是 f=±1；在线性 SVM 规定好得分尺度后，两条虚线间的垂直带宽为2/‖w‖。Soft margin允许点进入带内甚至被错分。C是这种违例的惩罚权重；gamma则控制RBF核的相似度衰减。先记录图中变化，课外再解释为什么。')
    para(d,'启动阅读资源：scikit-learn《Support Vector Machines》1.4.7.1 SVC（看目标函数中的两项）和1.4.6 Kernel functions（看核的定义）：'+SVMREF+'；《RBF SVM parameters》的参数解释与边界图：'+RBFREF+'。可先用AI翻译，仍需回到图 S1 核查。')
    answer(d,'术语澄清：为理解方案 L 与 R，请结合启动概念读页和图 S1，写两句分别说明“分类边界”和“虚线带”的含义，并指出在哪一种核下虚线间距可直接当作平面几何带宽。',[['边界含义',None],['虚线带含义',None]])
    picture(d,'figure-s1.png','图 S1','总纲课堂练习。每点为人为模拟的二维形态样本，u、v无量纲，不能解释成真实细胞指标；120点由固定种子9的双月形模拟产生，噪声尺度0.20。图示同72个训练点，余下24验证、24测试；训练集均值和SD用于全部缩放。实线f=0、虚线f=±1。服务 Q1、Q3、Q4。')
    answer(d,'启动观察：图 S1 左、中图仅改变线性核的 C，右图为RBF设置。请在启动观察表每行写两句，分别记录可见差异和一个仍需核验的解释；把观察用于判断研究组委托中的方案 L 与 R，暂不使用 Packet A–D 的数字。',[['C与直线边界',None],['线性虚线带宽',None],['线性与RBF形状',None]],height=.66)
    answer(d,'3W表：为补齐研究组委托中方案 L、P、R、D 的选择依据，请按三个概念簇逐格写一句。“关于这个问题，我已经知道什么？”列引用表 S1 或图 S1；“要正确回答问题，我还不知道什么？”列只写影响模型选择的问题；“下一步我要学习什么、查找什么或分析什么？”列指明资源章节或探究模块和核验动作。',[['示例（另一门课）','已知温度以℃记录。','不知道两仪器是否一致。','查仪器说明并比较同一标准样本。'],['Q1–Q2 边界与样本',None,None,None],['Q3 容错',None,None,None],['Q4–Q6 核与适用范围',None,None,None]],headers=['概念簇','关于这个问题，我已经知道什么？','要正确回答问题，我还不知道什么？','下一步我要学习什么、查找什么或分析什么？'],widths=[1.22,1.85,1.85,2.048],height=.75)
    answer(d,'个人学习计划：从3W表选一项最影响研究组方案 L、P、R、D 的未知问题。用四行写问题、具体资源、分析动作与完成标准；标准必须能用本册一个表或图检验，例如“能不看笔记解释图中某两种设置的差别”。',[['待解决问题',None],['章节或网页',None],['核验动作',None],['完成标准',None]],height=.38)
    answer(d,'启动退出条：只依据表 S1 和图 S1，指出最可能推翻总答案追踪表的“T0 初选”行 初选的一项未知事实，并写出你首先要完成的探究问题编号及理由；共两句。',[['先补什么证据',None]],height=.65)

    heading(d,'总答案追踪表',new=True)
    para(d,'在 Phase 1 先填 T0：从“研究组的委托”表内的四个互斥方案 L、P、R、D 中选择一项，“当前判断”写方案和一句理由；“支持证据”引用表 S1 或图 S1；“反证或限制”写目前依据的不足；“什么发现会改变判断”写一个未知事实；“更新阶段”写启动。T1–T6在对应模块完成后各填一次，每格一条短句，不复制模块中的计算步骤。')
    table(d,['追踪行','当前判断','支持证据','反证或限制','什么发现会改变判断','更新阶段'],[[label,None,None,None,None,None] for label in ['T0 初选']+[q['tracker_row'] for q in QS]],widths=[.85,1.55,1.1,1.1,1.65,.718],answer=True,height=.93)
    para(d,'保存 T0 作为初始记录。T1–T4写结论对方案 L、P、R、D 的含义；T5写冻结的方案和参数；T6只可保留T5方案或改为D。翻转课堂内的最终修订写在“课堂修订”区。')

    heading(d,'Phase 2 Packet A 几何与数据',new=True)
    para(d,'现在开放 Packet A，完成 Q1；表 A1 补充启动阶段没有的数值范围、单位和记录核验。仅开放到 Packet A，不预读 B–D。')
    evidence(d,'A',a)
    heading(d,'Q1 两条分界线的抗扰动差异',2)
    para(d,'学习重点：加权和决定边界，归一化距离才表达几何远近。补充阅读：scikit-learn《Support Vector Machines》1.4.7 Mathematical formulation中的线性可分示意图。资源：'+SVMREF+'#mathematical-formulation。')
    para(d,'距离读页：点到线的垂直距离 d=|f(x)|/‖w‖；对带标签的样本，y f(x)/‖w‖为带符号间隔，负数表示误分。把w和b同时乘正数，分类线不变，原始得分却变了。对一个固定线性边界，若输入扰动的欧氏长度小于点到边界的距离，该点不会跨线；这不保证未来样本仍离线足够远。')
    eq(d,'d(x,H)=|wᵀx+b|/‖w‖')
    para(d,'在线性可分的硬间隔SVM中，可把得分尺度约定为所有训练点满足yᵢf(xᵢ)≥1，最近点取等号。此时每侧最小间隔为1/‖w‖，最小化½‖w‖²就等价于最大化间隔。较大的间隔让已有点容忍更大的扰动，也限制分界面的选择；这是可能有助于泛化的理由，不是未知样本准确率的保证。训练样本是否有代表性、特征尺度和噪声仍会影响泛化。')
    answer(d,'Q1 距离计算：为比较候选线a与b，请用表 A2 的六个输入算每条线最近样本的距离，并在下表逐行记录最近点、分子、分母与结果；第三行把线a的得分改写成5u，检验同一几何边界是否改变。',[['a：f=u',None,None,None,None],['b：f=u−0.8',None,None,None,None],['a重写：f=5u',None,None,None,None]],headers=['边界','最近样本','最小 |f|','‖w‖','最小距离'],widths=[1.5,1.4,1.4,1.268,1.4],height=.42)
    answer(d,'Q1 几何与现实：依据表 A2 比较a与b在扰动长度小于0.3时对六点标签的保障，再结合表 A1 说明为何不能直接把这种保障变成脊柱新样本准确率；写两句，各引用一个证据编号。',[['扰动与外推',None]],height=.8)
    tracker_update(d,QS[0],'几何结论及它对方案 L、P、R、D 的影响')

    heading(d,'Packet B 支持向量与容错',new=True)
    para(d,'开放 Packet B，依次完成 Q2 与 Q3。保持 Packet C、D 未读；本阶段只解释边界如何由样本和C共同决定。')
    heading(d,'Q2 哪些样本约束边界',2)
    para(d,'学习重点：支持向量是参与决定边界的训练样本。在线性可分的硬间隔图景中，最近样本限制空白带；软间隔中，支持向量还可能位于带内或误分一侧。它们不是被选中的特征，也不一定是“典型病例”。补充阅读：scikit-learn《Support Vector Machines》1.4.1 Classification 的支持向量示例：'+SVMREF+'#classification。')
    # B1 belongs to Q2; B2 and B1 figure follow in Q3.
    caption(d,'表 B1','同表 A2 的六个无量纲输入；每行一次线性SVM重拟合，C=10000、容差10⁻⁹、不缩放。边界v系数约为0。服务 Q2。')
    table(d,['操作','剩余样本','边界 u','带宽'],[[r['operation'],r['remaining'],fmt(r['boundary_u']),fmt(r['width'])] for r in a['deletions']])
    answer(d,'Q2 样本影响：为判断模型依赖的证据，请对照表 A2 与表 B1，比较移除 N1 与 P1 的位置和带宽变化；写两条数值比较，并用一句话说明这些结果能否证明“删除任何支持向量都必定改变边界”。',[['N1与P1的比较',None]],height=1.0)
    tracker_update(d,QS[1],'哪些样本约束边界及其对你选模的意义')
    heading(d,'Q3 C 控制的究竟是哪一种代价',2)
    para(d,'学习重点：soft margin（软间隔）允许间隔违例。本节标签yᵢ取−1或+1（模拟图中的类0在公式里编码为−1，类1为+1）。令 mᵢ=yᵢf(xᵢ)，hinge loss（合页损失）为 max(0,1−mᵢ)。当0<mᵢ<1时已判对但仍有损失；mᵢ<0才是误分。最低松弛量ξᵢ等于这种损失。')
    eq(d,'最小化 J = ½‖w‖² + C Σᵢ max(0,1−yᵢf(xᵢ))')
    para(d,'读公式时只需分辨两项：权重长度项与违例代价项。固定特征表示、样本和核，C越大越重视减少违例总量；这不等价于让每个分类指标单调提高。标准SVM得分尺度下，线性带宽=2/‖w‖；soft margin的带里可以有样本，所以它未必是“完全空白”的区域。补充资源：'+SVMREF+'#svc。')
    caption(d,'表 B2','同72个标准化模拟训练点的线性C路径；错误指类别判错，违例指m<1（容差10⁻⁵）；支持向量按求解器非零对偶系数统计。带宽无量纲。服务 Q3。')
    table(d,['C','训练错误数','间隔违例数','支持向量数','带宽'],[[r.C,r.errors,r.violations,r.sv,fmt(r.width)] for r in a['sim'][a['sim'].kernel=='linear'].itertuples()])
    picture(d,'figure-b1.png','图 B1','与表 B2 相同的72点、尺度与参数。实线f=0，虚线f=±1，空圈为支持向量。服务 Q3。')
    answer(d,'Q3 损失计算：为了区分研究组模型的“错分”与“间隔违例”，设四个教学样本的带标签得分m分别为1.4、0.4、−0.2、−1.0，逐行计算 max(0,1−m)，并判断类别是否判对、是否违例；这些输入为独立算例，不能当作表 B2 的四条原记录。',[['1.4',None,None,None],['0.4',None,None,None],['−0.2',None,None,None],['−1.0',None,None,None]],headers=['m的给定值','合页损失','是否判对','是否违例'],widths=[1.5,1.8,1.8,1.868],height=.38)
    answer(d,'Q3 参数比较：用表 B2 和图 B1 比较C=0.01与100的带宽、训练错误和支持向量数。写三句解释这种变化如何影响方案 L 的容错选择，并说明为何不能只选带宽最大的设置。',[['C选择的理由',None]],height=1.0)
    tracker_update(d,QS[2],'C权衡及尚需哪种验证证据')

    heading(d,'Packet C 核选择与验证')
    para(d,'开放 Packet C，完成 Q4、Q5，再填写证据卡。写完总答案追踪表的“T5 冻结方案”行后才能打开 Packet D。')
    heading(d,'Q4 三种核怎样表示相似性',2)
    para(d,'核读页：内积xᵀz是对应坐标乘积之和。若把输入映射成φ(x)，可在新坐标中使用线性边界；kernel trick用 K(x,z)=φ(x)ᵀφ(z) 直接求这个内积，避免逐项构造全部新坐标。核是满足相应数学条件的内积表达，并非任意相似度都可直接作为SVM核。')
    eq(d,'线性 K(x,z)=xᵀz')
    eq(d,'多项式 K(x,z)=(gamma·xᵀz+r)ᵈ')
    eq(d,'RBF K(x,z)=exp(−gamma·‖x−z‖²)')
    para(d,'多项式的次数d控制可表达的交互阶数，r是常数项。RBF的gamma越大，相似度随距离衰减越快。核SVM仍在隐式特征空间选择超平面，在原始坐标中可以呈现曲线。')
    eq(d,'f(x)=Σᵢ∈SV αᵢyᵢK(xᵢ,x)+b')
    para(d,'SV表示支持向量集合，α是训练求得的系数；这里只解释预测结构，不要求推导对偶优化。RBF图中f=±1是得分等高线，不能直接按原始平面间距与线性带宽比较。阅读资源：'+SVMREF+'#kernel-functions；参数可视化：'+RBFREF+'。')
    picture(d,'figure-c1.png','图 C1','同72个标准化模拟训练点；三核均C=1。多项式d=2、gamma=1、r=1；RBF gamma=1。线和圆圈含义同图 B1；曲线虚线不是原始空间等距离带。服务 Q4。')
    caption(d,'表 C1','三核在相同模拟数据上的结果；训练72、验证24（每类12）。每行一种拟合；参数同图 C1。服务 Q4。')
    s=a['sim'][(a['sim'].C==1)&(a['sim'].gamma==1)]
    table(d,['核','训练错误数','支持向量数','验证平衡准确率'],[[{'linear':'线性','poly':'二次多项式','rbf':'RBF'}[r.kernel],r.errors,r.sv,fmt(r.validation_ba)] for r in s.itertuples()])
    answer(d,'Q4 二次核计算：为解释方案 P 的数学依据，使用另一组给定输入x=(1,2)、z=(2,0)，令K=(xᵀz)²、φ(x)=(x₁²,√2x₁x₂,x₂²)。分别计算核值、两个映射向量和映射内积；最后写一句说明直接计算核值与映射内积为何相等，并指出本算例的常数项r=0与图 C1 的r=1不是同一个核设置。',[['xᵀz 与 K',None],['φ(x) 与 φ(z)',None],['φ(x)ᵀφ(z)',None],['恒等式与图的差别',None]],height=.42)
    answer(d,'Q4 三核比较：结合图 C1 与表 C1，为方案 L、P、R 各写一条“边界形状—验证数值—不能推出的结论”；随后在同一答题区用一句话比较gamma=0.1与10时距离为1的两点的RBF相似度，说明gamma改变的对象与C有什么不同。',[['三核与gamma',None]],height=1.15)
    tracker_update(d,QS[3],'选择哪种核的条件及模拟证据的限度')
    heading(d,'Q5 在脊柱数据上冻结方案',2)
    para(d,'学习重点：训练集估计均值、SD和模型参数；验证集比较核与超参数；测试集检查冻结方案。StandardScaler用训练均值和SD变换每个变量。尺度变化会改变距离和同一gamma的含义；本例控制核参数不变来观察这种影响，不能据此声称缩放总能提高分数。')
    para(d,'平衡准确率是异常召回率与正常召回率的平均，专为避免多数类掩盖另一类表现。SVM得分的正负用于分类，数值不是患病概率。阅读资源：'+SVMREF+'#tips-on-practical-use。')
    # Render the C2 and heatmap portion only, avoiding duplicate C1.
    caption(d,'表 C2','六特征脊柱数据，训练186（126异常、60正常）、验证62（42异常、20正常）。前3行是各核在预设网格中最高验证平衡准确率，同分按较小C、再较小gamma选定；末2行是固定参数的尺度对照。服务 Q5。')
    rows=[]
    for r in a['best']:
        rows.append([{'linear':'线性','poly':'二次多项式','rbf':'RBF'}[r['kernel']],f"{r['C']:g}",'不适用' if r['kernel']=='linear' else f"{r['gamma']:.3g}",int(r['train_errors']),fmt(r['validation_ba']),int(r['sv'])])
    for scaled in [False,True]:
        r=a['grid'][(a['grid'].kernel=='rbf')&(a['grid'].C==1)&(a['grid'].gamma==.1)&(a['grid'].scaled==scaled)].iloc[0]
        rows.append(['RBF 未缩放' if not scaled else 'RBF 已缩放','1','0.1',int(r.train_errors),fmt(r.validation_ba),int(r.sv)])
    table(d,['设置','C','gamma','训练错误','验证平衡准确率','支持向量'],rows,[1.65,.6,.8,1,1.55,1.368])
    validation_recalls(d,a)
    para(d,'网格固定：所有核C为0.01、0.1、1、10、100；二次多项式d=2、gamma=1/6、r=1；RBF gamma为0.01、0.1、1、10。六特征均用训练集缩放。跨核的相近分数不按微小差异强行排名。支持向量数量描述当前模型，不能单独当作泛化评分。')
    picture(d,'figure-c2.png','图 C2','每格是一组RBF参数在相同62条脊柱验证记录上的平衡准确率；186条训练记录及缩放固定。服务 Q5。',5.8)
    answer(d,'Q5 验证选择：为在方案 L、P、R、D 中作冻结决定，比较表 C2 最后两行的训练错误与验证平衡准确率，再从图 C2 指出两组不同gamma的验证结果。写三句，说明每项比较允许怎样的选择、为何不能直接沿用图 C1 模拟数据上的胜者。',[['参数与尺度证据',None]],height=1.0)
    para(d,'Q5 追踪更新：在总答案追踪表“T5 冻结方案”行，“当前判断”从研究组委托的方案 L、P、R、D 中选一项并写明核、C、gamma及缩放；“支持证据”引用表 C2 或图 C2 的数值；“反证或限制”写验证样本的局限；“什么发现会改变判断”写测试中会使你改为D的一项发现；“更新阶段”写Q5。此后不能用表 D1 再挑核。')
    answer(d,'证据卡：为核验Q3至Q5中的一个模型选择理由，请从AI回答或官方文档摘录一句关于C、gamma或缩放的具体说法（不超过25个英文词或一句中文），用表 B2、表 C2 或图 C2 的一项数字以及一个可追溯的官方章节检查它；写原句、数据核验、来源链接和必要限定各一行。',[['待核查原句',None],['数据核验',None],['官方来源',None],['适用限定',None]],height=.36)

    heading(d,'Packet D 新记录与适用范围',new=True)
    para(d,'只有总答案追踪表的“T5 冻结方案”行已完成才能阅读。三核测试结果是研究组预先安排的教学比较；本次评估允许保留T5方案或暂缓，不能据测试表更换核、参数或阈值。')
    evidence(d,'D',a)
    heading(d,'Q6 是否保留冻结方案以及能迁移多远',2)
    para(d,'学习重点：几何间隔解释固定模型对某类输入扰动的抵抗能力；泛化还依赖未来样本是否来自可比过程。中小样本时核SVM常是可行候选；高维中仍可使用间隔约束，但无用特征、标签噪声和小n仍会伤害预测。非线性核不直接给出原始变量的一组线性效应。')
    para(d,'规模读页：核训练可能涉及样本两两比较，随n增长代价明显上升；表 D2 只是完整核矩阵的存储算术，实际SVC有缓存等机制，不代表实际占用必为8n²。大n可考虑线性专用求解器；高p与大n是两种不同困难。阅读：'+SVMREF+'#complexity 及同页开头优点和局限。')
    answer(d,'Q6 错误与扰动：为复核T5冻结方案，用表 D1 的TP、FN、TN、FP计算该模型两类召回率（若T5选D，预先以线性核作核验对象）；再用图 D1 写一句说明平均表现与预测翻转描述的差别。结果分别写分子/分母、比值和一句解释。',[['异常召回率',None],['正常召回率',None],['扰动指标含义',None]],height=.46)
    answer(d,'Q6 迁移判断：依据表 D2，为“80样本、2000基因”和“100000样本、6特征”各写两句，分别判断核SVM是否适合作为起点并解释计算或统计限制；至少有一句说明支持向量或分类成功为何不能直接给出病因或生物标志物。',[['小样本基因表达',None],['大样本研究表',None]],height=.65)
    answer(d,'Q6 总纲作业原题：用自己的话解释“为什么 SVM 不是简单地找一条刚好分开两类的线？”写150–220字，至少引用表 A2、表 B2、图 C1、表 D1 中两项证据，把margin、支持向量、容错与泛化连成解释；综合备忘录只引用本题，不重复抄写。',[['原题作答',None]],height=1.55)
    para(d,'Q6 追踪更新：在总答案追踪表“T6 适用范围”行，“当前判断”写保留T5冻结的方案或改为D；“支持证据”引用表 D1 或图 D1；“反证或限制”联系表 D2 写一项迁移限制；“什么发现会改变判断”写需要补充的独立证据；“更新阶段”写Q6。这里不重新选择核。')

    heading(d,'课外综合 研究备忘录初稿')
    para(d,'证据卡与本节合计23分钟，包含在120分钟课外预算内。研究备忘录按CERB组织：Claim是采取的方案，Evidence是证据，Reasoning是证据怎样支持行动，Boundary是结论适用边界。')
    answer(d,'决策检验矩阵：为组织研究备忘录，结合图 C1、表 C2 和表 D1，给方案 L、P、R 各填一个已观察到的收益和一项已观察到的代价，两格都写证据编号。只总结已发生的观察，未来何时改选留在总答案追踪表，不在此重复。',[['L',None,None],['P',None,None],['R',None,None]],headers=['方案','已观察到的收益','已观察到的代价'],widths=[.65,3.159,3.159],height=.58)
    answer(d,'初稿：写一份给脊柱研究组的300–450字研究备忘录，按“行动主张—证据与推理—边界”三段作答。主张只能保留T5冻结方案或改为D；用总答案追踪表T1至T6各一项结果，整合几何、关键样本、C、三种核、真实验证及规模限制，并引用Q6原题作答而不抄写；缺少任一探究问题的理由均需补足。',[['给研究组的备忘录',None]],height=2.65)

    heading(d,'Phase 3 翻转课堂与课堂修订',new=True)
    para(d,'本阶段45分钟：0–4分钟回看初稿；4–20分钟教师随机选一至两位同学白板讲解；20–30分钟同伴质询；30–38分钟教师追问与综合；38–43分钟个人修订；43–45分钟退出条。每人都准备说明全套学习与分析过程，可自主组织讲解，无固定汇报脚本或小组角色。')
    answer(d,'同伴质询记录：针对白板讲解中一个关于方案 L、P、R、D 的理由，引用本册一个表或图，写你提出或收到的一个问题及回应；如果未被点名，记录自己对同一问题的回答。',[['问题与回应',None]],height=.8)
    answer(d,'课堂修订：比较你的研究备忘录初稿和本次课堂质询，写100–150字替换或补充段落，指出一个发生改变的论证及对应证据编号；若维持判断，回应最强反对意见。修订后的行动仍只能是T5冻结方案或D，并说明T1至T6中受影响的行。',[['备忘录修订段落',None]],height=1.2)
    answer(d,'结束退出条：结合图 B1 与图 C1，各写一句说明“线性带宽更大”与“核边界更弯”分别为什么不足以单独批准模型；本题检验你能否限制几何直觉的使用范围。',[['两条限制',None]],height=.7)
    heading(d,'数据获取与辅助实验资源',new=True)
    para(d,'必做题所需数值和图片已经包含在本册；不要求学生编写新代码。辅助Notebook用于离线复现，不增加必做题。')
    para(d,'数据入口：'+SOURCE+'；DOI：10.24432/C5K89B；许可CC BY 4.0。引用Barreto与Neto的数据记录。下载文件column_2C.dat，每行6个连续值和一个NO/AB标签，空白分隔，无表头；源页称无缺失，本地未见?、NA或空缺。column_3C.dat为同一批记录的三类标签，不合并。')
    para(d,'数据获取与辅助实验：assets/pbl-svm-margin/source-data/column_2C.dat；整理表assets/pbl-svm-margin/vertebral-column.csv；辅助实验本notebook/pbl-svm-margin-lab.ipynb。编号是课堂行号，不是患者标识。标准化、划分和模拟代码均随包提供。')
    para(d,'可选延伸（不计入120分钟）：交叉验证、bootstrap区间、概率校准和外部验证方案将在第10至12讲进一步学习。本讲不把这些技术当作完成Q1至Q6的先修要求。')
    d.save(ROOT/(SLUG+'-student-worksheet.docx'))
    return d

def evidence_doc(a):
    d=newdoc('Packet A D Evidence Pack')
    para(d,'这是与学生学习单同编号的证据汇编，便于教师分阶段发放或学生检索。证据不附标准答案。先发启动材料和Packet A，再按Q1、Q2–Q3、Q4–Q5、Q6的进度开放后续材料。学习单内已含全部必做证据。')
    para(d,'源数据未注明单位与测量可靠性，无法声称真实仪器扰动分布。完整行未重复也不能证明受试者独立。所有结果服务于教学和内部方法比较，不是临床诊断、部署或病因证据。公式与核设置依据：'+SVMREF+'。')
    para(d,PROBLEM);table(d,['方案','行动'],OPTIONS,[.7,6.268])
    for q in QS:para(d,f"{q['id']}：{q['question']} 对应{q['tracker_row']}。")
    for packet in 'ABCD':
        heading(d,'Packet '+packet,new=True);evidence(d,packet,a)
        related=[q for q in QS if q['release_after']==packet]
        for q in related:para(d,f"证据复核 {q['id']}：使用{ '、'.join(q['evidence_ids']) }完成学习单{q['id']}，把依据与可改变判断的未知事实写在总答案追踪表“{q['tracker_row']}”；不要另交重复答案。")
    heading(d,'来源与适用边界',2)
    para(d,'真实数据：'+SOURCE+'，DOI 10.24432/C5K89B，CC BY4.0，核查日期2026-09-17。模拟数据为固定种子9的教学构造，没有真实患者或真实细胞标签。')
    d.save(ROOT/(SLUG+'-evidence-pack.docx'))

def teacher_doc(a,design):
    d=newdoc('PBL Flipped Classroom Teacher Plan')
    para(d,'面向生命科学学院大四直博生。目标是让学生通过证据理解SVM边界，并为脊柱研究组选择一个离线基线。学生已修微积分、数理统计和前八讲；向量、内积和长度在学习单中就地解释，不要求凸优化基础。')
    para(d,'时间为课堂45分钟、自学120分钟、翻转课堂45分钟。worksheet是唯一必做载体；Notebook为辅助复现，review内试填答案不得发给学生。')
    heading(d,'目标覆盖与提问',2)
    for lo in OBJECTIVES:
        qids=','.join(q['id'] for q in QS if lo[0] in q['objective_ids']);para(d,lo[0]+'：'+lo[1]+'。可观察结果：'+lo[2]+' 对应'+qids+'。')
    heading(d,'课堂启动的45分钟',2)
    table(d,['分钟','实施','检查个人产出'],[['0–6','呈现委托、互斥方案和已知数据。','T0初选，允许D但要求理由。'],['6–10','只澄清得分、边界、虚线带与术语。','术语澄清两句。'],['10–25','使用图S1完成总纲三个二维观察任务。','启动观察三行。'],['25–34','让学生辨认影响选择的概念缺口。','3W三个概念簇。'],['34–41','检查学习计划是否有可检验标准。','资源、动作、标准。'],['41–45','收集最可能推翻初选的未知。','启动退出条。']],[.75,3.1,3.118])
    para(d,'首次讲解控制在4分钟，其余用短追问。图S1已经包含C改变、linear/RBF和间隔变化，不能把总纲“课堂练习”全部转移到课外。')
    heading(d,'课外释放和完整课件的使用',2)
    para(d,'学习单是完整母版，印刷时分段发放，LMS使用随包提供的阶段版：启动+追踪表、Packet A、Packet B、Packet C、Packet D+综合、翻转课堂。阶段版均取自同一学习单，追踪表随身保留。不要在课前公开完整母版、完整证据包或Notebook的后续输出，否则阶段封存仅靠自律。')
    table(d,['模块','分钟','开放与检查'],[[q['id'],q['minutes'],'Packet '+q['release_after']+'；检查'+q['tracker_row']] for q in QS]+[['证据卡与初稿',23,'Q5后核查来源，Q6后完成综合。']],[1,1,4.968])
    para(d,'必读概念解释已在学习单就地提供；外部文档用于独立核验。AI可解释术语和公式，学生需要引用真实表图数值。不给模块额外添加重复总结框；暂定结论与未知因素均写入对应追踪行。')
    heading(d,'教师参考答案与追问',2)
    para(d,'Q1：候选线a最近距离1，b最近距离0.2（P1），5u重写后仍为1；0.3以内扰动对a全部点有保证，对b的P1无统一保证。几何保障是固定模型、给定范数与已有点的结论，不是未来误差界。源页单位空白也限制跨数据距离比较。')
    para(d,'Q2：移除N1不改变u=0及带宽2；移除P1后边界u=1、带宽4。N1是远点，P1约束正类最近侧。不能由此说每个支持向量删除都会改变边界：重复或冗余约束可能留下同一解。支持向量是样本，不是生物标志物。')
    para(d,'Q3：给定m的hinge依次为0、0.6、1.2、2.0。0.4判对但违例；−0.2和−1.0误分。降低C重视权重项，带宽可增大，却不保证验证表现。支持向量数不是模型复杂度的通用单调指标。')
    para(d,'Q4：xᵀz=2，K=4；φ(x)=(1,2√2,4)，φ(z)=(4,0,0)，映射内积为4。这是r=0、gamma=1的二次核算例，图C1使用r=1，两者不能混算。RBF距离1时，gamma0.1相似度约0.905，gamma10约0.0000454。gamma控制相似度范围，C控制训练违例代价。')
    para(d,'Q5：由生成数据自动记录的各核验证选择如下。L和R达到本课预设条件；P的正常召回率为12/20=0.60，未达到0.65，不能在不修改条件或模型的情况下冻结P。允许学生因简洁选择L，或有条件选择R；若给出D，必须指出阻碍离线比较的具体证据缺口，不因“不是临床数据”一概停学。')
    table(d,['核','C','gamma','验证平衡准确率'],[[{'linear':'线性','poly':'二次多项式','rbf':'RBF'}[r['kernel']],f"{r['C']:g}",'不适用' if r['kernel']=='linear' else f"{r['gamma']:.3g}",fmt(r['validation_ba'])] for r in a['best']])
    para(d,'Q6：每个召回率都要报告分母。正常只有20条，一条改变5个百分点，不宜把小差距解释成普遍优势。压力测试是单变量人为变化，不等于真实测量误差或外部验证。80×2000情境可尝试受约束线性模型但p远大于n仍需严格验证；100000条的非线性核代价大，可讨论线性专用方法；不要求学生开发这些方法。')
    heading(d,'翻转课堂的45分钟',2)
    para(d,'0–4回看；4–20随机选1–2人白板讲解；20–30同伴质询；30–38教师综合；38–43个人修订；43–45退出条。所有学生准备完整过程，教师不预设唯一讲解顺序。')
    para(d,'可用追问：为何除以权重长度？判对为何仍有损失？支持向量能否当特征？改变单位时gamma是否仍是同一种设置？RBF虚线为何不能用尺量出与linear同义的margin？高p与大n为什么导致不同问题？测试结束后若想换核，需要何种新的评估？')
    heading(d,'作业与评分',2)
    para(d,'个人提交完整学习单。Q6保留总纲原题；综合备忘录须使用T1–T6，不再次回答同一道作业。迁移题已在Q6表D2，不另加一份项目。')
    table(d,['维度','权重','判据'],[['几何与容错','30%','Q1距离归一化；Q2关键样本；Q3判对但违例。'],['核与参数','25%','Q4三核与内积恒等式；Q5尺度和验证。'],['适用与迁移','15%','Q6区分高p、大n、模型解释与生物机制。'],['综合与修订','25%','追踪证据完整，初稿整合全部Q，能回应反证。'],['来源核验','5%','原句、数据、官方章节与限定相符。']],[1.2,.7,5.068])
    heading(d,'实施边界与风险响应',2)
    para(d,'若学生线性代数薄弱，先展开w₁u+w₂v+b，使用向量长度定义；不删掉归一化距离。若公式吃力，让学生先读数值表再回到目标函数两项。若时间超出，删选修阅读，不删三种核、支持向量、C或总纲课堂任务。')
    para(d,'代码数据记录：186/62/62固定分层划分；所有候选只用训练拟合、验证选择，不在test后重选。正文不展开CV、bootstrap或校准；这些作为第10–12讲的可选延伸。主案例是历史脊柱标签识别，无因果或临床部署证明。')
    d.save(ROOT/(SLUG+'-teaching-plan-flipped-classroom.docx'))

def rationale_doc(design):
    d=newdoc('Purpose Design Rationale Expected Outcomes')
    para(d,'本版以学生能独立理解问题、查找证据并写出可检查答案为设计目标。课程总纲第9讲是SVM算法课，核心工作围绕几何间隔、关键样本、容错与核方法；评价指标只服务于选模，避免把整讲改写成通用临床部署审计。')
    heading(d,'旧设计的复核与修订',2)
    para(d,'旧plan.md选脊柱数据，svm.md另选模拟细胞数据；本版统一为真实脊柱主案例和明确标注的二维模拟辅助。保留两份旧稿作为历史来源，执行规范以本版analysis-design和本版四份Word为准。')
    para(d,'旧计划的polynomial只放概念卡，无法验证学生是否掌握。本版Q4加入具体内积恒等式与三核同数据比较。高维优势、大数据代价和解释局限由Q6场景题实际考核。旧计划120–150分钟且含大量通用验证内容，本版必做自学120分钟，CV、bootstrap、calibration移至选修。')
    heading(d,'总问题与问题链覆盖',2)
    para(d,'研究人员要在下一轮离线比较中采用一个SVM基线，方案L/P/R/D互斥且可操作。初选基于真实规模和已展示的二维边界，避免盲猜。六个探究问题逐步建立选择理由；最终备忘录必须使用T1至T6，不能凭一张性能表跳过几何、核和适用条件。')
    for q in QS:para(d,q['id']+'：'+q['contribution_to_synthesis']+'；通过'+q['student_output']+'留下独立证据。')
    heading(d,'数据与分析为何适合',2)
    para(d,'UCI脊柱数据310条、6连续特征，易于独立加载和短时间重复分析。来源明确两类由三类合并，单位字段为空；本地核对记录这些限制而不自行补齐。二维双月形模拟仅用于直观比较，六点算例用于精确计算；两者均不命名为真实细胞生物标志物。')
    para(d,'C路径固定样本、表示和核；三核比较保持相同拆分；尺度对照固定核参数，解释时提醒同一gamma随尺度改变已代表不同相似度。真实数据上的核表现无须符合“非线性总是更好”的叙事。压力测试只扰动骨盆半径，避免独立扰动三个相关角度破坏其几何关系。')
    heading(d,'学生独立学习与可读性',2)
    para(d,'每个模块先给本地概念读页，再给精确官方章节、带编号证据和有明确形式的作答任务。计算题预置项目行，判断题保留开放回答；不在证据标题或图注里写出待判断结论。总答案追踪表保存结论和未知因素，模块不另抄一遍。')
    para(d,'学生可以用AI，但必须把一句具体说法与案例数字和官方来源相互核验。教师随机选1–2人白板讲解，让每位学生准备全流程；不设固定分工。字体、留白与答题表延续第7讲A4、0.65英寸边距的阅读节奏。')
    heading(d,'预期学习结果与不能声称的结果',2)
    para(d,'可观察结果是：正确算距离和hinge，解释支持向量，区分C/gamma，比较linear/poly/RBF，识别缩放与验证作用，对高p和大n作不同判断。评分强调证据与推理。没有真实教学试验，不能声称本包已证明提升考试成绩或科研能力；模拟学生试填仅用于发现题意、顺序和负担问题。')
    heading(d,'检验与迁移',2)
    para(d,'发布前检查目标覆盖、证据释放时点、DOCX题目可读性、独立模拟试填、图表一致性和页面排版。试填不是学习效果研究。Q6的小样本基因表达与大样本研究表要求迁移概念，不能把SVM分类系数或支持向量直接解释成病因。')
    para(d,'数据与算法来源：'+SOURCE+'；'+SVMREF+'；'+RBFREF+'。课程来源：全课程plan.docx第9讲与本目录两份旧计划，用户2026-09-17要求按最新版技能重新制作。')
    d.save(ROOT/(SLUG+'-design-rationale.docx'))

def notebook():
    import nbformat as nb
    n=nb.v4.new_notebook();c=[]
    c.append(nb.v4.new_markdown_cell('# 第9讲 SVM辅助复现\n所有必做证据已在worksheet中；本Notebook用于离线核查。数据不用于临床诊断或部署。运行后按Q1–Q6解释结果。'))
    c.append(nb.v4.new_code_cell("from pathlib import Path\nimport importlib.util\nROOT=Path.cwd().parent if Path.cwd().name=='notebook' else Path.cwd()\nspec=importlib.util.spec_from_file_location('lesson9',ROOT/'scripts'/'pbl-svm-margin-build.py')\nlesson=importlib.util.module_from_spec(spec); spec.loader.exec_module(lesson)\n# analyze()完整加载本地数据、训练内缩放、固定拆分、重训所有候选、生成B–D表图。\na=lesson.analyze()\nprint(a['summary'])"))
    for q,key,cols in [('Q1','audit',None),('Q2','deletions',None),('Q3','sim',['kernel','C','errors','violations','sv','width']),('Q4','sim',['kernel','C','gamma','validation_ba']),('Q5','grid',['kernel','C','gamma','scaled','validation_ba']),('Q6','test',None)]:
        c.append(nb.v4.new_markdown_cell('## '+q+'\n'+next(z['question'] for z in QS if z['id']==q)+'\n核验计算结果与worksheet对应表；指出一项你原先判断需要的限定。'))
        c.append(nb.v4.new_code_cell("import pandas as pd\nresult=pd.DataFrame(a[%r])\ndisplay(%s)"%(key,'result' if cols is None else 'result[%r]'%cols)))
        if q=='Q1':
            c.append(nb.v4.new_code_cell("import numpy as np\npoints=a['toy']\nrows=[]\nfor name,w,b in [('a',np.array([1.,0.]),0.),('b',np.array([1.,0.]),-.8),('5a',np.array([5.,0.]),0.)]:\n    distance=np.abs(points@w+b)/np.linalg.norm(w)\n    rows.append({'边界':name,'最小距离':distance.min(),'最近点行号':(np.flatnonzero(np.isclose(distance,distance.min()))+1).tolist()})\ndisplay(pd.DataFrame(rows))"))
        if q=='Q3':
            c.append(nb.v4.new_code_cell("m=np.array([1.4,.4,-.2,-1.])\ndisplay(pd.DataFrame({'带标签得分':m,'合页损失':np.maximum(0,1-m),'判对':m>0,'间隔违例':m<1}))"))
        figures={'Q1':['s1'],'Q3':['b1'],'Q4':['c1'],'Q5':['c2'],'Q6':['d1']}.get(q,[])
        for name in figures:
            c.append(nb.v4.new_code_cell("from IPython.display import Image, display\ndisplay(Image(filename=str(lesson.AS/'figure-%s.png')))"%name))
    c.append(nb.v4.new_code_cell("import numpy as np\nx=np.array([1.,2.]); z=np.array([2.,0.])\nphi=lambda t: np.array([t[0]**2,np.sqrt(2)*t[0]*t[1],t[1]**2])\nassert np.isclose((x@z)**2,phi(x)@phi(z))\nassert len(a['raw'])==310\n# 验证已生成表D与内存中重新计算的结果相同\npd.testing.assert_frame_equal(pd.read_csv(lesson.AS/'packet-d-test.csv'),a['test'])\ndisplay(a['stress'])"))
    c.append(nb.v4.new_markdown_cell('## 可执行分析源码\n下一单元展示本课分析函数的完整源码，检查只使用train拟合缩放、validation选参及共同扰动。无需联网。'))
    c.append(nb.v4.new_code_cell('import inspect\nprint(inspect.getsource(lesson.analyze))'))
    n.cells=c;n.metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'))
    nb.write(n,ROOT/'notebook'/(SLUG+'-lab.ipynb'))

def release_copies():
    # Keep same content, select ranges at Heading 1 release boundaries.
    import copy
    src=Document(ROOT/(SLUG+'-student-worksheet.docx'))
    parts={};stage='启动与追踪表'
    for e in src.element.body:
        if e.tag==qn('w:sectPr'):continue
        texts=''.join(e.itertext()) if False else ''.join(e.xpath('.//w:t/text()'))
        if e.tag==qn('w:p'):
            sty=e.find(qn('w:pPr'))
            sid=sty.find(qn('w:pStyle')).get(qn('w:val')) if sty is not None and sty.find(qn('w:pStyle')) is not None else ''
            if sid=='Heading1':
                if texts.startswith('Phase 2 Packet A'):stage='Packet-A'
                elif texts.startswith('Packet B'):stage='Packet-B'
                elif texts.startswith('Packet C'):stage='Packet-C'
                elif texts.startswith('Packet D'):stage='Packet-D与综合'
                elif texts.startswith('Phase 3'):stage='翻转课堂与入口'
        parts.setdefault(stage,[]).append(copy.deepcopy(e))
    dest=ROOT/'student-release';dest.mkdir(exist_ok=True)
    for i,(stage,els) in enumerate(parts.items(),1):
        d=Document(ROOT/(SLUG+'-student-worksheet.docx'))
        for e in list(d.element.body):
            if e.tag!=qn('w:sectPr'):d.element.body.remove(e)
        for e in els:d.element.body.insert(len(d.element.body)-1,e)
        first=d.element.body[0];pb=first.find('.//'+qn('w:pageBreakBefore'))
        if pb is not None:pb.getparent().remove(pb)
        # Word requires a closing paragraph after a final table; keep it one point
        # so a release slice does not acquire a blank trailing page.
        if d.element.body[-2].tag==qn('w:tbl'):
            p=d.add_paragraph();p.paragraph_format.space_before=Pt(0);p.paragraph_format.space_after=Pt(0)
            p.paragraph_format.line_spacing=Pt(1);font(p.add_run(''),1)
            p.paragraph_format.keep_with_next=False
        d.save(dest/f'{i:02d}-{stage}.docx')

def validate(design):
    for an in design['analyses']:
        for f in an['artifacts']:assert (AS/f).exists(),f
    for suffix in ['student-worksheet','evidence-pack','teaching-plan-flipped-classroom','design-rationale']:
        d=Document(ROOT/(SLUG+'-'+suffix+'.docx'));txt='\n'.join(p.text for p in d.paragraphs)+'\n'+'\n'.join(c.text for t in d.tables for r in t.rows for c in r.cells)
        assert '\ufffd' not in txt and 'SVM' in txt
        (REVIEW/(suffix+'-text.txt')).write_text(txt,encoding='utf-8')
    from PIL import Image
    for p in AS.glob('figure-*.png'):
        assert Image.open(p).width>=900
    for q in QS:assert all(any(i['id']==e for i in design['evidence_items']) for e in q['evidence_ids'])
    reports=[]
    for mode,target,report in [('design',AS/'analysis-design.json',REVIEW/(SLUG+'-design-check.md')),('lint',ROOT/(SLUG+'-student-worksheet.docx'),REVIEW/(SLUG+'-lint.md'))]:
        cmd=[sys.executable,str(SKILL/'scripts'/'worksheet_check.py'),mode,str(target)]
        if mode=='design':cmd+=['--homework-minutes','120']
        else:cmd+=['-o',str(report)]
        r=subprocess.run(cmd,text=True,encoding='utf-8',capture_output=True)
        if mode=='design':report.write_text(r.stdout,encoding='utf-8')
        print(mode,r.returncode,r.stderr.strip());reports.append(r.returncode)
    assert not any(reports),'Fix QA errors before release'

def manifest(design):
    o={k:f'{SLUG}-{s}.docx' for k,s in [('student_worksheet','student-worksheet'),('evidence_pack','evidence-pack'),('teaching_plan','teaching-plan-flipped-classroom'),('design_rationale','design-rationale')]};o['notebook']='notebook/'+SLUG+'-lab.ipynb'
    d=dict(topic='第9讲 SVM与Margin',slug='svm-margin',mode='algorithm choice',dataset=dict(name='UCI Vertebral Column',source=SOURCE,task='binary classification',target='AB=1,NO=0'),outputs=o,assets=[str(p.relative_to(ROOT)).replace('\\','/') for p in AS.rglob('*') if p.is_file()],boundary='教学研究；不构成临床、因果或部署结论。',core_question=PROBLEM,problem_architecture=design['problem_architecture'],course_coverage=design['course_coverage'],customized=True,course_brief=['plan.md','svm.md','../../../plan.docx'],analysis_design='assets/pbl-svm-margin/analysis-design.json',build_script='scripts/pbl-svm-margin-build.py',qa=dict(lint='review/'+SLUG+'-lint.md',design='review/'+SLUG+'-design-check.md',pilot_fields='review/pilot-fields.json',pilot_fill='review/'+SLUG+'-pilot-fill.docx',pilot_report='review/'+SLUG+'-pilot-report.md',status='awaiting-independent-pilot-and-render'),student_release=[str(p.relative_to(ROOT)).replace('\\','/') for p in (ROOT/'student-release').glob('*.docx')])
    dump(ROOT/'package-manifest.json',d)

def main():
    a=analyze();design=make_design();worksheet(a);evidence_doc(a);teacher_doc(a,design);rationale_doc(design);notebook();release_copies();validate(design);manifest(design)
    print('Four documents and auxiliary assets generated; pilot and visual QA still required.')

if __name__=='__main__':main()
