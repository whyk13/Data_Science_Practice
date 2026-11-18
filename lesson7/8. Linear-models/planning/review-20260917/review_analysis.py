"""Planning-only audit. All outputs remain beside this script; no final package writes."""
from pathlib import Path
import io
import json
import urllib.request
import zipfile
import hashlib

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import expit
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score, confusion_matrix, brier_score_loss

ROOT = Path(__file__).resolve().parent
PACKAGE = ROOT.parent.parent
SOURCE = PACKAGE / 'assets/pbl-linear-models-lda-logistic-regression/statlog-heart.csv'
data = pd.read_csv(SOURCE)
target = 'heart_disease_present'
features = list(data.drop(columns=target).columns)
continuous = ['age', 'resting_bp', 'serum_cholesterol', 'max_heart_rate', 'oldpeak']
names = {'age':'年龄', 'resting_bp':'静息血压', 'serum_cholesterol':'血清胆固醇', 'max_heart_rate':'运动最大心率', 'oldpeak':'ST 段压低'}
levels = {'sex':[0,1], 'chest_pain':[1,2,3,4], 'fasting_blood_sugar':[0,1], 'resting_ecg':[0,1,2], 'exercise_angina':[0,1], 'st_slope':[1,2,3], 'major_vessels':[0,1,2,3], 'thal':[3,6,7], target:[0,1]}

# Verify local numeric values and label conversion against the original source.
url = 'https://archive.ics.uci.edu/static/public/145/statlog%2Bheart.zip'
with urllib.request.urlopen(url, timeout=30) as response:
    payload = response.read()
with zipfile.ZipFile(io.BytesIO(payload)) as archive:
    raw = pd.read_csv(io.BytesIO(archive.read('heart.dat')), sep=r'\s+', header=None)
    codebook = archive.read('heart.doc').decode('utf-8', errors='replace')
raw.columns = features + [target]
raw[target] = raw[target] - 1
raw_equal = bool(np.array_equal(raw.to_numpy(), data.to_numpy()))
assert raw_equal, 'Local data mismatch; halt for source reconciliation.'

audit = {
    'source_url':url, 'source_archive_sha256':hashlib.sha256(payload).hexdigest(),
    'local_matches_raw_after_target_minus_one':raw_equal,
    'shape':list(data.shape), 'n_features':len(features),
    'missing_cells':int(data.isna().sum().sum()),
    'duplicate_full_records':int(data.duplicated().sum()),
    'duplicate_feature_rows':int(data[features].duplicated().sum()),
    'patient_linkage':'No patient ID, centre or time fields; no duplicate row does not prove patient independence.',
    'target_counts':data[target].value_counts().sort_index().to_dict(),
    'source_codebook':codebook,
    'schema_discrepancies':[
        'Old teaching dictionary labels thal as thalassemia; original codebook only supports 3 normal, 6 fixed defect, 7 reversible defect. Remove unsupported translation.',
        'major_vessels is an observed 0-3 integer count, not a continuous measurement even if standardized for a model.',
        'Statlog does not publish patient IDs, temporal provenance or category frequencies. Category frequencies below are local observations, not externally verified recruitment counts.',
    ],
    'levels': {k:{'documented':v,'observed_counts':data[k].value_counts().sort_index().to_dict(),'nonmissing_n':int(data[k].notna().sum()),'out_of_codebook_n':int((~data[k].isin(v)).sum())} for k,v in levels.items()},
    'continuous_ranges':{k:{'min':float(data[k].min()),'max':float(data[k].max()),'zero_n':int((data[k]==0).sum())} for k in continuous},
    'sentinel_interpretation':'No ?/NaN in source table. Zero is valid for binary fields and some measurements (e.g. oldpeak); no zero was silently recoded. Nonpositive age, blood pressure, cholesterol, maximum heart rate checked separately.',
    'impossible_nonpositive_n':{k:int((data[k]<=0).sum()) for k in continuous if k!='oldpeak'},
}

# Fixed, simple teaching partition: same 54-row holdout as the old package;
# author has previously seen it, so it is not fresh independent validation.
dev, test = train_test_split(np.arange(len(data)), test_size=.2, stratify=data[target], random_state=2026)
train, val = train_test_split(dev, test_size=.25, stratify=data.iloc[dev][target], random_state=2026)
x, y = data[features], data[target]
audit['splits']={k:{'n':len(ix),'positive_n':int(y.iloc[ix].sum())} for k,ix in [('train',train),('validation',val),('test',test)]}
audit['test_status']='Same historical teaching holdout previously inspected by author; no new untouched test claim. New exploratory runs use train/validation only.'

def logistic():
    return LogisticRegression(C=np.inf, max_iter=10000, tol=1e-9)

models={
    'Logistic（5项）':make_pipeline(StandardScaler(),logistic()),
    'LDA（5项）':make_pipeline(StandardScaler(),LinearDiscriminantAnalysis()),
}
rows=[]
fitted={}
for name, model in models.items():
    model.fit(x.iloc[train][continuous], y.iloc[train])
    fitted[name]=model
    p=model.predict_proba(x.iloc[val][continuous])[:,1]
    for t in [.5,.2]:
        tn,fp,fn,tp=confusion_matrix(y.iloc[val],p>=t,labels=[0,1]).ravel()
        rows.append({'模型':name,'验证例数':len(val),'阈值':t,'AUC':roc_auc_score(y.iloc[val],p),'TN':int(tn),'FP':int(fp),'FN':int(fn),'TP':int(tp),'敏感度':tp/(tp+fn),'特异度':tn/(tn+fp),'Brier':brier_score_loss(y.iloc[val],p)})

cat=['chest_pain','resting_ecg','st_slope','thal']
pre=ColumnTransformer([
    ('numeric',StandardScaler(),continuous+['major_vessels']),
    ('binary','passthrough',['sex','fasting_blood_sugar','exercise_angina']),
    ('categorical',OneHotEncoder(categories=[[1,2,3,4],[0,1,2],[1,2,3],[3,6,7]],drop='first',sparse_output=False),cat),
])
full=make_pipeline(pre,logistic()).fit(x.iloc[train],y.iloc[train])
p=full.predict_proba(x.iloc[val])[:,1]
tn,fp,fn,tp=confusion_matrix(y.iloc[val],p>=.5,labels=[0,1]).ravel()
rows.append({'模型':'Logistic（13项）','验证例数':len(val),'阈值':.5,'AUC':roc_auc_score(y.iloc[val],p),'TN':int(tn),'FP':int(fp),'FN':int(fn),'TP':int(tp),'敏感度':tp/(tp+fn),'特异度':tn/(tn+fp),'Brier':brier_score_loss(y.iloc[val],p)})
from sklearn.base import clone
full_lda=make_pipeline(clone(pre),LinearDiscriminantAnalysis()).fit(x.iloc[train],y.iloc[train])
for model_name,model,thresholds in [('LDA（13项）',full_lda,[.5,.2]),('Logistic（13项）',full,[.2])]:
    p=model.predict_proba(x.iloc[val])[:,1]
    for threshold in thresholds:
        tn,fp,fn,tp=confusion_matrix(y.iloc[val],p>=threshold,labels=[0,1]).ravel()
        rows.append({'模型':model_name,'验证例数':len(val),'阈值':threshold,'AUC':roc_auc_score(y.iloc[val],p),'TN':int(tn),'FP':int(fp),'FN':int(fn),'TP':int(tp),'敏感度':tp/(tp+fn),'特异度':tn/(tn+fp),'Brier':brier_score_loss(y.iloc[val],p)})
metrics=pd.DataFrame(rows)
metrics.to_csv(ROOT/'planning-validation.csv',index=False,encoding='utf-8-sig')

lr=fitted['Logistic（5项）']; scaler=lr[0]; beta=lr[1].coef_[0]
coefs=pd.DataFrame({'变量':[names[c] for c in continuous],'训练均值':scaler.mean_,'训练SD':scaler.scale_,'每SD系数':beta,'每原单位系数':beta/scaler.scale_,'每SD_OR':np.exp(beta)})
coefs.to_csv(ROOT/'planning-coefficients.csv',index=False,encoding='utf-8-sig')
full_coefs=pd.DataFrame({'feature':full[0].get_feature_names_out(),'beta':full[1].coef_[0], 'OR':np.exp(full[1].coef_[0])})
full_coefs.to_csv(ROOT/'planning-full-model-coefficients.csv',index=False)

# Unit reparameterization is distinct from the deliberately redundant feature.
unit_train=x.iloc[train][continuous].copy();unit_val=x.iloc[val][continuous].copy()
unit_train['age']/=10;unit_val['age']/=10
unit_model=make_pipeline(StandardScaler(),logistic()).fit(unit_train,y.iloc[train])
p0=lr.predict_proba(x.iloc[val][continuous])[:,1]
audit['unit_change_max_probability_difference']=float(np.max(np.abs(p0-unit_model.predict_proba(unit_val)[:,1])))
rng=np.random.default_rng(2026)
stress_train=x.iloc[train][continuous].copy();stress_val=x.iloc[val][continuous].copy()
stress_train['age_repeat']=stress_train['age']+rng.normal(0,.25,len(train))
stress_val['age_repeat']=stress_val['age']+rng.normal(0,.25,len(val))
stress=make_pipeline(StandardScaler(),logistic()).fit(stress_train,y.iloc[train])
audit['collinearity_train_validation_only']={
    'training_age_correlation':float(stress_train[['age','age_repeat']].corr().iloc[0,1]),
    'original_age_coefficient':float(beta[0]),
    'stress_age_coefficient':float(stress[1].coef_[0,0]),
    'stress_age_repeat_coefficient':float(stress[1].coef_[0,-1]),
    'original_validation_auc':float(roc_auc_score(y.iloc[val],p0)),
    'stress_validation_auc':float(roc_auc_score(y.iloc[val],stress.predict_proba(stress_val)[:,1])),
    'synthetic_disclosure':'Additional age measurement = age + independent N(0, 0.25^2 years^2) noise; original data unchanged.',
}
group_stats=x.iloc[train][continuous].assign(结局=y.iloc[train].to_numpy()).groupby('结局').agg(['mean','std','skew'])
group_stats.to_csv(ROOT/'planning-training-assumptions.csv',encoding='utf-8-sig')
audit['validation_metrics']=rows
audit['coefficients']=coefs.to_dict('records')
(ROOT/'planning-data-audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')

plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei','DejaVu Sans']
plt.rcParams['axes.unicode_minus']=False
fig, ax=plt.subplots(1,2,figsize=(12,4.7),constrained_layout=True)
small=metrics[(metrics['阈值']==.5)&metrics['模型'].isin(['Logistic（13项）','LDA（13项）'])]
pos=np.arange(len(small)); width=.25
for j,col in enumerate(['AUC','敏感度','特异度']):
    ax[0].bar(pos+(j-1)*width,small[col],width,label=col)
ax[0].set_xticks(pos,small['模型']);ax[0].set_ylim(0,1.08)
ax[0].set_ylabel('验证集指标（0–1）');ax[0].set_xlabel('原始13项特征；分类编码后18列；阈值 0.5')
ax[0].legend(loc='lower left');ax[0].set_title('预分析：同一验证集上的两种模型（54例）')
ax[1].barh(coefs['变量'],full[1].coef_[0,:5],color='#286b83');ax[1].axvline(0,color='gray',lw=1)
ax[1].set_xlabel('Logistic 系数：每增加训练集 1 SD');ax[1].set_title('13项模型的五个连续变量系数（训练162例）')
fig.suptitle('第8讲新版方案预分析｜数据：UCI Statlog Heart｜固定划分，无测试集结果')
fig.savefig(ROOT/'planning-main-evidence.png',dpi=170);plt.close(fig)
print(json.dumps({k:audit[k] for k in ['local_matches_raw_after_target_minus_one','missing_cells','duplicate_full_records','splits','validation_metrics','collinearity_train_validation_only','unit_change_max_probability_difference']},ensure_ascii=False,indent=2))
