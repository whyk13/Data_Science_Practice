"""Read-only source audit; all audit outputs remain in this lesson's planning folder."""
from pathlib import Path
import hashlib
import json
import importlib.util
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import confusion_matrix, average_precision_score, precision_recall_curve, auc, log_loss

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'planning' / 'review-20260917'
OUT.mkdir(exist_ok=True)
SOURCE = ROOT / 'planning/source-data/acute-inflammations/diagnosis.data'
names = ['temperature_c','nausea','lumbar_pain','urine_pushing','micturition_pains','burning_urethra','bladder_inflammation','nephritis']
raw = pd.read_csv(SOURCE, sep='\t', encoding='utf-16', header=None, names=names, decimal=',')
stored = pd.read_csv(ROOT / 'assets/pbl-machine-learning-basics-naive-bayes/acute-inflammations.csv')
pd.testing.assert_frame_equal(raw, stored, check_dtype=False)
df = raw.copy()
for c in names[1:]:
    assert set(df[c].dropna()) == {'yes','no'}
    df[c] = df[c].map({'yes':1,'no':0})
df['fever_ge_38'] = (df.temperature_c >= 38).astype(int)
features = ['fever_ge_38'] + names[1:6]
x,y = df[features],df.bladder_inflammation
dev,te = train_test_split(np.arange(len(df)),test_size=.2,random_state=7,stratify=y)
tr,va = train_test_split(dev,test_size=.25,random_state=7,stratify=y.iloc[dev])
patterns = [tuple(row) for row in x.to_numpy()]
rawpatterns = [tuple(row) for row in raw[names[:6]].to_numpy()]
model = BernoulliNB(alpha=1).fit(x.iloc[tr],y.iloc[tr])
p = model.predict_proba(x.iloc[va])[:,1]
thresholds=[]
for t in [.1,.25,.5,.75,.9]:
    tn,fp,fn,tp=confusion_matrix(y.iloc[va],p>=t).ravel()
    thresholds.append(dict(threshold=t,TN=int(tn),FP=int(fp),FN=int(fn),TP=int(tp),sensitivity=tp/(tp+fn),specificity=tn/(tn+fp),precision=tp/(tp+fp)))
pd.DataFrame(thresholds).to_csv(OUT/'validation-threshold-audit.csv',index=False)
audit = dict(n=len(df),features=6,labels=2,source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),source_encoding='UTF-16 with BOM; codebook says ASCII',source_matches_course_csv=True,missing_cells=int(raw.isna().sum().sum()),temperature_range=[float(raw.temperature_c.min()),float(raw.temperature_c.max())],positive_counts={c:int(df[c].sum()) for c in names[6:]},raw_duplicate_excess=int(raw.duplicated().sum()),raw_rows_in_duplicate_groups=int(raw.duplicated(keep=False).sum()),raw_unique_input_profiles=len(set(rawpatterns)),binary_unique_input_profiles=len(set(patterns)),binary_unique_labelled_profiles=len(df[features+['bladder_inflammation']].drop_duplicates()),binary_profiles_with_conflicting_primary_label=int((df.groupby(features).bladder_inflammation.nunique()>1).sum()),splits={})
for key,idx in [('train',tr),('validation',va),('test',te)]:
    audit['splits'][key]=dict(n=len(idx),positive=int(y.iloc[idx].sum()),unique_binary_profiles=len({patterns[i] for i in idx}),binary_profiles_previously_seen_in_train=None if key=='train' else sum(patterns[i] in {patterns[j] for j in tr} for i in idx),raw_input_profiles_previously_seen_in_train=None if key=='train' else sum(rawpatterns[i] in {rawpatterns[j] for j in tr} for i in idx))
audit['test_cm']=confusion_matrix(y.iloc[te],model.predict(x.iloc[te])).tolist()
audit['validation_log_loss']=float(log_loss(y.iloc[va],p))
prec,rec,_=precision_recall_curve(y.iloc[va],p)
audit['validation_average_precision']=float(average_precision_score(y.iloc[va],p))
audit['validation_pr_auc_trapezoid']=float(auc(rec,prec))
toy_y=np.array([1,0,1,0]); toy_p=np.array([.9,.8,.7,.1])
prec,rec,_=precision_recall_curve(toy_y,toy_p)
audit['explicit_synthetic_pr_comparison']=dict(labels=toy_y.tolist(),scores=toy_p.tolist(),average_precision=float(average_precision_score(toy_y,toy_p)),pr_auc_trapezoid=float(auc(rec,prec)))
(OUT/'source-and-analysis-audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(audit,ensure_ascii=False,indent=2))
print(pd.DataFrame(thresholds).to_string(index=False))
checker = ROOT.parents[2]/'.codex/skills/pbl-ml-package/scripts/worksheet_check.py'
spec=importlib.util.spec_from_file_location('worksheet_checker',checker)
qa=importlib.util.module_from_spec(spec);spec.loader.exec_module(qa)
report,bad=qa.design(ROOT/'assets/pbl-machine-learning-basics-naive-bayes/analysis-design.json',120)
(OUT/'existing-design-check.md').write_text(report,encoding='utf-8')
print(report)
