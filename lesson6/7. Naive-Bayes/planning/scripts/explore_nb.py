from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    log_loss,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB


PLANNING_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = (
    PLANNING_ROOT / "source-data" / "acute-inflammations" / "diagnosis.data"
)
COLUMNS = [
    "temperature",
    "nausea",
    "lumbar_pain",
    "urine_pushing",
    "micturition_pains",
    "burning_urethra",
    "bladder_inflammation",
    "nephritis",
]


df = pd.read_csv(DATA_PATH, sep="\t", header=None, names=COLUMNS, encoding="utf-16")
df["temperature"] = (
    df["temperature"].str.replace(",", ".", regex=False).astype(float)
)
for column in COLUMNS[1:]:
    df[column] = (df[column].str.strip() == "yes").astype(int)

features = df[
    [
        "nausea",
        "lumbar_pain",
        "urine_pushing",
        "micturition_pains",
        "burning_urethra",
    ]
].copy()
features.insert(0, "fever_ge_38", (df["temperature"] >= 38).astype(int))

print("shape", df.shape)
print("missing", df.isna().sum().to_dict())
print("positive counts", df[["bladder_inflammation", "nephritis"]].sum().to_dict())

for target in ["bladder_inflammation", "nephritis"]:
    y = df[target]
    x_tv, x_test, y_tv, y_test = train_test_split(
        features, y, test_size=0.20, random_state=7, stratify=y
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_tv, y_tv, test_size=0.25, random_state=7, stratify=y_tv
    )
    for alpha in [0.1, 1.0, 2.0]:
        model = BernoulliNB(alpha=alpha).fit(x_train, y_train)
        test_probability = model.predict_proba(x_test)[:, 1]
        test_prediction = test_probability >= 0.50
        metrics = {
            "accuracy": accuracy_score(y_test, test_prediction),
            "sensitivity": recall_score(y_test, test_prediction),
            "specificity": recall_score(y_test, test_prediction, pos_label=0),
            "precision": precision_score(y_test, test_prediction, zero_division=0),
            "f1": f1_score(y_test, test_prediction),
            "roc_auc": roc_auc_score(y_test, test_probability),
            "pr_auc": average_precision_score(y_test, test_probability),
            "brier": brier_score_loss(y_test, test_probability),
        }
        print(
            target,
            "alpha",
            alpha,
            "sizes",
            (len(y_train), len(y_val), len(y_test)),
            "test_prevalence",
            round(y_test.mean(), 3),
            "metrics",
            {name: round(value, 3) for name, value in metrics.items()},
            "confusion_matrix",
            confusion_matrix(y_test, test_prediction).tolist(),
        )

    base_model = BernoulliNB(alpha=1.0).fit(x_train, y_train)
    duplicate_column = "nausea"
    x_train_duplicate = x_train.copy()
    x_train_duplicate[f"{duplicate_column}_duplicate"] = x_train_duplicate[
        duplicate_column
    ]
    duplicate_model = BernoulliNB(alpha=1.0).fit(x_train_duplicate, y_train)
    all_profiles = pd.DataFrame(
        [
            [(profile >> bit) & 1 for bit in range(features.shape[1])]
            for profile in range(2 ** features.shape[1])
        ],
        columns=features.columns,
    )
    all_profiles_duplicate = all_profiles.copy()
    all_profiles_duplicate[f"{duplicate_column}_duplicate"] = all_profiles_duplicate[
        duplicate_column
    ]
    base_probability = base_model.predict_proba(all_profiles)[:, 1]
    duplicate_probability = duplicate_model.predict_proba(all_profiles_duplicate)[:, 1]
    profile_audit = all_profiles.copy()
    profile_audit["base_probability"] = base_probability
    profile_audit["duplicate_probability"] = duplicate_probability
    profile_audit["absolute_change"] = np.abs(
        profile_audit["duplicate_probability"] - profile_audit["base_probability"]
    )
    print(
        target,
        "largest duplicate-feature probability changes\n",
        profile_audit.nlargest(5, "absolute_change").round(4).to_string(index=False),
    )
    for name, model, xv in [
        ("base", base_model, x_val),
        (
            "duplicate",
            duplicate_model,
            x_val.assign(nausea_duplicate=x_val[duplicate_column]),
        ),
    ]:
        probabilities = model.predict_proba(xv)[:, 1]
        print(
            target,
            name,
            "validation probability summary",
            np.round(np.quantile(probabilities, [0, 0.25, 0.5, 0.75, 1]), 4),
            "log_loss",
            round(log_loss(y_val, probabilities), 5),
            "brier",
            round(brier_score_loss(y_val, probabilities), 5),
        )

    print(target, "conditional correlations")
    for disease_status in [0, 1]:
        print(
            "class",
            disease_status,
            "\n",
            features[y == disease_status].corr().round(2).to_string(),
        )
