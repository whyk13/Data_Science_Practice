from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
DATA_PATH = ROOT / "source-data" / "acute-inflammations" / "diagnosis.data"
TARGET = "bladder_inflammation"
RANDOM_STATE = 7
FEATURES = [
    "fever_ge_38",
    "nausea",
    "lumbar_pain",
    "urine_pushing",
    "micturition_pains",
    "burning_urethra",
]


def metric_row(y_true, probability, threshold=0.5):
    prediction = probability >= threshold
    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, prediction),
        "sensitivity": recall_score(y_true, prediction, zero_division=0),
        "specificity": recall_score(
            y_true, prediction, pos_label=0, zero_division=0
        ),
        "precision": precision_score(y_true, prediction, zero_division=0),
        "f1": f1_score(y_true, prediction, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probability),
        "pr_auc": average_precision_score(y_true, probability),
        "brier": brier_score_loss(y_true, probability),
        "log_loss": log_loss(y_true, probability),
    }


def wilson_interval(successes, total, z=1.959963984540054):
    proportion = successes / total
    denominator = 1 + z**2 / total
    center = (proportion + z**2 / (2 * total)) / denominator
    half_width = (
        z
        * np.sqrt(
            proportion * (1 - proportion) / total + z**2 / (4 * total**2)
        )
        / denominator
    )
    return center - half_width, center + half_width


columns = [
    "temperature",
    "nausea",
    "lumbar_pain",
    "urine_pushing",
    "micturition_pains",
    "burning_urethra",
    "bladder_inflammation",
    "nephritis",
]
data = pd.read_csv(
    DATA_PATH, sep="\t", header=None, names=columns, encoding="utf-16"
)
data["temperature"] = (
    data["temperature"].str.replace(",", ".", regex=False).astype(float)
)
for column in columns[1:]:
    data[column] = (data[column].str.strip() == "yes").astype(int)
data["fever_ge_38"] = (data["temperature"] >= 38).astype(int)

x = data[FEATURES]
y = data[TARGET]
x_train_validation, x_test, y_train_validation, y_test = train_test_split(
    x, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)
x_train, x_validation, y_train, y_validation = train_test_split(
    x_train_validation,
    y_train_validation,
    test_size=0.25,
    random_state=RANDOM_STATE,
    stratify=y_train_validation,
)

model = BernoulliNB(alpha=1.0).fit(x_train, y_train)
dummy = DummyClassifier(strategy="prior").fit(x_train, y_train)
test_probability = model.predict_proba(x_test)[:, 1]
dummy_probability = dummy.predict_proba(x_test)[:, 1]

model_metrics = metric_row(y_test, test_probability)
dummy_metrics = metric_row(y_test, dummy_probability)
metric_table = pd.DataFrame(
    [
        {"model": "Prior-only baseline", **dummy_metrics},
        {"model": "Bernoulli naive Bayes", **model_metrics},
    ]
)
metric_table.to_csv(ARTIFACTS / "planning-clean-test-metrics.csv", index=False)

threshold_table = pd.DataFrame(
    [
        metric_row(y_validation, model.predict_proba(x_validation)[:, 1], threshold)
        for threshold in [0.10, 0.25, 0.50, 0.75, 0.90]
    ]
)
threshold_table.to_csv(ARTIFACTS / "planning-validation-thresholds.csv", index=False)

clean_prediction = test_probability >= 0.5
clean_matrix = confusion_matrix(y_test, clean_prediction)
tn, fp, fn, tp = clean_matrix.ravel()
interval_table = pd.DataFrame(
    [
        {
            "metric": "accuracy",
            "estimate": (tp + tn) / len(y_test),
            "ci_low": wilson_interval(tp + tn, len(y_test))[0],
            "ci_high": wilson_interval(tp + tn, len(y_test))[1],
            "denominator": len(y_test),
        },
        {
            "metric": "sensitivity",
            "estimate": tp / (tp + fn),
            "ci_low": wilson_interval(tp, tp + fn)[0],
            "ci_high": wilson_interval(tp, tp + fn)[1],
            "denominator": tp + fn,
        },
        {
            "metric": "specificity",
            "estimate": tn / (tn + fp),
            "ci_low": wilson_interval(tn, tn + fp)[0],
            "ci_high": wilson_interval(tn, tn + fp)[1],
            "denominator": tn + fp,
        },
    ]
)
interval_table.to_csv(ARTIFACTS / "planning-clean-test-wilson-ci.csv", index=False)

# A deliberately redundant feature: the same nausea response is entered twice.
x_train_duplicate = x_train.assign(nausea_duplicate=x_train["nausea"])
duplicate_model = BernoulliNB(alpha=1.0).fit(x_train_duplicate, y_train)
all_profiles = pd.DataFrame(
    [
        [(profile >> bit) & 1 for bit in range(len(FEATURES))]
        for profile in range(2 ** len(FEATURES))
    ],
    columns=FEATURES,
)
all_profiles_duplicate = all_profiles.assign(
    nausea_duplicate=all_profiles["nausea"]
)
profile_audit = all_profiles.copy()
profile_audit["original_probability"] = model.predict_proba(all_profiles)[:, 1]
profile_audit["duplicate_probability"] = duplicate_model.predict_proba(
    all_profiles_duplicate
)[:, 1]
profile_audit["probability_change"] = (
    profile_audit["duplicate_probability"]
    - profile_audit["original_probability"]
)
profile_audit["decision_changed"] = (
    (profile_audit["original_probability"] >= 0.5)
    != (profile_audit["duplicate_probability"] >= 0.5)
)
profile_audit.to_csv(ARTIFACTS / "planning-duplicate-feature-audit.csv", index=False)

# Stress test: independently flip each binary symptom entry with a known probability.
rng = np.random.default_rng(20260903)
stress_rows = []
for flip_rate in [0.00, 0.02, 0.05, 0.10, 0.15, 0.20]:
    repeated_metrics = []
    for repetition in range(500):
        flip_mask = rng.random(x_test.shape) < flip_rate
        stressed = np.logical_xor(x_test.to_numpy(dtype=bool), flip_mask).astype(int)
        stressed = pd.DataFrame(stressed, columns=FEATURES, index=x_test.index)
        stressed_probability = model.predict_proba(stressed)[:, 1]
        row = metric_row(y_test, stressed_probability)
        repeated_metrics.append(
            [row["accuracy"], row["sensitivity"], row["specificity"]]
        )
    repeated_metrics = np.asarray(repeated_metrics)
    for metric_index, metric_name in enumerate(
        ["accuracy", "sensitivity", "specificity"]
    ):
        values = repeated_metrics[:, metric_index]
        stress_rows.append(
            {
                "flip_rate": flip_rate,
                "metric": metric_name,
                "mean": values.mean(),
                "q025": np.quantile(values, 0.025),
                "q975": np.quantile(values, 0.975),
            }
        )
stress_table = pd.DataFrame(stress_rows)
stress_table.to_csv(ARTIFACTS / "planning-measurement-error-stress.csv", index=False)

conditional_correlations = []
for disease_status in [0, 1]:
    correlation = x[y == disease_status].corr()
    for row_index, row_name in enumerate(FEATURES):
        for column_index in range(row_index + 1, len(FEATURES)):
            column_name = FEATURES[column_index]
            conditional_correlations.append(
                {
                    "target_status": disease_status,
                    "feature_1": row_name,
                    "feature_2": column_name,
                    "phi_correlation": correlation.loc[row_name, column_name],
                }
            )
pd.DataFrame(conditional_correlations).to_csv(
    ARTIFACTS / "planning-conditional-dependence.csv", index=False
)

dataset_summary = pd.DataFrame(
    [
        {
            "dataset": "UCI Acute Inflammations",
            "samples": len(data),
            "original_features": 6,
            "model_features": len(FEATURES),
            "target": TARGET,
            "positive_n": int(y.sum()),
            "negative_n": int((1 - y).sum()),
            "missing_cells": int(data.isna().sum().sum()),
            "split_train": len(y_train),
            "split_validation": len(y_validation),
            "split_test": len(y_test),
        }
    ]
)
dataset_summary.to_csv(ARTIFACTS / "planning-dataset-summary.csv", index=False)

plt.style.use("seaborn-v0_8-whitegrid")
figure, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

axis = axes[0, 0]
image = axis.imshow(clean_matrix, cmap="Blues", vmin=0)
for row_index in range(2):
    for column_index in range(2):
        axis.text(
            column_index,
            row_index,
            str(clean_matrix[row_index, column_index]),
            ha="center",
            va="center",
            fontsize=18,
            fontweight="bold",
        )
axis.set_xticks([0, 1], ["Predicted no", "Predicted yes"])
axis.set_yticks([0, 1], ["Actual no", "Actual yes"])
axis.set_title("A. Locked test looks perfect (n=24)", loc="left", fontweight="bold")
axis.set_xlabel("Prediction")
axis.set_ylabel("Reference label")
figure.colorbar(image, ax=axis, shrink=0.8)

axis = axes[0, 1]
metric_names = ["accuracy", "sensitivity", "specificity", "precision"]
positions = np.arange(len(metric_names))
width = 0.36
axis.bar(
    positions - width / 2,
    [dummy_metrics[name] for name in metric_names],
    width,
    label="Prior-only baseline",
    color="#9aa4b2",
)
axis.bar(
    positions + width / 2,
    [model_metrics[name] for name in metric_names],
    width,
    label="Naive Bayes",
    color="#2563eb",
)
axis.set_xticks(positions, [name.replace("_", "\n") for name in metric_names])
axis.set_ylim(0, 1.08)
axis.set_ylabel("Metric value")
axis.set_title("B. Accuracy alone hides baseline failure", loc="left", fontweight="bold")
axis.legend(frameon=False, loc="lower right")

axis = axes[1, 0]
colors = np.where(profile_audit["decision_changed"], "#dc2626", "#2563eb")
axis.scatter(
    profile_audit["original_probability"],
    profile_audit["duplicate_probability"],
    c=colors,
    alpha=0.8,
    edgecolor="white",
    linewidth=0.4,
)
axis.plot([0, 1], [0, 1], color="#111827", linestyle="--", linewidth=1)
axis.axvline(0.5, color="#6b7280", linestyle=":", linewidth=1)
axis.axhline(0.5, color="#6b7280", linestyle=":", linewidth=1)
axis.set_xlim(-0.03, 1.03)
axis.set_ylim(-0.03, 1.03)
axis.set_xlabel("P(disease) with original features")
axis.set_ylabel("P(disease) after duplicating nausea")
changed = int(profile_audit["decision_changed"].sum())
axis.set_title(
    f"C. Redundant evidence changes {changed}/64 possible decisions",
    loc="left",
    fontweight="bold",
)

axis = axes[1, 1]
palette = {
    "accuracy": "#2563eb",
    "sensitivity": "#dc2626",
    "specificity": "#059669",
}
for metric_name, metric_color in palette.items():
    subset = stress_table[stress_table["metric"] == metric_name]
    axis.plot(
        subset["flip_rate"] * 100,
        subset["mean"],
        marker="o",
        color=metric_color,
        label=metric_name.capitalize(),
    )
    axis.fill_between(
        subset["flip_rate"] * 100,
        subset["q025"],
        subset["q975"],
        color=metric_color,
        alpha=0.10,
    )
axis.set_ylim(0, 1.05)
axis.set_xlabel("Independent symptom-entry flip rate (%)")
axis.set_ylabel("Mean metric over 500 simulations")
axis.set_title("D. Measurement error exposes brittleness", loc="left", fontweight="bold")
axis.legend(frameon=False, ncol=3, loc="lower left")

figure.suptitle(
    "Why a 100% test score is not enough: UCI Acute Inflammations",
    fontsize=17,
    fontweight="bold",
)
figure.savefig(
    ARTIFACTS / "main-result-clean-vs-audit.png", dpi=180, bbox_inches="tight"
)
plt.close(figure)

print(dataset_summary.to_string(index=False))
print("\nClean test metrics\n", metric_table.round(4).to_string(index=False))
print("\nWilson intervals\n", interval_table.round(4).to_string(index=False))
print("\nValidation thresholds\n", threshold_table.round(4).to_string(index=False))
print(
    "\nDuplicate feature audit:",
    int(profile_audit["decision_changed"].sum()),
    "of 64 possible profiles changed class at threshold 0.5; max probability shift",
    round(profile_audit["probability_change"].abs().max(), 4),
)
print("\nStress test means\n", stress_table.pivot(index="flip_rate", columns="metric", values="mean").round(4).to_string())
print("\nMain figure", ARTIFACTS / "main-result-clean-vs-audit.png")
