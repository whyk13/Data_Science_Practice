from __future__ import annotations

import io
import json
import urllib.request
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


OUT = Path(__file__).resolve().parent
SOURCE_URL = "https://archive.ics.uci.edu/static/public/145/statlog%2Bheart.zip"
RANDOM_STATE = 2026

FEATURES = [
    "age",
    "sex",
    "chest_pain",
    "resting_bp",
    "serum_cholesterol",
    "fasting_blood_sugar",
    "resting_ecg",
    "max_heart_rate",
    "exercise_angina",
    "oldpeak",
    "st_slope",
    "major_vessels",
    "thal",
]
CONTINUOUS = [
    "age",
    "resting_bp",
    "serum_cholesterol",
    "max_heart_rate",
    "oldpeak",
    "major_vessels",
]
BINARY = ["sex", "fasting_blood_sugar", "exercise_angina"]
CATEGORICAL = ["chest_pain", "resting_ecg", "st_slope", "thal"]
CATEGORY_LEVELS = [[1, 2, 3, 4], [0, 1, 2], [1, 2, 3], [3, 6, 7]]


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    with urllib.request.urlopen(SOURCE_URL, timeout=30) as response:
        payload = response.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        raw = archive.read("heart.dat")
    frame = pd.read_csv(io.BytesIO(raw), sep=r"\s+", header=None)
    frame.columns = FEATURES + ["target_raw"]
    x = frame[FEATURES].copy()
    y = (frame["target_raw"] == 2).astype(int).rename("heart_disease_present")
    return x, y


def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("continuous", StandardScaler(), CONTINUOUS),
            ("binary", "passthrough", BINARY),
            (
                "categorical",
                OneHotEncoder(
                    categories=CATEGORY_LEVELS,
                    drop="first",
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                CATEGORICAL,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_models() -> dict[str, Pipeline]:
    return {
        "Logistic regression": Pipeline(
            [
                ("preprocess", make_preprocessor()),
                (
                    "model",
                    LogisticRegression(
                        C=np.inf,
                        solver="lbfgs",
                        max_iter=5000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "LDA": Pipeline(
            [
                ("preprocess", make_preprocessor()),
                ("model", LinearDiscriminantAnalysis(solver="svd")),
            ]
        ),
        "Gaussian NB": Pipeline(
            [
                ("preprocess", make_preprocessor()),
                ("model", GaussianNB()),
            ]
        ),
        "Prevalence baseline": Pipeline(
            [
                ("preprocess", make_preprocessor()),
                ("model", DummyClassifier(strategy="prior")),
            ]
        ),
    }


def classification_metrics(y_true: pd.Series | np.ndarray, prob: np.ndarray, threshold: float) -> dict[str, float | int]:
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "roc_auc": roc_auc_score(y_true, prob),
        "pr_auc": average_precision_score(y_true, prob),
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "sensitivity": recall_score(y_true, pred, pos_label=1),
        "specificity": recall_score(y_true, pred, pos_label=0),
        "precision": precision_score(y_true, pred, zero_division=0),
        "brier": brier_score_loss(y_true, prob),
        "log_loss": log_loss(y_true, prob, labels=[0, 1]),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "weighted_error_cost": int(fp + 5 * fn),
        "threshold": float(threshold),
    }


def select_cost_threshold(y_true: pd.Series, prob: np.ndarray) -> tuple[float, pd.DataFrame]:
    rows = []
    for threshold in np.round(np.linspace(0.05, 0.95, 181), 3):
        m = classification_metrics(y_true, prob, float(threshold))
        rows.append(
            {
                "threshold": threshold,
                "false_positive": m["fp"],
                "false_negative": m["fn"],
                "weighted_error_cost": m["weighted_error_cost"],
                "sensitivity": m["sensitivity"],
                "specificity": m["specificity"],
            }
        )
    table = pd.DataFrame(rows)
    best = table.sort_values(
        ["weighted_error_cost", "sensitivity", "specificity"],
        ascending=[True, False, False],
    ).iloc[0]
    return float(best["threshold"]), table


def repeated_cv_predictions(model: Pipeline, x: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=RANDOM_STATE)
    rows = []
    for fold, (train_idx, val_idx) in enumerate(cv.split(x, y), start=1):
        fitted = clone(model).fit(x.iloc[train_idx], y.iloc[train_idx])
        prob = fitted.predict_proba(x.iloc[val_idx])[:, 1]
        m = classification_metrics(y.iloc[val_idx], prob, 0.5)
        m["fold"] = fold
        rows.append(m)
    return pd.DataFrame(rows)


def bootstrap_coefficients(model: Pipeline, x: pd.DataFrame, y: pd.Series, n_boot: int = 500) -> pd.DataFrame:
    fitted = clone(model).fit(x, y)
    feature_names = fitted.named_steps["preprocess"].get_feature_names_out()
    reference = fitted.named_steps["model"].coef_[0]
    rng = np.random.default_rng(RANDOM_STATE)
    estimates = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(x), len(x))
        if np.unique(y.iloc[idx]).size < 2:
            continue
        boot = clone(model).fit(x.iloc[idx], y.iloc[idx])
        estimates.append(boot.named_steps["model"].coef_[0])
    arr = np.vstack(estimates)
    rows = []
    for j, name in enumerate(feature_names):
        vals = arr[:, j]
        sign = np.sign(reference[j])
        sign_stability = np.mean(np.sign(vals) == sign) if sign != 0 else np.nan
        rows.append(
            {
                "feature": name,
                "coefficient_full_development": reference[j],
                "odds_ratio_full_development": float(np.exp(np.clip(reference[j], -20, 20))),
                "bootstrap_median": np.median(vals),
                "bootstrap_q025": np.quantile(vals, 0.025),
                "bootstrap_q975": np.quantile(vals, 0.975),
                "sign_stability": sign_stability,
            }
        )
    return pd.DataFrame(rows).sort_values("coefficient_full_development", key=np.abs, ascending=False)


def bootstrap_test_ci(y_true: pd.Series, probabilities: dict[str, np.ndarray], thresholds: dict[str, float], n_boot: int = 2000) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    idx0 = np.flatnonzero(np.asarray(y_true) == 0)
    idx1 = np.flatnonzero(np.asarray(y_true) == 1)
    records = []
    for _ in range(n_boot):
        sample_idx = np.concatenate(
            [
                rng.choice(idx0, size=len(idx0), replace=True),
                rng.choice(idx1, size=len(idx1), replace=True),
            ]
        )
        rng.shuffle(sample_idx)
        yb = np.asarray(y_true)[sample_idx]
        for name, prob in probabilities.items():
            m = classification_metrics(yb, np.asarray(prob)[sample_idx], thresholds[name])
            for metric in ["roc_auc", "pr_auc", "sensitivity", "specificity", "brier"]:
                records.append({"model": name, "metric": metric, "value": m[metric]})
    boot = pd.DataFrame(records)
    return (
        boot.groupby(["model", "metric"])["value"]
        .agg(
            estimate="median",
            ci_low=lambda s: s.quantile(0.025),
            ci_high=lambda s: s.quantile(0.975),
        )
        .reset_index()
    )


def main() -> None:
    x, y = load_data()
    x_dev, x_test, y_dev, y_test = train_test_split(
        x,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    models = make_models()

    cv_tables = []
    for name, model in models.items():
        table = repeated_cv_predictions(model, x_dev, y_dev)
        table.insert(0, "model", name)
        cv_tables.append(table)
    cv = pd.concat(cv_tables, ignore_index=True)
    cv.to_csv(OUT / "temporary-repeated-cv.csv", index=False)

    oof_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    selected_thresholds: dict[str, float] = {}
    threshold_tables: list[pd.DataFrame] = []
    for name in ["Logistic regression", "LDA"]:
        oof_prob = cross_val_predict(
            models[name], x_dev, y_dev, cv=oof_cv, method="predict_proba"
        )[:, 1]
        threshold, table = select_cost_threshold(y_dev, oof_prob)
        selected_thresholds[name] = threshold
        table.insert(0, "model", name)
        threshold_tables.append(table)
    threshold_data = pd.concat(threshold_tables, ignore_index=True)
    threshold_data.to_csv(OUT / "temporary-threshold-audit.csv", index=False)

    fitted_models: dict[str, Pipeline] = {}
    test_probabilities: dict[str, np.ndarray] = {}
    test_rows = []
    for name, model in models.items():
        fitted = clone(model).fit(x_dev, y_dev)
        prob = fitted.predict_proba(x_test)[:, 1]
        fitted_models[name] = fitted
        test_probabilities[name] = prob
        threshold = selected_thresholds.get(name, 0.5)
        row = {"model": name, **classification_metrics(y_test, prob, threshold)}
        test_rows.append(row)
    test_results = pd.DataFrame(test_rows)
    test_results.to_csv(OUT / "temporary-locked-test.csv", index=False)

    coeffs = bootstrap_coefficients(models["Logistic regression"], x_dev, y_dev)
    coeffs.to_csv(OUT / "temporary-logistic-coefficient-bootstrap.csv", index=False)

    ci_models = {k: v for k, v in test_probabilities.items() if k in selected_thresholds}
    cis = bootstrap_test_ci(y_test, ci_models, selected_thresholds)
    cis.to_csv(OUT / "temporary-locked-test-bootstrap-ci.csv", index=False)

    continuous_corr = x[CONTINUOUS].corr()
    off_diag = continuous_corr.where(~np.eye(len(CONTINUOUS), dtype=bool)).stack()
    class_skew = x[CONTINUOUS].assign(target=y).groupby("target").skew()
    covariance_norms = {}
    standardized_cont = pd.DataFrame(
        StandardScaler().fit_transform(x[CONTINUOUS]), columns=CONTINUOUS
    )
    cov0 = standardized_cont[y == 0].cov().to_numpy()
    cov1 = standardized_cont[y == 1].cov().to_numpy()
    covariance_norms["relative_frobenius_difference"] = float(
        np.linalg.norm(cov1 - cov0, ord="fro")
        / ((np.linalg.norm(cov1, ord="fro") + np.linalg.norm(cov0, ord="fro")) / 2)
    )

    summary = {
        "source_url": SOURCE_URL,
        "n_samples": int(len(x)),
        "n_features_raw": int(x.shape[1]),
        "n_continuous": len(CONTINUOUS),
        "n_binary": len(BINARY),
        "n_categorical": len(CATEGORICAL),
        "missing_cells": int(x.isna().sum().sum()),
        "class_counts": {"absence": int((y == 0).sum()), "presence": int((y == 1).sum())},
        "development_n": int(len(x_dev)),
        "locked_test_n": int(len(x_test)),
        "max_abs_continuous_pair_correlation": float(off_diag.abs().max()),
        "continuous_pairs_abs_corr_ge_0_5": int((off_diag.abs() >= 0.5).sum() / 2),
        "continuous_feature_class_skewness": {
            str(idx): {k: float(v) for k, v in row.items()}
            for idx, row in class_skew.iterrows()
        },
        "covariance_check": covariance_norms,
        "selected_thresholds_from_development_oof": selected_thresholds,
        "cv_summary": (
            cv.groupby("model")
            .agg(
                auc_mean=("roc_auc", "mean"),
                auc_sd=("roc_auc", "std"),
                sensitivity_mean=("sensitivity", "mean"),
                sensitivity_sd=("sensitivity", "std"),
                specificity_mean=("specificity", "mean"),
                specificity_sd=("specificity", "std"),
                brier_mean=("brier", "mean"),
                brier_sd=("brier", "std"),
            )
            .reset_index()
            .to_dict(orient="records")
        ),
        "locked_test": test_results.to_dict(orient="records"),
    }
    (OUT / "temporary-analysis-summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

    order = ["Prevalence baseline", "Gaussian NB", "LDA", "Logistic regression"]
    data = [cv.loc[cv.model == name, "roc_auc"].to_numpy() for name in order]
    axes[0, 0].boxplot(data, tick_labels=order, showmeans=True)
    axes[0, 0].set_ylim(0.45, 1.01)
    axes[0, 0].set_ylabel("ROC-AUC")
    axes[0, 0].set_title("A. Repeated CV: discrimination is not the whole decision")
    axes[0, 0].tick_params(axis="x", rotation=18)

    for name in ["Logistic regression", "LDA", "Gaussian NB"]:
        fpr, tpr, _ = roc_curve(y_test, test_probabilities[name])
        auc = roc_auc_score(y_test, test_probabilities[name])
        axes[0, 1].plot(fpr, tpr, lw=2, label=f"{name} (AUC={auc:.3f})")
    axes[0, 1].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0, 1].set_xlabel("False-positive rate")
    axes[0, 1].set_ylabel("Sensitivity")
    axes[0, 1].set_title("B. Locked test ROC curves")
    axes[0, 1].legend(fontsize=8)

    for name, color in [("Logistic regression", "#1f77b4"), ("LDA", "#d95f02")]:
        part = threshold_data[threshold_data.model == name]
        axes[1, 0].plot(
            part.threshold,
            part.weighted_error_cost,
            label=name,
            color=color,
            lw=2,
        )
        axes[1, 0].axvline(selected_thresholds[name], color=color, ls="--", alpha=0.8)
    axes[1, 0].set_xlabel("Probability threshold selected on development data")
    axes[1, 0].set_ylabel("5 × false negatives + false positives")
    axes[1, 0].set_title("C. Decision threshold depends on error costs")
    axes[1, 0].legend(fontsize=8)

    plot_coeffs = coeffs.head(8).sort_values("bootstrap_median")
    xerr = np.vstack(
        [
            plot_coeffs.bootstrap_median - plot_coeffs.bootstrap_q025,
            plot_coeffs.bootstrap_q975 - plot_coeffs.bootstrap_median,
        ]
    )
    axes[1, 1].errorbar(
        plot_coeffs.bootstrap_median,
        np.arange(len(plot_coeffs)),
        xerr=xerr,
        fmt="o",
        color="#2c7fb8",
        ecolor="#7fcdbb",
        capsize=3,
    )
    axes[1, 1].axvline(0, color="black", lw=1)
    axes[1, 1].set_yticks(np.arange(len(plot_coeffs)), plot_coeffs.feature)
    axes[1, 1].set_xlabel("Log-odds coefficient (bootstrap median and 95% interval)")
    axes[1, 1].set_title("D. Coefficient interpretation is uncertain and conditional")

    fig.suptitle(
        "Lesson 8 planning figure: similar linear models, different assumptions and decision consequences",
        fontsize=15,
        fontweight="bold",
    )
    fig.savefig(OUT / "temporary-main-result.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
