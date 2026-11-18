#!/usr/bin/env python3
"""Build the complete Lesson 7 PBL package.

This is a course-specific builder. It regenerates the objective-mapped evidence
assets and four DOCX deliverables, writes the analysis contract and manifest,
and validates the approved executable notebook without replacing its hand-edited
instructional prose.

The repository currently uses two Python runtimes: the course Python has the ML
stack, while the bundled Codex document runtime has python-docx. This script
orchestrates both automatically and can be launched with either runtime.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path


SLUG = "machine-learning-basics-naive-bayes"
COURSE_DIR = Path(__file__).resolve().parents[1]
ASSET_DIR = COURSE_DIR / "assets" / f"pbl-{SLUG}"
NOTEBOOK_DIR = COURSE_DIR / "notebook"
TEACHER_NOTEBOOK = NOTEBOOK_DIR / f"pbl-{SLUG}-lab.ipynb"
STUDENT_NOTEBOOK = NOTEBOOK_DIR / f"pbl-{SLUG}-lab-student.ipynb"
WORKPLAN = COURSE_DIR / "WORKPLAN.md"
DATA_PATH = ASSET_DIR / "acute-inflammations.csv"
RANDOM_STATE = 7
UCI_URL = "https://archive.ics.uci.edu/dataset/184/acute+inflammations"

OUTPUTS = {
    "worksheet": COURSE_DIR / f"pbl-{SLUG}-student-worksheet.docx",
    "evidence": COURSE_DIR / f"pbl-{SLUG}-evidence-pack.docx",
    "teacher": COURSE_DIR / f"pbl-{SLUG}-teaching-plan-flipped-classroom.docx",
    "rationale": COURSE_DIR / f"pbl-{SLUG}-design-rationale.docx",
}

FEATURES = [
    "fever_ge_38",
    "nausea",
    "lumbar_pain",
    "urine_pushing",
    "micturition_pains",
    "burning_urethra",
]
TARGET = "bladder_inflammation"

LEARNING_OBJECTIVES = [
    {
        "id": "LO1",
        "source_excerpt": "什么是数据模型；数据输入、特征、标签",
        "observable_outcome": "说清 observation unit、feature、label，以及模型的输入和输出。",
    },
    {
        "id": "LO2",
        "source_excerpt": "什么是训练、预测和泛化；什么是损失函数；训练集、验证集、测试集",
        "observable_outcome": "区分 training、prediction、generalization 和 loss，并说明三类数据集的职责。",
    },
    {
        "id": "LO3",
        "source_excerpt": "朴素贝叶斯：基于 Bayes theorem；特征之间条件独立",
        "observable_outcome": "用 prior、likelihood、posterior、条件独立和 Laplace smoothing 解释并手算一次 NB 预测。",
    },
    {
        "id": "LO4",
        "source_excerpt": "模型评估：accuracy、sensitivity、specificity、precision、recall、F1、ROC-AUC、PR-AUC、calibration",
        "observable_outcome": "从混淆矩阵计算并解释主要分类指标，并区分阈值分类、排序和概率校准。",
    },
    {
        "id": "LO5",
        "source_excerpt": "优点与局限：简单、快速、可解释；条件独立假设常常不完全成立",
        "observable_outcome": "借助条件依赖、重复证据和概率质量审查 NB 的长处与局限。",
    },
    {
        "id": "LO6",
        "source_excerpt": "作业：解释为什么医学诊断模型不能只看 accuracy",
        "observable_outcome": "综合 locked test、不确定性、患病率和错误代价，形成有边界的研究决定。",
    },
]

ANALYSES = [
    {
        "id": "A1",
        "objective_ids": ["LO1"],
        "question": "一行和各列分别代表什么，模型输入和输出什么？",
        "domain": "data_structure",
        "method": "数据字典、标签分布、缺失审计与 provenance 核对",
        "applicability": "把生物医学描述转换为可计算的监督学习任务。",
        "artifacts": ["dataset-summary.csv"],
        "packet": "A",
        "interpretation_boundary": "一行是专家构造的症状—标签组合，不是真实患者记录。",
        "notebook_ref": "cell-13-packet-a-data-audit",
    },
    {
        "id": "A2",
        "objective_ids": ["LO2"],
        "question": "train/validation/test 各负责什么；accuracy 相同时 loss 能否不同？",
        "domain": "validation_robustness",
        "method": "分层拆分重叠检查与同准确率概率预测的 cross-entropy 对照",
        "applicability": "让训练、预测、泛化和 loss 形成可观察证据。",
        "artifacts": ["packet-b-split-audit.csv", "packet-b-loss-contrast.csv"],
        "packet": "B",
        "interpretation_boundary": "内部拆分模拟未见样本，但不能证明跨人群泛化。",
        "notebook_ref": "cells-8-17-loss-and-split",
    },
    {
        "id": "A3",
        "objective_ids": ["LO3"],
        "question": "NB 从训练数据估计了什么，posterior 如何得到？",
        "domain": "model_behavior",
        "method": "导出 prior/likelihood 参数并与手算 posterior 核对",
        "applicability": "直接揭示 Bernoulli NB 的生成式概率机制。",
        "artifacts": ["packet-b-naive-bayes-parameters.csv", "packet-b-posterior-handcheck.csv"],
        "packet": "B",
        "interpretation_boundary": "posterior 的数值依赖模型假设和训练数据。",
        "notebook_ref": "cells-19-21-naive-bayes-handcheck",
    },
    {
        "id": "A4",
        "objective_ids": ["LO4"],
        "question": "模型犯哪类错误，阈值改变了什么，各指标分别回答什么？",
        "domain": "model_behavior",
        "method": "基线对照、混淆矩阵、阈值路径、ROC/PR 与概率损失",
        "applicability": "把指标选择连接到筛查决策与错误代价。",
        "artifacts": [
            "packet-b-validation-metrics.csv",
            "packet-b-validation-confusion-matrix.png",
            "packet-b-threshold-path.csv",
            "packet-b-threshold-tradeoff.png",
        ],
        "packet": "B",
        "interpretation_boundary": "任何单一指标都不能概括用途、患病率和不确定性。",
        "notebook_ref": "cells-24-27-validation-metrics",
    },
    {
        "id": "A5",
        "objective_ids": ["LO5"],
        "question": "相关或重复症状是否会被当成多份独立证据？",
        "domain": "similarity_distance",
        "method": "类别内 phi 相关、重复特征消融、Brier/log loss 与校准诊断",
        "applicability": "以可见反例审查条件独立假设及 posterior 质量。",
        "artifacts": [
            "packet-c-conditional-dependence.csv",
            "packet-c-conditional-dependence.png",
            "packet-c-duplicate-feature-audit.csv",
            "packet-c-duplicate-feature-shift.png",
            "packet-c-probability-quality.csv",
            "packet-c-calibration-audit.png",
        ],
        "packet": "C",
        "interpretation_boundary": "假设不成立不等于分类必然失败；24 个 validation 样本不足以稳定估计校准。",
        "notebook_ref": "cells-29-33-dependence-and-duplicate",
    },
    {
        "id": "A6",
        "objective_ids": ["LO6"],
        "question": "100% 点估计足以支持哪一级行动？",
        "domain": "validation_robustness",
        "method": "锁定测试、Wilson 区间、报告边界；测量误差模拟作为挑战",
        "applicability": "把漂亮分数转换为带不确定性和边界的研究决定。",
        "artifacts": [
            "packet-d-locked-test-metrics.csv",
            "packet-d-locked-test-confusion-matrix.png",
            "packet-d-wilson-ci.csv",
            "packet-d-reporting-checklist.csv",
            "packet-d-measurement-error-stress.csv",
            "packet-d-measurement-error-stress.png",
        ],
        "packet": "D",
        "interpretation_boundary": "内部教学数据不能支持临床诊断、部署、因果或患者照护结论。",
        "notebook_ref": "cells-37-40-locked-test-and-uncertainty",
    },
    {
        "id": "A7",
        "objective_ids": ["LO4", "LO6"],
        "question": "目标标签或患病率改变后，指标解释如何变化？",
        "domain": "sampling_representativeness",
        "method": "nephritis 近迁移与 1% 患病率下 PPV 计算",
        "applicability": "检验学生能否把评价逻辑迁移到相邻任务和新情境。",
        "artifacts": [
            "transfer-nephritis-validation-thresholds.csv",
            "transfer-nephritis-locked-test-metrics.csv",
            "transfer-low-prevalence-precision.csv",
        ],
        "packet": "D",
        "interpretation_boundary": "同一规则数据上的近迁移仍不等于外部验证；PPV 情境是教学计算。",
        "notebook_ref": "cells-41-42-transfer",
    },
]


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def find_python_with(module: str, preferred: list[Path]) -> Path:
    candidates = [Path(sys.executable), *preferred]
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate).lower()
        if key in seen or not candidate.exists():
            continue
        seen.add(key)
        probe = subprocess.run(
            [str(candidate), "-c", f"import {module}"],
            capture_output=True,
            text=True,
        )
        if probe.returncode == 0:
            return candidate
    raise RuntimeError(f"No Python runtime with module {module!r} was found")


def metric_row(y_true, probability, threshold=0.5):
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        brier_score_loss,
        f1_score,
        log_loss,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    import numpy as np

    prediction = (np.asarray(probability) >= threshold).astype(int)
    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, prediction),
        "sensitivity_recall": recall_score(y_true, prediction, zero_division=0),
        "specificity": recall_score(y_true, prediction, pos_label=0, zero_division=0),
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
    half = z * math.sqrt(
        proportion * (1 - proportion) / total + z**2 / (4 * total**2)
    ) / denominator
    return center - half, center + half


def generate_analysis_assets() -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn.calibration import calibration_curve
    from sklearn.dummy import DummyClassifier
    from sklearn.metrics import (
        accuracy_score,
        brier_score_loss,
        confusion_matrix,
        log_loss,
    )
    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import BernoulliNB

    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {"figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False}
    )
    raw = pd.read_csv(DATA_PATH)
    data = raw.copy()
    binary_columns = [
        "nausea",
        "lumbar_pain",
        "urine_pushing",
        "micturition_pains",
        "burning_urethra",
        "bladder_inflammation",
        "nephritis",
    ]
    for column in binary_columns:
        data[column] = data[column].map({"no": 0, "yes": 1})
    data["fever_ge_38"] = (data["temperature_c"] >= 38.0).astype(int)

    pd.DataFrame(
        [
            {
                "dataset": "UCI Acute Inflammations",
                "observation_unit": "expert-constructed symptom-label profile (not a real patient record)",
                "samples": len(data),
                "original_features": 6,
                "targets": 2,
                "primary_target": TARGET,
                "primary_positive_n": int(data[TARGET].sum()),
                "primary_negative_n": int((1 - data[TARGET]).sum()),
                "missing_cells": int(data.isna().sum().sum()),
                "source": UCI_URL,
                "license": "CC BY 4.0",
                "doi": "10.24432/C5V59S",
            }
        ]
    ).to_csv(ASSET_DIR / "dataset-summary.csv", index=False)

    loss_truth = np.array([1, 0, 1, 0])
    loss_candidates = {
        "A: cautious wrong prediction": np.array([0.90, 0.10, 0.60, 0.60]),
        "B: overconfident wrong prediction": np.array([0.90, 0.10, 0.60, 0.99]),
    }
    loss_rows = []
    for name, probability in loss_candidates.items():
        loss_rows.append(
            {
                "model": name,
                "accuracy": accuracy_score(loss_truth, probability >= 0.5),
                "cross_entropy_log_loss": log_loss(loss_truth, probability),
                "probability_on_wrong_case": probability[-1],
            }
        )
    pd.DataFrame(loss_rows).to_csv(ASSET_DIR / "packet-b-loss-contrast.csv", index=False)

    X = data[FEATURES].copy()
    y = data[TARGET].copy()
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_tv, y_tv, test_size=0.25, random_state=RANDOM_STATE, stratify=y_tv
    )
    split_audit = pd.DataFrame(
        {
            "split": ["train", "validation", "locked test"],
            "n": [len(y_train), len(y_validation), len(y_test)],
            "positive_n": [int(y_train.sum()), int(y_validation.sum()), int(y_test.sum())],
            "positive_rate": [y_train.mean(), y_validation.mean(), y_test.mean()],
            "overlap_with_other_splits": [0, 0, 0],
            "role": ["fit parameters", "select/check", "one final estimate"],
        }
    )
    split_audit.to_csv(ASSET_DIR / "packet-b-split-audit.csv", index=False)

    alpha = 1.0
    model = BernoulliNB(alpha=alpha).fit(X_train, y_train)
    parameter_rows = []
    for class_position, disease_class in enumerate(model.classes_):
        class_count = int((y_train == disease_class).sum())
        parameter_rows.append(
            {
                "parameter_type": "prior",
                "class": int(disease_class),
                "feature": "(class prior)",
                "train_count": class_count,
                "probability": np.exp(model.class_log_prior_[class_position]),
                "laplace_alpha": alpha,
            }
        )
        for feature_position, feature in enumerate(FEATURES):
            parameter_rows.append(
                {
                    "parameter_type": "likelihood",
                    "class": int(disease_class),
                    "feature": feature,
                    "train_count": class_count,
                    "probability": np.exp(
                        model.feature_log_prob_[class_position, feature_position]
                    ),
                    "laplace_alpha": alpha,
                }
            )
    pd.DataFrame(parameter_rows).to_csv(
        ASSET_DIR / "packet-b-naive-bayes-parameters.csv", index=False
    )

    profile = {
        "fever_ge_38": 0,
        "nausea": 1,
        "lumbar_pain": 1,
        "urine_pushing": 0,
        "micturition_pains": 1,
        "burning_urethra": 1,
    }
    x_row = np.array([profile[name] for name in FEATURES], dtype=int)
    log_scores = model.class_log_prior_.copy()
    for class_position, _ in enumerate(model.classes_):
        for feature_position, _feature in enumerate(FEATURES):
            p_one = np.exp(model.feature_log_prob_[class_position, feature_position])
            log_scores[class_position] += (
                np.log(p_one)
                if x_row[feature_position] == 1
                else np.log(1 - p_one)
            )
    shifted = np.exp(log_scores - log_scores.max())
    posterior = shifted / shifted.sum()
    sklearn_posterior = model.predict_proba(pd.DataFrame([profile]))[0, 1]
    pd.DataFrame(
        [
            {
                **profile,
                "manual_posterior_y1": posterior[1],
                "sklearn_posterior_y1": sklearn_posterior,
                "absolute_difference": abs(posterior[1] - sklearn_posterior),
                "threshold": 0.5,
                "predicted_class": int(posterior[1] >= 0.5),
            }
        ]
    ).to_csv(ASSET_DIR / "packet-b-posterior-handcheck.csv", index=False)

    baseline = DummyClassifier(strategy="prior").fit(X_train, y_train)
    p_nb = model.predict_proba(X_validation)[:, 1]
    p_base = baseline.predict_proba(X_validation)[:, 1]
    validation_metrics = pd.DataFrame(
        [
            {"model": "prior-only baseline", **metric_row(y_validation, p_base)},
            {"model": "Bernoulli naive Bayes", **metric_row(y_validation, p_nb)},
        ]
    )
    validation_metrics.to_csv(ASSET_DIR / "packet-b-validation-metrics.csv", index=False)

    cm_validation = confusion_matrix(y_validation, p_nb >= 0.5)
    fig, ax = plt.subplots(figsize=(5.3, 4.4))
    image = ax.imshow(cm_validation, cmap="Blues", vmin=0)
    for row in range(2):
        for col in range(2):
            ax.text(col, row, cm_validation[row, col], ha="center", va="center", fontsize=18, fontweight="bold")
    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["Predicted no", "Predicted yes"],
        yticklabels=["Actual no", "Actual yes"],
        xlabel="Prediction",
        ylabel="Reference label",
        title="Packet B: validation confusion matrix (threshold=0.50)",
    )
    fig.colorbar(image, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(ASSET_DIR / "packet-b-validation-confusion-matrix.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    threshold_grid = np.linspace(0.02, 0.98, 49)
    threshold_results = pd.DataFrame(
        [metric_row(y_validation, p_nb, threshold=t) for t in threshold_grid]
    )
    threshold_results.to_csv(ASSET_DIR / "packet-b-threshold-path.csv", index=False)
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    for metric, color in [
        ("sensitivity_recall", "#dc2626"),
        ("specificity", "#059669"),
        ("precision", "#7c3aed"),
        ("accuracy", "#2563eb"),
    ]:
        ax.plot(threshold_results["threshold"], threshold_results[metric], label=metric, color=color, linewidth=2)
    ax.axvline(0.5, color="#111827", linestyle="--", label="selected=0.50")
    ax.set(xlabel="Decision threshold", ylabel="Metric value", ylim=(-0.03, 1.05), title="Packet B: threshold changes the error trade-off")
    ax.legend(ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(ASSET_DIR / "packet-b-threshold-tradeoff.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    dependence_rows = []
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for axis, disease_status in zip(axes, [0, 1]):
        corr = X[y == disease_status].corr()
        shown = np.ma.masked_invalid(corr.to_numpy())
        image = axis.imshow(shown, cmap="coolwarm", vmin=-1, vmax=1)
        axis.set_xticks(range(len(FEATURES)), FEATURES, rotation=55, ha="right")
        axis.set_yticks(range(len(FEATURES)), FEATURES)
        axis.set_title(f"Within class Y={disease_status}")
        for row in range(len(FEATURES)):
            for col in range(len(FEATURES)):
                value = corr.iloc[row, col]
                if not np.isnan(value):
                    axis.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=7)
        for row in range(len(FEATURES)):
            for col in range(row + 1, len(FEATURES)):
                dependence_rows.append(
                    {
                        "target_status": disease_status,
                        "feature_1": FEATURES[row],
                        "feature_2": FEATURES[col],
                        "phi_correlation": corr.iloc[row, col],
                    }
                )
    fig.colorbar(image, ax=axes, shrink=0.82, label="Phi correlation")
    fig.suptitle("Packet C: feature dependence conditional on the target", fontweight="bold")
    fig.savefig(ASSET_DIR / "packet-c-conditional-dependence.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    dependence_table = pd.DataFrame(dependence_rows)
    dependence_table.to_csv(ASSET_DIR / "packet-c-conditional-dependence.csv", index=False)

    duplicated_feature = "nausea"
    X_train_duplicate = X_train.copy()
    X_train_duplicate[f"{duplicated_feature}_duplicate"] = X_train_duplicate[duplicated_feature]
    duplicate_model = BernoulliNB(alpha=alpha).fit(X_train_duplicate, y_train)
    all_profiles = pd.DataFrame(
        [[(profile_id >> bit) & 1 for bit in range(len(FEATURES))] for profile_id in range(2 ** len(FEATURES))],
        columns=FEATURES,
    )
    all_profiles_duplicate = all_profiles.copy()
    all_profiles_duplicate[f"{duplicated_feature}_duplicate"] = all_profiles_duplicate[duplicated_feature]
    duplicate_audit = all_profiles.copy()
    duplicate_audit["original_probability"] = model.predict_proba(all_profiles)[:, 1]
    duplicate_audit["duplicate_probability"] = duplicate_model.predict_proba(all_profiles_duplicate)[:, 1]
    duplicate_audit["probability_change"] = duplicate_audit["duplicate_probability"] - duplicate_audit["original_probability"]
    duplicate_audit["decision_changed_at_0.5"] = (
        (duplicate_audit["original_probability"] >= 0.5)
        != (duplicate_audit["duplicate_probability"] >= 0.5)
    )
    duplicate_audit.to_csv(ASSET_DIR / "packet-c-duplicate-feature-audit.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.3, 5.4))
    colors = np.where(duplicate_audit["decision_changed_at_0.5"], "#dc2626", "#2563eb")
    ax.scatter(duplicate_audit["original_probability"], duplicate_audit["duplicate_probability"], c=colors, alpha=0.78, edgecolor="white", linewidth=0.4)
    ax.plot([0, 1], [0, 1], "--", color="#111827", linewidth=1)
    ax.axvline(0.5, color="#6b7280", linestyle=":")
    ax.axhline(0.5, color="#6b7280", linestyle=":")
    ax.set(xlim=(-0.03, 1.03), ylim=(-0.03, 1.03), xlabel="P(disease) with original features", ylabel="P(disease) after duplicating nausea", title="Packet C: redundant evidence can change a decision")
    fig.tight_layout()
    fig.savefig(ASSET_DIR / "packet-c-duplicate-feature-shift.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    validation_duplicate = X_validation.copy()
    validation_duplicate[f"{duplicated_feature}_duplicate"] = validation_duplicate[duplicated_feature]
    p_duplicate = duplicate_model.predict_proba(validation_duplicate)[:, 1]
    probability_quality = pd.DataFrame(
        [
            {
                "model": "original",
                "accuracy": accuracy_score(y_validation, p_nb >= 0.5),
                "brier": brier_score_loss(y_validation, p_nb),
                "log_loss": log_loss(y_validation, p_nb),
            },
            {
                "model": "duplicate nausea",
                "accuracy": accuracy_score(y_validation, p_duplicate >= 0.5),
                "brier": brier_score_loss(y_validation, p_duplicate),
                "log_loss": log_loss(y_validation, p_duplicate),
            },
        ]
    )
    probability_quality.to_csv(ASSET_DIR / "packet-c-probability-quality.csv", index=False)
    fig, ax = plt.subplots(figsize=(5.8, 5.0))
    ax.plot([0, 1], [0, 1], "--", color="#6b7280", label="perfect calibration")
    for label, probability, color in [
        ("original", p_nb, "#2563eb"),
        ("duplicate symptom", p_duplicate, "#dc2626"),
    ]:
        observed, predicted = calibration_curve(y_validation, probability, n_bins=4, strategy="quantile")
        ax.plot(predicted, observed, marker="o", linewidth=2, color=color, label=label)
    ax.set(xlabel="Mean predicted probability", ylabel="Observed positive fraction", xlim=(-0.03, 1.03), ylim=(-0.03, 1.03), title="Packet C: calibration audit (validation n=24)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(ASSET_DIR / "packet-c-calibration-audit.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    p_test = model.predict_proba(X_test)[:, 1]
    test_prediction = (p_test >= 0.5).astype(int)
    pd.DataFrame([metric_row(y_test, p_test, 0.5)]).to_csv(ASSET_DIR / "packet-d-locked-test-metrics.csv", index=False)
    cm_test = confusion_matrix(y_test, test_prediction)
    tn, fp, fn, tp = cm_test.ravel()
    interval_rows = []
    for name, successes, total in [
        ("accuracy", tp + tn, len(y_test)),
        ("sensitivity", tp, tp + fn),
        ("specificity", tn, tn + fp),
    ]:
        low, high = wilson_interval(successes, total)
        interval_rows.append({"metric": name, "estimate": successes / total, "ci_low": low, "ci_high": high, "denominator": total})
    pd.DataFrame(interval_rows).to_csv(ASSET_DIR / "packet-d-wilson-ci.csv", index=False)
    fig, ax = plt.subplots(figsize=(5.3, 4.4))
    image = ax.imshow(cm_test, cmap="Blues", vmin=0)
    for row in range(2):
        for col in range(2):
            ax.text(col, row, cm_test[row, col], ha="center", va="center", fontsize=18, fontweight="bold")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Predicted no", "Predicted yes"], yticklabels=["Actual no", "Actual yes"], xlabel="Prediction", ylabel="Reference label", title="Packet D: locked test (threshold=0.50, n=24)")
    fig.colorbar(image, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(ASSET_DIR / "packet-d-locked-test-confusion-matrix.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame(
        [
            {"item": "data provenance stated", "required": True, "reason": "expert-rule data are not a clinical cohort"},
            {"item": "split and locked-test rule stated", "required": True, "reason": "protect the final estimate"},
            {"item": "confusion matrix and threshold reported", "required": True, "reason": "show error types"},
            {"item": "sensitivity, specificity, precision and F1 reported", "required": True, "reason": "avoid a single-score claim"},
            {"item": "probability quality and uncertainty discussed", "required": True, "reason": "a point estimate is insufficient"},
            {"item": "clinical/deployment/causal boundary stated", "required": True, "reason": "teaching evidence only"},
        ]
    ).to_csv(ASSET_DIR / "packet-d-reporting-checklist.csv", index=False)

    flip_rates = [0.00, 0.02, 0.05, 0.10, 0.15, 0.20]
    n_simulations = 500
    rng = np.random.default_rng(20260903)
    x_base = X_test.to_numpy(dtype=bool)
    y_base = y_test.to_numpy(dtype=bool)
    stress_rows = []
    for flip_rate in flip_rates:
        masks = rng.random((n_simulations, *x_base.shape)) < flip_rate
        stressed = np.logical_xor(x_base[None, :, :], masks).astype(int)
        flat = pd.DataFrame(stressed.reshape(-1, len(FEATURES)), columns=FEATURES)
        probability = model.predict_proba(flat)[:, 1].reshape(n_simulations, len(y_base))
        prediction = probability >= 0.5
        truth = y_base[None, :]
        tp_sim = (prediction & truth).sum(axis=1)
        tn_sim = ((~prediction) & (~truth)).sum(axis=1)
        fp_sim = (prediction & (~truth)).sum(axis=1)
        fn_sim = ((~prediction) & truth).sum(axis=1)
        values = {
            "accuracy": (tp_sim + tn_sim) / len(y_base),
            "sensitivity": tp_sim / (tp_sim + fn_sim),
            "specificity": tn_sim / (tn_sim + fp_sim),
        }
        for metric_name, metric_values in values.items():
            stress_rows.append({"flip_rate": flip_rate, "metric": metric_name, "mean": metric_values.mean(), "q025": np.quantile(metric_values, 0.025), "q975": np.quantile(metric_values, 0.975)})
    stress_table = pd.DataFrame(stress_rows)
    stress_table.to_csv(ASSET_DIR / "packet-d-measurement-error-stress.csv", index=False)
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    colors = {"accuracy": "#2563eb", "sensitivity": "#dc2626", "specificity": "#059669"}
    for metric_name, color in colors.items():
        subset = stress_table[stress_table["metric"] == metric_name]
        x_percent = subset["flip_rate"].to_numpy() * 100
        ax.plot(x_percent, subset["mean"], marker="o", linewidth=2, color=color, label=metric_name)
        ax.fill_between(x_percent, subset["q025"], subset["q975"], color=color, alpha=0.10)
    ax.set(xlabel="Independent symptom-entry flip rate (%)", ylabel="Metric value", ylim=(0.45, 1.03), title=f"Packet D: measurement-error stress test ({n_simulations} simulations)")
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(ASSET_DIR / "packet-d-measurement-error-stress.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    axis = axes[0, 0]
    axis.imshow(cm_test, cmap="Blues", vmin=0)
    for row in range(2):
        for col in range(2):
            axis.text(col, row, cm_test[row, col], ha="center", va="center", fontsize=17, fontweight="bold")
    axis.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Pred no", "Pred yes"], yticklabels=["Actual no", "Actual yes"], title="A. Locked test looks perfect (n=24)")
    axis = axes[0, 1]
    names = ["accuracy", "sensitivity_recall", "specificity", "precision"]
    positions = np.arange(len(names))
    width = 0.36
    axis.bar(positions - width / 2, [validation_metrics.iloc[0][name] for name in names], width, color="#9aa4b2", label="prior baseline")
    axis.bar(positions + width / 2, [validation_metrics.iloc[1][name] for name in names], width, color="#2563eb", label="naive Bayes")
    axis.set_xticks(positions, [name.replace("_recall", "") for name in names], rotation=18)
    axis.set_ylim(0, 1.08)
    axis.set_ylabel("Metric value")
    axis.legend(frameon=False)
    axis.set_title("B. One metric cannot describe the task")
    axis = axes[1, 0]
    scatter_colors = np.where(duplicate_audit["decision_changed_at_0.5"], "#dc2626", "#2563eb")
    axis.scatter(duplicate_audit["original_probability"], duplicate_audit["duplicate_probability"], c=scatter_colors, alpha=0.75, edgecolor="white", linewidth=0.3)
    axis.plot([0, 1], [0, 1], "--", color="#111827", linewidth=1)
    axis.axvline(0.5, color="#6b7280", linestyle=":")
    axis.axhline(0.5, color="#6b7280", linestyle=":")
    axis.set(xlabel="Original P(disease)", ylabel="P(disease) after duplicate symptom", title="C. Redundant evidence can change decisions")
    axis = axes[1, 1]
    for metric_name, color in colors.items():
        subset = stress_table[stress_table["metric"] == metric_name]
        axis.plot(subset["flip_rate"] * 100, subset["mean"], marker="o", linewidth=2, color=color, label=metric_name)
    axis.set(xlabel="Symptom-entry flip rate (%)", ylabel="Mean metric", ylim=(0.65, 1.03), title="D. Measurement error exposes brittleness")
    axis.legend(frameon=False)
    fig.suptitle("Packet D synthesis: why a 100% score is not enough", fontsize=16, fontweight="bold")
    fig.savefig(ASSET_DIR / "main-result-clean-vs-audit.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    y_transfer = data["nephritis"].copy()
    Xt_tv, Xt_test, yt_tv, yt_test = train_test_split(X, y_transfer, test_size=0.20, random_state=RANDOM_STATE, stratify=y_transfer)
    Xt_train, Xt_validation, yt_train, yt_validation = train_test_split(Xt_tv, yt_tv, test_size=0.25, random_state=RANDOM_STATE, stratify=yt_tv)
    transfer_model = BernoulliNB(alpha=alpha).fit(Xt_train, yt_train)
    p_transfer_validation = transfer_model.predict_proba(Xt_validation)[:, 1]
    pd.DataFrame([metric_row(yt_validation, p_transfer_validation, threshold=t) for t in [0.10, 0.25, 0.50, 0.75, 0.90]]).to_csv(ASSET_DIR / "transfer-nephritis-validation-thresholds.csv", index=False)
    p_transfer_test = transfer_model.predict_proba(Xt_test)[:, 1]
    pd.DataFrame([metric_row(yt_test, p_transfer_test, threshold=0.5)]).to_csv(ASSET_DIR / "transfer-nephritis-locked-test-metrics.csv", index=False)
    prevalence = 0.01
    sensitivity = 0.95
    specificity = 0.95
    precision = sensitivity * prevalence / (sensitivity * prevalence + (1 - specificity) * (1 - prevalence))
    pd.DataFrame([{"prevalence": prevalence, "sensitivity": sensitivity, "specificity": specificity, "precision_ppv": precision, "false_positive_share_among_positive_predictions": 1 - precision}]).to_csv(ASSET_DIR / "transfer-low-prevalence-precision.csv", index=False)

    print(f"Generated analysis assets in {ASSET_DIR}")


def write_analysis_design() -> None:
    payload = {
        "study_goal": "从统计检验走向预测模型，建立定义任务、训练与评价模型、解释朴素贝叶斯及审查证据边界的共同语言。",
        "learning_objectives": LEARNING_OBJECTIVES,
        "decision_problem": "判断一个只报告内部 locked-test accuracy=100% 的症状分类模型，现有证据究竟支持哪一级使用。",
        "driving_question": "对使用六项症状预测膀胱炎症的 Bernoulli 朴素贝叶斯模型，应依据错误类型、概率质量、不确定性、假设和数据来源，决定只作教学基线、继续验证还是进入前瞻性筛查试点；任一关键证据不足时必须收缩结论。",
        "entry_type": "direct_problem",
        "topic_type": ["algorithm choice", "model evaluation", "validation strategy"],
        "key_concepts": [
            "supervised learning task and data model",
            "training, prediction and generalization",
            "train/validation/locked test",
            "loss versus evaluation metric",
            "Bayes theorem and Bernoulli naive Bayes",
            "confusion-matrix metrics and threshold",
            "discrimination, calibration and uncertainty",
            "conditional independence and redundant evidence",
        ],
        "misconceptions": [
            "High accuracy establishes medical usefulness",
            "Sensitivity, specificity and precision are interchangeable",
            "High ROC-AUC implies a correct threshold and calibrated probability",
            "Validation and locked test can both be repeatedly used for tuning",
            "Conditional-dependence failure means classification must fail, or perfect classification proves posterior probabilities are valid risks",
        ],
        "failure_modes": [
            "Repeated tuning against validation or test results",
            "Reporting accuracy without the confusion matrix, threshold and error costs",
            "Double-counting correlated or duplicated symptoms",
            "Claiming population or clinical performance from a tiny internal test",
            "Treating expert-rule profiles as a representative patient cohort",
        ],
        "data_context": {
            "dataset_name": "UCI Acute Inflammations",
            "source": UCI_URL,
            "doi": "10.24432/C5V59S",
            "license": "CC BY 4.0",
            "task": "binary symptom-based classification and evidence audit",
            "observation_unit": "one expert-constructed symptom-label profile, not a real patient record",
            "sample_count": 120,
            "variable_roles": {
                "features": FEATURES,
                "raw_continuous_feature": ["temperature_c"],
                "primary_target": [TARGET],
                "transfer_target": ["nephritis"],
            },
            "grouping_fields": [],
            "target": {"name": TARGET, "positive_n": 59, "negative_n": 61},
            "missing_cells": 0,
            "provenance_boundary": "Created by a medical expert to test an expert system; not a clinical cohort.",
        },
        "analyses": ANALYSES,
        "split_plan": {
            "required": True,
            "unit": "expert-constructed profile",
            "train_n": 72,
            "validation_n": 24,
            "locked_test_n": 24,
            "random_state": RANDOM_STATE,
            "rationale": "Generalization, threshold selection and test leakage are explicit learning objectives.",
        },
        "packets": {
            "A": {
                "purpose": "Define the data object and expose provenance limits",
                "reveals": ["dataset structure", "class balance", "missingness", "expert-rule provenance"],
                "withholds": ["model results", "locked test"],
            },
            "B": {
                "purpose": "Fit and inspect naive Bayes, then evaluate the first model evidence",
                "reveals": ["split audit", "loss contrast", "NB parameters", "validation metrics", "threshold trade-off"],
                "withholds": ["dependence stress", "locked test"],
            },
            "C": {
                "purpose": "Challenge the independence and probability-quality assumptions",
                "reveals": ["within-class dependence", "duplicate-feature shift", "calibration diagnostic"],
                "withholds": ["locked test and Wilson intervals"],
            },
            "D": {
                "purpose": "Synthesize the frozen-test evidence, uncertainty and reporting boundary",
                "reveals": ["locked test", "Wilson intervals", "reporting checklist", "optional stress test", "transfer context"],
                "withholds": [],
            },
        },
        "boundaries": [
            "Teaching-only evidence",
            "No clinical diagnosis or patient-care conclusion",
            "No deployment-readiness claim",
            "No causal inference",
            "Synthetic measurement-error stress does not replace external validation",
        ],
    }
    (ASSET_DIR / "analysis-design.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def read_csv_records(name: str) -> list[dict[str, str]]:
    with (ASSET_DIR / name).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def fnum(value, digits=3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number):
        return "NA"
    return f"{number:.{digits}f}"


def import_docx_symbols() -> None:
    global Document, WD_ALIGN_PARAGRAPH, WD_BREAK, WD_CELL_VERTICAL_ALIGNMENT
    global WD_ORIENTATION, WD_SECTION, WD_TABLE_ALIGNMENT, WD_ROW_HEIGHT_RULE
    global Inches, Pt, RGBColor, Cm, OxmlElement, qn
    from docx import Document
    from docx.enum.section import WD_ORIENTATION, WD_SECTION
    from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_ROW_HEIGHT_RULE, WD_TABLE_ALIGNMENT
    from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Cm, Inches, Pt, RGBColor


BLUE = "2B73B6"
NAVY = "1F4E79"
LIGHT_BLUE = "E7EEF6"
PALE_BLUE = "F3F7FB"
PALE_YELLOW = "FFF3CD"
PALE_RED = "FCE8E6"
PALE_GREEN = "E8F5E9"
GRAY = "667085"


def set_run_font(run, size=10, bold=False, color=None, italic=False, ascii_font="Arial"):
    run.font.name = ascii_font
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic
    if color:
        run.font.color.rgb = RGBColor.from_string(color)


def shade_cell(cell, fill: str):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_margins(cell, top=70, start=85, bottom=70, end=85):
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for margin, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn(f"w:{margin}"))
        if node is None:
            node = OxmlElement(f"w:{margin}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def repeat_header(row):
    tr_pr = row._tr.get_or_add_trPr()
    tbl_header = OxmlElement("w:tblHeader")
    tbl_header.set(qn("w:val"), "true")
    tr_pr.append(tbl_header)


def prevent_row_split(row):
    tr_pr = row._tr.get_or_add_trPr()
    cant_split = OxmlElement("w:cantSplit")
    tr_pr.append(cant_split)


def set_repeat_table_header(table):
    if table.rows:
        repeat_header(table.rows[0])
    for row in table.rows:
        prevent_row_split(row)


def configure_doc(doc, landscape=False, footer_text="教学用途 · 非临床建议 · UCI Acute Inflammations (CC BY 4.0)"):
    section = doc.sections[0]
    section.page_width = Cm(29.7 if landscape else 21.0)
    section.page_height = Cm(21.0 if landscape else 29.7)
    section.orientation = WD_ORIENTATION.LANDSCAPE if landscape else WD_ORIENTATION.PORTRAIT
    margin = Inches(0.55 if landscape else 0.65)
    section.top_margin = margin
    section.bottom_margin = margin
    section.left_margin = margin
    section.right_margin = margin
    section.header_distance = Inches(0.3)
    section.footer_distance = Inches(0.3)

    normal = doc.styles["Normal"]
    normal.font.name = "Arial"
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    normal.font.size = Pt(10)
    normal.paragraph_format.space_after = Pt(4)
    normal.paragraph_format.line_spacing = 1.1
    for style_name, size, color in (("Title", 24, NAVY), ("Heading 1", 16, BLUE), ("Heading 2", 13, BLUE), ("Heading 3", 11, NAVY)):
        style = doc.styles[style_name]
        style.font.name = "Arial"
        style._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor.from_string(color)
        style.paragraph_format.space_before = Pt(10)
        style.paragraph_format.space_after = Pt(5)
        style.paragraph_format.keep_with_next = True

    header = section.header
    hp = header.paragraphs[0]
    hp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    hr = hp.add_run("AI for Biomedical Data Science · PBL 2026")
    set_run_font(hr, 8.5, color=GRAY)
    footer = section.footer
    fp = footer.paragraphs[0]
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    fr = fp.add_run(footer_text)
    set_run_font(fr, 8, color=GRAY)


def add_title_block(doc, kicker: str, title: str, subtitle: str):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(kicker.upper())
    set_run_font(r, 10, bold=True, color=BLUE)
    p = doc.add_paragraph()
    p.style = doc.styles["Title"]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(3)
    r = p.add_run(title)
    set_run_font(r, 24, bold=True, color=NAVY)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(12)
    r = p.add_run(subtitle)
    set_run_font(r, 11, italic=True, color=GRAY)


def add_callout(doc, label: str, text: str, fill=PALE_YELLOW):
    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = True
    cell = table.cell(0, 0)
    shade_cell(cell, fill)
    set_cell_margins(cell, 90, 100, 90, 100)
    paragraph = cell.paragraphs[0]
    r = paragraph.add_run(f"{label}  ")
    set_run_font(r, 10, bold=True, color=NAVY)
    r = paragraph.add_run(text)
    set_run_font(r, 10)
    doc.add_paragraph().paragraph_format.space_after = Pt(0)


def add_table(doc, headers, rows, widths=None, font_size=9, header_fill=LIGHT_BLUE, alignments=None):
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = widths is None
    for index, header in enumerate(headers):
        cell = table.rows[0].cells[index]
        shade_cell(cell, header_fill)
        set_cell_margins(cell)
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(str(header))
        set_run_font(r, font_size, bold=True, color=NAVY)
        if widths:
            cell.width = Inches(widths[index])
    for row_values in rows:
        cells = table.add_row().cells
        for index, value in enumerate(row_values):
            cell = cells[index]
            set_cell_margins(cell)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            p = cell.paragraphs[0]
            if alignments:
                p.alignment = alignments[index]
            r = p.add_run(str(value))
            set_run_font(r, font_size)
            if widths:
                cell.width = Inches(widths[index])
    set_repeat_table_header(table)
    doc.add_paragraph().paragraph_format.space_after = Pt(0)
    return table


def add_bullets(doc, items, level=0):
    for item in items:
        p = doc.add_paragraph(style="List Bullet" if level == 0 else "List Bullet 2")
        r = p.add_run(str(item))
        set_run_font(r, 10)


def add_numbered(doc, items):
    for index, item in enumerate(items, start=1):
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Inches(0.18)
        p.paragraph_format.first_line_indent = Inches(-0.18)
        r = p.add_run(f"{index}.  {item}")
        set_run_font(r, 10)


def add_lines(doc, count=3, prompt=None):
    if prompt:
        p = doc.add_paragraph()
        r = p.add_run(prompt)
        set_run_font(r, 10, bold=True, color=NAVY)
    for _ in range(count):
        p = doc.add_paragraph("________________________________________________________________________________")
        p.paragraph_format.space_after = Pt(2)
        for run in p.runs:
            set_run_font(run, 8, color="98A2B3")


def add_picture(doc, filename: str, width_inches: float, caption: str, alt_text: str | None = None):
    path = ASSET_DIR / filename
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    inline_shape = run.add_picture(str(path), width=Inches(width_inches))
    doc_pr = inline_shape._inline.docPr
    doc_pr.set("descr", alt_text or caption)
    cp = doc.add_paragraph()
    cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cp.paragraph_format.keep_with_next = False
    r = cp.add_run(caption)
    set_run_font(r, 8.5, italic=True, color=GRAY)


def add_packet_audit_questions(doc):
    add_table(
        doc,
        ["审计问题", "个人记录"],
        [
            ["现在可以得出什么？", ""],
            ["还不能得出什么？", ""],
            ["应如何修订当前决定、解释或流程？", ""],
            ["什么证据会改变当前决定？", ""],
        ],
        widths=[2.2, 4.7],
        font_size=9,
    )


def add_packet_audit_prompts(doc):
    """Compact evidence-pack reminder; the writable audit table lives in the worksheet."""
    add_callout(
        doc,
        "审计提示",
        "阅读后依次判断：现在可以得出什么？还不能得出什么？应如何修订当前决定、解释或流程？什么新证据会改变当前决定？",
        PALE_BLUE,
    )


def add_page_break(doc):
    doc.add_page_break()


def build_student_worksheet() -> None:
    doc = Document()
    configure_doc(doc)
    add_title_block(
        doc,
        "Problem-Based Learning · Machine Learning Foundations",
        "机器学习基础与朴素贝叶斯",
        "Problem-Based Learning Student Worksheet · 个人工作表",
    )
    add_callout(
        doc,
        "使用边界",
        "本材料仅用于教学。数据是医学专家按规则构造的症状—标签组合，不得用于临床诊断、患者照护、部署判断或因果结论。",
        PALE_RED,
    )
    doc.add_heading("1. 案例元数据", level=1)
    add_table(
        doc,
        ["项目", "内容"],
        [
            ["数据集", "UCI Acute Inflammations；120 条组合，6 个原始特征"],
            ["观察单位", "一条专家构造的症状—标签组合，不是真实患者记录"],
            ["主标签", "bladder_inflammation：59 阳性，61 阴性"],
            ["核心决策", "教学/方法基线、继续验证，还是前瞻性筛查试点？"],
            ["最终产出", "个人 CERB 模型审计 Memo + 迁移作业"],
        ],
        widths=[1.35, 5.55],
    )
    doc.add_heading("2. 学习路线", level=1)
    add_numbered(
        doc,
        [
            "Section 1（45 分钟）：先判断，再复习 ML 基础并提出证据需求。",
            "课间自学：只完成 Packet A 与 Bayes/NB 预习，不提前运行 Packet B–D。",
            "Section 2（45 分钟）：课堂完成模型训练、指标计算、假设审计和 locked test 综合。",
            "课后：完成 nephritis 近迁移、低患病率新情境与最终 CERB。",
        ],
    )
    doc.add_heading("3. Section 1 课堂任务表（总计 45 分钟）", level=1)
    add_table(
        doc,
        ["时间", "个人任务", "必须留下的证据"],
        [
            ["0–6", "阅读问题触发，作出初始决定", "选择、理由、置信度"],
            ["6–20", "复习预测流程、数据划分、loss 与评价", "案例映射与三个判断"],
            ["20–32", "区分推断与预测，完成 3W 与竞争解释", "至少三种解释/决策"],
            ["32–40", "制定学习契约并填写 Evidence Card", "问题、来源与核验标准"],
            ["40–45", "Exit ticket", "当前最缺的一项证据"],
        ],
        widths=[0.75, 3.75, 2.4],
        font_size=8.8,
    )

    doc.add_heading("4. STEP 1｜问题触发与初始判断", level=1)
    add_callout(
        doc,
        "情境",
        "研究小组使用 6 项症状训练 Bernoulli 朴素贝叶斯模型预测膀胱炎症。摘要只写了一句：‘模型在锁定测试集上的 accuracy 是 100%。’你必须决定现有证据支持哪一级行动。",
    )
    p = doc.add_paragraph()
    r = p.add_run("我的初始立场：")
    set_run_font(r, 10, bold=True, color=NAVY)
    r = p.add_run("  □ 进入前瞻性筛查试点  □ 仅作教学/方法基线并继续验证  □ 直接否定  □ 用途与错误代价尚未定义")
    set_run_font(r, 10)
    add_lines(doc, 3, "理由（不能只重复 accuracy）：")
    add_lines(doc, 1, "当前置信度（0–100%）：")
    doc.add_heading("5. 向模型团队提出三个问题", level=1)
    add_table(
        doc,
        ["问题", "为什么会改变我的决定", "希望得到的证据形式"],
        [["", "", ""], ["", "", ""], ["", "", ""]],
        widths=[2.4, 2.4, 2.1],
    )
    doc.add_heading("6. 澄清与定义", level=1)
    add_table(
        doc,
        ["术语", "在本案例中的工作定义", "常见误用"],
        [
            ["observation unit", "", "把一行直接称为真实患者"],
            ["feature / label", "", "把标签泄漏进特征"],
            ["training / prediction", "", "把训练表现当泛化表现"],
            ["validation / locked test", "", "看完 test 再改模型"],
            ["loss / metric", "", "把优化目标等同于全部评价"],
            ["generalization", "", "把内部拆分等同于跨中心验证"],
        ],
        widths=[1.45, 3.2, 2.25],
        font_size=8.5,
    )

    add_page_break(doc)
    doc.add_heading("7. 机器学习基础快速复习", level=1)
    add_callout(doc, "工作流", "数据输入 → 定义 sample/feature/label → train 拟合 → validation 选方案与阈值 → 冻结方案 → locked test 评价 → 说明泛化边界。", PALE_BLUE)
    add_table(
        doc,
        ["层次", "概念/指标", "主要问题"],
        [
            ["数据与流程", "sample / feature / label", "模型究竟接收什么、预测什么？"],
            ["学习过程", "training / prediction / generalization", "参数如何得到，结论要推向哪里？"],
            ["单次预测差距", "cross-entropy / MSE", "预测错得有多严重？"],
            ["固定阈值", "accuracy / sensitivity / specificity / precision / F1", "当前决策犯了什么错？"],
            ["跨阈值排序", "ROC-AUC / PR-AUC", "阳性能否总体排在阴性前？"],
            ["概率质量", "calibration / Brier / log loss", "概率数值本身可信吗？"],
        ],
        widths=[1.1, 2.65, 3.15],
        font_size=8.6,
    )
    doc.add_heading("8. Driving question 框架", level=1)
    add_table(
        doc,
        ["组成", "我的表述"],
        [
            ["对象与应用场景", ""],
            ["需要作出的决定", ""],
            ["必须查看的可观察证据", ""],
            ["什么失败条件会迫使我收缩结论", ""],
        ],
        widths=[2.05, 4.85],
    )
    doc.add_heading("9. 参考阅读路线与 3W", level=1)
    add_bullets(doc, ["Bayes theorem：prior → likelihood → posterior。", "Bernoulli NB：二元特征、条件独立、Laplace smoothing。", "混淆矩阵与阈值：错误类型及其代价。", "三类评价：分类、排序、概率质量。", "泛化边界：样本量、来源、测量误差和分布变化。"])
    add_table(
        doc,
        ["What we know", "What we need to know", "What we will do"],
        [["", "", ""], ["", "", ""], ["", "", ""]],
        widths=[2.3, 2.3, 2.3],
        font_size=8.6,
    )

    doc.add_heading("10. 竞争解释 / 决策地图", level=1)
    add_table(
        doc,
        ["解释或决策", "若成立应看到什么", "什么证据会推翻", "选错的后果"],
        [
            ["H1：模型学到可泛化模式", "", "", ""],
            ["H2：规则数据过于干净", "", "", ""],
            ["H3：测试太小，100% 不稳定", "", "", ""],
            ["H4：相关症状被重复计证据", "", "", ""],
        ],
        widths=[1.85, 1.75, 1.75, 1.55],
        font_size=8.2,
    )
    doc.add_heading("11. 个人学习契约", level=1)
    add_table(
        doc,
        ["字段", "个人记录"],
        [
            ["我要解决的两个问题", ""],
            ["准备查阅的来源", ""],
            ["完成判据", ""],
            ["如何独立核验", ""],
        ],
        widths=[1.8, 5.1],
    )
    doc.add_heading("12. 指标主张 Evidence Card", level=1)
    add_callout(doc, "待审查主张", "‘医学筛查应该优先看 sensitivity。’来源可以是教材、官方文档、论文或 AI，但必须回到本数据的混淆矩阵独立核验。", PALE_GREEN)
    add_table(
        doc,
        ["字段", "个人记录"],
        [["来源/工具", ""], ["原始主张", ""], ["支持证据", ""], ["反证/限制", ""], ["最终可信度", ""]],
        widths=[1.6, 5.3],
    )
    add_lines(doc, 2, "Section 1 Exit ticket：当前最缺的证据是什么？")

    add_page_break(doc)
    doc.add_heading("13. Packet A｜数据与‘漂亮结论’", level=1)
    add_callout(doc, "释放规则", "Section 2 之前完成本页；先写审计结论，不提前运行 Packet B–D。", PALE_BLUE)
    summary = read_csv_records("dataset-summary.csv")[0]
    add_table(
        doc,
        ["项目", "证据"],
        [
            ["样本/特征/目标", f"{summary['samples']} 条组合；{summary['original_features']} 个原始特征；{summary['targets']} 个目标"],
            ["主标签分布", f"阳性 {summary['primary_positive_n']}；阴性 {summary['primary_negative_n']}"],
            ["缺失", f"缺失单元格 {summary['missing_cells']}"],
            ["观察单位", "一条专家构造的症状—标签组合"],
            ["来源与许可", "UCI Acute Inflammations；DOI 10.24432/C5V59S；CC BY 4.0"],
        ],
        widths=[1.6, 5.3],
    )
    raw_rows = read_csv_records("acute-inflammations.csv")[:5]
    add_table(
        doc,
        ["temp", "nausea", "lumbar", "urine", "micturition", "burning", "bladder"],
        [[r["temperature_c"], r["nausea"], r["lumbar_pain"], r["urine_pushing"], r["micturition_pains"], r["burning_urethra"], r["bladder_inflammation"]] for r in raw_rows],
        font_size=7.7,
    )
    add_packet_audit_questions(doc)

    add_page_break(doc)
    doc.add_heading("14. Packet B1｜Loss、拆分与 NB 机制", level=1)
    add_callout(doc, "CHECKPOINT", "只使用 train 拟合；validation 可比较方案；locked test 此时完全不看。", PALE_YELLOW)
    loss_rows = read_csv_records("packet-b-loss-contrast.csv")
    add_table(doc, ["概率模型", "accuracy", "cross-entropy", "错例概率"], [[r["model"], fnum(r["accuracy"]), fnum(r["cross_entropy_log_loss"]), fnum(r["probability_on_wrong_case"])] for r in loss_rows], widths=[3.05, 1.0, 1.35, 1.5], font_size=8.5)
    add_lines(doc, 2, "两个模型 accuracy 相同，为什么 loss 不同？")
    split_rows = read_csv_records("packet-b-split-audit.csv")
    add_table(doc, ["split", "n", "positive n", "positive rate", "职责"], [[r["split"], r["n"], r["positive_n"], fnum(r["positive_rate"]), r["role"]] for r in split_rows], widths=[1.25, 0.6, 1.0, 1.1, 2.95], font_size=8.3)
    add_callout(doc, "Bernoulli NB", "P(Y=c|X=x) ∝ P(Y=c) × ∏ P(Xj=xj|Y=c)。所谓‘朴素’：给定类别后，把各特征近似当作条件独立；alpha=1 的 Laplace smoothing 避免零计数把整类概率压成零。", PALE_BLUE)
    profile = read_csv_records("packet-b-posterior-handcheck.csv")[0]
    add_table(doc, ["症状组合", "手算区"], [["fever=0, nausea=1, lumbar=1, urine=0, micturition=1, burning=1", "score0=________  score1=________  P(Y=1|x)=________  class=____"]], widths=[3.9, 3.0], font_size=8.3)
    add_lines(doc, 2, "模型从 train 中实际估计了哪些量？")

    add_page_break(doc)
    doc.add_heading("15. Packet B2｜Validation 指标与阈值", level=1)
    metric_rows = read_csv_records("packet-b-validation-metrics.csv")
    add_table(doc, ["model", "acc", "sens", "spec", "prec", "F1", "ROC", "PR", "Brier", "log loss"], [[r["model"], *[fnum(r[k]) for k in ["accuracy", "sensitivity_recall", "specificity", "precision", "f1", "roc_auc", "pr_auc", "brier", "log_loss"]]] for r in metric_rows], font_size=7.2)
    add_picture(doc, "packet-b-validation-confusion-matrix.png", 4.8, "图 B1（LO4）：validation 混淆矩阵，threshold=0.50。", "Validation confusion matrix with 12 true negatives and 12 true positives")
    add_lines(doc, 2, "由混淆矩阵计算 sensitivity 与 specificity，并解释 prior-only baseline 为什么不适合筛查：")
    add_picture(doc, "packet-b-threshold-tradeoff.png", 6.4, "图 B2（LO4）：阈值改变 sensitivity、specificity、precision 与 accuracy。", "Threshold trade-off curves for four classification metrics")
    add_packet_audit_questions(doc)

    doc.add_heading("16. Packet C｜条件独立与概率可信度", level=1)
    add_callout(doc, "解释要求", "条件独立不等于特征在总体上互不相关；它要求给定类别后，各特征不再相互提供信息。假设被违反不等于分类必然失败。", PALE_BLUE)
    add_picture(doc, "packet-c-conditional-dependence.png", 6.6, "图 C1（LO5）：按标签分层后的特征 phi 相关。", "Two within-class feature correlation heatmaps")
    dep_rows = read_csv_records("packet-c-conditional-dependence.csv")
    dep_rows = sorted(dep_rows, key=lambda r: abs(float(r["phi_correlation"])) if r["phi_correlation"] else -1, reverse=True)[:6]
    add_table(doc, ["Y", "feature 1", "feature 2", "phi"], [[r["target_status"], r["feature_1"], r["feature_2"], fnum(r["phi_correlation"])] for r in dep_rows], widths=[0.55, 2.25, 2.25, 1.0], font_size=8.4)
    add_lines(doc, 2, "phi=1 或 NA 分别意味着什么？为什么都不能草率解释？")
    add_picture(doc, "packet-c-duplicate-feature-shift.png", 4.6, "图 C2（LO5）：重复录入 nausea 后 posterior 的变化。", "Original versus duplicated-feature probabilities; decision-changing profiles highlighted")
    quality = read_csv_records("packet-c-probability-quality.csv")
    add_table(doc, ["model", "accuracy", "Brier", "log loss"], [[r["model"], fnum(r["accuracy"]), fnum(r["brier"]), fnum(r["log_loss"])] for r in quality], widths=[2.5, 1.25, 1.25, 1.25], font_size=8.5)
    add_packet_audit_questions(doc)

    add_page_break(doc)
    doc.add_heading("17. Packet C 决策矩阵", level=1)
    add_table(
        doc,
        ["可选行动", "理由/假设", "期待证据", "验证方式", "停止条件或副作用"],
        [
            ["保留全部特征", "", "", "", ""],
            ["删除重复/高度相关症状", "", "", "", ""],
            ["保留分类，但不解释原始 posterior", "", "", "", ""],
            ["先收集外部数据再决定", "", "", "", ""],
        ],
        font_size=7.6,
    )
    doc.add_heading("18. 概念辨析", level=1)
    add_table(
        doc,
        ["概念", "它回答的问题", "本例风险"],
        [
            ["accuracy", "当前阈值下总体分对多少？", "掩盖类别比例和错误代价"],
            ["sensitivity / recall", "真阳性中找出多少？", "不说明阳性预测里多少是真的"],
            ["specificity", "真阴性中排除多少？", "不说明漏诊"],
            ["precision", "阳性预测中多少是真的？", "随患病率变化"],
            ["ROC-AUC / PR-AUC", "跨阈值排序怎样？", "不确定具体阈值，也不等于校准"],
            ["calibration", "概率数值与频率吻合吗？", "n=24 时很不稳定"],
            ["log loss / Brier", "概率离标签多远？", "不能代替用途与错误代价"],
        ],
        widths=[1.75, 2.75, 2.4],
        font_size=8.3,
    )
    add_picture(doc, "packet-c-calibration-audit.png", 4.5, "图 C3（LO5）：validation n=24 的校准诊断，仅用于发现问题。", "Small-sample calibration diagnostic for original and duplicated-feature models")

    doc.add_heading("19. Packet D｜Locked test 与统计不确定性", level=1)
    add_callout(doc, "释放规则", "只有模型、特征和阈值全部冻结后才查看本页。", PALE_RED)
    add_picture(doc, "packet-d-locked-test-confusion-matrix.png", 4.8, "图 D1（LO6）：locked test 混淆矩阵，n=24。", "Locked-test confusion matrix showing 24 correct classifications")
    ci_rows = read_csv_records("packet-d-wilson-ci.csv")
    add_table(doc, ["metric", "estimate", "95% Wilson low", "95% Wilson high", "denominator"], [[r["metric"], fnum(r["estimate"]), fnum(r["ci_low"]), fnum(r["ci_high"]), r["denominator"]] for r in ci_rows], widths=[1.55, 1.2, 1.5, 1.5, 1.15], font_size=8.4)
    add_lines(doc, 2, "为什么 24/24 全对仍不能说总体 sensitivity=100%？")
    add_packet_audit_questions(doc)
    p = doc.add_paragraph()
    r = p.add_run("当前决定：")
    set_run_font(r, 10, bold=True, color=NAVY)
    r = p.add_run("  □ 临床试点  □ 教学/方法研究  □ 要求真实外部验证  □ 停止使用 NB")
    set_run_font(r, 10)
    add_lines(doc, 1, "依据：")

    doc.add_heading("20. 课后挑战｜测量误差压力测试", level=1)
    add_picture(doc, "packet-d-measurement-error-stress.png", 6.4, "图 D2（LO6，挑战）：独立症状录入翻转的合成压力测试。", "Stress curves showing metric decline as symptom-entry errors increase")
    add_lines(doc, 3, "这个模拟可以揭示什么？为什么不能替代真实外部验证？")
    doc.add_heading("21. 课后迁移作业", level=1)
    add_numbered(doc, ["将目标换成 nephritis；只在 train 上拟合。", "在 validation 上比较阈值，报告混淆矩阵、sensitivity、specificity、precision 和 F1。", "找出至少一组违反条件独立的特征。", "方案冻结后一次性查看 locked test。", "回答：为什么医学诊断模型不能只看 accuracy？"])
    doc.add_heading("22. 新情境迁移｜低患病率筛查", level=1)
    add_callout(doc, "情境", "遗传病患病率为 1%，模型 sensitivity=95%、specificity=95%。计算 precision，并解释阳性预测中为什么仍会有大量假阳性。", PALE_GREEN)
    p = doc.add_paragraph("Precision = (Sensitivity × Prevalence) / [(Sensitivity × Prevalence) + (1 − Specificity) × (1 − Prevalence)]")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for run in p.runs:
        set_run_font(run, 9.5, bold=True, color=NAVY)
    add_lines(doc, 3, "计算与解释：")

    add_page_break(doc)
    doc.add_heading("23. CERB 模型审计 Memo", level=1)
    add_table(doc, ["要素", "个人写作"], [["Claim", ""], ["Evidence（分类）", ""], ["Evidence（概率/假设）", ""], ["Evidence（不确定性/泛化）", ""], ["Reasoning", ""], ["Boundary", ""]], widths=[1.85, 5.05])
    doc.add_heading("24. Section 2 收束记录", level=1)
    add_table(doc, ["项目", "个人记录"], [["Re-vote：新选择与置信度", ""], ["最强反对意见", ""], ["我的回应及引用证据", ""], ["仍然回答不了的问题", ""]], widths=[2.15, 4.75])
    doc.add_heading("25. Exit ticket", level=1)
    add_numbered(doc, ["‘朴素’朴素在哪里？", "数据划分最容易被用错的地方是什么？", "为什么 discrimination 和 calibration 不能互相替代？", "迁移到基因表达分类时，第一步应检查什么？"])
    add_lines(doc, 5)
    add_page_break(doc)
    doc.add_heading("26. 中英文工作词汇", level=1)
    add_table(doc, ["English", "本课工作含义"], [["observation unit", "一条可独立解释的观测；本数据中不是患者"], ["feature / label", "输入变量 / 希望预测的已知结果"], ["training / prediction", "估计参数 / 用固定模型产生输出"], ["validation / locked test", "选择方案 / 冻结后最终评价"], ["loss", "单次预测与标签之间的数值差距"], ["prior / likelihood / posterior", "观察前、给定类别的证据项、观察后的类别概率"], ["conditional independence", "给定类别后近似把各特征看作彼此独立"], ["calibration", "预测概率与实际频率的吻合程度"]], widths=[2.25, 4.65], font_size=8.6)
    doc.save(OUTPUTS["worksheet"])


def build_evidence_pack() -> None:
    doc = Document()
    configure_doc(doc, landscape=True)
    add_title_block(doc, "Packet A–D · Evidence Pack", "机器学习基础与朴素贝叶斯", "Staged Evidence Pack · 分阶段证据包")
    add_callout(doc, "释放提醒", "不要一次性发放全部 Packet。A 用于课间数据审计；B、C、D 在 Section 2 按 checkpoint 逐步释放。", PALE_RED)
    doc.add_heading("1. 证据路线与用途边界", level=1)
    add_table(doc, ["Packet", "核心问题", "释放证据", "暂不提供"], [["A", "数据对象和来源允许什么结论？", "结构、标签分布、缺失、provenance", "模型表现"], ["B", "NB 学到了什么，第一次评价说明什么？", "loss、拆分、参数、validation、阈值", "假设压力与 test"], ["C", "条件独立和 posterior 可信吗？", "依赖、重复证据、概率质量", "locked test"], ["D", "100% 点估计足以支持什么行动？", "locked test、Wilson、报告边界", "无" ]], widths=[0.8, 3.0, 4.0, 2.1], font_size=8.5)
    add_callout(doc, "边界", "数据由医学专家为专家系统测试而构造，不是真实临床队列。所有结果只支持教学与方法讨论，不支持临床诊断、部署、因果或患者照护建议。", PALE_YELLOW)
    doc.add_heading("2. 数据来源", level=1)
    add_bullets(doc, ["UCI Machine Learning Repository: Acute Inflammations (Dataset ID 184).", "DOI: 10.24432/C5V59S；License: CC BY 4.0。", f"Source: {UCI_URL}", "120 条组合，6 个原始特征，2 个疾病标签；无缺失。"])

    add_page_break(doc)
    doc.add_heading("Packet A｜数据结构、变量语义与 provenance", level=1)
    summary = read_csv_records("dataset-summary.csv")[0]
    add_table(doc, ["dataset", "observation unit", "n", "features", "primary target", "positive / negative", "missing", "license"], [[summary["dataset"], "expert-constructed symptom-label profile", summary["samples"], summary["original_features"], summary["primary_target"], f"{summary['primary_positive_n']} / {summary['primary_negative_n']}", summary["missing_cells"], summary["license"]]], font_size=8.1)
    add_table(doc, ["变量", "角色", "尺度/编码", "本课处理", "解释边界"], [["temperature_c", "feature", "continuous °C", "预先声明 temperature≥38 → fever_ge_38", "不是从全数据搜索最优切点"], ["nausea", "feature", "yes/no", "0/1", "主观症状可能有测量误差"], ["lumbar_pain", "feature", "yes/no", "0/1", "不能解释为因果机制"], ["urine_pushing", "feature", "yes/no", "0/1", "可能与其他症状相关"], ["micturition_pains", "feature", "yes/no", "0/1", "可能与其他症状相关"], ["burning_urethra", "feature", "yes/no", "0/1", "可能与其他症状相关"], ["bladder_inflammation", "primary target", "yes/no", "0/1", "专家规则标签"], ["nephritis", "transfer target", "yes/no", "0/1", "仍是同一规则数据"]], widths=[1.8, 1.5, 1.45, 3.0, 2.3], font_size=8.2)
    doc.add_heading("Packet A 审计提示", level=2)
    add_packet_audit_prompts(doc)

    add_page_break(doc)
    doc.add_heading("Packet B1｜Loss、拆分与 NB 参数", level=1)
    loss_rows = read_csv_records("packet-b-loss-contrast.csv")
    add_table(doc, ["模型", "accuracy", "cross-entropy", "错误样本上的 P(Y=1)"], [[r["model"], fnum(r["accuracy"]), fnum(r["cross_entropy_log_loss"]), fnum(r["probability_on_wrong_case"])] for r in loss_rows], widths=[3.8, 1.4, 1.7, 2.1], font_size=8.5)
    add_callout(doc, "观察", "两组类别判断完全相同，但过度自信的错误受到更大 log-loss 惩罚。Accuracy 看不到这部分概率质量。", PALE_BLUE)
    split_rows = read_csv_records("packet-b-split-audit.csv")
    add_table(doc, ["split", "n", "positive n", "positive rate", "overlap", "role"], [[r["split"], r["n"], r["positive_n"], fnum(r["positive_rate"]), r["overlap_with_other_splits"], r["role"]] for r in split_rows], widths=[1.6, 0.7, 1.2, 1.4, 1.0, 3.1], font_size=8.5)
    parameter_rows = read_csv_records("packet-b-naive-bayes-parameters.csv")
    selected = [r for r in parameter_rows if r["parameter_type"] == "prior" or r["feature"] in ("nausea", "urine_pushing", "micturition_pains")]
    add_table(doc, ["type", "class", "feature", "train count", "estimated probability", "alpha"], [[r["parameter_type"], r["class"], r["feature"], r["train_count"], fnum(r["probability"]), r["laplace_alpha"]] for r in selected], widths=[1.25, 0.8, 2.2, 1.1, 1.8, 0.8], font_size=8.2)
    hand = read_csv_records("packet-b-posterior-handcheck.csv")[0]
    add_callout(doc, "后验核对", f"指定症状组合的手算 P(Y=1|x)={fnum(hand['manual_posterior_y1'],4)}；sklearn={fnum(hand['sklearn_posterior_y1'],4)}；绝对差={float(hand['absolute_difference']):.2e}。", PALE_GREEN)

    doc.add_heading("Packet B2｜Validation 分类证据与阈值", level=1)
    metric_rows = read_csv_records("packet-b-validation-metrics.csv")
    add_table(doc, ["model", "acc", "sens", "spec", "precision", "F1", "ROC-AUC", "PR-AUC", "Brier", "log loss"], [[r["model"], *[fnum(r[k]) for k in ["accuracy", "sensitivity_recall", "specificity", "precision", "f1", "roc_auc", "pr_auc", "brier", "log_loss"]]] for r in metric_rows], font_size=7.5)
    add_picture(doc, "packet-b-validation-confusion-matrix.png", 4.5, "图 B1（LO4）：Validation confusion matrix，threshold=0.50。", "Validation confusion matrix")
    add_picture(doc, "packet-b-threshold-tradeoff.png", 7.2, "图 B2（LO4）：阈值路径显示错误类型之间的权衡。", "Threshold trade-off plot")
    add_packet_audit_prompts(doc)

    add_page_break(doc)
    doc.add_heading("Packet C｜条件依赖、重复证据与概率质量", level=1)
    add_picture(doc, "packet-c-conditional-dependence.png", 8.6, "图 C1（LO5）：给定类别后的特征依赖。NaN 通常代表类别内零方差，而不是‘没有相关’。", "Within-class dependence heatmaps")
    dep_rows = read_csv_records("packet-c-conditional-dependence.csv")
    dep_rows = sorted(dep_rows, key=lambda r: abs(float(r["phi_correlation"])) if r["phi_correlation"] else -1, reverse=True)[:8]
    add_table(doc, ["Y", "feature 1", "feature 2", "phi"], [[r["target_status"], r["feature_1"], r["feature_2"], fnum(r["phi_correlation"])] for r in dep_rows], widths=[0.7, 2.6, 2.6, 1.1], font_size=8.4)
    duplicate = read_csv_records("packet-c-duplicate-feature-audit.csv")
    changed = sum(r["decision_changed_at_0.5"].lower() == "true" for r in duplicate)
    max_shift = max(abs(float(r["probability_change"])) for r in duplicate)
    add_callout(doc, "重复特征审计", f"把 nausea 完整复制为第二列后，64 种可能症状组合中有 {changed} 种跨过 0.50 阈值；最大概率变化为 {max_shift:.3f}。", PALE_YELLOW)
    add_picture(doc, "packet-c-duplicate-feature-shift.png", 5.3, "图 C2（LO5）：同一证据被重复计入可改变 posterior 与类别决定。", "Duplicate-feature probability shift")
    add_picture(doc, "packet-c-calibration-audit.png", 4.8, "图 C3（LO5）：validation n=24 的校准曲线只能作诊断，不能作稳定校准证明。", "Calibration diagnostic")
    quality = read_csv_records("packet-c-probability-quality.csv")
    add_table(doc, ["model", "accuracy", "Brier", "log loss"], [[r["model"], fnum(r["accuracy"]), fnum(r["brier"]), fnum(r["log_loss"])] for r in quality], widths=[3.1, 1.4, 1.4, 1.4], font_size=8.6)
    add_packet_audit_prompts(doc)

    add_page_break(doc)
    doc.add_heading("Packet D｜Locked test、Wilson 区间与报告边界", level=1)
    add_picture(doc, "packet-d-locked-test-confusion-matrix.png", 4.6, "图 D1（LO6）：冻结方案在 24 个内部 test 组合上 24/24 正确。", "Locked-test confusion matrix")
    ci_rows = read_csv_records("packet-d-wilson-ci.csv")
    add_table(doc, ["metric", "estimate", "95% Wilson low", "95% Wilson high", "denominator"], [[r["metric"], fnum(r["estimate"]), fnum(r["ci_low"]), fnum(r["ci_high"]), r["denominator"]] for r in ci_rows], widths=[2.0, 1.6, 2.0, 2.0, 1.4], font_size=8.6)
    add_callout(doc, "关键解释", "Point estimate=1.00 不等于总体真实性=1.00。Sensitivity 与 specificity 各只有 12 个分母，Wilson 下限约 0.758。", PALE_RED)
    checklist = read_csv_records("packet-d-reporting-checklist.csv")
    add_table(doc, ["报告项目", "必须", "为什么"], [[r["item"], r["required"], r["reason"]] for r in checklist], widths=[3.7, 1.0, 5.1], font_size=8.3)
    add_packet_audit_prompts(doc)

    add_page_break(doc)
    doc.add_heading("Packet D 挑战证据与迁移", level=1)
    add_picture(doc, "packet-d-measurement-error-stress.png", 7.5, "图 D2（挑战）：合成症状录入误差暴露脆弱性，但不能替代外部验证。", "Synthetic measurement-error stress test")
    transfer = read_csv_records("transfer-low-prevalence-precision.csv")[0]
    add_table(doc, ["prevalence", "sensitivity", "specificity", "precision/PPV", "false-positive share among positive predictions"], [[fnum(transfer["prevalence"]), fnum(transfer["sensitivity"]), fnum(transfer["specificity"]), fnum(transfer["precision_ppv"]), fnum(transfer["false_positive_share_among_positive_predictions"])]], font_size=8.5)
    add_callout(doc, "迁移解释", "患病率 1% 时，即使 sensitivity 与 specificity 都是 95%，precision 也只有约 16.1%。Precision 随患病率变化，不能脱离目标人群解释。", PALE_GREEN)
    add_picture(doc, "main-result-clean-vs-audit.png", 8.4, "综合图：为什么规则化内部数据上的 100% accuracy 仍不足以支持临床试点。", "Four-panel synthesis of locked test, metrics, duplicate evidence and stress test")
    doc.add_heading("来源与引用", level=1)
    p = doc.add_paragraph(f"Czerniak, J. (2003). Acute Inflammations [Dataset]. UCI Machine Learning Repository. DOI 10.24432/C5V59S. {UCI_URL} (CC BY 4.0).")
    for run in p.runs:
        set_run_font(run, 9)
    doc.save(OUTPUTS["evidence"])


def build_teacher_plan() -> None:
    doc = Document()
    configure_doc(doc)
    add_title_block(doc, "PBL + Flipped Classroom", "机器学习基础与朴素贝叶斯", "Teacher Plan · 教师实施方案")
    add_callout(doc, "课程定位", "面向生命科学专业、即将进入大四的直博生；不假定 CS/算法背景，但要求基于证据完成较高强度的自主学习。", PALE_BLUE)
    add_callout(doc, "使用边界", "本课数据和模型仅用于教学与方法讨论，不支持临床诊断、部署、因果或患者照护结论。", PALE_RED)
    doc.add_heading("1. 概览", level=1)
    add_table(doc, ["项目", "安排"], [["课时结构", "Section 1 课堂 45 min + 课间自学 + Section 2 课堂 Lab 45 min + 迁移作业"], ["数据", "UCI Acute Inflammations；120 条专家规则组合"], ["核心问题", "只报告 locked-test accuracy=100% 时，现有证据支持哪一级行动？"], ["完成方式", "学生个人完成；可同伴质询，但不设置固定组内角色"], ["最终证据", "运行结果、Packet 四问、指标主张 Evidence Card、CERB Memo"]], widths=[1.45, 5.45])
    doc.add_heading("2. 可观察学习目标", level=1)
    add_table(doc, ["LO", "学生能够……", "主要证据"], [[lo["id"], lo["observable_outcome"], next(a["method"] for a in ANALYSES if lo["id"] in a["objective_ids"])] for lo in LEARNING_OBJECTIVES], widths=[0.6, 3.8, 2.5], font_size=8.5)
    doc.add_heading("3. 教师讲授边界", level=1)
    add_bullets(doc, ["基础复习控制在 14 分钟内，只澄清概念的作用与关系。", "不把高 accuracy 当课程结论；学生必须先作初始判断。", "不完整推导所有算法，也不要求记忆 API。", "课堂训练和指标计算由学生运行；教师只在 checkpoint 汇总。", "始终重申数据不是临床队列。"])

    add_page_break(doc)
    doc.add_heading("4. Section 1 时间表（45 分钟）", level=1)
    add_table(doc, ["时间", "学生任务", "教师动作", "可观察输出"], [["0–6", "读触发并作初始决定", "只澄清任务，不补充 test 大小", "选择、理由、置信度"], ["6–20", "复习流程、划分、loss、指标", "用相同 accuracy/不同 loss 对照点拨", "案例映射与三项判断"], ["20–32", "推断 vs 预测；3W；竞争解释", "追问每种解释需要什么证据", "至少三种解释/决策"], ["32–40", "学习契约与 Evidence Card", "强调来源可含 AI，但必须独立核验", "问题、来源、核验标准"], ["40–45", "Exit ticket", "收集最缺证据，决定下次切入点", "一项关键证据缺口"]], widths=[0.7, 2.25, 2.55, 1.4], font_size=8.1)
    doc.add_heading("5. Section 1 提问脚本", level=1)
    add_bullets(doc, ["100% 是在多少样本上？分母是什么？", "错的是谁：FN 还是 FP？当前用途更怕哪一种？", "概率 0.9 与‘真实风险 90%’之间还缺什么？", "若模型只复现专家规则，仍有什么教学价值？", "哪些证据一旦出现，会让你改变当前决定？"])
    doc.add_heading("6. 课间自学与 LMS 规则", level=1)
    add_numbered(doc, ["只释放 Packet A、Bayes/NB 概念页及学生 Notebook 对应前置部分。", "要求每位学生独立提交 Packet A 四问和公式预习记录。", "Packet B–D 保持锁定；不要让学生提前用 test 调整阈值。", "Evidence Card 审查‘筛查应优先看 sensitivity’，AI 仅为可选来源。"])

    add_page_break(doc)
    doc.add_heading("7. Section 2：课堂 Lab 与证据综合（45 分钟）", level=1)
    add_table(doc, ["时间", "活动", "教师 checkpoint", "个人输出"], [["0–25", "Packet B：拆分、拟合、手算 posterior、混淆矩阵与 sens/spec", "确认只用 train 拟合；test 未打开", "运行结果与指标解释"], ["25–33", "Packet C：依赖、重复证据、概率质量", "区分‘分类仍正确’与‘posterior 可信’", "一项支持、一项限制"], ["33–38", "Packet D：locked test 与 Wilson", "方案冻结后才释放；强调分母", "修订决定"], ["38–45", "Re-vote、peer challenge、CERB、exit", "围绕证据而非分数高低综合", "最终结论与边界"]], widths=[0.7, 2.8, 2.25, 1.15], font_size=8.1)
    doc.add_heading("8. 课堂练习操作要求", level=1)
    add_numbered(doc, ["把 temperature_c≥38 转成预先声明的 fever_ge_38；其他症状转 0/1。", "固定 random_state=7，分层得到 train/validation/locked test=72/24/24。", "用 BernoulliNB(alpha=1.0) 训练；prior-only baseline 作对照。", "手算指定症状组合 posterior，并与 sklearn 输出核对。", "由 confusion matrix 计算 sensitivity 和 specificity；再观察阈值路径。"])
    doc.add_heading("9. 同伴质询与汇报主题", level=1)
    add_bullets(doc, ["哪一个证据最支持继续研究？", "哪一个证据最限制临床解释？", "如果漏诊代价更高，阈值与指标优先级怎么变？", "条件独立被违反时，为什么不能只说‘模型失效’？", "什么样的外部验证才会改变决定？"])

    add_page_break(doc)
    doc.add_heading("10. Packet 教师综合锚点", level=1)
    add_table(doc, ["Packet", "可得结论", "不可得结论", "合理修订"], [["A", "可定义教学二分类任务；标签接近平衡；无缺失", "代表真实患者或真实数据质量", "只作为教学/方法基线"], ["B", "NB 在当前 validation 明显优于 prior baseline；阈值影响错误类型", "真实 sensitivity=100% 或外部泛化", "保留简单基线，冻结方案后再看 test"], ["C", "类别内依赖与重复症状能改变 posterior/决定", "条件依赖必然导致分类失败；n=24 可稳定校准", "分类可保留，但个体概率解释需收缩"], ["D", "24/24 证明模型复现当前规则数据；Wilson 显示不确定性", "临床效用、跨中心代表性或部署就绪", "要求预注册真实外部验证"]], widths=[0.7, 2.4, 2.6, 1.2], font_size=8.0)
    doc.add_heading("11. 关键数值答案", level=1)
    add_bullets(doc, ["Loss 对照：两个模型 accuracy=0.75；较谨慎错例 log loss≈0.410，过度自信错例≈1.332。", "Validation：NB TP=12、TN=12、FP=0、FN=0；prior baseline sensitivity=0、specificity=1。", "后验手算：指定症状组合 P(Y=1|x)≈0.3755。", "重复 nausea：64 种组合中 5 种改变 0.50 阈值决定；最大概率变化≈0.221。", "Locked test：24/24；accuracy Wilson 下限≈0.862；sens/spec 各自下限≈0.758。", "患病率 1%、sens/spec=95% 时 precision≈16.1%。"])
    doc.add_heading("12. 误区响应", level=1)
    add_table(doc, ["学生说法", "教师追问/响应"], [["accuracy 都 100%，没有别的可看", "分母、阈值、错误类型、区间、概率质量和来源分别是什么？"], ["条件独立不成立，所以 NB 一定错", "当前分类与 posterior 数值各发生了什么？"], ["ROC-AUC=1 就能用", "具体阈值、校准、患病率与错误代价在哪里？"], ["test 也可以帮助选阈值", "选过以后，它还能代表冻结方案的独立评价吗？"], ["一个缺失都没有，数据质量很好", "真实临床数据为什么如此整齐？provenance 能解释吗？"]], widths=[2.6, 4.3], font_size=8.6)

    add_page_break(doc)
    doc.add_heading("13. 作业与迁移", level=1)
    add_bullets(doc, ["近迁移：将目标改为 nephritis，保持同一拆分纪律并报告完整指标。", "新情境：计算低患病率筛查的 precision，解释患病率的作用。", "挑战：解释测量误差压力测试能揭示什么、不能替代什么。", "最终：以 CERB 回答为什么医学诊断模型不能只看 accuracy。"])
    doc.add_heading("14. 评价量规（100 分）", level=1)
    add_table(doc, ["维度", "权重", "达标表现"], [["任务与流程定义", "15", "正确说明 observation unit、X/Y、train/validation/test 与泄漏边界"], ["NB 机制", "20", "正确解释 prior/likelihood/posterior、条件独立、Laplace 并完成手算"], ["分类评价", "25", "正确计算并解释混淆矩阵指标、阈值、AUC 与 calibration 分工"], ["假设与概率审计", "15", "用依赖、重复特征和概率质量支持判断"], ["不确定性与迁移", "15", "解释 Wilson、患病率和外部验证边界"], ["CERB 表达", "10", "主张、证据、推理和边界一致；允许据证据改观点"]], widths=[1.5, 0.8, 4.6], font_size=8.4)
    doc.add_heading("15. 教师检查清单", level=1)
    add_bullets(doc, ["学生先判断，后看证据。", "Packet 严格按 A→B→C→D 释放。", "课堂核心不被压力测试挤占。", "不使用固定组内角色；每位学生都有个人证据记录。", "评分奖励证据与边界，不奖励单纯跑出最高分。", "结束前再次声明教学、临床、部署和因果边界。"])
    doc.save(OUTPUTS["teacher"])


def build_design_rationale() -> None:
    doc = Document()
    configure_doc(doc)
    add_title_block(doc, "Purpose · Design Rationale · Expected Outcomes", "机器学习基础与朴素贝叶斯", "PBL Design Rationale · 设计说明")
    doc.add_heading("1. 教学问题", level=1)
    p = doc.add_paragraph("生命科学学生往往能记住若干算法名和指标定义，却不一定能把研究问题写成预测任务，也容易把漂亮的内部 accuracy 当成可推广结论。传统讲授若先给出完整定义和公式，再安排一次 API 实验，学生缺少必须调用这些概念作判断的理由。")
    for run in p.runs: set_run_font(run, 10)
    add_callout(doc, "本课目标", "把‘会运行朴素贝叶斯’提升为‘能定义任务、解释模型、选择证据并限制结论’。", PALE_BLUE)
    doc.add_heading("2. 为什么适合 PBL", level=1)
    add_bullets(doc, ["100% accuracy 是诱人但证据不足的真实决策触发，学生必须先表态。", "train/validation/test、loss、混淆矩阵、Bayes 和 calibration 都因解决当前问题而出现。", "Packet A–D 逐步改变可见证据，允许并要求学生修订判断。", "同一结果同时支持‘模型复现规则’和限制‘模型可用于临床’，天然要求边界推理。", "最终 CERB 把代码输出转化为可审查的科研论证。"])
    doc.add_heading("3. 学习者与难度校准", level=1)
    add_table(doc, ["已有基础", "本课不假定", "本课施加的认知压力"], [["微积分、数理统计、前六讲的数据与推断概念", "CS 数据结构、优化算法、软件工程经验", "独立运行代码并解释输入/输出/假设"], ["条件概率与基本统计指标", "熟悉 sklearn API", "手算 posterior 并核对实现"], ["能读表格和基础图", "临床预测建模经验", "在冲突证据下选择指标并写边界"]], widths=[2.15, 2.05, 2.7], font_size=8.6)

    add_page_break(doc)
    doc.add_heading("4. 数据集与分析设计理由", level=1)
    add_table(doc, ["设计选择", "理由", "教学收益", "限制"], [["UCI Acute Inflammations", "小、公开、医学语义直观、可离线运行", "一节课内完成数据审计、手算和模型评价", "专家规则构造，不代表真实患者"], ["体温二值化", "与 Bernoulli NB 的二元输入一致，切点预先声明", "让预处理和泄漏问题可见", "丢失连续信息"], ["72/24/24 分层拆分", "课程明确要求训练、验证、测试与泛化", "把调参和最终评价职责分开", "样本仍小，不能提供稳定外推"], ["prior baseline", "揭示只看 accuracy/specificity 的危险", "错误类型与基线意识", "不是完整模型比较"], ["重复症状反例", "直接模拟独立证据被重复计算", "区分分类正确与概率可信", "不等于真实世界错误机制"], ["Wilson 区间", "零错误时仍显示有限样本的不确定性", "点估计与总体参数分离", "不是外部验证"]], widths=[1.55, 2.25, 2.2, 1.15], font_size=8.0)
    doc.add_heading("5. 渐进式证据弧", level=1)
    add_table(doc, ["阶段", "学生起始认识", "新证据", "期望修订"], [["触发", "100% 可能等于‘很好’", "只有一句 accuracy", "提出分母、错误、概率与来源问题"], ["Packet A", "干净数据可能质量高", "无缺失、近乎平衡、专家规则来源", "区分整齐与代表性"], ["Packet B", "模型分对即可", "基线、混淆矩阵、阈值、loss", "按用途解释错误类型与概率损失"], ["Packet C", "独立假设错则模型必然无效", "条件依赖、重复证据、校准诊断", "分类能力与 posterior 可信度分开"], ["Packet D", "24/24 等于真实 100%", "Wilson 区间与报告边界", "收缩到教学/方法基线和外部验证"]], widths=[1.0, 2.2, 2.4, 1.6], font_size=8.1)

    add_page_break(doc)
    doc.add_heading("6. 目标—证据覆盖", level=1)
    add_table(doc, ["LO", "数据问题", "方法/产物", "Packet", "解释边界"], [["/".join(a["objective_ids"]), a["question"], f"{a['method']}；{', '.join(a['artifacts'][:2])}", a["packet"], a["interpretation_boundary"]] for a in ANALYSES], widths=[0.75, 2.25, 2.35, 0.7, 1.05], font_size=7.4)
    doc.add_heading("7. 五类学习结果", level=1)
    add_table(doc, ["PBL 结果", "本课如何支持"], [["灵活知识", "同一指标在筛查、低患病率和不同阈值下意义不同"], ["问题解决", "从一句结果反向确定还需要什么证据"], ["自主学习", "个人契约、阅读路线、Evidence Card 与可运行 Notebook"], ["讨论协作", "同伴质询围绕主张、证据和边界，不依赖固定角色"], ["元认知", "初始投票、checkpoint、re-vote 和 CERB 记录观点变化"]], widths=[1.6, 5.3], font_size=8.7)

    add_page_break(doc)
    doc.add_heading("8. 预期结果", level=1)
    add_bullets(doc, ["学生能准确写出监督学习任务和数据流。", "学生能解释 Bernoulli NB 的概率机制，而不把训练等同于梯度下降。", "学生能从混淆矩阵计算 sensitivity 与 specificity，并说明其他指标各自回答什么。", "学生能解释条件依赖为什么可能影响 posterior 而不必立刻破坏分类。", "学生能用区间、患病率和外部验证边界反驳单一 accuracy 结论。", "学生能提交一份内部逻辑一致的 CERB 研究判断。"])
    doc.add_heading("9. 本课不声称的结果", level=1)
    add_callout(doc, "非目标", "不声称该模型具有临床效用、部署就绪、跨中心泛化能力、个体风险校准或因果解释；也不把一节课训练等同于完整的临床预测模型开发。", PALE_RED)
    doc.add_heading("10. 迁移设计", level=1)
    add_bullets(doc, ["近迁移：在同一数据上把标签改为 nephritis，重复完整评价纪律。", "情境迁移：低患病率筛查展示 precision 对目标人群的依赖。", "远迁移提示：基因表达分类首先按患者划分，并把标准化和特征选择放入训练流程内部。", "后续课程接口：第 8 讲线性模型、第 11 讲过拟合与第 12 讲验证策略可继续沿用本讲的任务—证据—边界语言。"])
    doc.add_heading("11. 数据来源", level=1)
    p = doc.add_paragraph(f"Czerniak, J. (2003). Acute Inflammations [Dataset]. UCI Machine Learning Repository. DOI: 10.24432/C5V59S. License: CC BY 4.0. {UCI_URL}")
    for run in p.runs: set_run_font(run, 9)
    doc.save(OUTPUTS["rationale"])


def preserve_and_stamp_notebooks() -> None:
    if not TEACHER_NOTEBOOK.exists():
        raise FileNotFoundError(f"Approved teacher notebook is missing: {TEACHER_NOTEBOOK}")
    notebook = json.loads(TEACHER_NOTEBOOK.read_text(encoding="utf-8"))
    metadata = notebook.setdefault("metadata", {})
    metadata["pbl_package"] = {
        "slug": SLUG,
        "role": "teacher_review_filled",
        "course_brief": "../../../plan.docx#lecture-7",
        "workplan": "../WORKPLAN.md",
        "analysis_design": f"../assets/pbl-{SLUG}/analysis-design.json",
        "manual_prose_preserved": True,
    }
    TEACHER_NOTEBOOK.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    if STUDENT_NOTEBOOK.exists():
        student = json.loads(STUDENT_NOTEBOOK.read_text(encoding="utf-8"))
        student_metadata = student.setdefault("metadata", {})
        student_metadata["pbl_package"] = {
            "slug": SLUG,
            "role": "student_blank",
            "teacher_answers_removed": True,
            "analysis_design": f"../assets/pbl-{SLUG}/analysis-design.json",
        }
        STUDENT_NOTEBOOK.write_text(
            json.dumps(student, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
        )


def write_manifest() -> None:
    mapped_assets = sorted(
        {
            artifact
            for analysis in ANALYSES
            for artifact in analysis["artifacts"]
        }
    )
    manifest = {
        "topic": "机器学习基础与朴素贝叶斯",
        "slug": SLUG,
        "mode": "algorithm_and_model_evaluation_pbl",
        "dataset": {
            "name": "UCI Acute Inflammations",
            "source": UCI_URL,
            "doi": "10.24432/C5V59S",
            "license": "CC BY 4.0",
            "task": "binary classification and evidence audit",
            "target": TARGET,
            "observation_unit": "expert-constructed symptom-label profile, not a real patient record",
        },
        "outputs": {
            "student_worksheet": OUTPUTS["worksheet"].name,
            "teaching_plan": OUTPUTS["teacher"].name,
            "evidence_pack": OUTPUTS["evidence"].name,
            "design_rationale": OUTPUTS["rationale"].name,
            "notebook": TEACHER_NOTEBOOK.relative_to(COURSE_DIR).as_posix(),
            "student_notebook": STUDENT_NOTEBOOK.relative_to(COURSE_DIR).as_posix(),
        },
        "assets": list(
            dict.fromkeys(
                [
                    f"assets/pbl-{SLUG}/analysis-design.json",
                    f"assets/pbl-{SLUG}/dataset-summary.csv",
                    f"assets/pbl-{SLUG}/acute-inflammations.csv",
                    f"assets/pbl-{SLUG}/main-result-clean-vs-audit.png",
                    *[f"assets/pbl-{SLUG}/{name}" for name in mapped_assets],
                ]
            )
        ),
        "boundary": "Teaching only; no clinical diagnosis, deployment, causal, or patient-care claim.",
        "core_question": "When a medical prediction model reports 100% accuracy, what evidence is needed before changing the research or screening decision?",
        "customized": True,
        "course_brief": "../../../plan.docx#lecture-7",
        "workplan": "WORKPLAN.md",
        "analysis_design": f"assets/pbl-{SLUG}/analysis-design.json",
        "build_script": f"scripts/pbl-{SLUG}-build.py",
        "source_of_truth_note": "The teacher notebook contains approved manual prose; the build preserves its cells and stamps package metadata rather than replacing the instructional text.",
    }
    (COURSE_DIR / "package-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def extract_doc_text(path: Path) -> str:
    document = Document(path)
    parts = [paragraph.text for paragraph in document.paragraphs]
    for table in document.tables:
        for row in table.rows:
            parts.extend(cell.text for cell in row.cells)
    return "\n".join(parts)


def validate_package() -> None:
    from PIL import Image

    design = json.loads((ASSET_DIR / "analysis-design.json").read_text(encoding="utf-8"))
    manifest = json.loads((COURSE_DIR / "package-manifest.json").read_text(encoding="utf-8"))
    objective_ids = {item["id"] for item in design["learning_objectives"]}
    covered = {objective_id for analysis in design["analyses"] for objective_id in analysis["objective_ids"]}
    if objective_ids != covered:
        raise AssertionError(f"Objective coverage mismatch: expected {objective_ids}, covered {covered}")
    for packet in ("B", "C", "D"):
        if not any(analysis["packet"] == packet for analysis in design["analyses"]):
            raise AssertionError(f"Packet {packet} has no mapped analysis")

    for analysis in design["analyses"]:
        for artifact in analysis["artifacts"]:
            path = ASSET_DIR / artifact
            if not path.exists() or path.stat().st_size == 0:
                raise AssertionError(f"Missing mapped artifact: {path}")

    for output_path in OUTPUTS.values():
        if not output_path.exists() or output_path.stat().st_size < 15_000:
            raise AssertionError(f"DOCX missing or unexpectedly small: {output_path}")

    required_doc_terms = {
        "worksheet": ["Section 1", "Packet A", "Packet B", "Packet C", "Packet D", "CERB", "使用边界"],
        "evidence": ["Evidence Pack", "Packet A", "Packet B", "Packet C", "Packet D", "CC BY 4.0"],
        "teacher": ["Teacher Plan", "Section 1", "Section 2", "评价量规", "教师检查清单"],
        "rationale": ["Design Rationale", "为什么适合 PBL", "目标—证据覆盖", "本课不声称的结果"],
    }
    for key, terms in required_doc_terms.items():
        text = extract_doc_text(OUTPUTS[key])
        for term in terms:
            if term not in text:
                raise AssertionError(f"{OUTPUTS[key].name} is missing required term {term!r}")
        if "不得用于临床" not in text and "不支持临床" not in text and "不声称该模型具有临床" not in text and "非临床" not in text:
            raise AssertionError(f"{OUTPUTS[key].name} lacks an explicit clinical boundary")
        if "�" in text:
            raise AssertionError(f"Mangled Unicode detected in {OUTPUTS[key].name}")

    teacher_nb = json.loads(TEACHER_NOTEBOOK.read_text(encoding="utf-8"))
    if len(teacher_nb.get("cells", [])) < 40:
        raise AssertionError("Teacher notebook is too sparse")
    code_cells = [cell for cell in teacher_nb["cells"] if cell.get("cell_type") == "code"]
    if not code_cells or any(cell.get("execution_count") is None for cell in code_cells):
        raise AssertionError("Teacher notebook is not fully executed")
    if any(output.get("output_type") == "error" for cell in code_cells for output in cell.get("outputs", [])):
        raise AssertionError("Teacher notebook contains an execution error")
    notebook_source = "\n".join("".join(cell.get("source", [])) for cell in teacher_nb["cells"])
    for required in ("Packet B", "Packet C", "Packet D", "BernoulliNB", "CERB", "clinical"):
        if required not in notebook_source:
            raise AssertionError(f"Teacher notebook lacks {required!r}")

    image_names = {artifact for analysis in design["analyses"] for artifact in analysis["artifacts"] if artifact.endswith(".png")}
    image_names.add("main-result-clean-vs-audit.png")
    for name in image_names:
        with Image.open(ASSET_DIR / name) as image:
            if image.width < 900:
                raise AssertionError(f"Plot is below 900 px width: {name} ({image.width}px)")

    if manifest.get("customized") is not True:
        raise AssertionError("Manifest must mark the package customized")
    print("Package validation passed")


def build_documents_and_metadata() -> None:
    import_docx_symbols()
    write_analysis_design()
    build_student_worksheet()
    build_evidence_pack()
    build_teacher_plan()
    build_design_rationale()
    preserve_and_stamp_notebooks()
    write_manifest()
    validate_package()


def orchestrate() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-only", action="store_true")
    parser.add_argument("--docs-only", action="store_true")
    args = parser.parse_args()
    if args.analysis_only:
        generate_analysis_assets()
        write_analysis_design()
        return
    if args.docs_only:
        build_documents_and_metadata()
        return

    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    if has_module("sklearn") and has_module("matplotlib"):
        generate_analysis_assets()
        write_analysis_design()
    else:
        analysis_python = find_python_with(
            "sklearn",
            [Path(r"D:\Python\python.exe")],
        )
        subprocess.run(
            [str(analysis_python), str(Path(__file__).resolve()), "--analysis-only"],
            cwd=COURSE_DIR,
            check=True,
        )

    if has_module("docx"):
        build_documents_and_metadata()
    else:
        doc_python_candidates = sorted(
            Path.home().glob(
                ".cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe"
            )
        )
        document_python = find_python_with("docx", doc_python_candidates)
        subprocess.run(
            [str(document_python), str(Path(__file__).resolve()), "--docs-only"],
            cwd=COURSE_DIR,
            check=True,
        )


if __name__ == "__main__":
    orchestrate()
