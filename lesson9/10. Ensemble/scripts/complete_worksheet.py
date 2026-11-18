"""Reproduce WDBC worksheet evidence without opening locked test labels.

Run from any directory: python path/to/scripts/complete_worksheet.py
Outputs: ../results/ and ../figures/. No input data or teacher scripts are changed.
"""
from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

import demo_ensemble as demo

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "results"
FIG = BASE / "figures"
SEED = 1010
COLORS = ["#777777", "#0072B2", "#009E73", "#CC79A7", "#D55E00", "#E69F00", "#56B4E9"]
LABELS = {"single_tree": "Tree (depth 4)", "bagging": "Bagging", "random_forest": "Random forest",
          "adaboost": "AdaBoost", "gradient_boosting": "Gradient boosting", "logistic": "Logistic", "rbf_svm": "RBF-SVM"}


def save_csv(frame: pd.DataFrame, name: str) -> None:
    frame.to_csv(OUT / f"{name}.csv", index=False, float_format="%.12g")


def save_fig(fig: plt.Figure, name: str) -> None:
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(FIG / f"{name}.{suffix}", dpi=350, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def panel(ax: plt.Axes, title: str) -> None:
    ax.set_title(title, loc="left", fontweight="bold", pad=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="0.92", linewidth=.6)
    ax.set_axisbelow(True)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.labelsize": 9,
                         "axes.titlesize": 10, "legend.fontsize": 8, "axes.linewidth": .7,
                         "lines.linewidth": 1.5, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "svg.fonttype": "none"})
    original = {}
    for path in BASE.rglob("*"):
        rel = path.relative_to(BASE)
        if path.is_file() and rel.parts[0] in {"data", "data-ctg", "slides", "scripts"} and "__pycache__" not in rel.parts and path.name != Path(__file__).name:
            # Protected labels and raw data are not even hashed/read by this analysis.
            if "teacher" not in rel.parts and "raw" not in rel.parts:
                original[str(rel)] = hashlib.sha256(path.read_bytes()).hexdigest()
    train, valid, features, label, _ = demo.load("wdbc")
    X, y, Xv, yv = train[features], train[label], valid[features], valid[label]
    dev = pd.concat([train, valid], ignore_index=True)
    assert len(features) == 30 and not set(train.sample_id) & set(valid.sample_id)
    assert not dev[features].isna().any().any()
    print("1/6: validation and shared five-fold CV", flush=True)
    comparison = demo.fit_table(X, y, Xv, yv)
    save_csv(comparison, "model-comparison")
    fitted = {name: model.fit(X, y) for name, model in demo.models().items()}
    pred = []
    for name, model in fitted.items():
        for sid, target, score, cls in zip(valid.sample_id, yv, demo.score(model, Xv), model.predict(Xv)):
            pred.append(dict(model=name, sample_id=sid, target=int(target), score=score, prediction=int(cls)))
    save_csv(pd.DataFrame(pred), "validation-predictions")
    folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(dev[features], dev[label]))
    cv_rows, cv_pred, fold_ids = [], [], []
    for fold, (tr, va) in enumerate(folds, 1):
        fold_ids.extend(dict(sample_id=sid, fold=fold) for sid in dev.iloc[va].sample_id)
        for name, model in demo.models().items():
            model.fit(dev.iloc[tr][features], dev.iloc[tr][label])
            score = demo.score(model, dev.iloc[va][features])
            cls = model.predict(dev.iloc[va][features])
            cv_rows.append(dict(model=name, fold=fold, roc_auc=roc_auc_score(dev.iloc[va][label], score),
                                balanced_accuracy=balanced_accuracy_score(dev.iloc[va][label], cls)))
            cv_pred.extend(dict(model=name, fold=fold, sample_id=sid, target=int(target), score=s, prediction=int(c))
                           for sid, target, s, c in zip(dev.iloc[va].sample_id, dev.iloc[va][label], score, cls))
    cv = pd.DataFrame(cv_rows)
    save_csv(cv, "cv-folds")
    save_csv(pd.DataFrame(cv_pred), "cv-predictions")
    save_csv(pd.DataFrame(fold_ids), "cv-membership")
    cv_summary = cv.groupby("model", sort=False).agg(roc_auc_mean=("roc_auc", "mean"), roc_auc_sd=("roc_auc", "std"),
                                                                 ba_mean=("balanced_accuracy", "mean"), ba_sd=("balanced_accuracy", "std")).reset_index()
    save_csv(cv_summary, "cross-validation")
    depth_rows = []
    for depth in (1, 2, 3, 4, None):
        tree = DecisionTreeClassifier(max_depth=depth, random_state=SEED).fit(X, y)
        depth_rows.append(dict(depth=str(depth) if depth else "unlimited", leaves=tree.get_n_leaves(),
                               train_accuracy=accuracy_score(y, tree.predict(X)), validation_accuracy=accuracy_score(yv, tree.predict(Xv)),
                               train_auc=roc_auc_score(y, tree.predict_proba(X)[:, 1]), validation_auc=roc_auc_score(yv, tree.predict_proba(Xv)[:, 1])))
    depths = pd.DataFrame(depth_rows)
    save_csv(depths, "tree-depth")
    rng = np.random.default_rng(SEED)
    root_rows = []
    for iteration in range(10):
        idx = rng.choice(len(y), len(y), replace=True)
        tree = DecisionTreeClassifier(max_depth=2, random_state=SEED).fit(X.iloc[idx], y.iloc[idx])
        root_rows.append(dict(refit=iteration + 1, feature=features[tree.tree_.feature[0]], threshold=tree.tree_.threshold[0]))
    save_csv(pd.DataFrame(root_rows), "bootstrap-roots")
    print("2/6: 1000 bootstrap simulations (n=5 and n=341)", flush=True)
    boot_rows = []
    for n in (5, len(y)):
        rng = np.random.default_rng(SEED)
        for iteration in range(1000):
            fraction = len(np.unique(rng.choice(n, n, replace=True))) / n
            boot_rows.append(dict(n=n, iteration=iteration + 1, unique_fraction=fraction, oob_fraction=1-fraction))
    bootstrap = pd.DataFrame(boot_rows)
    save_csv(bootstrap, "bootstrap-simulation")
    print("3/6: 30 shared bootstrap refits per model", flush=True)
    rng = np.random.default_rng(SEED)
    samples = np.array([rng.choice(len(y), len(y), replace=True) for _ in range(30)])
    np.savez_compressed(OUT / "training-bootstrap-indices.npz", indices=samples)
    stability, stability_predictions, case_sd = [], [], {}
    # LR and GB are added so the final recommendation also has measured stability evidence.
    for name in ("single_tree", "bagging", "random_forest", "logistic", "gradient_boosting"):
        probs = np.array([clone(demo.models()[name]).fit(X.iloc[idx], y.iloc[idx]).predict_proba(Xv)[:, 1] for idx in samples])
        sd = probs.std(axis=0, ddof=0)  # same descriptive definition as the teacher demo
        case_sd[name] = sd
        share = (probs >= .5).mean(axis=0)
        stability.append(dict(model=name, prediction_sd_mean=sd.mean(), prediction_sd_max=sd.max(), class_flip_rate=((share > 0) & (share < 1)).mean()))
        stability_predictions.extend(dict(model=name, refit=r + 1, sample_id=sid, probability=p)
                                     for r, row in enumerate(probs) for sid, p in zip(valid.sample_id, row))
        print("  finished", name, flush=True)
    stability = pd.DataFrame(stability)
    save_csv(stability, "prediction-stability")
    save_csv(pd.DataFrame(stability_predictions), "bootstrap-predictions")
    print("4/6: ensemble size and feature importance", flush=True)
    counts = demo.tree_count_curve(X, y, Xv, yv)
    save_csv(counts, "tree-count-curve")
    forest = clone(demo.models()["random_forest"]).set_params(n_estimators=300).fit(X, y)
    perm = permutation_importance(forest, Xv, yv, scoring="roc_auc", n_repeats=20, random_state=SEED)
    importance = pd.DataFrame(dict(feature=features, impurity_importance=forest.feature_importances_,
                                   permutation_auc_drop_mean=perm.importances_mean, permutation_auc_drop_sd=perm.importances_std))
    save_csv(importance.sort_values("impurity_importance", ascending=False), "feature-importance")
    save_csv(pd.DataFrame(perm.importances, index=features).rename_axis("feature").reset_index(), "permutation-repeats")
    corr = X.corr()
    corr.rename_axis("feature").to_csv(OUT / "train-feature-correlations.csv")
    print("5/6: boosting curves, learning_rate=0.03, 0.1, 0.3; 500 stages", flush=True)
    boost_rows = []
    for lr in (.03, .1, .3):
        model = GradientBoostingClassifier(n_estimators=500, learning_rate=lr, max_depth=3, random_state=SEED).fit(X, y)
        for stage, (pt, pv) in enumerate(zip(model.staged_predict_proba(X), model.staged_predict_proba(Xv)), 1):
            boost_rows.append(dict(learning_rate=lr, stage=stage, train_log_loss=log_loss(y, pt), validation_log_loss=log_loss(yv, pv),
                                   validation_auc=roc_auc_score(yv, pv[:, 1])))
    boost = pd.DataFrame(boost_rows)
    save_csv(boost, "boosting-curves")
    print("6/6: export six figures as PNG, PDF and SVG", flush=True)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.65), layout="constrained")
    for col, mark, color, text in (("train_accuracy", "o", COLORS[1], "Train"), ("validation_accuracy", "s", COLORS[4], "Validation")):
        axes[0].plot(range(5), depths[col], marker=mark, color=color, label=text)
    axes[0].set(xticks=range(5), xticklabels=["1", "2", "3", "4", "None"], xlabel="Maximum tree depth", ylabel="Accuracy", ylim=(.88, 1.01))
    axes[0].legend(frameon=False)
    panel(axes[0], "a  Depth and generalization")
    b = bootstrap[bootstrap.n == len(y)]
    axes[1].hist(b.unique_fraction, bins=18, color=COLORS[1], alpha=.8, edgecolor="white", linewidth=.4)
    axes[1].axvline(1-(1-1/len(y))**len(y), color=COLORS[4], linestyle="--", label="Finite-n expectation")
    axes[1].set(xlabel="Fraction of distinct training records", ylabel="Count (1000 resamples)")
    axes[1].legend(frameon=False)
    panel(axes[1], "b  Bootstrap coverage (n = 341)")
    save_fig(fig, "fig1-depth-bootstrap")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), layout="constrained")
    selected = ["single_tree", "bagging", "random_forest"]
    for j, name in enumerate(selected):
        axes[0].scatter(comparison.set_index("model").loc[name, "roc_auc"], j, s=45, color=COLORS[j], zorder=3)
        axes[1].boxplot(case_sd[name], positions=[j], widths=.5, patch_artist=True, showfliers=False,
                        boxprops=dict(facecolor=COLORS[j], alpha=.5), medianprops=dict(color="black"))
        axes[1].scatter(np.full(len(yv), j) + np.random.default_rng(SEED).uniform(-.12, .12, len(yv)), case_sd[name], s=4, alpha=.4, color=COLORS[j], rasterized=True)
    axes[0].set(yticks=range(3), yticklabels=[LABELS[n] for n in selected], xlabel="Validation ROC-AUC", xlim=(.92, 1.005))
    axes[0].invert_yaxis()
    axes[1].set(xticks=range(3), xticklabels=["Tree", "Bagging", "RF"], ylabel="Prediction SD across 30 refits", ylim=(-.01, .52))
    panel(axes[0], "a  Same validation records (n = 114)")
    panel(axes[1], "b  Training perturbation stability")
    save_fig(fig, "fig2-ensemble-stability")

    fig, ax = plt.subplots(figsize=(5.6, 2.65), layout="constrained")
    for name, color, marker in (("bagging", COLORS[1], "o"), ("random_forest", COLORS[2], "s")):
        sub = counts[counts.model == name]
        ax.plot(sub.n_trees, sub.validation_roc_auc, color=color, marker=marker, label=LABELS[name])
    ax.set(xscale="log", xticks=[1, 10, 50, 100, 300], xticklabels=[1, 10, 50, 100, 300], xlabel="Number of trees (log scale)", ylabel="Validation ROC-AUC", ylim=(.9, 1.005))
    panel(ax, "Ensemble size: rapid gains, then a plateau")
    ax.legend(frameon=False)
    save_fig(fig, "fig3-tree-count")

    top = importance.nlargest(7, "impurity_importance").feature.tolist()
    order = importance.set_index("feature").loc[top].iloc[::-1]
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.1), layout="constrained", gridspec_kw={"width_ratios": [1.15, 1.15, 1]})
    yy = np.arange(len(order))
    axes[0].barh(yy, order.impurity_importance, color=COLORS[1], height=.6)
    axes[0].set(yticks=yy, yticklabels=order.index, xlabel="Mean decrease in impurity")
    axes[1].errorbar(order.permutation_auc_drop_mean, yy, xerr=order.permutation_auc_drop_sd, fmt="o", color=COLORS[4], capsize=2, markersize=4)
    axes[1].axvline(0, color="0.6", linewidth=.7)
    axes[1].set(yticks=yy, yticklabels=[], xlabel="Validation AUC drop (mean ± SD)")
    axes[1].locator_params(axis="x", nbins=4)
    triad = ["worst radius", "worst perimeter", "worst area"]
    matrix = corr.loc[triad, triad].to_numpy()
    im = axes[2].imshow(matrix, cmap="Blues", vmin=0, vmax=1)
    short = ["Radius", "Perimeter", "Area"]
    axes[2].set(xticks=range(3), xticklabels=short, yticks=range(3), yticklabels=short)
    axes[2].tick_params(axis="x", labelrotation=35)
    for i in range(3):
        for j in range(3):
            axes[2].text(j, i, f"{matrix[i,j]:.3f}", ha="center", va="center", color="white", fontsize=8)
    for ax, title in zip(axes, ["a  Training importance", "b  Permutation: 20 shuffles", "c  Training Pearson r"]):
        panel(ax, title)
    axes[2].grid(False)
    save_fig(fig, "fig4-importance-correlation")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), layout="constrained")
    for lr, color in zip((.03, .1, .3), [COLORS[1], COLORS[4], COLORS[2]]):
        sub = boost[boost.learning_rate == lr]
        axes[0].plot(sub.stage, sub.validation_log_loss, color=color, label=f"lr={lr:g}, validation")
        axes[0].plot(sub.stage, sub.train_log_loss, color=color, linestyle="--", alpha=.8)
        axes[1].plot(sub.stage, sub.validation_auc, color=color, label=f"lr={lr:g}")
    axes[0].set(xlabel="Boosting stage", ylabel="Log-loss (lower is better)")
    axes[1].set(xlabel="Boosting stage", ylabel="Validation ROC-AUC", ylim=(.97, 1.002))
    axes[0].legend(frameon=False)
    axes[1].legend(frameon=False)
    panel(axes[0], "a  Solid: validation; dashed: train")
    panel(axes[1], "b  Ranking can hide probability errors")
    save_fig(fig, "fig5-boosting-curves")

    fig, ax = plt.subplots(figsize=(6.4, 3.4), layout="constrained")
    for j, (name, color) in enumerate(zip(LABELS, COLORS)):
        row = cv_summary.set_index("model").loc[name]
        ax.scatter(cv[cv.model == name].roc_auc, np.full(5, j), color=color, s=17, alpha=.5, zorder=3)
        ax.errorbar(row.roc_auc_mean, j+.16, xerr=row.roc_auc_sd, fmt="D", color=color, capsize=3, markersize=4)
    ax.set(yticks=range(len(LABELS)), yticklabels=list(LABELS.values()), xlabel="Development ROC-AUC (dots: folds; diamond: mean ± SD)", xlim=(.85, 1.006))
    ax.invert_yaxis()
    panel(ax, "Shared stratified five-fold CV (n = 455)")
    save_fig(fig, "fig6-fair-comparison")

    report = {"seed": SEED, "dataset": "WDBC", "train_n": len(y), "train_positive": int(y.sum()), "validation_n": len(yv),
              "validation_positive": int(yv.sum()), "n_features": len(features), "missing_cells": int(dev[features].isna().sum().sum()),
              "duplicate_feature_rows_development": int(dev[features].duplicated().sum()), "test_labels_opened": False,
              "cv": "5 shared stratified folds on train+validation; no tuning; scaler fit within each training fold",
              "stability": "30 shared row-bootstrap resamples of train; algorithm seed fixed; per-record probability SD ddof=0, then mean; flip=any class change across refits",
              "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__, "matplotlib": matplotlib.__version__},
              "model_parameters": {name: str(m.get_params()) for name, m in demo.models().items()}, "unchanged_input_sha256": original,
              "pubfig": "not available: pip found no matching distribution; matplotlib fallback"}
    for rel, digest in original.items():
        assert hashlib.sha256((BASE / rel).read_bytes()).hexdigest() == digest, rel
    (OUT / "run-info.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(comparison.round(4).to_string(index=False), flush=True)
    print(cv_summary.round(4).to_string(index=False), flush=True)
    print(stability.round(4).to_string(index=False), flush=True)
    print("Saved", OUT, "and", FIG, flush=True)


if __name__ == "__main__":
    main()
