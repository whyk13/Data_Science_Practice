"""Check saved metrics, links and protected input hashes; no model refitting."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "results"


def main() -> None:
    checks: list[str] = []
    report = json.loads((OUT / "run-info.json").read_text(encoding="utf-8"))
    for name, digest in report["unchanged_input_sha256"].items():
        assert hashlib.sha256((BASE / name).read_bytes()).hexdigest() == digest, name
    assert not report["test_labels_opened"]
    checks.append(f"{len(report['unchanged_input_sha256'])} input hashes unchanged; test labels not opened")
    predictions = pd.read_csv(OUT / "validation-predictions.csv")
    table = pd.read_csv(OUT / "model-comparison.csv").set_index("model")
    target = pd.read_csv(BASE / "data/student/validation.csv").set_index("sample_id").malignant
    for name, rows in predictions.groupby("model"):
        assert len(rows) == 114 and rows.sample_id.nunique() == 114
        assert np.array_equal(rows.target, target.loc[rows.sample_id])
        assert np.isclose(roc_auc_score(rows.target, rows.score), table.loc[name, "roc_auc"])
        assert np.isclose(balanced_accuracy_score(rows.target, rows.prediction), table.loc[name, "balanced_accuracy"])
    checks.append("7 validation AUC/BA values recomputed from 798 stored predictions")
    cvp = pd.read_csv(OUT / "cv-predictions.csv")
    cv = pd.read_csv(OUT / "cv-folds.csv").set_index(["model", "fold"])
    membership = pd.read_csv(OUT / "cv-membership.csv").set_index("sample_id").fold
    for name, rows in cvp.groupby("model"):
        assert len(rows) == 455 and rows.sample_id.nunique() == 455
        assert np.array_equal(rows.fold, membership.loc[rows.sample_id])
        for fold, block in rows.groupby("fold"):
            assert len(block) == 91
            assert np.isclose(roc_auc_score(block.target, block.score), cv.loc[(name, fold), "roc_auc"])
            assert np.isclose(balanced_accuracy_score(block.target, block.prediction), cv.loc[(name, fold), "balanced_accuracy"])
    summary = pd.read_csv(OUT / "cross-validation.csv").set_index("model")
    for name, rows in cv.reset_index().groupby("model"):
        assert np.isclose(rows.roc_auc.mean(), summary.loc[name, "roc_auc_mean"])
        assert np.isclose(rows.roc_auc.std(ddof=1), summary.loc[name, "roc_auc_sd"])
    checks.append("35 CV fold metrics, shared membership and 7 mean/SD pairs verified")
    bp = pd.read_csv(OUT / "bootstrap-predictions.csv")
    st = pd.read_csv(OUT / "prediction-stability.csv").set_index("model")
    for name, rows in bp.groupby("model"):
        p = rows.pivot(index="refit", columns="sample_id", values="probability").to_numpy()
        assert p.shape == (30, 114) and np.isfinite(p).all()
        assert np.isclose(p.std(axis=0, ddof=0).mean(), st.loc[name, "prediction_sd_mean"])
        cls = p >= .5
        assert np.isclose((cls.any(axis=0) & ~cls.all(axis=0)).mean(), st.loc[name, "class_flip_rate"])
    idx = np.load(OUT / "training-bootstrap-indices.npz")["indices"]
    assert idx.shape == (30, 341) and idx.min() >= 0 and idx.max() < 341
    checks.append("5 stability summaries recomputed from 17100 predictions; shared bootstrap indices valid")
    worksheet = (BASE / "student-worksheet.md").read_text(encoding="utf-8")
    for link in re.findall(r"\]\(([^)]+)\)", worksheet):
        if not link.startswith("https://"):
            assert (BASE / link).exists(), link
    assert not re.search(r"\|[ \t]*\|", worksheet), "empty table cell"
    assert all(f"## {letter}." in worksheet for letter in "ABCDEFGH")
    memo = worksheet.split("**研究备忘录：**")[1].strip()
    chinese = len(re.findall(r"[\u4e00-\u9fff]", memo))
    assert 250 <= chinese <= 400 and len(memo) <= 400
    for name in ("logistic", "rbf_svm", "random_forest", "gradient_boosting"):
        assert f"{summary.loc[name, 'roc_auc_mean']:.4f} ± {summary.loc[name, 'roc_auc_sd']:.4f}" in worksheet
        assert f"{table.loc[name, 'balanced_accuracy']:.4f}" in worksheet
    checks.append(f"A-H, filled tables, all local links, F table values and memo ({len(memo)} characters; {chinese} Chinese) verified")
    pngs = list((BASE / "figures").glob("*.png"))
    assert len(pngs) == 6
    for path in pngs:
        with Image.open(path) as img:
            assert img.width > 1500 and img.height > 700
            img.verify()
        assert path.with_suffix(".pdf").read_bytes().startswith(b"%PDF-")
        assert "<svg" in path.with_suffix(".svg").read_text(encoding="utf-8")
    checks.append("6 PNGs and corresponding PDF/SVG exports structurally valid; visual checks recorded in figure-catalog.md")
    result = {"passed": True, "checks": checks, "memo_characters": len(memo), "memo_chinese_characters": chinese}
    (OUT / "verification.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
