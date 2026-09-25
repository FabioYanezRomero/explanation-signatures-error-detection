#!/usr/bin/env python3
"""Baseline-relative error-detection metrics for the revised manuscript.

For every per-module dataset (``outputs/use_case/module_datasets/<dataset>/module_dataset_*.csv``)
this script reports, under the same stratified 10-fold protocol as the paper:

* the majority-class baseline and, for each feature set, accuracy, balanced accuracy,
  AUROC, AUPRC of the error class, MCC and precision/recall/F1 of the error class;
* nested models -- confidence-only (MSP, top-2 margin, entropy), explanation-only and
  confidence + explanation -- with a paired bootstrap CI for the AUROC gain that the
  explanation features bring on top of confidence (delta-AUROC);
* two error definitions for the GNN surrogates: disagreement with the BERT teacher
  (``is_correct``, the label the surrogates were trained on) and error against the gold
  label (``prediction_class == label_id``);
* for the quadrant analyses (Dimensions 3 and 4), Jensen-Shannon divergence between the
  quadrant distributions of correct and incorrect predictions and Cramer's V of the
  quadrant x correctness contingency table, replacing the former Separability index
  (which reduced to sqrt(2) times a weighted standard deviation).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import chi2_contingency
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from use_case.feature_config import ALLOWED_FEATURES  # type: ignore  # noqa: E402

warnings.filterwarnings("ignore")

MARGIN_FEATURE = "consistency_baseline_margin"
CONFIDENCE_FEATURES = ["prediction_confidence", MARGIN_FEATURE, "auc_origin_entropy"]
EXPLANATION_FEATURES = sorted(ALLOWED_FEATURES - {MARGIN_FEATURE})
PAPER_FEATURES = sorted(ALLOWED_FEATURES)

FEATURE_SETS: Dict[str, List[str]] = {
    "confidence": CONFIDENCE_FEATURES,
    "explanation": EXPLANATION_FEATURES,
    "confidence+explanation": CONFIDENCE_FEATURES + EXPLANATION_FEATURES,
    "paper (4 dimensions)": PAPER_FEATURES,
}
SINGLE_FEATURES = [
    "prediction_confidence",
    MARGIN_FEATURE,
    "auc_origin_entropy",
    "auc_deletion_auc",
    "auc_insertion_auc",
    "fidelity_minus",
    "fidelity_plus",
    "fidelity_asymmetry",
    "consistency_preservation_necessity",
    "consistency_preservation_sufficiency",
    "progression_maskout_drop_k5",
    "progression_sufficiency_drop_k5",
]
CONSISTENCY_QUADRANTS = ["Sufficient-Necessary", "Sufficient-Redundant", "Insufficient-Necessary", "Insufficient-Redundant"]
FIDELITY_QUADRANTS = ["Faithful", "Redundant", "Incomplete", "Unfaithful"]


def consistency_quadrant(suff_ratio: float, nec_ratio: float) -> str:
    suff_positive = suff_ratio >= 0
    nec_positive = nec_ratio >= 0
    if suff_positive and nec_positive:
        return "Sufficient-Redundant"
    if suff_positive and not nec_positive:
        return "Sufficient-Necessary"
    if not suff_positive and nec_positive:
        return "Insufficient-Redundant"
    return "Insufficient-Necessary"


def fidelity_quadrant(f_plus: float, f_minus: float, threshold: float = 0.5) -> str:
    """Dimension-4 quadrant from the stored normalised drops.

    The pipeline stores ``fidelity_plus = 1 - c(x_S)/c(x)`` (drop when keeping only the
    explanation) and ``fidelity_minus = 1 - c(x_minus_S)/c(x)`` (drop when removing it).
    Sufficiency is therefore ``1 - fidelity_plus`` and necessity is ``fidelity_minus``;
    an explanation is Faithful when both are at least ``threshold``.
    """
    f_plus = 0.0 if pd.isna(f_plus) else float(f_plus)
    f_minus = 0.0 if pd.isna(f_minus) else float(f_minus)
    sufficiency = 1.0 - f_plus
    necessity = f_minus
    if sufficiency >= threshold and necessity >= threshold:
        return "Faithful"
    if sufficiency >= threshold and necessity < threshold:
        return "Redundant"
    if sufficiency < threshold and necessity >= threshold:
        return "Incomplete"
    return "Unfaithful"


def _clean(X: np.ndarray) -> np.ndarray:
    return np.clip(np.nan_to_num(X.astype(float), nan=0.0, posinf=0.0, neginf=0.0), -50.0, 50.0)


def out_of_fold_probs(X: np.ndarray, y: np.ndarray, *, folds: int, seed: int) -> np.ndarray:
    """Out-of-fold P(correct) under the paper's protocol (scaled, class-balanced LR)."""
    n_splits = min(folds, int(np.bincount(y).min()))
    if n_splits < 2:
        raise ValueError("not enough minority samples for cross-validation")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, class_weight="balanced"))
    return cross_val_predict(model, _clean(X), y, cv=cv, method="predict_proba")[:, 1]


def classification_metrics(y: np.ndarray, p_correct: np.ndarray) -> Dict[str, float]:
    """Metrics with the ERROR class as the positive class of interest."""
    y_err = 1 - y
    p_err = 1.0 - p_correct
    pred_err = (p_err >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y_err, pred_err),
        "balanced_accuracy": balanced_accuracy_score(y_err, pred_err),
        "auroc": roc_auc_score(y_err, p_err),
        "auprc_error": average_precision_score(y_err, p_err),
        "mcc": matthews_corrcoef(y_err, pred_err),
        "precision_error": precision_score(y_err, pred_err, zero_division=0),
        "recall_error": recall_score(y_err, pred_err, zero_division=0),
        "f1_error": f1_score(y_err, pred_err, zero_division=0),
    }


def paired_bootstrap_delta_auroc(
    y: np.ndarray, p_base: np.ndarray, p_full: np.ndarray, *, reps: int, seed: int
) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    deltas = []
    n = len(y)
    for _ in range(reps):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        deltas.append(roc_auc_score(y[idx], p_full[idx]) - roc_auc_score(y[idx], p_base[idx]))
    if not deltas:
        return float("nan"), float("nan"), float("nan")
    arr = np.asarray(deltas)
    return float(arr.mean()), float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def expected_calibration_error(conf: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            ece += mask.mean() * abs(conf[mask].mean() - correct[mask].mean())
    return float(ece)


def paired_bootstrap_delta(y_err, p_base, p_full, *, reps, seed, metric):
    rng = np.random.default_rng(seed)
    deltas = []
    n = len(y_err)
    for _ in range(reps):
        idx = rng.integers(0, n, n)
        if y_err[idx].sum() == 0 or y_err[idx].sum() == n:
            continue
        deltas.append(metric(y_err[idx], p_full[idx]) - metric(y_err[idx], p_base[idx]))
    if not deltas:
        return float("nan"), float("nan"), float("nan")
    arr = np.asarray(deltas)
    return float(arr.mean()), float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def quadrant_divergence(labels: Sequence[str], correct: np.ndarray, order: Sequence[str]) -> Dict[str, float]:
    labels_arr = np.asarray(labels)
    correct_mask = np.asarray(correct).astype(bool)
    counts = np.array(
        [[np.sum((labels_arr == q) & correct_mask), np.sum((labels_arr == q) & ~correct_mask)] for q in order],
        dtype=float,
    )  # rows: quadrants, columns: [correct, incorrect]
    p_correct = counts[:, 0] / max(counts[:, 0].sum(), 1.0)
    p_incorrect = counts[:, 1] / max(counts[:, 1].sum(), 1.0)
    js_bits = float(jensenshannon(p_correct, p_incorrect, base=2) ** 2)
    nonzero = counts[counts.sum(axis=1) > 0]
    if nonzero.shape[0] >= 2 and nonzero.sum(axis=0).min() > 0:
        chi2, p_value, _, _ = chi2_contingency(nonzero, correction=False)
        n = nonzero.sum()
        k = min(nonzero.shape) - 1
        cramers_v = math.sqrt(chi2 / (n * k)) if k > 0 else float("nan")
    else:
        chi2, p_value, cramers_v = float("nan"), float("nan"), float("nan")
    # Legacy index for reference: weighted SD of per-quadrant %correct, weights = quadrant prevalence.
    totals = counts.sum(axis=1)
    weights = totals / max(totals.sum(), 1.0)
    pct_correct = np.where(totals > 0, 100.0 * counts[:, 0] / np.maximum(totals, 1), 0.0)
    mean = float(np.dot(pct_correct, weights))
    weighted_sd = math.sqrt(float(np.dot(weights, (pct_correct - mean) ** 2)))
    return {
        "js_divergence_bits": js_bits,
        "cramers_v": cramers_v,
        "chi2": chi2,
        "chi2_p_value": p_value,
        "legacy_weighted_sd": weighted_sd,
        "legacy_separability": weighted_sd * math.sqrt(2.0),
        "n_correct": int(counts[:, 0].sum()),
        "n_incorrect": int(counts[:, 1].sum()),
        "pct_correct_by_quadrant": json.dumps({q: round(float(v), 2) for q, v in zip(order, pct_correct)}),
        "quadrant_share_pct": json.dumps({q: round(float(100 * w), 2) for q, w in zip(order, weights)}),
    }


def analyse_module(df: pd.DataFrame, meta: Dict[str, str], *, folds: int, reps: int, seed: int):
    rows, nested_rows, quad_rows, single_rows, regime_rows = [], [], [], [], []
    targets = {"teacher_agreement": df["is_correct"].astype(int).to_numpy()}
    if "label_id" in df.columns and df["label_id"].notna().all():
        targets["gold_label"] = (df["prediction_class"].astype(int) == df["label_id"].astype(int)).astype(int).to_numpy()
    alignment = float(np.mean(df["label"] == df["label_id"])) if "label_id" in df.columns else float("nan")

    for target_name, y in targets.items():
        base = dict(meta, target=target_name, n=len(y), n_errors=int((1 - y).sum()), error_rate=float(1 - y.mean()),
                    teacher_gold_alignment=alignment)
        majority = classification_metrics(y, np.ones_like(y, dtype=float))
        rows.append(dict(base, feature_set="majority baseline", n_features=0, **majority))
        probs: Dict[str, np.ndarray] = {}
        for set_name, cols in FEATURE_SETS.items():
            cols_present = [c for c in cols if c in df.columns]
            if not cols_present:
                continue
            try:
                p = out_of_fold_probs(df[cols_present].to_numpy(), y, folds=folds, seed=seed)
            except ValueError as exc:
                print(f"  ! {meta['dataset']}/{meta['method']}/{meta['graph']} [{target_name}] {set_name}: {exc}")
                continue
            probs[set_name] = p
            rows.append(dict(base, feature_set=set_name, n_features=len(cols_present), **classification_metrics(y, p)))
        if "confidence" in probs and "confidence+explanation" in probs:
            y_err = 1 - y
            mean, lo, hi = paired_bootstrap_delta_auroc(y_err, 1 - probs["confidence"], 1 - probs["confidence+explanation"], reps=reps, seed=seed)
            nested_rows.append(dict(base,
                auroc_confidence=roc_auc_score(y_err, 1 - probs["confidence"]),
                auroc_explanation=roc_auc_score(y_err, 1 - probs["explanation"]) if "explanation" in probs else float("nan"),
                auroc_confidence_plus_explanation=roc_auc_score(y_err, 1 - probs["confidence+explanation"]),
                delta_auroc=mean, delta_auroc_ci_low=lo, delta_auroc_ci_high=hi, bootstrap_reps=reps))
        # Calibration and the high-confidence regime (errors the confidence cannot flag)
        conf = df["prediction_confidence"].to_numpy(dtype=float)
        conf_cols = [c for c in CONFIDENCE_FEATURES if c in df.columns]
        for thr in (None, 0.9, 0.95):
            mask = np.ones(len(y), dtype=bool) if thr is None else conf > thr
            ys = y[mask]
            n_err = int((1 - ys).sum())
            row = dict(base, regime="all" if thr is None else f"confidence>{thr}", n_regime=int(mask.sum()), n_errors_regime=n_err,
                       error_rate_regime=float(1 - ys.mean()) if len(ys) else float("nan"),
                       ece=expected_calibration_error(conf, y) if thr is None else float("nan"),
                       share_of_errors_in_regime=float(mask[y == 0].mean()) if (y == 0).any() else float("nan"))
            if n_err >= 10 and len(ys) - n_err >= 10:
                try:
                    p_c = out_of_fold_probs(df.loc[mask, conf_cols].to_numpy(), ys, folds=folds, seed=seed)
                    p_ce = out_of_fold_probs(df.loc[mask, conf_cols + [c for c in EXPLANATION_FEATURES if c in df.columns]].to_numpy(), ys, folds=folds, seed=seed)
                    y_err = 1 - ys
                    row.update(auroc_confidence=roc_auc_score(y_err, 1 - p_c), auroc_confidence_plus_explanation=roc_auc_score(y_err, 1 - p_ce),
                               auprc_confidence=average_precision_score(y_err, 1 - p_c), auprc_confidence_plus_explanation=average_precision_score(y_err, 1 - p_ce))
                    d, lo, hi = paired_bootstrap_delta(y_err, 1 - p_c, 1 - p_ce, reps=reps, seed=seed, metric=roc_auc_score)
                    row.update(delta_auroc=d, delta_auroc_ci_low=lo, delta_auroc_ci_high=hi)
                    d, lo, hi = paired_bootstrap_delta(y_err, 1 - p_c, 1 - p_ce, reps=reps, seed=seed, metric=average_precision_score)
                    row.update(delta_auprc=d, delta_auprc_ci_low=lo, delta_auprc_ci_high=hi)
                except ValueError:
                    pass
            regime_rows.append(row)
        for feat in SINGLE_FEATURES:
            if feat in df.columns:
                values = _clean(df[feat].to_numpy())
                auc = roc_auc_score(y, values)
                single_rows.append(dict(base, feature=feat, auroc_correct=auc, auroc_error_oriented=max(auc, 1 - auc)))
        # Quadrant divergences (Dimensions 3 and 4)
        suff = df["consistency_sufficiency_ratio"].replace([np.inf, -np.inf], 0.0).fillna(0.0)
        nec = df["consistency_necessity_ratio"].replace([np.inf, -np.inf], 0.0).fillna(0.0)
        cons_labels = [consistency_quadrant(s, n) for s, n in zip(suff, nec)]
        fid_labels = [fidelity_quadrant(a, b) for a, b in zip(df["fidelity_plus"], df["fidelity_minus"])]
        quad_rows.append(dict(base, dimension="Dimension 3 (consistency)", **quadrant_divergence(cons_labels, y, CONSISTENCY_QUADRANTS)))
        quad_rows.append(dict(base, dimension="Dimension 4 (fidelity)", **quadrant_divergence(fid_labels, y, FIDELITY_QUADRANTS)))
    return rows, nested_rows, quad_rows, single_rows, regime_rows


def discover_modules(root: Path):
    for dataset_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name != "coefficients"):
        prefix = f"module_dataset_{dataset_dir.name}_"
        for csv_path in sorted(dataset_dir.glob("module_dataset_*.csv")):
            remainder = csv_path.stem.replace(prefix, "", 1)
            method, graph = remainder.rsplit("_", 1)
            yield {"dataset": dataset_dir.name, "method": method, "graph": graph}, csv_path


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--module-root", type=Path, default=Path("outputs/use_case/module_datasets"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/use_case/revision_metrics"))
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--bootstrap-reps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--only-quadrants", action="store_true", help="Recompute only the quadrant association tables (fast).")
    args = parser.parse_args(argv)

    all_rows, all_nested, all_quad, all_single, all_regime = [], [], [], [], []
    for meta, csv_path in discover_modules(args.module_root):
        print(f"• {meta['dataset']} / {meta['method']} / {meta['graph']}")
        df = pd.read_csv(csv_path)
        if args.only_quadrants:
            for target_name, y in {"teacher_agreement": df["is_correct"].astype(int).to_numpy(),
                                   "gold_label": (df["prediction_class"].astype(int) == df["label_id"].astype(int)).astype(int).to_numpy()}.items():
                base = dict(meta, target=target_name, n=len(y), n_errors=int((1 - y).sum()), error_rate=float(1 - y.mean()))
                suff = df["consistency_sufficiency_ratio"].replace([np.inf, -np.inf], 0.0).fillna(0.0)
                nec = df["consistency_necessity_ratio"].replace([np.inf, -np.inf], 0.0).fillna(0.0)
                cons_labels = [consistency_quadrant(a, b) for a, b in zip(suff, nec)]
                fid_labels = [fidelity_quadrant(a, b) for a, b in zip(df["fidelity_plus"], df["fidelity_minus"])]
                all_quad.append(dict(base, dimension="Dimension 3 (consistency)", **quadrant_divergence(cons_labels, y, CONSISTENCY_QUADRANTS)))
                all_quad.append(dict(base, dimension="Dimension 4 (fidelity)", **quadrant_divergence(fid_labels, y, FIDELITY_QUADRANTS)))
            continue
        r, n, q, s, g = analyse_module(df, meta, folds=args.folds, reps=args.bootstrap_reps, seed=args.seed)
        all_rows += r; all_nested += n; all_quad += q; all_single += s; all_regime += g

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.only_quadrants:
        pd.DataFrame(all_quad).to_csv(args.output_dir / "quadrant_divergence.csv", index=False)
        print(f"Wrote {len(all_quad)} quadrant rows -> {args.output_dir}")
        return
    pd.DataFrame(all_rows).to_csv(args.output_dir / "error_detection_metrics.csv", index=False)
    pd.DataFrame(all_nested).to_csv(args.output_dir / "nested_delta_auroc.csv", index=False)
    pd.DataFrame(all_quad).to_csv(args.output_dir / "quadrant_divergence.csv", index=False)
    pd.DataFrame(all_single).to_csv(args.output_dir / "single_feature_auroc.csv", index=False)
    pd.DataFrame(all_regime).to_csv(args.output_dir / "calibration_and_confidence_regimes.csv", index=False)
    print(f"Wrote {len(all_rows)} metric rows, {len(all_nested)} nested rows, {len(all_quad)} quadrant rows -> {args.output_dir}")


if __name__ == "__main__":
    main()
