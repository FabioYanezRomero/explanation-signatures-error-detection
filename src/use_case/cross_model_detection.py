#!/usr/bin/env python3
"""Cross-model error detection: can GNN-surrogate information improve the detection of
the *language model's* errors beyond the language model's own confidence?

Target: BERT error with respect to the gold label (instance-aligned across modules).
Base model: BERT confidence features (MSP, top-2 margin, entropy).
Nested additions (each evaluated as base + addition, paired bootstrap of the AUROC gain):
  * BERT explanation features (TokenSHAP)          -> within-model reference
  * GNN confidence features                         -> a second opinion
  * GNN confidence + GNN explanation features
  * GNN disagreement flag (pred_GNN != pred_BERT)
  * committee: confidence + disagreement of the four topologies of one explainer
Same protocol as revision_metrics.py (10-fold stratified CV, scaled class-balanced LR).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from use_case.revision_metrics import (  # type: ignore  # noqa: E402
    CONFIDENCE_FEATURES,
    EXPLANATION_FEATURES,
    out_of_fold_probs,
    paired_bootstrap_delta,
)

DATASETS = ["setfit_ag_news", "stanfordnlp_sst2"]
GRAPHS = ["constituency", "syntactic", "window", "skipgrams"]
METHODS = ["graphsvx", "subgraphx"]
KEY = "global_graph_index"


def load(root: Path, ds: str, method: str, graph: str) -> pd.DataFrame:
    f = root / ds / f"module_dataset_{ds}_{method}_{graph}.csv"
    return pd.read_csv(f)


def prefixed(df: pd.DataFrame, cols: List[str], prefix: str) -> pd.DataFrame:
    cols = [c for c in cols if c in df.columns]
    out = df[[KEY] + cols].copy()
    out.columns = [KEY] + [f"{prefix}{c}" for c in cols]
    return out


def evaluate(y: np.ndarray, X_base: np.ndarray, additions: Dict[str, np.ndarray], *, folds: int, reps: int, seed: int, mask=None):
    """Return rows: AUROC/AUPRC of base and of base+addition, with paired bootstrap deltas."""
    y_err = 1 - y
    p_base = out_of_fold_probs(X_base, y, folds=folds, seed=seed)
    rows = []
    sel = np.ones(len(y), bool) if mask is None else mask
    base_auroc = roc_auc_score(y_err[sel], 1 - p_base[sel])
    base_auprc = average_precision_score(y_err[sel], 1 - p_base[sel])
    for name, X_add in additions.items():
        p_full = out_of_fold_probs(np.hstack([X_base, X_add]), y, folds=folds, seed=seed)
        d_roc, lo_roc, hi_roc = paired_bootstrap_delta(y_err[sel], 1 - p_base[sel], 1 - p_full[sel], reps=reps, seed=seed, metric=roc_auc_score)
        d_pr, lo_pr, hi_pr = paired_bootstrap_delta(y_err[sel], 1 - p_base[sel], 1 - p_full[sel], reps=reps, seed=seed, metric=average_precision_score)
        rows.append(dict(addition=name, n_added=X_add.shape[1], auroc_base=base_auroc, auroc_full=roc_auc_score(y_err[sel], 1 - p_full[sel]),
                         delta_auroc=d_roc, delta_auroc_ci_low=lo_roc, delta_auroc_ci_high=hi_roc,
                         auprc_base=base_auprc, auprc_full=average_precision_score(y_err[sel], 1 - p_full[sel]),
                         delta_auprc=d_pr, delta_auprc_ci_low=lo_pr, delta_auprc_ci_high=hi_pr))
    return rows


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--module-root", type=Path, default=Path("outputs/use_case/module_datasets"))
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--bootstrap-reps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(argv)

    out = []
    for ds in DATASETS:
        bert = load(a.module_root, ds, "token_shap_llm", "tokens").sort_values(KEY).reset_index(drop=True)
        y = (bert.prediction_class.astype(int) == bert.label_id.astype(int)).astype(int).to_numpy()
        conf_b = prefixed(bert, CONFIDENCE_FEATURES, "bert_")
        expl_b = prefixed(bert, EXPLANATION_FEATURES, "bert_")
        regimes = {"all": np.ones(len(y), bool),
                   "bert_confidence>0.9": bert.prediction_confidence.to_numpy() > 0.9,
                   "bert_confidence>0.95": bert.prediction_confidence.to_numpy() > 0.95}
        for method in METHODS:
            gnn = {g: load(a.module_root, ds, method, g) for g in GRAPHS}
            merged = bert[[KEY, "prediction_class"]].merge(conf_b, on=KEY).merge(expl_b, on=KEY)
            for g in GRAPHS:
                d = gnn[g]
                d = d.assign(**{f"{g}_disagree": (d.prediction_class.astype(int) != bert.set_index(KEY).loc[d[KEY], "prediction_class"].to_numpy().astype(int)).astype(float)})
                merged = merged.merge(prefixed(d, CONFIDENCE_FEATURES + EXPLANATION_FEATURES + [f"{g}_disagree"], f"{g}_"), on=KEY)
            merged = merged.sort_values(KEY).reset_index(drop=True)
            assert len(merged) == len(bert), (ds, method, len(merged), len(bert))
            X_base = merged[[f"bert_{c}" for c in CONFIDENCE_FEATURES]].to_numpy()
            gconf = lambda g: merged[[f"{g}_{c}" for c in CONFIDENCE_FEATURES]].to_numpy()
            gexpl = lambda g: merged[[f"{g}_{c}" for c in EXPLANATION_FEATURES if f"{g}_{c}" in merged.columns]].to_numpy()
            gdis = lambda g: merged[[f"{g}_{g}_disagree"]].to_numpy()
            additions: Dict[str, np.ndarray] = {"bert_explanation": merged[[f"bert_{c}" for c in EXPLANATION_FEATURES if f"bert_{c}" in merged.columns]].to_numpy()}
            for g in GRAPHS:
                additions[f"{g}:gnn_confidence"] = gconf(g)
                additions[f"{g}:gnn_disagreement"] = gdis(g)
                additions[f"{g}:gnn_confidence+explanation"] = np.hstack([gconf(g), gexpl(g)])
                additions[f"{g}:gnn_confidence+explanation+bert_explanation"] = np.hstack([gconf(g), gexpl(g), additions["bert_explanation"]])
            additions["committee:confidence+disagreement(4 topologies)"] = np.hstack([np.hstack([gconf(g), gdis(g)]) for g in GRAPHS])
            additions["committee:confidence+explanation(4 topologies)"] = np.hstack([np.hstack([gconf(g), gexpl(g)]) for g in GRAPHS])
            for regime, mask in regimes.items():
                if (1 - y)[mask].sum() < 10:
                    continue
                for r in evaluate(y, X_base, additions, folds=a.folds, reps=a.bootstrap_reps, seed=a.seed, mask=mask):
                    out.append(dict(dataset=ds, method=method, regime=regime, n=int(mask.sum()), n_bert_errors=int((1 - y)[mask].sum()), **r))
            print(ds, method, "done", flush=True)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out).to_csv(a.output, index=False)
    print("written", a.output)


if __name__ == "__main__":
    main()
