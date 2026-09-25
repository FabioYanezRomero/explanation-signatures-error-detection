#!/usr/bin/env python3
"""Non-linear counterpart of the nested detectors: gradient boosting on confidence vs confidence+explanation.

10-fold stratified out-of-fold probabilities, class-balanced HistGradientBoosting, paired bootstrap
CI of the AUROC gain. Gold-label errors. Run from a directory containing outputs/.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from use_case.revision_metrics import CONFIDENCE_FEATURES, EXPLANATION_FEATURES  # noqa: E402

def oof(X, y, folds, seed):
    p = np.zeros(len(y))
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, y):
        clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=3, class_weight="balanced", random_state=seed)
        clf.fit(X[tr], y[tr]); p[te] = clf.predict_proba(X[te])[:, 1]
    return p

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--module-root", type=Path, default=Path("outputs/use_case/module_datasets"))
    ap.add_argument("--output", type=Path, required=True); ap.add_argument("--folds", type=int, default=10); ap.add_argument("--bootstrap-reps", type=int, default=1000); ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(); rng = np.random.default_rng(a.seed); rows = []
    for f in sorted(a.module_root.glob("*/module_dataset_*.csv")):
        ds, rest = f.parent.name, f.stem.replace(f"module_dataset_{f.parent.name}_", "")
        method, graph = rest.rsplit("_", 1)
        df = pd.read_csv(f); y = (df.prediction_class != df.label_id).astype(int).to_numpy()
        conf = [c for c in CONFIDENCE_FEATURES if c in df.columns]; expl = [c for c in EXPLANATION_FEATURES if c in df.columns]
        Xc = df[conf].fillna(0).to_numpy(); Xce = df[conf + expl].fillna(0).to_numpy()
        pc, pce = oof(Xc, y, a.folds, a.seed), oof(Xce, y, a.folds, a.seed)
        d_auc, d_ap = [], []
        for _ in range(a.bootstrap_reps):
            i = rng.integers(0, len(y), len(y))
            if y[i].min() == y[i].max(): continue
            d_auc.append(roc_auc_score(y[i], pce[i]) - roc_auc_score(y[i], pc[i])); d_ap.append(average_precision_score(y[i], pce[i]) - average_precision_score(y[i], pc[i]))
        r = dict(dataset=ds, method=method, graph=graph, n=len(y), n_errors=int(y.sum()), auroc_confidence=roc_auc_score(y, pc), auroc_confidence_plus_explanation=roc_auc_score(y, pce),
                 delta_auroc=roc_auc_score(y, pce) - roc_auc_score(y, pc), delta_auroc_ci_low=np.percentile(d_auc, 2.5), delta_auroc_ci_high=np.percentile(d_auc, 97.5),
                 auprc_confidence=average_precision_score(y, pc), auprc_confidence_plus_explanation=average_precision_score(y, pce), delta_auprc=average_precision_score(y, pce) - average_precision_score(y, pc),
                 delta_auprc_ci_low=np.percentile(d_ap, 2.5), delta_auprc_ci_high=np.percentile(d_ap, 97.5))
        rows.append(r); print(ds, method, graph, {k: round(v, 3) for k, v in r.items() if k.startswith(("auroc", "delta_auroc"))}, flush=True)
    a.output.parent.mkdir(parents=True, exist_ok=True); pd.DataFrame(rows).to_csv(a.output, index=False); print("written", a.output)

if __name__ == "__main__":
    main()
