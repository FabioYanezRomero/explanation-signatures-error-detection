"""Sign and magnitude of the Dimension-2 mask-out drop as error signals.

For every explainer--representation configuration, reports the error-oriented
AUROC of the mask-out drop at k in {1, 5, 10} used as a signed feature and as an
unsigned magnitude, and the share of correct and incorrect predictions whose
confidence *rises* by more than ``--rise`` when the top-k elements are removed.
For the language model it also fits the nested class-balanced logistic detector
(10-fold, out-of-fold AUROC) with the confidence features alone and with the
unsigned drops added, with a paired-bootstrap interval for the gain.

Usage (host side, cwd = outputs_finetuned_embeddings/)::

    python src/use_case/maskout_drop_sign.py \
        --module-root outputs/use_case/module_datasets \
        --output revision/results_ft/maskout_drop_sign.csv
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATASETS = ["setfit_ag_news", "stanfordnlp_sst2"]
CONF = ["prediction_confidence", "consistency_baseline_margin", "auc_origin_entropy"]
KS = (1, 5, 10)


def oof(X: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, class_weight="balanced"))
    cv = StratifiedKFold(10, shuffle=True, random_state=seed)
    return cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--module-root", type=Path, default=Path("outputs/use_case/module_datasets"))
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--rise", type=float, default=0.05)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    drop_cols = [f"progression_maskout_drop_k{k}" for k in KS]
    rows = []
    for ds in DATASETS:
        for f in sorted(glob.glob(str(a.module_root / ds / f"module_dataset_{ds}_*.csv"))):
            name = Path(f).stem.replace(f"module_dataset_{ds}_", "")
            method, graph = name.rsplit("_", 1) if not name.startswith("token_shap") else ("token_shap_llm", "tokens")
            d = pd.read_csv(f, usecols=["label_id", "prediction_class"] + CONF + drop_cols).fillna(0.0)
            err = (d.prediction_class != d.label_id).astype(int).to_numpy()
            row = dict(dataset=ds, method=method, graph=graph, n=len(d), n_errors=int(err.sum()))
            for k in KS:
                x = d[f"progression_maskout_drop_k{k}"].to_numpy()
                row[f"auroc_signed_k{k}"] = roc_auc_score(err, x)
                row[f"auroc_abs_k{k}"] = roc_auc_score(err, np.abs(x))
                row[f"rise_share_correct_k{k}"] = float((x[err == 0] < -a.rise).mean())
                row[f"rise_share_error_k{k}"] = float((x[err == 1] < -a.rise).mean())
            if method == "token_shap_llm":
                y = 1 - err
                base = oof(d[CONF].to_numpy(), y, a.seed)
                full = oof(np.column_stack([d[CONF].to_numpy(), np.abs(d[drop_cols].to_numpy())]), y, a.seed)
                rng = np.random.default_rng(a.seed)
                n = len(y)
                boots = []
                for _ in range(a.boot):
                    i = rng.integers(0, n, n)
                    if y[i].min() == y[i].max():
                        continue
                    boots.append(roc_auc_score(y[i], full[i]) - roc_auc_score(y[i], base[i]))
                row["nested_auroc_conf"] = roc_auc_score(y, base)
                row["nested_auroc_conf_plus_absdrop"] = roc_auc_score(y, full)
                row["nested_delta"] = row["nested_auroc_conf_plus_absdrop"] - row["nested_auroc_conf"]
                row["nested_delta_ci_low"], row["nested_delta_ci_high"] = np.percentile(boots, [2.5, 97.5])
            rows.append(row)
    out = pd.DataFrame(rows)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.output, index=False)
    print(out.round(3).to_string())


if __name__ == "__main__":
    main()
