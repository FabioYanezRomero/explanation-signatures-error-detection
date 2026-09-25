"""Standardised coefficients of the nested confidence + explanation detector.

Fits the class-balanced logistic detector of the manuscript (StandardScaler +
LogisticRegression, y = 1 for a correct prediction w.r.t. the gold label) on the
confidence features (MSP, top-2 margin, entropy) plus every explanation feature,
pooled over predicted classes, for each explainer--representation configuration,
and reports the point coefficient of every feature with a percentile interval
from ``--boot`` bootstrap refits.

Usage (host side, cwd = outputs_finetuned_embeddings/)::

    python src/use_case/nested_coefficients.py \
        --module-root outputs/use_case/module_datasets \
        --output revision/results_ft/nested_coefficients.csv
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_config import ALLOWED_FEATURES, dimension_for_feature  # noqa: E402

DATASETS = ["setfit_ag_news", "stanfordnlp_sst2"]
MARGIN = "consistency_baseline_margin"
CONF = ["prediction_confidence", MARGIN, "auc_origin_entropy"]
EXPL = sorted(ALLOWED_FEATURES - {MARGIN})


def fit(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, class_weight="balanced"))
    m.fit(X, y)
    return m[-1].coef_[0]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--module-root", type=Path, default=Path("outputs/use_case/module_datasets"))
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--paper-model", action="store_true",
                    help="fit the four-dimension detector instead (explanation features plus the top-2 margin, no MSP/entropy), "
                         "i.e. the model of Table 5's 'Four dimensions' columns pooled over predicted classes")
    a = ap.parse_args(argv)

    cols = sorted(ALLOWED_FEATURES) if a.paper_model else CONF + EXPL
    rows = []
    for ds in DATASETS:
        for f in sorted(glob.glob(str(a.module_root / ds / f"module_dataset_{ds}_*.csv"))):
            name = Path(f).stem.replace(f"module_dataset_{ds}_", "")
            method, graph = ("token_shap_llm", "tokens") if name.startswith("token_shap") else name.rsplit("_", 1)
            d = pd.read_csv(f, usecols=["label_id", "prediction_class"] + cols)
            d[cols] = d[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
            X = d[cols].to_numpy(dtype=float)
            y = (d.prediction_class == d.label_id).astype(int).to_numpy()
            point = fit(X, y)
            rng = np.random.default_rng(a.seed)
            boots = []
            for _ in range(a.boot):
                i = rng.integers(0, len(y), len(y))
                if y[i].min() == y[i].max():
                    continue
                boots.append(fit(X[i], y[i]))
            boots = np.array(boots)
            lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
            for j, c in enumerate(cols):
                rows.append(
                    dict(
                        dataset=ds,
                        method=method,
                        graph=graph,
                        feature=c,
                        block=("confidence" if (c in CONF and not a.paper_model) or c == MARGIN else dimension_for_feature(c)),
                        coef=point[j],
                        ci_low=lo[j],
                        ci_high=hi[j],
                        significant_95=bool(lo[j] > 0 or hi[j] < 0),
                    )
                )
            print(ds, method, graph, "done", file=sys.stderr)
    out = pd.DataFrame(rows)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.output, index=False)


if __name__ == "__main__":
    main()
