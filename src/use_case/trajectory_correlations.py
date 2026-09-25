"""What do the Dimension-1 trajectories track? Per-instance Spearman correlations.

For every explainer--representation configuration, correlates the deletion and
insertion AUC (over the explanation budget) with the model's confidence in the
predicted class and with the explanation-only confidence of Dimension 4
(``prediction_confidence - fidelity_plus``), and reports the mean deletion AUC
next to the mean confidence.

Usage (cwd must contain outputs/)::

    python src/use_case/trajectory_correlations.py --output revision/results_ft/trajectory_correlations.csv
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

DATASETS = ["setfit_ag_news", "stanfordnlp_sst2"]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--module-root", type=Path, default=Path("outputs/use_case/module_datasets"))
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args(argv)
    rows = []
    for ds in DATASETS:
        for f in sorted(glob.glob(str(a.module_root / ds / f"module_dataset_{ds}_*.csv"))):
            name = Path(f).stem.replace(f"module_dataset_{ds}_", "")
            method, graph = ("token_shap_llm", "tokens") if name.startswith("token_shap") else name.rsplit("_", 1)
            d = pd.read_csv(f, usecols=["prediction_confidence", "auc_deletion_auc", "auc_insertion_auc", "fidelity_plus"]).dropna()
            c_S = d.prediction_confidence - d.fidelity_plus
            rows.append(
                dict(
                    dataset=ds, method=method, graph=graph, n=len(d),
                    spearman_deletion_confidence=spearmanr(d.auc_deletion_auc, d.prediction_confidence)[0],
                    spearman_insertion_confidence=spearmanr(d.auc_insertion_auc, d.prediction_confidence)[0],
                    spearman_insertion_explanation_only_confidence=spearmanr(d.auc_insertion_auc, c_S)[0],
                    spearman_deletion_explanation_only_confidence=spearmanr(d.auc_deletion_auc, c_S)[0],
                    mean_deletion_auc=d.auc_deletion_auc.mean(), mean_insertion_auc=d.auc_insertion_auc.mean(),
                    mean_confidence=d.prediction_confidence.mean(), mean_explanation_only_confidence=c_S.mean(),
                )
            )
    out = pd.DataFrame(rows)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.output, index=False)
    print(out.round(3).to_string(index=False))

    # Same surrogate, two explainers: is each trajectory a property of the model or of the explanation?
    cross = []
    for ds in DATASETS:
        for g in ["constituency", "syntactic", "window", "skipgrams"]:
            fa = a.module_root / ds / f"module_dataset_{ds}_graphsvx_{g}.csv"
            fb = a.module_root / ds / f"module_dataset_{ds}_subgraphx_{g}.csv"
            if not (fa.exists() and fb.exists()):
                continue
            cols = ["global_graph_index", "auc_deletion_auc", "auc_insertion_auc"]
            m = pd.read_csv(fa, usecols=cols).dropna().merge(pd.read_csv(fb, usecols=cols).dropna(), on="global_graph_index", suffixes=("_graphsvx", "_subgraphx"))
            cross.append(
                dict(
                    dataset=ds, graph=g, n=len(m),
                    mean_deletion_graphsvx=m.auc_deletion_auc_graphsvx.mean(), mean_deletion_subgraphx=m.auc_deletion_auc_subgraphx.mean(),
                    spearman_deletion_between_explainers=spearmanr(m.auc_deletion_auc_graphsvx, m.auc_deletion_auc_subgraphx)[0],
                    mean_insertion_graphsvx=m.auc_insertion_auc_graphsvx.mean(), mean_insertion_subgraphx=m.auc_insertion_auc_subgraphx.mean(),
                    spearman_insertion_between_explainers=spearmanr(m.auc_insertion_auc_graphsvx, m.auc_insertion_auc_subgraphx)[0],
                )
            )
    cross_out = pd.DataFrame(cross)
    cross_path = a.output.with_name(a.output.stem + "_cross_explainer.csv")
    cross_out.to_csv(cross_path, index=False)
    print(cross_out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
