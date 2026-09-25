"""Dimension 3 read as decisions: does the prediction survive the perturbation?

For every explainer--representation configuration and separately for correct
and incorrect predictions (gold label), reports the share of instances whose
decision flips when the explanation is removed (margin after removal < 0), the
share whose decision is reproduced by the explanation alone (margin of the
explanation alone > 0), and the median margin ratios m(x\\S)/m(x) and m(x_S)/m(x).

Usage (cwd must contain outputs/)::

    python src/use_case/margin_flips.py --output revision/results_ft/margin_flips.csv
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

DATASETS = ["setfit_ag_news", "stanfordnlp_sst2"]
COLS = ["label_id", "prediction_class", "consistency_baseline_margin", "consistency_preservation_necessity", "consistency_preservation_sufficiency"]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--module-root", type=Path, default=Path("outputs/use_case/module_datasets"))
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--tex", type=Path, default=None, help="also write LaTeX table rows here")
    a = ap.parse_args(argv)
    rows = []
    for ds in DATASETS:
        for f in sorted(glob.glob(str(a.module_root / ds / f"module_dataset_{ds}_*.csv"))):
            name = Path(f).stem.replace(f"module_dataset_{ds}_", "")
            method, graph = ("token_shap_llm", "tokens") if name.startswith("token_shap") else name.rsplit("_", 1)
            d = pd.read_csv(f, usecols=COLS).dropna()
            err = (d.prediction_class != d.label_id).to_numpy()
            m = d.consistency_baseline_margin.to_numpy()
            after_removal = d.consistency_preservation_necessity.to_numpy()
            alone = d.consistency_preservation_sufficiency.to_numpy()
            with np.errstate(divide="ignore", invalid="ignore"):
                r_rem = after_removal / m
                r_alone = alone / m
            for label, mask in (("error", err), ("correct", ~err)):
                rows.append(
                    dict(
                        dataset=ds, method=method, graph=graph, outcome=label, n=int(mask.sum()),
                        flip_on_removal=float((after_removal[mask] < 0).mean()),
                        decision_kept_by_explanation_alone=float((alone[mask] > 0).mean()),
                        median_margin_ratio_after_removal=float(np.nanmedian(r_rem[mask])),
                        median_margin_ratio_explanation_alone=float(np.nanmedian(r_alone[mask])),
                    )
                )
    out = pd.DataFrame(rows)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.output, index=False)
    print(out.round(3).to_string(index=False))

    # LaTeX rows for the supplementary table (one row per explainer and graph type)
    if a.tex:
        M = {"subgraphx": "SubgraphX", "graphsvx": "GraphSVX", "token_shap_llm": "TokenSHAP"}
        G = {"constituency": "Constituency", "syntactic": "Syntactic", "window": "Window", "skipgrams": "Skip-gram", "tokens": "Tokens"}
        lines = []
        for method in ["subgraphx", "graphsvx", "token_shap_llm"]:
            graphs = [g for g in G if ((out.method == method) & (out.graph == g)).any()]
            for i, g in enumerate(graphs):
                cells = []
                for ds in DATASETS:
                    for col in ("flip_on_removal", "decision_kept_by_explanation_alone"):
                        for outcome in ("correct", "error"):
                            r = out[(out.method == method) & (out.graph == g) & (out.dataset == ds) & (out.outcome == outcome)]
                            cells.append(f"{100 * r[col].iloc[0]:.0f}" if len(r) else "--")
                mcell = (rf"\multirow{{{len(graphs)}}}{{*}}{{{M[method]}}}" if len(graphs) > 1 else M[method]) if i == 0 else ""
                lines.append(" & ".join([mcell, G[g]] + cells) + r" \\")
            lines.append(r"\midrule")
        lines.pop()
        a.tex.parent.mkdir(parents=True, exist_ok=True)
        a.tex.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
