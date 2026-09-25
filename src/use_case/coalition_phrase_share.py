"""Why SubgraphX coalitions are the least sufficient on constituency graphs.

For each graph explainer on the constituency graphs, measures the share of
phrase (non-terminal) nodes in the whole graph and inside the explanation
(the first K = round(sparsity * N) ranked nodes), the share of explanations in
which phrase nodes are the majority, and, among correct predictions, how often
the explanation alone reproduces the decision (Dimension-3 margin > 0) for
phrase-dominated versus word-dominated explanations.

Usage (cwd must contain outputs/)::

    python src/use_case/coalition_phrase_share.py --output revision/results_ft/coalition_phrase_share.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DATASETS = ["setfit_ag_news", "stanfordnlp_sst2"]
NON_TERMINALS = {"S", "SBAR", "SINV", "SQ", "SBARQ", "FRAG", "ROOT"}


def is_phrase(tok: str) -> bool:
    tok = tok.strip()
    return tok.startswith("«") or tok in NON_TERMINALS


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--module-root", type=Path, default=Path("outputs/use_case/module_datasets"))
    ap.add_argument("--tokens-gnn", type=Path, default=Path("outputs/analytics/tokens_v2"))
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args(argv)
    rows = []
    for method in ["subgraphx", "graphsvx"]:
        for ds in DATASETS:
            tok = pd.read_csv(a.tokens_gnn / method / ds / "constituency.csv", usecols=["global_graph_index", "num_nodes", "ranked_tokens"])
            mod = pd.read_csv(a.module_root / ds / f"module_dataset_{ds}_{method}_constituency.csv",
                              usecols=["global_graph_index", "label_id", "prediction_class", "sparsity", "consistency_preservation_sufficiency"])
            m = tok.merge(mod, on="global_graph_index")
            in_graph, in_expl, kept, correct = [], [], [], []
            for r in m.itertuples():
                toks = str(r.ranked_tokens).split(",")
                K = max(1, int(round(r.sparsity * int(r.num_nodes))))
                flags = [is_phrase(t) for t in toks]
                in_graph.append(np.mean(flags))
                in_expl.append(np.mean(flags[:K]))
                kept.append(r.consistency_preservation_sufficiency > 0)
                correct.append(r.prediction_class == r.label_id)
            in_graph, in_expl, kept, correct = map(np.array, (in_graph, in_expl, kept, correct))
            dom = in_expl > 0.5
            rows.append(
                dict(
                    method=method, dataset=ds, n=len(m),
                    phrase_share_graph=in_graph.mean(), phrase_share_explanation=in_expl.mean(),
                    share_phrase_dominated_explanations=dom.mean(),
                    decision_kept_correct_phrase_dominated=kept[correct & dom].mean() if (correct & dom).any() else np.nan,
                    decision_kept_correct_word_dominated=kept[correct & ~dom].mean() if (correct & ~dom).any() else np.nan,
                )
            )
    out = pd.DataFrame(rows)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.output, index=False)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
