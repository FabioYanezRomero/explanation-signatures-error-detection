"""Gold-accuracy gap between each GCN surrogate and its BERT teacher.

For every dataset and graph type, pairs the surrogate's test predictions with
BERT's (module datasets keyed by ``global_graph_index``) and reports

* surrogate-only and BERT-only gold errors,
* the gap in accuracy points with a paired-bootstrap 95% interval and an exact
  McNemar p-value,
* the disagreement rate with the teacher, the share of disagreements that fall
  on predictions with BERT confidence below ``--low-conf``, and the accuracy of
  both models on the disagreements.

``label_id`` is the gold label; ``label`` in the surrogate files is the teacher
label, so ``is_correct`` there means agreement with the teacher and is not used.

Usage (host side, cwd = outputs_finetuned_embeddings/)::

    python src/use_case/surrogate_gap_vs_bert.py \
        --module-root outputs/use_case/module_datasets \
        --output revision/results_ft/surrogate_gap_vs_bert.csv
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

DATASETS = ["setfit_ag_news", "stanfordnlp_sst2"]
GRAPHS = ["constituency", "syntactic", "window", "skipgrams"]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--module-root", type=Path, default=Path("outputs/use_case/module_datasets"))
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--method", default="graphsvx", help="explainer whose module dataset supplies the surrogate predictions (identical across explainers)")
    ap.add_argument("--low-conf", type=float, default=0.9)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    rows = []
    for ds in DATASETS:
        llm = glob.glob(str(a.module_root / ds / f"module_dataset_{ds}_token_shap_llm_*.csv"))[0]
        b = pd.read_csv(llm, usecols=["global_graph_index", "label_id", "prediction_class", "prediction_confidence"])
        b = b.rename(columns={"prediction_class": "bert_pred", "prediction_confidence": "bert_conf"})
        b["bert_ok"] = b.bert_pred == b.label_id
        for g in GRAPHS:
            f = a.module_root / ds / f"module_dataset_{ds}_{a.method}_{g}.csv"
            s = pd.read_csv(f, usecols=["global_graph_index", "label_id", "prediction_class"])
            s = s.rename(columns={"prediction_class": "surr_pred", "label_id": "label_id_s"})
            m = b.merge(s, on="global_graph_index")
            n = len(m)
            m["surr_ok"] = m.surr_pred == m.label_id_s
            surr_only = int((~m.surr_ok & m.bert_ok).sum())
            bert_only = int((m.surr_ok & ~m.bert_ok).sum())
            p = binomtest(surr_only, surr_only + bert_only, 0.5).pvalue if surr_only + bert_only else 1.0
            d = (m.bert_ok.astype(int) - m.surr_ok.astype(int)).to_numpy()
            rng = np.random.default_rng(a.seed)
            boots = np.array([d[rng.integers(0, n, n)].mean() for _ in range(a.boot)]) * 100
            dis = m.bert_pred != m.surr_pred
            low = m.bert_conf < a.low_conf
            rows.append(
                dict(
                    dataset=ds,
                    graph=g,
                    n=n,
                    bert_accuracy=m.bert_ok.mean(),
                    surrogate_accuracy=m.surr_ok.mean(),
                    surrogate_only_errors=surr_only,
                    bert_only_errors=bert_only,
                    gap_points=(surr_only - bert_only) / n * 100,
                    gap_ci_low=np.percentile(boots, 2.5),
                    gap_ci_high=np.percentile(boots, 97.5),
                    mcnemar_p=p,
                    disagreement_rate=dis.mean(),
                    share_low_conf_all=low.mean(),
                    share_low_conf_among_disagreements=low[dis].mean(),
                    disagreement_rate_within_low_conf=dis[low].mean(),
                    bert_accuracy_on_disagreements=m.bert_ok[dis].mean(),
                    surrogate_accuracy_on_disagreements=m.surr_ok[dis].mean(),
                )
            )
    out = pd.DataFrame(rows)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.output, index=False)
    print(out.round(4).to_string())


if __name__ == "__main__":
    main()
