"""On BERT's gold-label errors, how does each surrogate behave? (distilled vs gold-trained control)

For every graph type, among the test instances that BERT misclassifies, reports the
fraction on which the surrogate is also wrong, the fraction on which it predicts the
same (wrong) class as BERT, and the fraction on which it does so with confidence > 0.9.

Usage (cwd must contain outputs/)::

    python src/use_case/control_agreement_on_bert_errors.py --output revision/results_ft/control_agreement_on_bert_errors.csv
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import pandas as pd

DATASETS = ["setfit_ag_news", "stanfordnlp_sst2"]
GRAPHS = ["constituency", "syntactic", "window", "skipgrams"]
KEY = "global_graph_index"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--module-root", type=Path, default=Path("outputs/use_case/module_datasets"))
    ap.add_argument("--control-root", type=Path, default=Path("outputs/gnn_models_control/true_label"))
    ap.add_argument("--threshold", type=float, default=0.9)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args(argv)
    rows = []
    for ds in DATASETS:
        b = pd.read_csv(a.module_root / ds / f"module_dataset_{ds}_token_shap_llm_tokens.csv", usecols=[KEY, "label_id", "prediction_class"]).sort_values(KEY).reset_index(drop=True)
        gold, bp = b.label_id.to_numpy(), b.prediction_class.to_numpy()
        m = bp != gold
        for g in GRAPHS:
            d = pd.read_csv(a.module_root / ds / f"module_dataset_{ds}_graphsvx_{g}.csv", usecols=[KEY, "prediction_class", "prediction_confidence"]).sort_values(KEY).reset_index(drop=True)
            cands = [c for c in glob.glob(str(a.control_root / "**" / g / "test_predictions.csv"), recursive=True) if ds.split("_", 1)[1] in c]
            if not cands:
                continue
            p = pd.read_csv(cands[0]).sort_values("position").reset_index(drop=True)
            assert (p.true_label.to_numpy() == gold).all()
            for model, pred, conf in (("distilled", d.prediction_class.to_numpy(), d.prediction_confidence.to_numpy()),
                                      ("gold-trained", p.prediction_class.to_numpy(), p.prediction_confidence.to_numpy())):
                rows.append(dict(dataset=ds, graph=g, model=model, n_bert_errors=int(m.sum()),
                                 also_wrong=float((pred[m] != gold[m]).mean()),
                                 same_class_as_bert=float((pred[m] == bp[m]).mean()),
                                 same_class_confident=float(((pred[m] == bp[m]) & (conf[m] > a.threshold)).mean())))
    out = pd.DataFrame(rows)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.output, index=False)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
