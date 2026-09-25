#!/usr/bin/env python3
"""Ablation table: node features from the pre-trained encoder (original submission) vs the fine-tuned teacher.

Ranges over the eight configurations (two explainers x four topologies) of each dataset, from the revision CSVs
of both runs. Writes tab_ablation.tex (rows: dataset x feature setting).
"""
import argparse
from pathlib import Path
import pandas as pd

D = {"setfit_ag_news": "AG News", "stanfordnlp_sst2": "SST-2"}

def rng(s, nd=2, pct=False):
    lo, hi = s.min(), s.max()
    if pct:
        lo, hi = round(100 * lo), round(100 * hi)
        return f"{lo}" if lo == hi else f"{lo}--{hi}"
    return f"{lo:.{nd}f}" if f"{lo:.{nd}f}" == f"{hi:.{nd}f}" else f"{lo:.{nd}f}--{hi:.{nd}f}"

def load(res):
    n = pd.read_csv(res / "nested_delta_auroc.csv"); n = n[(n.target == "gold_label") & (n.method != "token_shap_llm")]
    e = pd.read_csv(res / "error_detection_metrics.csv"); e = e[(e.target == "gold_label") & (e.method != "token_shap_llm")]
    c = pd.read_csv(res / "cross_model_bert_errors.csv")
    k = pd.read_csv(res / "control_error_overlap.csv"); k = k[k.model.isin(["teacher (original)", "gold"])]
    cal = pd.read_csv(res / "calibration_and_confidence_regimes.csv"); cal = cal[(cal.target == "gold_label") & (cal.method != "token_shap_llm")]
    g = pd.read_csv(res / "nested_gbm.csv") if (res / "nested_gbm.csv").exists() else None
    return n, e, c, k, cal, g

def row(ds, label, n, e, c, k, cal, g):
    n_ = n[n.dataset == ds]; e_ = e[e.dataset == ds]; c_ = c[c.dataset == ds]; k_ = k[k.dataset == ds]; cal_ = cal[cal.dataset == ds]
    dist, gold = k_[k_.model == "teacher (original)"], k_[k_.model == "gold"]
    conf = e_[e_.feature_set == "confidence"]; expl = e_[e_.feature_set == "explanation"]; both = e_[e_.feature_set == "confidence+explanation"]
    nsig = int((n_.delta_auroc_ci_low > 0).sum())
    xm = f"{int((c_.delta_auroc_ci_low > 0).sum())} / {int((c_.delta_auprc_ci_low > 0).sum())}"
    hc = cal_[cal_.regime == "confidence>0.9"]
    cells = [label, rng(dist.agreement_with_bert, 3), rng(dist.accuracy_gold, 3), rng(conf.auroc, 2), rng(expl.auroc, 2), rng(both.auroc, 2),
             f"{n_.delta_auroc.min():+.3f} to {n_.delta_auroc.max():+.3f} ({nsig})", xm,
             rng(cal_[cal_.regime == "all"].ece, 3), rng(hc.share_of_errors_in_regime, pct=True),
             f"{rng(dist.bert_errors_reproduced, pct=True)} / {rng(gold.bert_errors_reproduced, pct=True)}",
             f"{rng(dist.confident_agreeing_on_bert_errors, pct=True)} / {rng(gold.confident_agreeing_on_bert_errors, pct=True)}"]
    return cells

LABELS = ['Agreement with the teacher', 'Accuracy (gold labels)', 'AUROC, surrogate confidence', 'AUROC, explanation features', 'AUROC, both', 'Nested $\\Delta$AUROC (configurations with CI $>$ 0)', 'Cross-model gains, AUROC / AUPRC (of 114)', 'ECE (gold labels)', 'Errors with confidence $>$ 0.9 (\\%)', 'BERT errors reproduced, D / G (\\%)', 'Confident and agreeing on BERT errors, D / G (\\%)']

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--base", type=Path, required=True); ap.add_argument("--finetuned", type=Path, required=True); ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args(); B, F = load(a.base), load(a.finetuned)
    cols = [row(ds, lab, *R)[1:] for ds in D for lab, R in (("pre-trained", B), ("fine-tuned", F))]  # 4 columns x 11 metrics
    lines = [" & ".join([LABELS[k]] + [c[k] for c in cols]) + r" \\" for k in range(len(LABELS))]
    lines.append(r"\bottomrule"); a.output.write_text("\n".join(lines) + "\n"); print("\n".join(lines))

if __name__ == "__main__":
    main()
