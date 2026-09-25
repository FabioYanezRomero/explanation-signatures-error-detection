"""Table 6 (main text): node-feature ablation in four blocks, one column per dataset x feature set.

Blocks: surrogate quality (agreement with the teacher, gold accuracy: median over the four
topologies), calibration (ECE median; errors with confidence > 0.9, range over the eight
explainer x topology configurations), error detection (AUROC of the confidence-only and of
the confidence + explanation detector, ranges; number of the eight configurations whose
nested gain has a 95% interval above zero) and behaviour on BERT's errors (same wrong class
as BERT, and with confidence > 0.9; distilled / gold-trained, ranges over the four topologies).

Usage::

    python src/use_case/ablation_table_blocks.py --base revision/results --ft revision/results_ft \
        --output revision/manuscript_v2/tables_ft/tab_ablation.tex
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DS = ["setfit_ag_news", "stanfordnlp_sst2"]


def rng(s, nd=2, pct=False):
    lo, hi = s.min(), s.max()
    if pct:
        lo, hi = 100 * lo, 100 * hi
        return f"{lo:.0f}--{hi:.0f}" if round(lo) != round(hi) else f"{lo:.0f}"
    return f"{lo:.{nd}f}--{hi:.{nd}f}" if f"{lo:.{nd}f}" != f"{hi:.{nd}f}" else f"{lo:.{nd}f}"


def load(res: Path):
    n = pd.read_csv(res / "nested_delta_auroc.csv"); n = n[(n.target == "gold_label") & (n.method != "token_shap_llm")]
    k = pd.read_csv(res / "control_error_overlap.csv"); k = k[k.model.isin(["teacher (original)", "gold"])]
    cal = pd.read_csv(res / "calibration_and_confidence_regimes.csv"); cal = cal[(cal.target == "gold_label") & (cal.method != "token_shap_llm")]
    ag = pd.read_csv(res / "control_agreement_on_bert_errors.csv")
    return n, k, cal, ag


def column(ds, n, k, cal, ag):
    n_, k_, cal_, ag_ = n[n.dataset == ds], k[k.dataset == ds], cal[cal.dataset == ds], ag[ag.dataset == ds]
    dist, gold = k_[k_.model == "teacher (original)"], k_[k_.model == "gold"]
    ad, ag_g = ag_[ag_.model == "distilled"], ag_[ag_.model == "gold-trained"]
    hc = cal_[cal_.regime == "confidence>0.9"]
    ece = cal_[(cal_.regime == "all") & (cal_.method == "graphsvx")].ece  # one surrogate per topology
    return [f"{dist.agreement_with_bert.median():.2f}", f"{dist.accuracy_gold.median():.2f}",
            f"{ece.median():.3f}", rng(hc.share_of_errors_in_regime, pct=True) + r"\%",
            rng(n_.auroc_confidence, 2), rng(n_.auroc_confidence_plus_explanation, 2), str(int((n_.delta_auroc_ci_low > 0).sum())),
            f"{rng(ad.same_class_as_bert, pct=True)} / {rng(ag_g.same_class_as_bert, pct=True)}",
            f"{rng(ad.same_class_confident, pct=True)} / {rng(ag_g.same_class_confident, pct=True)}"]


BLOCKS = [("Surrogate quality", ["Agreement with the teacher", "Accuracy (gold labels)"]),
          ("Calibration (gold labels)", ["ECE", r"Errors with confidence \(>0.9\)"]),
          ("Error detection (AUROC)", ["Surrogate confidence", "Confidence + explanation", "Configurations with a significant gain (of 8)"]),
          ("On BERT's errors (distilled / gold-trained)", ["Same wrong class as BERT", r"\ldots and confident (\(>0.9\))"])]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True, help="results of the pre-trained-feature run")
    ap.add_argument("--ft", type=Path, required=True, help="results of the fine-tuned-feature run")
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args(argv)
    cols = []
    for ds in DS:
        for res in (a.base, a.ft):
            cols.append(column(ds, *load(res)))
    lines, i = [], 0
    for title, labels in BLOCKS:
        lines.append(rf"\emph{{{title}}} & & & & \\")  # plain cell: \multicolumn at row start breaks under \input
        for lab in labels:
            lines.append(" & ".join([r"\quad " + lab] + [c[i] for c in cols]) + r" \\")
            i += 1
        lines.append(r"\addlinespace")
    lines.pop()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
