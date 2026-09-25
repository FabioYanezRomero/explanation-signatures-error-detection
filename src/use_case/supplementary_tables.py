"""Supplementary tables not produced by manuscript_tables.py (fine-tuned run).

S4/S5  supp_coefficients_{agnews,sst2}.tex : standardised coefficients of the explanation features in the
        combined (confidence + explanation) detector, one column per configuration, * = 95% interval excludes zero.
S7     supp_calibration_teacher.tex : calibration and high-confidence regime with respect to the teacher's labels.
S8     supp_gbm.tex : gradient-boosting nested detector (gold label).
S9b    supp_second_opinion.tex : surrogate as a second opinion on BERT (disagreements).
S11    supp_seed_replicates.tex : seed replicates of the distilled and gold-trained surrogates.

Usage::

    python src/use_case/supplementary_tables.py --results revision/results_ft --output-dir revision/manuscript_v2/tables_ft
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

D = {"setfit_ag_news": "AG News", "stanfordnlp_sst2": "SST-2"}
M = {"subgraphx": "SubgraphX", "graphsvx": "GraphSVX", "token_shap_llm": "TokenSHAP"}
G = {"constituency": "Constituency", "syntactic": "Syntactic", "window": "Window", "skipgrams": "Skip-gram", "tokens": "Tokens"}
GO = ["constituency", "syntactic", "window", "skipgrams", "tokens"]
FEAT = [  # (feature, label, dimension)
    ("auc_deletion_auc", "Deletion AUC", 1), ("auc_insertion_auc", "Insertion AUC", 1),
    ("progression_maskout_drop_k1", r"Mask-out drop \(k{=}1\)", 2), ("progression_maskout_drop_k3", r"Mask-out drop \(k{=}3\)", 2),
    ("progression_maskout_drop_k5", r"Mask-out drop \(k{=}5\)", 2), ("progression_maskout_drop_k10", r"Mask-out drop \(k{=}10\)", 2),
    ("progression_sufficiency_drop_k1", r"Sufficiency drop \(k{=}1\)", 2), ("progression_sufficiency_drop_k3", r"Sufficiency drop \(k{=}3\)", 2),
    ("progression_sufficiency_drop_k5", r"Sufficiency drop \(k{=}5\)", 2), ("progression_sufficiency_drop_k10", r"Sufficiency drop \(k{=}10\)", 2),
    ("progression_concentration_top3", "Concentration top-3", 2), ("progression_concentration_top5", "Concentration top-5", 2), ("progression_concentration_top10", "Concentration top-10", 2),
    ("consistency_preservation_necessity", r"Margin after removal \(m(x_{\setminus S})\)", 3), ("consistency_preservation_sufficiency", r"Margin of the explanation \(m(x_S)\)", 3),
    ("fidelity_plus", r"Sufficiency \(F^{+}\)", 4), ("fidelity_minus", r"Necessity \(F^{-}\)", 4), ("fidelity_asymmetry", r"Asymmetry \(A\)", 4),
]


def f3(x):
    return "--" if pd.isna(x) else f"{x:.3f}"


def ci(d, lo, hi):
    return "--" if pd.isna(d) else rf"\({d:+.3f}\) [\({lo:+.3f}, {hi:+.3f}\)]"


def write(path, lines):  # every file is a complete tabular body: rows ... \bottomrule (LaTeX rejects \midrule/\bottomrule right after \input)
    path.write_text("\n".join(lines + [r"\bottomrule"]) + "\n")


def coefficients(res: Path, out: Path):
    d = pd.read_csv(res / "nested_coefficients.csv")
    d = d[d.block != "confidence"]
    for ds, fn in (("setfit_ag_news", "supp_coefficients_agnews.tex"), ("stanfordnlp_sst2", "supp_coefficients_sst2.tex")):
        sub = d[d.dataset == ds]
        cols = [(m, g) for m in ["subgraphx", "graphsvx"] for g in GO[:4]] + [("token_shap_llm", "tokens")]
        cols = [(m, g) for m, g in cols if ((sub.method == m) & (sub.graph == g)).any()]
        lines, last_dim = [], None
        for f, lab, dim in FEAT:
            if dim != last_dim:
                if last_dim is not None:
                    lines.append(r"\addlinespace")
                lines.append(rf"\emph{{Dimension {dim}}}" + " &" * len(cols) + r" \\")
                last_dim = dim
            cells = []
            for m, g in cols:
                r = sub[(sub.method == m) & (sub.graph == g) & (sub.feature == f)]
                if r.empty:
                    cells.append("--")
                else:
                    r = r.iloc[0]
                    cells.append(f"{r.coef:+.2f}" + ("*" if r.significant_95 else ""))
            lines.append(" & ".join([r"\quad " + lab] + cells) + r" \\")
        header = " & ".join([""] + [f"{M[m]} {G[g]}" if m != "token_shap_llm" else "TokenSHAP" for m, g in cols]) + r" \\"
        write(out / fn, [header, r"\midrule"] + lines)


def calibration_teacher(res: Path, out: Path):
    cal = pd.read_csv(res / "calibration_and_confidence_regimes.csv")
    cal = cal[(cal.target == "teacher_agreement") & (cal.method != "token_shap_llm")]
    lines = []
    for ds in D:
        sub = cal[cal.dataset == ds]
        first = True
        for m in ["subgraphx", "graphsvx"]:
            for i, g in enumerate(GO[:4]):
                a = sub[(sub.method == m) & (sub.graph == g) & (sub.regime == "all")]
                h = sub[(sub.method == m) & (sub.graph == g) & (sub.regime == "confidence>0.9")]
                if a.empty or h.empty:
                    continue
                a, h = a.iloc[0], h.iloc[0]
                lines.append(" & ".join([D[ds] if first else "", M[m] if i == 0 else "", G[g], f3(a.ece), str(int(a.n_errors_regime)),
                                         f"{int(h.n_errors_regime)} ({h.share_of_errors_in_regime:.2f})", f"{f3(h.auroc_confidence)} / {f3(h.auroc_confidence_plus_explanation)}",
                                         ci(h.delta_auroc, h.delta_auroc_ci_low, h.delta_auroc_ci_high)]) + r" \\")
                first = False
        lines.append(r"\midrule")
    lines.pop()
    write(out / "supp_calibration_teacher.tex", lines)


def gbm(res: Path, out: Path):
    g = pd.read_csv(res / "nested_gbm.csv")
    lines = []
    for ds in D:
        first = True
        for m in ["subgraphx", "graphsvx", "token_shap_llm"]:
            gs = [x for x in GO if ((g.dataset == ds) & (g.method == m) & (g.graph == x)).any()]
            for i, x in enumerate(gs):
                r = g[(g.dataset == ds) & (g.method == m) & (g.graph == x)].iloc[0]
                lines.append(" & ".join([D[ds] if first else "", M[m] if i == 0 else "", G[x], f3(r.auroc_confidence), f3(r.auroc_confidence_plus_explanation),
                                         ci(r.delta_auroc, r.delta_auroc_ci_low, r.delta_auroc_ci_high), ci(r.delta_auprc, r.delta_auprc_ci_low, r.delta_auprc_ci_high)]) + r" \\")
                first = False
        lines.append(r"\midrule")
    lines.pop()
    write(out / "supp_gbm.tex", lines)


def second_opinion(res: Path, out: Path):
    s = pd.read_csv(res / "surrogate_gap_vs_bert.csv")
    lines = []
    for ds in D:
        sub = s[s.dataset == ds]
        for i, g in enumerate(GO[:4]):
            r = sub[sub.graph == g]
            if r.empty:
                continue
            r = r.iloc[0]
            n_dis = int(round(r.disagreement_rate * r.n))
            lines.append(" & ".join([D[ds] if i == 0 else "", G[g], f"{100 * r.bert_accuracy:.1f}", f"{100 * r.surrogate_accuracy:.1f}",
                                     f"{n_dis} ({100 * r.disagreement_rate:.1f}\\%)", f"{100 * r.bert_accuracy_on_disagreements:.0f}", f"{100 * r.surrogate_accuracy_on_disagreements:.0f}",
                                     f"{r.bert_accuracy_on_disagreements / max(r.surrogate_accuracy_on_disagreements, 1e-9):.1f}"]) + r" \\")
        lines.append(r"\midrule")
    lines.pop()
    write(out / "supp_second_opinion.tex", lines)


def seed_replicates(res: Path, out: Path):
    k = pd.read_csv(res / "control_error_overlap.csv")
    names = {"teacher (original)": "distilled", "teacher (replicate)": "distilled, seed replicate", "gold": "gold-trained", "gold [gnn_models_control_seed1]": "gold-trained, seed replicate"}
    lines = []
    for ds in D:
        for i, g in enumerate(GO[:4]):
            sub = k[(k.dataset == ds) & (k.graph == g)]
            if sub.empty:
                continue
            for j, (key, name) in enumerate(names.items()):
                r = sub[sub.model == key]
                if r.empty:
                    continue
                r = r.iloc[0]
                lines.append(" & ".join([D[ds] if (i == 0 and j == 0) else "", G[g] if j == 0 else "", name, f3(r.accuracy_gold), f"{r.agreement_with_bert:.3f}",
                                         f"{r.bert_errors_reproduced:.2f}", f"{r.jaccard_errors:.2f}", f"{r.confident_agreeing_on_bert_errors:.2f}"]) + r" \\")
            lines.append(r"\addlinespace")
        lines[-1] = r"\midrule"
    lines.pop()
    write(out / "supp_seed_replicates.tex", lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    a = ap.parse_args(argv)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    coefficients(a.results, a.output_dir)
    calibration_teacher(a.results, a.output_dir)
    gbm(a.results, a.output_dir)
    second_opinion(a.results, a.output_dir)
    seed_replicates(a.results, a.output_dir)
    print("written to", a.output_dir)


if __name__ == "__main__":
    main()
