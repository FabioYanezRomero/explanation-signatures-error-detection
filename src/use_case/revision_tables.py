#!/usr/bin/env python3
"""Assemble the revision tables (Markdown + LaTeX) from the LR summary and revision metrics."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

METHOD_NAMES = {"subgraphx": "SubgraphX", "graphsvx": "GraphSVX", "token_shap_llm": "TokenSHAP (BERT)"}
GRAPH_NAMES = {"constituency": "Constituency", "syntactic": "Syntactic", "window": "Window", "skipgrams": "Skip-gram", "tokens": "Tokens"}
DATASET_NAMES = {"setfit_ag_news": "AG News", "stanfordnlp_sst2": "SST-2"}
ORDER = ["subgraphx", "graphsvx", "token_shap_llm"]
GORDER = ["constituency", "syntactic", "window", "skipgrams", "tokens"]


def _label(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Dataset"] = df["dataset"].map(DATASET_NAMES)
    df["Explainer"] = df["method"].map(METHOD_NAMES)
    df["Graph"] = df["graph"].map(GRAPH_NAMES)
    df["_m"] = df["method"].map(ORDER.index)
    df["_g"] = df["graph"].map(GORDER.index)
    return df.sort_values(["dataset", "_m", "_g"])


def to_md(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(floatfmt.format(v) if isinstance(v, float) else str(v) for v in r.values) + " |")
    return "\n".join(out)


def to_tex(df: pd.DataFrame, caption: str, label: str, floatfmt: str = "{:.3f}") -> str:
    cols = list(df.columns)
    lines = [r"\begin{table}[H]", r"\centering", r"\small", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             r"\begin{tabular}{" + "l" * 3 + "r" * (len(cols) - 3) + "}", r"\toprule",
             " & ".join(c.replace("_", r"\_") for c in cols) + r" \\", r"\midrule"]
    for _, r in df.iterrows():
        lines.append(" & ".join(floatfmt.format(v) if isinstance(v, float) else str(v).replace("_", r"\_") for v in r.values) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-dir", type=Path, default=Path("outputs/use_case/revision_metrics"))
    ap.add_argument("--lr-summary", type=Path, default=Path("outputs/use_case/module_datasets/error_signal_classification_summary.csv"))
    ap.add_argument("--output-dir", type=Path, default=Path("outputs/use_case/revision_metrics/tables"))
    a = ap.parse_args(argv)
    a.output_dir.mkdir(parents=True, exist_ok=True)

    metrics = pd.read_csv(a.metrics_dir / "error_detection_metrics.csv")
    nested = pd.read_csv(a.metrics_dir / "nested_delta_auroc.csv")
    quad = pd.read_csv(a.metrics_dir / "quadrant_divergence.csv")
    regime = pd.read_csv(a.metrics_dir / "calibration_and_confidence_regimes.csv")
    lr = pd.read_csv(a.lr_summary)

    md, tex = [], []

    # Table A: paper protocol (LR, 4 dimensions) with baselines, both stratifications
    base = metrics[(metrics.feature_set == "majority baseline")][["dataset", "method", "graph", "target", "error_rate"]]
    paper = metrics[(metrics.feature_set == "paper (4 dimensions)")]
    pooled = lr[lr.label == "all"][["dataset", "method", "graph", "pooled_model_accuracy_mean", "accuracy_mean", "accuracy_std", "bootstrap_mean", "bootstrap_std"]]
    for target, title in [("gold_label", "errors w.r.t. the gold label"), ("teacher_agreement", "disagreement with the BERT teacher")]:
        t = paper[paper.target == target].merge(base[base.target == target][["dataset", "method", "graph", "error_rate"]], on=["dataset", "method", "graph"], suffixes=("", "_b"))
        t = _label(t)
        t["Majority acc."] = 1 - t["error_rate"]
        tab = t[["Dataset", "Explainer", "Graph", "Majority acc.", "accuracy", "balanced_accuracy", "auroc", "auprc_error", "mcc", "recall_error"]].rename(columns={
            "accuracy": "Accuracy", "balanced_accuracy": "Bal. acc.", "auroc": "AUROC", "auprc_error": "AUPRC (err)", "mcc": "MCC", "recall_error": "Recall (err)"})
        md.append(f"## Table A ({target}): stratified logistic regression on the four dimensions, {title}\n\n" + to_md(tab))
        tex.append(to_tex(tab, f"Error detection with the four-dimension logistic regression ({title}); pooled model, 10-fold CV.", f"tab:lr_{target}"))
    t = _label(pooled)
    tab = t[["Dataset", "Explainer", "Graph", "pooled_model_accuracy_mean", "accuracy_mean", "accuracy_std", "bootstrap_mean", "bootstrap_std"]].rename(columns={
        "pooled_model_accuracy_mean": "Pooled acc.", "accuracy_mean": "Per-pred-class acc.", "accuracy_std": "SD", "bootstrap_mean": "Bootstrap acc.", "bootstrap_std": "Bootstrap SD"})
    md.append("## Table A2: paper's LR script (teacher target), stratified by PREDICTED class, 200 bootstraps\n\n" + to_md(tab))
    tex.append(to_tex(tab, "Original Table 5 protocol recomputed with per-predicted-class models (teacher-agreement target).", "tab:lr_paper_protocol"))

    # Table B: nested models
    for target in ["gold_label", "teacher_agreement"]:
        t = _label(nested[nested.target == target])
        t["Delta AUROC [95% CI]"] = [f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]" for d, lo, hi in zip(t.delta_auroc, t.delta_auroc_ci_low, t.delta_auroc_ci_high)]
        tab = t[["Dataset", "Explainer", "Graph", "auroc_confidence", "auroc_explanation", "auroc_confidence_plus_explanation", "Delta AUROC [95% CI]"]].rename(columns={
            "auroc_confidence": "Confidence", "auroc_explanation": "Explanation", "auroc_confidence_plus_explanation": "Conf.+Expl."})
        md.append(f"## Table B ({target}): nested models, AUROC of error detection\n\n" + to_md(tab))
        tex.append(to_tex(tab, f"Nested models ({target}): confidence-only (MSP, top-2 margin, entropy) vs. explanation features vs. both; paired bootstrap CI for the gain.", f"tab:nested_{target}"))

    # Table C: quadrant divergence
    for target in ["gold_label", "teacher_agreement"]:
        t = _label(quad[quad.target == target])
        t["Dim."] = t["dimension"].str.extract(r"Dimension (\d)")
        piv = t.pivot_table(index=["Dataset", "Explainer", "Graph", "_m", "_g", "dataset"], columns="Dim.", values=["js_divergence_bits", "cramers_v", "legacy_separability"]).reset_index()
        piv.columns = [" ".join(c).strip() if c[1] else c[0] for c in piv.columns]
        piv = piv.sort_values(["dataset", "_m", "_g"])
        tab = piv[["Dataset", "Explainer", "Graph", "js_divergence_bits 3", "cramers_v 3", "legacy_separability 3", "js_divergence_bits 4", "cramers_v 4", "legacy_separability 4"]].rename(columns={
            "js_divergence_bits 3": "JS (D3)", "cramers_v 3": "Cramer V (D3)", "legacy_separability 3": "old Sep. (D3)", "js_divergence_bits 4": "JS (D4)", "cramers_v 4": "Cramer V (D4)", "legacy_separability 4": "old Sep. (D4)"})
        md.append(f"## Table C ({target}): quadrant association with correctness (Dimensions 3 and 4)\n\n" + to_md(tab))
        tex.append(to_tex(tab, f"Association between quadrant and correctness ({target}): Jensen-Shannon divergence (bits) and Cramer's V; the former Separability index is shown for reference.", f"tab:quadrants_{target}"))

    # Table D: calibration and high-confidence regime
    for target in ["gold_label", "teacher_agreement"]:
        t = _label(regime[regime.target == target])
        allr = t[t.regime == "all"][["Dataset", "Explainer", "Graph", "ece", "n_errors_regime"]].rename(columns={"ece": "ECE", "n_errors_regime": "Errors"})
        hi = t[t.regime == "confidence>0.9"].copy()
        hi["Delta AUROC [CI]"] = [f"{d:+.3f} [{lo:+.3f}, {h:+.3f}]" if pd.notna(d) else "n/a" for d, lo, h in zip(hi.delta_auroc, hi.delta_auroc_ci_low, hi.delta_auroc_ci_high)]
        hi["Delta AUPRC [CI]"] = [f"{d:+.3f} [{lo:+.3f}, {h:+.3f}]" if pd.notna(d) else "n/a" for d, lo, h in zip(hi.delta_auprc, hi.delta_auprc_ci_low, hi.delta_auprc_ci_high)]
        hi = hi[["Dataset", "Explainer", "Graph", "share_of_errors_in_regime", "n_errors_regime", "auroc_confidence", "auroc_confidence_plus_explanation", "Delta AUROC [CI]", "Delta AUPRC [CI]"]].rename(columns={
            "share_of_errors_in_regime": "Share of errors conf>0.9", "n_errors_regime": "Errors conf>0.9", "auroc_confidence": "AUROC conf.", "auroc_confidence_plus_explanation": "AUROC conf.+expl."})
        tab = allr.merge(hi, on=["Dataset", "Explainer", "Graph"])
        md.append(f"## Table D ({target}): calibration (ECE) and the high-confidence regime (confidence > 0.9)\n\n" + to_md(tab))
        tex.append(to_tex(tab, f"Calibration and over-confident errors ({target}).", f"tab:regime_{target}"))

    (a.output_dir / "revision_tables.md").write_text("\n\n".join(md) + "\n")
    (a.output_dir / "revision_tables.tex").write_text("\n\n".join(tex) + "\n")
    print(f"Wrote {a.output_dir/'revision_tables.md'} and .tex")


if __name__ == "__main__":
    main()
