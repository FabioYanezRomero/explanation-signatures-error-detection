#!/usr/bin/env python3
"""Emit the LaTeX tables of the revised manuscript from the final result CSVs.

Every number printed in the manuscript tables comes from these files, so text and
tables cannot diverge as long as the text is checked against the generated files.
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

M = {"subgraphx": "SubgraphX", "graphsvx": "GraphSVX", "token_shap_llm": "TokenSHAP"}
G = {"constituency": "Constituency", "syntactic": "Syntactic", "window": "Window", "skipgrams": "Skip-gram", "tokens": "Tokens"}
D = {"setfit_ag_news": "AG News", "stanfordnlp_sst2": "SST-2"}
MO = ["subgraphx", "graphsvx", "token_shap_llm"]
GO = ["constituency", "syntactic", "window", "skipgrams", "tokens"]


def order(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["_d"] = df["dataset"].map(list(D).index)
    df["_m"] = df["method"].map(MO.index)
    df["_g"] = df["graph"].map(GO.index)
    return df.sort_values(["_d", "_m", "_g"])


def f3(v) -> str:
    return "--" if pd.isna(v) else f"{v:.3f}"


def signed(v) -> str:
    return "--" if pd.isna(v) else f"{v:+.3f}"


def ci(d, lo, hi) -> str:
    if pd.isna(d):
        return "n/a"
    return rf"\({d:+.3f}\) [\({lo:+.3f}, {hi:+.3f}\)]"


def rows_by_dataset(df: pd.DataFrame, cells, ncols: int) -> str:
    """Build tabular rows with multirow dataset/explainer blocks."""
    out = []
    for dataset, block in df.groupby("dataset", sort=False):
        n_ds = len(block)
        first_ds = True
        for method, sub in block.groupby("method", sort=False):
            n_m = len(sub)
            first_m = True
            for _, r in sub.iterrows():
                dcell = rf"\multirow{{{n_ds}}}{{*}}{{{D[dataset]}}}" if first_ds else ""
                mcell = (rf"\multirow{{{n_m}}}{{*}}{{{M[method]}}}" if n_m > 1 else M[method]) if first_m else ""
                out.append(" & ".join([dcell, mcell, G[r["graph"]]] + cells(r)) + r" \\")
                first_ds = first_m = False
            out.append(rf"\cmidrule(lr){{2-{ncols}}}")
        out[-1] = r"\midrule"
    if out and out[-1] == r"\midrule":
        out.pop()
    return "\n".join(out)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-dir", type=Path, default=Path("outputs/use_case/revision_metrics"))
    ap.add_argument("--module-root", type=Path, default=Path("outputs/use_case/module_datasets"))
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--control-csv", type=Path, default=None)
    ap.add_argument("--agreement-csv", type=Path, default=None, help="control_agreement_on_bert_errors.csv: same-class column of tab:control")
    ap.add_argument("--cross-model-csv", type=Path, default=None)
    ap.add_argument("--overlap-dir", type=Path, default=None)
    a = ap.parse_args(argv)
    a.output_dir.mkdir(parents=True, exist_ok=True)

    metrics = pd.read_csv(a.metrics_dir / "error_detection_metrics.csv")
    nested = pd.read_csv(a.metrics_dir / "nested_delta_auroc.csv")
    quad = pd.read_csv(a.metrics_dir / "quadrant_divergence.csv")
    single = pd.read_csv(a.metrics_dir / "single_feature_auroc.csv")
    regime = pd.read_csv(a.metrics_dir / "calibration_and_confidence_regimes.csv")

    # ---------------- tab:error_detection (gold): four-dimension pooled detector + nested
    four = metrics[(metrics.target == "gold_label") & (metrics.feature_set == "paper (4 dimensions)")]
    maj = metrics[(metrics.target == "gold_label") & (metrics.feature_set == "majority baseline")][["dataset", "method", "graph", "accuracy"]].rename(columns={"accuracy": "majority"})
    t = four.merge(maj, on=["dataset", "method", "graph"]).merge(nested[nested.target == "gold_label"], on=["dataset", "method", "graph"], suffixes=("", "_n"))
    t = order(t)
    # the explanation-alone AUROC of the nested comparison equals the AUROC of the four-dimension detector (same features), so it is not repeated
    body = rows_by_dataset(t, lambda r: [f3(r.majority), f3(r.accuracy), f3(r.balanced_accuracy), f3(r.auroc), f3(r.auprc_error), f3(r.mcc),
                                        f3(r.auroc_confidence), f3(r.auroc_confidence_plus_explanation),
                                        ci(r.delta_auroc, r.delta_auroc_ci_low, r.delta_auroc_ci_high)], 12)
    (a.output_dir / "tab_error_detection_full.tex").write_text(body + "\n")  # supplementary: all metrics
    # main text: majority, accuracy, AUROC of the explanation alone, nested AUROCs; BERT's confidence in bold as the reference
    def _main_row(r):
        conf = f3(r.auroc_confidence)
        if r.method == "token_shap_llm":
            conf = r"\textbf{" + conf + "}"
        return [f3(r.majority), f3(r.accuracy), f3(r.auroc), conf, f3(r.auroc_confidence_plus_explanation)]  # Δ and its CI: full table
    body = rows_by_dataset(t, _main_row, 8)
    (a.output_dir / "tab_error_detection.tex").write_text(body + "\n")

    # ---------------- tab:error_detection_teacher (supplementary)
    # surrogates only: the language model has no teacher, so its rows (gold-label task) are not comparable here
    four_t = metrics[(metrics.target == "teacher_agreement") & (metrics.feature_set == "paper (4 dimensions)") & (metrics.method != "token_shap_llm")]
    maj_t = metrics[(metrics.target == "teacher_agreement") & (metrics.feature_set == "majority baseline")][["dataset", "method", "graph", "accuracy"]].rename(columns={"accuracy": "majority"})
    tt = order(four_t.merge(maj_t, on=["dataset", "method", "graph"]).merge(nested[nested.target == "teacher_agreement"], on=["dataset", "method", "graph"], suffixes=("", "_n")))
    body = rows_by_dataset(tt, lambda r: [f3(r.majority), f3(r.accuracy), f3(r.balanced_accuracy), f3(r.auroc), f3(r.auprc_error), f3(r.mcc),
                                         f3(r.auroc_confidence), f3(r.auroc_explanation), f3(r.auroc_confidence_plus_explanation),
                                         ci(r.delta_auroc, r.delta_auroc_ci_low, r.delta_auroc_ci_high)], 13)
    (a.output_dir / "tab_error_detection_teacher.tex").write_text(body + "\n")

    # ---------------- tab:quadrant_association (gold, D3 and D4 side by side, datasets as column groups)
    q = quad[quad.target == "gold_label"].copy()
    q["dim"] = q["dimension"].str.extract(r"Dimension (\d)").astype(int)
    lines = []
    for method in MO:
        graphs = [g for g in GO if ((q.method == method) & (q.graph == g)).any()]
        for i, g in enumerate(graphs):
            cells = []
            for ds in D:
                for dim in (3, 4):
                    r = q[(q.method == method) & (q.graph == g) & (q.dataset == ds) & (q.dim == dim)]
                    cells += [f3(r.js_divergence_bits.iloc[0]), f3(r.cramers_v.iloc[0])] if len(r) else ["--", "--"]
            mcell = (rf"\multirow{{{len(graphs)}}}{{*}}{{{M[method]}}}" if len(graphs) > 1 else M[method]) if i == 0 else ""
            lines.append(" & ".join([mcell, G[g]] + cells) + r" \\")
        lines.append(r"\midrule")
    lines.pop()
    (a.output_dir / "tab_quadrant_association.tex").write_text("\n".join(lines) + "\n")

    # ---------------- tab:single_feature (ranges over topologies)
    s = single[single.target == "gold_label"]
    feats = ["prediction_confidence", "auc_deletion_auc", "auc_insertion_auc", "progression_maskout_drop_k5", "progression_sufficiency_drop_k5",
             "consistency_preservation_necessity", "consistency_preservation_sufficiency"]
    lines = []
    for ds in D:
        for method in MO:
            sub = s[(s.dataset == ds) & (s.method == method)]
            cells = []
            for f in feats:
                v = sub[sub.feature == f].auroc_error_oriented
                cells.append(f3(v.iloc[0]) if len(v) == 1 else f"{v.min():.2f}--{v.max():.2f}")
            lines.append(" & ".join([D[ds], M[method]] + cells) + r" \\")
        lines.append(r"\midrule")
    lines.pop()
    (a.output_dir / "tab_single_feature.tex").write_text("\n".join(lines) + "\n")

    # ---------------- tab:single_feature_full (one row per configuration)
    feats_full = feats + ["fidelity_minus", "fidelity_plus"]  # Dimension-4 endpoints, supplementary only
    wide = s.pivot_table(index=["dataset", "method", "graph"], columns="feature", values="auroc_error_oriented").reset_index()
    wide = order(wide)
    body = rows_by_dataset(wide, lambda r: [f3(r[f]) for f in feats_full], 12)
    (a.output_dir / "tab_single_feature_full.tex").write_text(body + "\n")

    # ---------------- tab:signatures (from module datasets)
    def q3(su, ne):
        return "S-R" if su >= 0 and ne >= 0 else "S-N" if su >= 0 and ne < 0 else "I-R" if su < 0 and ne >= 0 else "I-N"
    sig = {}
    for f in glob.glob(str(a.module_root / "*/module_dataset_*.csv")):
        d = pd.read_csv(f)
        ds, method, graph = d.dataset_slug[0], d.method[0], d.graph_type[0]
        o = d.auc_origin_confidence
        suff = ((o - d.fidelity_plus) / o).clip(0, 1.5).mean()
        nec = d.fidelity_minus.mean()
        s3 = d.consistency_sufficiency_ratio.replace([np.inf, -np.inf], 0).fillna(0)
        n3 = d.consistency_necessity_ratio.replace([np.inf, -np.inf], 0).fillna(0)
        vc = pd.Series([q3(x, y) for x, y in zip(s3, n3)]).value_counts(normalize=True)
        sig[(ds, method, graph)] = (suff, nec, vc.index[0], 100 * vc.iloc[0], d.sparsity.mean())
    lines = []
    for method in MO:
        graphs = [g for g in GO if ("setfit_ag_news", method, g) in sig]
        for i, g in enumerate(graphs):
            cells = []
            for ds in D:
                su, ne, qd, pct, sp = sig[(ds, method, g)]
                cells += [f"{su:.2f}", f"{ne:.2f}", qd, f"{pct:.0f}"]
            mcell = (rf"\multirow{{{len(graphs)}}}{{*}}{{{M[method]}}}" if len(graphs) > 1 else M[method]) if i == 0 else ""
            lines.append(" & ".join([mcell, G[g]] + cells) + r" \\")
        lines.append(r"\midrule")
    lines.pop()
    (a.output_dir / "tab_signatures.tex").write_text("\n".join(lines) + "\n")
    sp = [v[4] for v in sig.values()]
    (a.output_dir / "sparsity_range.txt").write_text(f"{min(sp):.2f}--{max(sp):.2f}\n")

    # ---------------- tab:calibration (gold). Main text: AG News, no intervals, * = gain interval above zero.
    # Supplementary tab_calibration_full: both datasets with the intervals.
    full_lines = []
    for ds, name in (("setfit_ag_news", "tab_calibration.tex"), ("stanfordnlp_sst2", None)):
        r_all = regime[(regime.target == "gold_label") & (regime.regime == "all") & (regime.dataset == ds)][["method", "graph", "ece", "n_errors_regime"]]
        r_hi = regime[(regime.target == "gold_label") & (regime.regime == "confidence>0.9") & (regime.dataset == ds)]
        t = r_all.merge(r_hi, on=["method", "graph"], suffixes=("_all", ""))
        t["dataset"] = ds
        t = order(t)
        lines, dlines = [], []
        for method in MO:
            sub = t[t.method == method]
            for i, (_, r) in enumerate(sub.iterrows()):
                mcell = (rf"\multirow{{{len(sub)}}}{{*}}{{{M[method]}}}" if len(sub) > 1 else M[method]) if i == 0 else ""
                star = "*" if r.delta_auroc_ci_low > 0 else ""
                # regime detectors are reported for the surrogates only: each row detects its own model's errors within its own
                # high-confidence regime, so the language model's row is not comparable (its over-confident errors: Table S9)
                # main text: calibration only (ECE, errors, errors above 0.9); regime detectors in the full (supplementary) table
                lines.append(" & ".join([mcell, G[r.graph], f3(r.ece_all), str(int(r.n_errors_regime_all)),
                                         f"{int(r.n_errors_regime)} ({r.share_of_errors_in_regime:.2f})"]) + r" \\")
                dlines.append(" & ".join([mcell, G[r.graph], f3(r.ece_all), str(int(r.n_errors_regime_all)), f"{r.share_of_errors_in_regime:.2f}", str(int(r.n_errors_regime)),
                                          f"{f3(r.auroc_confidence)} / {f3(r.auroc_confidence_plus_explanation)}",
                                          ci(r.delta_auroc, r.delta_auroc_ci_low, r.delta_auroc_ci_high), ci(r.delta_auprc, r.delta_auprc_ci_low, r.delta_auprc_ci_high)]) + r" \\")
            lines.append(r"\midrule"); dlines.append(r"\cmidrule(lr){2-10}")  # inside a dataset block: do not cross the dataset label
        lines.pop(); dlines.pop()
        n = sum(1 for l in dlines if l.count("&") > 5); first = True
        for l in dlines:
            if l.count("&") > 5:
                l = (rf"\multirow{{{n}}}{{*}}{{{D[ds]}}} & " if first else " & ") + l; first = False
            full_lines.append(l)
        full_lines.append(r"\midrule")
    full_lines.pop()
    (a.output_dir / "tab_calibration_full.tex").write_text("\n".join(full_lines) + "\n")
    # main text (tab:calibration): both datasets, one row per surrogate (shared by the two explainers) plus BERT;
    # ECE, errors and errors above confidence 0.9 with respect to the teacher's labels and to the gold labels
    rg = regime[regime.method != "subgraphx"]
    lines = []
    for ds in D:
        for i, g in enumerate(["constituency", "syntactic", "window", "skipgrams", "tokens"]):
            sub = rg[(rg.dataset == ds) & (rg.graph == g)]
            def get(t, reg, col):
                r = sub[(sub.target == t) & (sub.regime == reg)]
                return r[col].iloc[0]
            dcell = rf"\multirow{{5}}{{*}}{{{D[ds]}}}" if i == 0 else ""
            n_test = int(get("gold_label", "all", "n"))
            def trio(t):  # ECE, count (share of the test set), count above 0.9 (share of the count)
                e = int(get(t, "all", "n_errors_regime")); hi = int(get(t, "confidence>0.9", "n_errors_regime"))
                return [f"{get(t, 'all', 'ece'):.3f}", f"{e} ({100 * e / n_test:.1f}\\%)", f"{hi} ({100 * hi / e:.0f}\\%)"]
            tcells = ["--", "--", "--"] if g == "tokens" else trio("teacher_agreement")
            gcells = trio("gold_label")
            lines.append(" & ".join([dcell, "BERT (teacher)" if g == "tokens" else G[g]] + tcells + gcells) + r" \\")
        lines.append(r"\midrule")
    lines.pop()
    (a.output_dir / "tab_calibration.tex").write_text("\n".join(lines) + "\n")


    # ---------------- tab:control (gold-trained control, main run only)
    ctrl = pd.read_csv(a.control_csv) if a.control_csv else None
    if ctrl is not None:
        ctrl = ctrl[~ctrl.model.str.contains(r"\[")]
        agree = pd.read_csv(a.agreement_csv) if a.agreement_csv else None
        lines, main_lines = [], []
        for ds in D:
            graphs = [g for g in GO if ((ctrl.dataset == ds) & (ctrl.graph == g)).any()]
            for i, g in enumerate(graphs):
                d0 = ctrl[(ctrl.dataset == ds) & (ctrl.graph == g) & (ctrl.model == "teacher (original)")].iloc[0]
                d1 = ctrl[(ctrl.dataset == ds) & (ctrl.graph == g) & (ctrl.model == "gold")].iloc[0]
                dcell = rf"\multirow{{{len(graphs)}}}{{*}}{{{D[ds]}}}" if i == 0 else ""
                lines.append(" & ".join([dcell, G[g], f"{d0.accuracy_gold:.3f} / {d1.accuracy_gold:.3f}",
                                         f"{d0.bert_errors_reproduced:.2f} / {d1.bert_errors_reproduced:.2f}",
                                         ci(d1.diff_vs_original_bert_errors_reproduced, d1.diff_vs_original_bert_errors_reproduced_ci_low, d1.diff_vs_original_bert_errors_reproduced_ci_high),
                                         f"{d0.confident_agreeing_on_bert_errors:.2f} / {d1.confident_agreeing_on_bert_errors:.2f}",
                                         ci(d1.diff_vs_original_confident_agreeing_on_bert_errors, d1.diff_vs_original_confident_agreeing_on_bert_errors_ci_low, d1.diff_vs_original_confident_agreeing_on_bert_errors_ci_high)]) + r" \\")
                # main text: no intervals, * = interval of G minus D excludes zero
                s1 = "*" if (d1.diff_vs_original_bert_errors_reproduced_ci_low > 0 or d1.diff_vs_original_bert_errors_reproduced_ci_high < 0) else ""
                s2 = "*" if (d1.diff_vs_original_confident_agreeing_on_bert_errors_ci_low > 0 or d1.diff_vs_original_confident_agreeing_on_bert_errors_ci_high < 0) else ""
                if agree is not None:  # same wrong class as BERT on its errors (no interval; both surrogates agree alike)
                    a0 = agree[(agree.dataset == ds) & (agree.graph == g) & (agree.model == "distilled")].iloc[0]
                    a1 = agree[(agree.dataset == ds) & (agree.graph == g) & (agree.model == "gold-trained")].iloc[0]
                    mid = f"{a0.same_class_as_bert:.2f} / {a1.same_class_as_bert:.2f}"
                else:
                    mid = f"{d0.bert_errors_reproduced:.2f} / {d1.bert_errors_reproduced:.2f}" + s1
                main_lines.append(" & ".join([dcell, G[g], f"{d0.accuracy_gold:.3f} / {d1.accuracy_gold:.3f}", mid,
                                              f"{d0.confident_agreeing_on_bert_errors:.2f} / {d1.confident_agreeing_on_bert_errors:.2f}" + s2]) + r" \\")
            lines.append(r"\midrule"); main_lines.append(r"\midrule")
        lines.pop(); main_lines.pop()
        (a.output_dir / "tab_control_full.tex").write_text("\n".join(lines) + "\n")
        (a.output_dir / "tab_control.tex").write_text("\n".join(main_lines) + "\n")

    # ---------------- tab:cross_model (supplementary): best gain per addition family, all instances and confidence>0.95
    if a.cross_model_csv:
        x = pd.read_csv(a.cross_model_csv)
        fam = {"bert_explanation": "BERT explanation (TokenSHAP)", "gnn_confidence": "Surrogate confidence", "gnn_disagreement": "Surrogate disagreement",
               "gnn_confidence+explanation": "Surrogate confidence + explanation", "committee:confidence+disagreement(4 topologies)": "Committee: confidence + disagreement",
               "committee:confidence+explanation(4 topologies)": "Committee: confidence + explanation"}
        x["family"] = x.addition.str.replace(r"^[a-z]+:", "", regex=True)
        x = x[x.family.isin(fam)]
        lines = []
        for ds in D:
            first = True
            for regime in ["all", "bert_confidence>0.95"]:
                sub = x[(x.dataset == ds) & (x.regime == regime)]
                for key, name in fam.items():
                    r = sub[sub.family == key]
                    if r.empty:
                        continue
                    r = r.loc[r.delta_auroc.idxmax()]
                    lines.append(" & ".join([D[ds] if first else "", "all" if regime == "all" else r"conf.\ \(>0.95\)", name, f3(r.auroc_base), f3(r.auroc_full),
                                             ci(r.delta_auroc, r.delta_auroc_ci_low, r.delta_auroc_ci_high), ci(r.delta_auprc, r.delta_auprc_ci_low, r.delta_auprc_ci_high)]) + r" \\")
                    first = False
            lines.append(r"\midrule")
        lines.pop()
        (a.output_dir / "tab_cross_model.tex").write_text("\n".join(lines) + "\n")

    # ---------------- tab:overlap (supplementary)
    if a.overlap_dir:
        pi = pd.read_csv(a.overlap_dir / "explanation_overlap_per_instance.csv")
        de = pd.read_csv(a.overlap_dir / "explanation_overlap_descriptive.csv")
        ne = pd.read_csv(a.overlap_dir / "explanation_overlap_nested.csv")
        lines = []
        for ds in D:
            for method in ["graphsvx", "subgraphx"]:
                graphs = [g for g in GO if ((pi.dataset == ds) & (pi.method == method) & (pi.graph == g)).any()]
                for i, g in enumerate(graphs):
                    s_ = pi[(pi.dataset == ds) & (pi.method == method) & (pi.graph == g)]
                    au = de[(de.dataset == ds) & (de.method == method) & (de.graph == g) & (de.target == "bert_correct") & (de.feature == "jaccard_k20")].auroc_error_oriented.iloc[0]
                    nb = ne[(ne.dataset == ds) & (ne.method == method) & (ne.graph == g) & (ne.target == "bert_correct") & (ne.regime == "all")].iloc[0]
                    ng = ne[(ne.dataset == ds) & (ne.method == method) & (ne.graph == g) & (ne.target == "gnn_correct_gold") & (ne.regime == "all")].iloc[0]
                    lines.append(" & ".join([D[ds] if (i == 0 and method == "graphsvx") else "", M[method] if i == 0 else "", G[g], f"{s_.jaccard_k20.mean():.3f}", f"{s_.spearman.mean():.3f}", f"{s_.top1_match.mean():.2f}",
                                             f3(au), ci(nb.delta_auroc, nb.delta_auroc_ci_low, nb.delta_auroc_ci_high), ci(ng.delta_auroc, ng.delta_auroc_ci_low, ng.delta_auroc_ci_high)]) + r" \\")
            lines.append(r"\midrule")
        lines.pop()
        (a.output_dir / "tab_overlap.tex").write_text("\n".join(lines) + "\n")

    print("tables written to", a.output_dir)


if __name__ == "__main__":
    main()
