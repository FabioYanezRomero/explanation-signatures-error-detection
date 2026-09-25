#!/usr/bin/env python3
"""Regenerate Figures 3-5 of the manuscript (matplotlib) from the module datasets.

Figure 3 / 4: error and correctness detection rates when thresholding the insertion /
deletion AUC, averaged over topologies per explainer (one panel per dataset).
Figure 5a / 5b: mean cumulative mask-out / sufficiency drop at k in {1,3,5,10}, by
explainer and correctness (one panel per dataset).
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

M = {"subgraphx": "SubgraphX", "graphsvx": "GraphSVX", "token_shap_llm": "TokenSHAP (BERT)"}
D = {"setfit_ag_news": "AG News", "stanfordnlp_sst2": "SST-2"}
COLORS = {"subgraphx": "#1b6ca8", "graphsvx": "#d1495b", "token_shap_llm": "#3c8d40"}
THRESHOLDS = np.round(np.arange(0.1, 1.01, 0.1), 1)


def load(module_root: Path) -> pd.DataFrame:
    frames = []
    for f in glob.glob(str(module_root / "*/module_dataset_*.csv")):
        d = pd.read_csv(f, usecols=lambda c: c in {"dataset_slug", "method", "graph_type", "is_correct", "label_id", "prediction_class",
                                                      "auc_deletion_auc", "auc_insertion_auc", "progression_maskout_progression_drop",
                                                      "progression_sufficiency_progression_drop"} or c.startswith("progression_maskout_drop_k") or c.startswith("progression_sufficiency_drop_k"))
        d["correct_gold"] = d.prediction_class == d.label_id
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def detection_rates(df: pd.DataFrame, col: str, direction: str):
    """Rate of errors flagged and of correct predictions retained at each threshold.

    direction 'below': flag as error when metric < threshold (deletion AUC low = necessary
    features were removed first; insertion AUC low = weak reconstruction)."""
    err, ok = [], []
    for thr in THRESHOLDS:
        flag = df[col] < thr if direction == "below" else df[col] >= thr
        e = df[~df.correct_gold]
        c = df[df.correct_gold]
        err.append((e[col] < thr).mean() if direction == "below" else (e[col] >= thr).mean())
        ok.append((c[col] >= thr).mean() if direction == "below" else (c[col] < thr).mean())
    return np.array(err), np.array(ok)


def figure_auc(df: pd.DataFrame, col: str, title: str, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, ds in zip(axes, D):
        for method in M:
            sub = df[(df.dataset_slug == ds) & (df.method == method)]
            if sub.empty:
                continue
            err, ok = detection_rates(sub, col, "below")
            ax.plot(THRESHOLDS, err, marker="o", color=COLORS[method], label=f"{M[method]}: errors flagged")
            ax.plot(THRESHOLDS, ok, marker="s", linestyle="--", color=COLORS[method], label=f"{M[method]}: correct retained")
        ax.set_title(D[ds]); ax.set_xlabel(f"{title} threshold"); ax.set_ylim(0, 1.02); ax.grid(alpha=0.3)
    axes[0].set_ylabel("rate")
    axes[1].legend(fontsize=7, loc="center right")
    fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)


def figure_thresholds_combined(df: pd.DataFrame, out: Path) -> None:
    """Main-text Figure 3: insertion (top row) and deletion (bottom row) AUC thresholds,
    one column per dataset, one shared legend."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 5.0), sharey=True)
    for row, (col, title) in enumerate((("auc_insertion_auc", "insertion AUC"), ("auc_deletion_auc", "deletion AUC"))):
        for ax, ds in zip(axes[row], D):
            for method in M:
                sub = df[(df.dataset_slug == ds) & (df.method == method)]
                if sub.empty:
                    continue
                err, ok = detection_rates(sub, col, "below")
                ax.plot(THRESHOLDS, err, marker="o", color=COLORS[method], label=f"{M[method]}: errors flagged")
                ax.plot(THRESHOLDS, ok, marker="s", linestyle="--", color=COLORS[method], label=f"{M[method]}: correct retained")
            if row == 0:
                ax.set_title(D[ds])
            ax.set_xlabel(f"{title} threshold"); ax.set_ylim(0, 1.02); ax.grid(alpha=0.3)
        axes[row][0].set_ylabel("rate")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.06, 1, 1)); fig.savefig(out, dpi=200); plt.close(fig)


def figure_progression(df: pd.DataFrame, prefix: str, ylabel: str, out: Path) -> None:
    ks = [1, 3, 5, 10]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, ds in zip(axes, D):
        for method in M:
            sub = df[(df.dataset_slug == ds) & (df.method == method)]
            if sub.empty:
                continue
            for correct, ls, mk in ((True, "-", "o"), (False, "--", "x")):
                s = sub[sub.correct_gold == correct]
                ys = [np.nanmean(np.clip(s[f"{prefix}_k{k}"], -1, 1)) for k in ks]
                ax.plot(ks, ys, linestyle=ls, marker=mk, color=COLORS[method], label=f"{M[method]}: {'correct' if correct else 'incorrect'}")
        ax.set_title(D[ds]); ax.set_xlabel("top-k elements"); ax.set_xticks(ks); ax.grid(alpha=0.3)
    axes[0].set_ylabel(ylabel)
    axes[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)


def figure_signatures(module_root: Path, out: Path) -> None:
    """Main-text figure for Dimension 4: mean sufficiency F+ against mean necessity F-,
    one point per explainer, graph type and dataset (same statistics as the signatures table)."""
    G = {"constituency": ("Constituency", "o"), "syntactic": ("Syntactic", "s"), "window": ("Window", "^"), "skipgrams": ("Skip-gram", "D"), "tokens": ("Tokens", "*")}
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.1), sharey=True)
    for ax, ds in zip(axes, D):
        for f in sorted(glob.glob(str(module_root / ds / "module_dataset_*.csv"))):
            d = pd.read_csv(f, usecols=["method", "graph_type", "prediction_confidence", "fidelity_plus", "fidelity_minus"]).dropna()
            method = d.method.iloc[0]; graph = d.graph_type.iloc[0]
            o = d.prediction_confidence
            fplus = ((o - d.fidelity_plus) / o).clip(0, 1.5).mean()
            fminus = d.fidelity_minus.mean()
            label, mk = G.get(graph, (graph, "o"))
            ax.scatter(fplus, fminus, color=COLORS[method], marker=mk, s=90 if mk == "*" else 55, edgecolor="black", linewidth=0.5, zorder=3)
        ax.set_title(D[ds]); ax.set_xlabel(r"sufficiency $F^{+}$ (confidence kept by the explanation alone)")
        ax.set_xlim(0.4, 1.02); ax.set_ylim(-0.01, 0.16); ax.grid(alpha=0.3)
        # inset: the same points on the full [0,1] x [0,1] scale with the four quadrants
        ins = ax.inset_axes([0.06, 0.50, 0.30, 0.44])
        for coll in ax.collections:
            off = coll.get_offsets()
            ins.scatter(off[:, 0], off[:, 1], color=coll.get_facecolor(), marker=coll.get_paths()[0], s=14, linewidth=0)
        ins.set_xlim(0, 1); ins.set_ylim(0, 1)
        ins.axhline(0.5, color="grey", linewidth=0.6); ins.axvline(0.5, color="grey", linewidth=0.6)
        ins.set_xticks([0, 0.5, 1]); ins.set_yticks([0, 0.5, 1]); ins.tick_params(labelsize=6)
        ins.text(0.75, 0.75, "sufficient\nnecessary", ha="center", va="center", fontsize=5.5, color="grey")
        ins.text(0.25, 0.75, "insufficient\nnecessary", ha="center", va="center", fontsize=5.5, color="grey")
        ins.text(0.75, 0.25, "sufficient\nnot necessary", ha="center", va="center", fontsize=5.5, color="grey")
        ins.text(0.25, 0.25, "insufficient\nnot necessary", ha="center", va="center", fontsize=5.5, color="grey")
    axes[0].set_ylabel(r"necessity $F^{-}$ (confidence lost without it)")
    from matplotlib.lines import Line2D
    h1 = [Line2D([], [], color=COLORS[m], marker="o", linestyle="", label=M[m]) for m in M]
    h2 = [Line2D([], [], color="grey", marker=mk, linestyle="", label=lab) for lab, mk in G.values()]
    fig.legend(handles=h1 + h2, loc="lower center", ncol=8, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig(out, dpi=200); plt.close(fig)


def figure_signatures_quadrants(module_root: Path, out: Path) -> None:
    """Alternative Dimension-4 figure: the four sufficiency--necessity quadrants on the full scale."""
    G = {"constituency": ("Constituency", "o"), "syntactic": ("Syntactic", "s"), "window": ("Window", "^"), "skipgrams": ("Skip-gram", "D"), "tokens": ("Tokens", "*")}
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    for ax, ds in zip(axes, D):
        for f in sorted(glob.glob(str(module_root / ds / "module_dataset_*.csv"))):
            d = pd.read_csv(f, usecols=["method", "graph_type", "prediction_confidence", "fidelity_plus", "fidelity_minus"]).dropna()
            method = d.method.iloc[0]; graph = d.graph_type.iloc[0]
            o = d.prediction_confidence
            fplus = ((o - d.fidelity_plus) / o).clip(0, 1.5).mean(); fminus = d.fidelity_minus.mean()
            label, mk = G.get(graph, (graph, "o"))
            ax.scatter(fplus, fminus, color=COLORS[method], marker=mk, s=110 if mk == "*" else 70, edgecolor="black", linewidth=0.5, zorder=3)
        ax.axhline(0.5, color="grey", linewidth=0.8); ax.axvline(0.5, color="grey", linewidth=0.8)
        for x, y, txt in ((0.75, 0.75, "sufficient\nnecessary"), (0.25, 0.75, "insufficient\nnecessary"), (0.75, 0.30, "sufficient\nnot necessary"), (0.25, 0.30, "insufficient\nnot necessary")):
            ax.text(x, y, txt, ha="center", va="center", fontsize=9, color="grey")
        ax.set_title(D[ds]); ax.set_xlabel(r"sufficiency $F^{+}$"); ax.set_xlim(0, 1.02); ax.set_ylim(-0.02, 1.0); ax.grid(alpha=0.3)
    axes[0].set_ylabel(r"necessity $F^{-}$")
    from matplotlib.lines import Line2D
    h1 = [Line2D([], [], color=COLORS[m], marker="o", linestyle="", label=M[m]) for m in M]
    h2 = [Line2D([], [], color="grey", marker=mk, linestyle="", label=lab) for lab, mk in G.values()]
    fig.legend(handles=h1 + h2, loc="lower center", ncol=8, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(out, dpi=200); plt.close(fig)


def figure_margin_flips(module_root: Path, out: Path) -> None:
    """Dimension 3 read as decisions: share of predictions whose decision flips when the
    explanation is removed (margin after removal < 0), correct vs incorrect, per configuration."""
    G = {"constituency": "Const.", "syntactic": "Synt.", "window": "Window", "skipgrams": "Skip-g.", "tokens": "Tokens"}
    order = [("subgraphx", g) for g in ["constituency", "syntactic", "window", "skipgrams"]] + \
            [("graphsvx", g) for g in ["constituency", "syntactic", "window", "skipgrams"]] + [("token_shap_llm", "tokens")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), sharey=True)
    for ax, ds in zip(axes, D):
        vals = {}
        for f in glob.glob(str(module_root / ds / "module_dataset_*.csv")):
            d = pd.read_csv(f, usecols=["method", "graph_type", "label_id", "prediction_class", "consistency_preservation_necessity"]).dropna()
            err = d.prediction_class != d.label_id
            flip = d.consistency_preservation_necessity < 0
            vals[(d.method.iloc[0], d.graph_type.iloc[0])] = (100 * flip[~err].mean(), 100 * flip[err].mean())
        xs = np.arange(len(order)); w = 0.38
        for i, key in enumerate(order):
            c_ok, c_err = vals.get(key, (np.nan, np.nan))
            ax.bar(xs[i] - w / 2, c_ok, w, color=COLORS[key[0]], alpha=0.45, edgecolor="black", linewidth=0.4)
            ax.bar(xs[i] + w / 2, c_err, w, color=COLORS[key[0]], edgecolor="black", linewidth=0.4)
        ax.set_xticks(xs); ax.set_xticklabels([G[g] for _, g in order], fontsize=8)
        ax.axvline(3.5, color="grey", linewidth=0.6, linestyle=":"); ax.axvline(7.5, color="grey", linewidth=0.6, linestyle=":")
        ax.set_ylim(0, 38)
        for x, lab in ((1.5, "SubgraphX"), (5.5, "GraphSVX"), (8, "TokenSHAP")):
            ax.text(x, 37, lab, ha="center", va="top", fontsize=8, color="dimgrey")
        ax.set_title(D[ds]); ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("decision flips when the\nexplanation is removed (%)")
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(facecolor="grey", alpha=0.45, edgecolor="black", label="correct predictions"), Patch(facecolor="grey", edgecolor="black", label="errors")],
               loc="lower center", ncol=2, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(out, dpi=200); plt.close(fig)


def figure_error_detection(nested_csv: Path, out: Path) -> None:
    """Figure 5: AUROC of the error detector trained on the explanation features, on the
    model's confidence, and on both, per configuration; bootstrap interval of the gain
    of adding the explanation; BERT's confidence as a horizontal reference."""
    G = {"constituency": "Const.", "syntactic": "Synt.", "window": "Window", "skipgrams": "Skip-g.", "tokens": "Tokens"}
    order = [("subgraphx", g) for g in ["constituency", "syntactic", "window", "skipgrams"]] + \
            [("graphsvx", g) for g in ["constituency", "syntactic", "window", "skipgrams"]] + [("token_shap_llm", "tokens")]
    n = pd.read_csv(nested_csv); n = n[n.target == "gold_label"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    for ax, ds in zip(axes, D):
        sub = n[n.dataset == ds].set_index(["method", "graph"])
        bert = sub.loc[("token_shap_llm", "tokens"), "auroc_confidence"]
        ax.axhline(bert, color=COLORS["token_shap_llm"], linestyle="--", linewidth=1.0, zorder=1)
        ax.text(-0.5, bert + 0.003, "BERT confidence", ha="left", va="bottom", fontsize=7.5, color=COLORS["token_shap_llm"])
        xs = np.arange(len(order)); dx = 0.22
        for i, key in enumerate(order):
            r = sub.loc[key]; c = COLORS[key[0]]
            ax.plot(xs[i] - dx, r.auroc_explanation, marker="s", color=c, markersize=6, linestyle="none", markerfacecolor="white", markeredgewidth=1.3, zorder=3)
            ax.plot(xs[i], r.auroc_confidence, marker="o", color=c, markersize=6, linestyle="none", zorder=3)
            both = r.auroc_confidence_plus_explanation
            ax.errorbar(xs[i] + dx, both, yerr=[[both - (r.auroc_confidence + r.delta_auroc_ci_low)], [(r.auroc_confidence + r.delta_auroc_ci_high) - both]],
                        fmt="D", color=c, markersize=5.5, capsize=2, linewidth=0.9, zorder=3)
        ax.set_xticks(xs); ax.set_xticklabels([G[g] for _, g in order], fontsize=8)
        ax.axvline(3.5, color="grey", linewidth=0.6, linestyle=":"); ax.axvline(7.5, color="grey", linewidth=0.6, linestyle=":")
        for x, lab in ((1.5, "SubgraphX"), (5.5, "GraphSVX"), (8, "TokenSHAP")):
            ax.text(x, 0.917, lab, ha="center", va="top", fontsize=8, color="dimgrey")
        ax.set_ylim(0.74, 0.92); ax.set_xlim(-0.6, 8.6); ax.set_title(D[ds]); ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("AUROC of the error detector")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="s", color="black", markerfacecolor="white", linestyle="none", label="trained on the explanation"),
         Line2D([], [], marker="o", color="black", linestyle="none", label="trained on the model's confidence"),
         Line2D([], [], marker="D", color="black", linestyle="none", label="trained on both (95% bootstrap interval of the gain)")]
    fig.legend(handles=h, loc="lower center", ncol=3, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(out, dpi=200); plt.close(fig)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--module-root", type=Path, default=Path("outputs/use_case/module_datasets"))
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--nested-csv", type=Path, default=None, help="nested_delta_auroc.csv; enables Figure 5")
    a = ap.parse_args(argv)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    df = load(a.module_root)
    figure_auc(df, "auc_insertion_auc", "insertion AUC", a.output_dir / "Figure_3.png")
    figure_auc(df, "auc_deletion_auc", "deletion AUC", a.output_dir / "Figure_4.png")
    figure_thresholds_combined(df, a.output_dir / "Figure_3_thresholds.png")
    figure_signatures(a.module_root, a.output_dir / "Figure_4_signatures.png")
    figure_signatures_quadrants(a.module_root, a.output_dir / "Figure_4_quadrants.png")
    figure_margin_flips(a.module_root, a.output_dir / "Figure_4_flips.png")
    if a.nested_csv:
        figure_error_detection(a.nested_csv, a.output_dir / "Figure_5_error_detection.png")
    figure_progression(df, "progression_maskout_drop", "mean mask-out drop (confidence lost)", a.output_dir / "Figure_5a.png")
    figure_progression(df, "progression_sufficiency_drop", "mean sufficiency drop (confidence lost)", a.output_dir / "Figure_5b.png")
    print("figures written to", a.output_dir)


if __name__ == "__main__":
    main()
