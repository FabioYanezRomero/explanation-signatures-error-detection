#!/usr/bin/env python3
"""Teacher-student explanation overlap as an error signal.

For every test instance we compare the words highlighted by the GNN explainer
(SubgraphX / GraphSVX on each topology) with the words highlighted by TokenSHAP on
BERT, and ask whether the agreement between the two explanations predicts (a) a
BERT error, (b) a GNN error w.r.t. the gold label, (c) a GNN disagreement with the
teacher -- beyond the confidence of the respective model.

Overlap features per instance (word strings, lower-cased, non-terminal nodes dropped):
  * jaccard_k20 : multiset Jaccard between the top-k words of both explainers, with
                  k = ceil(0.2 * #words of the BERT input) (the sparsity budget)
  * jaccard_top5: same with k = 5
  * spearman    : rank correlation of the words present in both rankings
  * top1_match  : 1 if the most important word coincides

Protocol identical to revision_metrics.py (10-fold stratified class-balanced LR,
paired bootstrap of AUROC / AUPRC gain, 1000 reps, seed 42).
"""
from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from use_case.revision_metrics import CONFIDENCE_FEATURES, EXPLANATION_FEATURES, out_of_fold_probs, paired_bootstrap_delta  # type: ignore  # noqa: E402

DATASETS = {"setfit_ag_news": "setfit_ag_news", "stanfordnlp_sst2": "stanfordnlp_sst2"}
GRAPHS = ["constituency", "syntactic", "window", "skipgrams"]
METHODS = ["graphsvx", "subgraphx"]
KEY = "global_graph_index"
OVERLAP = ["jaccard_k20", "jaccard_top5", "spearman", "top1_match"]
SPECIAL = {"[cls]", "[sep]", "[pad]", "<s>", "</s>", ""}


def words(cell) -> List[str]:
    if not isinstance(cell, str):
        return []
    out = []
    for w in cell.split(","):
        w = w.strip().lower()
        if not w or w in SPECIAL or w.startswith("«") or w.startswith("«"):
            continue
        out.append(w)
    return out


def is_terminal(tok: str) -> bool:
    return not (tok.startswith("«") or tok in {"S", "SBAR", "SINV", "SQ", "SBARQ", "FRAG", "ROOT"})


def gnn_words(cell) -> List[str]:
    if not isinstance(cell, str):
        return []
    return [w.strip().lower() for w in cell.split(",") if w.strip() and is_terminal(w.strip()) and w.strip().lower() not in SPECIAL]


def multiset_jaccard(a: List[str], b: List[str]) -> float:
    ca, cb = Counter(a), Counter(b)
    inter = sum((ca & cb).values())
    union = sum((ca | cb).values())
    return inter / union if union else float("nan")


def overlap_features(bert: List[str], gnn: List[str]) -> Dict[str, float]:
    if not bert or not gnn:
        return {f: float("nan") for f in OVERLAP}
    k = max(1, math.ceil(0.2 * len(bert)))
    common = [w for w in dict.fromkeys(bert) if w in set(gnn)]
    if len(common) >= 3:
        rb = [bert.index(w) for w in common]
        rg = [gnn.index(w) for w in common]
        rho = spearmanr(rb, rg).correlation
        rho = float(rho) if rho == rho else float("nan")
    else:
        rho = float("nan")
    return dict(jaccard_k20=multiset_jaccard(bert[:k], gnn[:k]), jaccard_top5=multiset_jaccard(bert[:5], gnn[:5]),
                spearman=rho, top1_match=float(bert[0] == gnn[0]))


def nested(y_correct: np.ndarray, X_base: np.ndarray, X_add: np.ndarray, *, folds, reps, seed, mask=None):
    y_err = 1 - y_correct
    p_b = out_of_fold_probs(X_base, y_correct, folds=folds, seed=seed)
    p_f = out_of_fold_probs(np.hstack([X_base, X_add]), y_correct, folds=folds, seed=seed)
    p_a = out_of_fold_probs(X_add, y_correct, folds=folds, seed=seed)
    sel = np.ones(len(y_err), bool) if mask is None else mask
    d_roc, lo_roc, hi_roc = paired_bootstrap_delta(y_err[sel], 1 - p_b[sel], 1 - p_f[sel], reps=reps, seed=seed, metric=roc_auc_score)
    d_pr, lo_pr, hi_pr = paired_bootstrap_delta(y_err[sel], 1 - p_b[sel], 1 - p_f[sel], reps=reps, seed=seed, metric=average_precision_score)
    return dict(auroc_base=roc_auc_score(y_err[sel], 1 - p_b[sel]), auroc_overlap_only=roc_auc_score(y_err[sel], 1 - p_a[sel]),
                auroc_full=roc_auc_score(y_err[sel], 1 - p_f[sel]), delta_auroc=d_roc, delta_auroc_ci_low=lo_roc, delta_auroc_ci_high=hi_roc,
                auprc_base=average_precision_score(y_err[sel], 1 - p_b[sel]), auprc_full=average_precision_score(y_err[sel], 1 - p_f[sel]),
                delta_auprc=d_pr, delta_auprc_ci_low=lo_pr, delta_auprc_ci_high=hi_pr)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens-llm", type=Path, default=Path("outputs/analytics/tokens/token_shap_llm"))
    ap.add_argument("--tokens-gnn", type=Path, default=Path("outputs/analytics/tokens_v2"))
    ap.add_argument("--module-root", type=Path, default=Path("outputs/use_case/module_datasets"))
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--bootstrap-reps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(argv)
    a.output_dir.mkdir(parents=True, exist_ok=True)

    desc_rows, nested_rows, per_instance = [], [], []
    for ds in DATASETS:
        llm = pd.read_csv(a.tokens_llm / ds / "tokens.csv", usecols=[KEY, "ranked_tokens"])
        bert_words = {int(r[KEY]): words(r.ranked_tokens) for _, r in llm.iterrows()}
        bmod = pd.read_csv(a.module_root / ds / f"module_dataset_{ds}_token_shap_llm_tokens.csv").sort_values(KEY).reset_index(drop=True)
        for method in METHODS:
            for g in GRAPHS:
                tok = pd.read_csv(a.tokens_gnn / method / ds / f"{g}.csv", usecols=[KEY, "ranked_tokens"])
                gmod = pd.read_csv(a.module_root / ds / f"module_dataset_{ds}_{method}_{g}.csv").sort_values(KEY).reset_index(drop=True)
                feats = {int(r[KEY]): overlap_features(bert_words.get(int(r[KEY]), []), gnn_words(r.ranked_tokens)) for _, r in tok.iterrows()}
                F = pd.DataFrame.from_dict(feats, orient="index").rename_axis(KEY).reset_index()
                m = gmod.merge(F, on=KEY).merge(bmod[[KEY, "label_id", "prediction_class", "prediction_confidence"] + CONFIDENCE_FEATURES[1:]].rename(
                    columns={c: f"bert_{c}" for c in ["label_id", "prediction_class", "prediction_confidence"] + CONFIDENCE_FEATURES[1:]}), on=KEY)
                assert len(m) == len(gmod), (ds, method, g, len(m), len(gmod))
                m["bert_correct"] = (m.bert_prediction_class == m.bert_label_id).astype(int)
                m["gnn_correct_gold"] = (m.prediction_class == m.label_id).astype(int)
                m["gnn_agrees_teacher"] = m.is_correct.astype(int)
                per_instance.append(m[[KEY] + OVERLAP + ["bert_correct", "gnn_correct_gold", "gnn_agrees_teacher"]].assign(dataset=ds, method=method, graph=g))
                # descriptive: mean overlap by outcome + single-feature AUROC (error-oriented)
                for target in ("bert_correct", "gnn_correct_gold", "gnn_agrees_teacher"):
                    y = m[target].to_numpy()
                    for f in OVERLAP:
                        v = m[f].to_numpy(dtype=float)
                        ok = ~np.isnan(v)
                        auroc = roc_auc_score(1 - y[ok], -v[ok]) if len(np.unique(y[ok])) == 2 else float("nan")
                        desc_rows.append(dict(dataset=ds, method=method, graph=g, target=target, feature=f, n_valid=int(ok.sum()),
                                              mean_when_correct=float(np.nanmean(v[y == 1])), mean_when_error=float(np.nanmean(v[y == 0])),
                                              auroc_error_oriented=auroc))
                # nested detectors
                X_over = m[OVERLAP].to_numpy(dtype=float)
                bases = {"bert_correct": m[[f"bert_{c}" for c in CONFIDENCE_FEATURES]].to_numpy(dtype=float),
                         "gnn_correct_gold": m[CONFIDENCE_FEATURES].to_numpy(dtype=float),
                         "gnn_agrees_teacher": m[CONFIDENCE_FEATURES].to_numpy(dtype=float)}
                regimes = {"bert_correct": {"all": None, "confidence>0.9": (m.bert_prediction_confidence > 0.9).to_numpy()},
                           "gnn_correct_gold": {"all": None, "confidence>0.9": (m.prediction_confidence > 0.9).to_numpy()},
                           "gnn_agrees_teacher": {"all": None}}
                for target, X_base in bases.items():
                    y = m[target].to_numpy()
                    for regime, mask in regimes[target].items():
                        if mask is not None and (1 - y)[mask].sum() < 10:
                            continue
                        try:
                            r = nested(y, X_base, X_over, folds=a.folds, reps=a.bootstrap_reps, seed=a.seed, mask=mask)
                        except ValueError as exc:
                            print("  !", ds, method, g, target, regime, exc); continue
                        nested_rows.append(dict(dataset=ds, method=method, graph=g, target=target, regime=regime,
                                                n=int(len(y) if mask is None else mask.sum()), n_errors=int((1 - y).sum() if mask is None else (1 - y)[mask].sum()), **r))
                    # GNN explanation features + overlap on top of confidence (does overlap add beyond the GNN's own explanation?)
                    if target != "bert_correct":
                        X_ce = np.hstack([X_base, m[[c for c in EXPLANATION_FEATURES if c in m.columns]].to_numpy(dtype=float)])
                        r = nested(y, X_ce, X_over, folds=a.folds, reps=a.bootstrap_reps, seed=a.seed)
                        nested_rows.append(dict(dataset=ds, method=method, graph=g, target=target, regime="all (base = confidence+explanation)",
                                                n=int(len(y)), n_errors=int((1 - y).sum()), **r))
                print(ds, method, g, "done", flush=True)
    pd.DataFrame(desc_rows).to_csv(a.output_dir / "explanation_overlap_descriptive.csv", index=False)
    pd.DataFrame(nested_rows).to_csv(a.output_dir / "explanation_overlap_nested.csv", index=False)
    pd.concat(per_instance).to_csv(a.output_dir / "explanation_overlap_per_instance.csv", index=False)
    print("written to", a.output_dir)


if __name__ == "__main__":
    main()
