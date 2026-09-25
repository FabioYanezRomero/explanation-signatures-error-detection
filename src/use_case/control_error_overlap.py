#!/usr/bin/env python3
"""Do distilled surrogates inherit the teacher's errors, or are those instances simply hard?

Compares, per dataset and topology, three GCNs with identical architecture and data:
  * teacher (original)  : trained on BERT's labels (the models used in the paper)
  * teacher (replicate) : same, retrained with the streaming trainer (seed baseline)
  * gold                : trained on the gold labels
against BERT's errors on the test split. Reported per model:
  accuracy vs gold, fraction of BERT errors reproduced (recall), Jaccard between the
  error sets, P(GNN error | BERT error) / P(GNN error | BERT correct) (lift), mean GNN
  confidence on BERT's errors, and share of BERT's errors on which the GNN is confident
  (>0.9) and agrees with BERT. Paired bootstrap CIs for gold - teacher differences.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

KEY = "global_graph_index"
DS = {"setfit_ag_news": ("SetFit", "ag_news"), "stanfordnlp_sst2": ("stanfordnlp", "sst2")}
GRAPHS = ["constituency", "syntactic", "window", "skipgrams"]


def stats(bert_ok: np.ndarray, bert_pred: np.ndarray, pred: np.ndarray, conf: np.ndarray, gold: np.ndarray) -> dict:
    ok = pred == gold
    be, ge = ~bert_ok, ~ok
    inter, union = (be & ge).sum(), (be | ge).sum()
    return dict(accuracy_gold=ok.mean(), n_errors=int(ge.sum()), bert_errors_reproduced=(be & ge).sum() / max(be.sum(), 1),
                jaccard_errors=inter / max(union, 1), p_err_given_bert_err=ge[be].mean(), p_err_given_bert_ok=ge[~be].mean(),
                lift=ge[be].mean() / max(ge[~be].mean(), 1e-9), agreement_with_bert=(pred == bert_pred).mean(),
                mean_conf_on_bert_errors=conf[be].mean(), confident_agreeing_on_bert_errors=((conf > 0.9) & (pred == bert_pred))[be].mean())


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--module-root", type=Path, default=Path("outputs/use_case/module_datasets"))
    ap.add_argument("--control-roots", type=Path, nargs="+", default=[Path("outputs/gnn_models_control")], help="one directory per seed; the first is the main run")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--bootstrap-reps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(argv)
    rng = np.random.default_rng(a.seed)
    rows = []
    for ds, (backbone, name) in DS.items():
        bert = pd.read_csv(a.module_root / ds / f"module_dataset_{ds}_token_shap_llm_tokens.csv", usecols=[KEY, "label_id", "prediction_class", "prediction_confidence"]).sort_values(KEY).reset_index(drop=True)
        gold = bert.label_id.to_numpy().astype(int)
        bert_pred = bert.prediction_class.to_numpy().astype(int)
        bert_ok = bert_pred == gold
        for g in GRAPHS:
            models = {}
            orig = pd.read_csv(a.module_root / ds / f"module_dataset_{ds}_graphsvx_{g}.csv", usecols=[KEY, "label_id", "prediction_class", "prediction_confidence"]).sort_values(KEY).reset_index(drop=True)
            assert (orig.label_id.to_numpy() == gold).all()
            models["teacher (original)"] = (orig.prediction_class.to_numpy().astype(int), orig.prediction_confidence.to_numpy())
            for root_i, root in enumerate(a.control_roots):
                for label, tag0 in (("y", "teacher (replicate)"), ("true_label", "gold")):
                    tag = tag0 if root_i == 0 else f"{tag0} [{root.name}]"
                    f = root / label / backbone / name / g / "test_predictions.csv"
                    if not f.exists():
                        continue
                    p = pd.read_csv(f)
                    if "position" not in p.columns:
                        print("  ! no position column (re-dump predictions)", f); continue
                    p = p.sort_values("position").reset_index(drop=True)
                    if len(p) != len(bert) or not (p.true_label.to_numpy() == gold).all():
                        print("  ! alignment problem", ds, g, label, len(p), len(bert), float((p.true_label.to_numpy() == gold).mean())); continue
                    models[tag] = (p.prediction_class.to_numpy().astype(int), p.prediction_confidence.to_numpy())
            for tag, (pred, conf) in models.items():
                r = stats(bert_ok, bert_pred, pred, conf, gold)
                # paired bootstrap of the difference vs the original teacher model
                if tag != "teacher (original)":
                    p0, c0 = models["teacher (original)"]
                    diffs = {k: [] for k in ("bert_errors_reproduced", "jaccard_errors", "lift", "confident_agreeing_on_bert_errors")}
                    n = len(gold)
                    for _ in range(a.bootstrap_reps):
                        idx = rng.integers(0, n, n)
                        s1 = stats(bert_ok[idx], bert_pred[idx], pred[idx], conf[idx], gold[idx]); s0 = stats(bert_ok[idx], bert_pred[idx], p0[idx], c0[idx], gold[idx])
                        for k in diffs:
                            diffs[k].append(s1[k] - s0[k])
                    for k, v in diffs.items():
                        r[f"diff_vs_original_{k}"] = float(np.mean(v)); r[f"diff_vs_original_{k}_ci_low"] = float(np.percentile(v, 2.5)); r[f"diff_vs_original_{k}_ci_high"] = float(np.percentile(v, 97.5))
                # overlap between the two teacher-trained models (seed baseline) and gold vs teacher replicate
                if tag == "teacher (replicate)" or tag.startswith("teacher (replicate) ["):
                    p0, _ = models["teacher (original)"]
                    e0, e1 = p0 != gold, pred != gold
                    r["jaccard_errors_with_original_teacher_model"] = (e0 & e1).sum() / max((e0 | e1).sum(), 1)
                if tag.startswith("gold") and "teacher (replicate)" in models:
                    p1, _ = models["teacher (replicate)"]
                    e0, e1 = p1 != gold, pred != gold
                    r["jaccard_errors_with_teacher_replicate"] = (e0 & e1).sum() / max((e0 | e1).sum(), 1)
                rows.append(dict(dataset=ds, graph=g, model=tag, n=len(gold), bert_accuracy=bert_ok.mean(), n_bert_errors=int((~bert_ok).sum()), **r))
                print(ds, g, tag, {k: round(v, 3) for k, v in r.items() if not k.startswith("diff")}, flush=True)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(a.output, index=False)
    print("written", a.output)


if __name__ == "__main__":
    main()
