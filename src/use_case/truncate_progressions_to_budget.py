"""Restrict the Dimension-1/2 progressions of every module dataset to the explanation budget.

For SubgraphX the explainer stores the confidence progressions over its full
node ranking (search credit is assigned to every explored node), whereas
GraphSVX and TokenSHAP store them over the top-20% elements only. This script
truncates every progression to K = round(sparsity * len) elements (a no-op for
GraphSVX and TokenSHAP, whose lists already have length K) and recomputes the
derived columns with the same formulas as ``build_module_datasets.py`` and
``Analytics/auc/extract_gnn.py``:

* ``auc_deletion_auc``, ``auc_insertion_auc`` (trapezoid, x normalised to [0, 1]),
  ``auc_deletion_aac``, ``auc_normalised_deletion_auc``, ``auc_normalised_insertion_auc``,
  ``auc_maskout_progression_len``, ``auc_sufficiency_progression_len``;
* ``progression_{maskout,sufficiency}_drop_k{1,3,5,10}`` and
  ``progression_concentration_top{1,3,5,10}``;
* the four list columns and their ``_len`` columns.

Originals are copied to ``<module-root>_fullspan/`` before being overwritten.

Usage::

    python src/use_case/truncate_progressions_to_budget.py --module-root outputs/use_case/module_datasets [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_config import PROGRESSION_TOP_K  # noqa: E402

LISTS = {
    "maskout_conf": "progression_maskout_progression_confidence",
    "maskout_drop": "progression_maskout_progression_drop",
    "suff_conf": "progression_sufficiency_progression_confidence",
    "suff_drop": "progression_sufficiency_progression_drop",
}


def parse(s) -> List[float]:
    if isinstance(s, str) and s.strip():
        try:
            return [float(v) for v in json.loads(s)]
        except Exception:
            return []
    return []


def auc(v: List[float]):
    n = len(v)
    if n == 0:
        return np.nan
    if n == 1:
        return float(v[0])
    w = 1.0 / (n - 1)
    return float(sum((v[i] + v[i + 1]) * 0.5 * w for i in range(n - 1)))


def at(seq: List[float], k: int) -> float:
    if not seq:
        return 0.0
    return float(seq[min(k, len(seq)) - 1])


def conc(seq: List[float], k: int) -> float:
    total = seq[-1] if seq else 0.0
    return at(seq, k) / total if total > 0 else 0.0


def process(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    lists = {key: df[col].map(parse) for key, col in LISTS.items()}
    n = lists["maskout_conf"].map(len)
    K = np.maximum(1, np.round(df["sparsity"].fillna(0).to_numpy() * n.to_numpy())).astype(int)
    K = np.minimum(K, np.maximum(n.to_numpy(), 1))
    changed = int((n.to_numpy() > K).sum())
    out = df.copy()
    trunc = {key: [s[:k] if s else s for s, k in zip(lists[key], K)] for key in LISTS}
    for key, col in LISTS.items():
        out[col] = [json.dumps(s) if s else "" for s in trunc[key]]
        out[col + "_len"] = [len(s) for s in trunc[key]]
    out["auc_maskout_progression_len"] = out[LISTS["maskout_conf"] + "_len"]
    out["auc_sufficiency_progression_len"] = out[LISTS["suff_conf"] + "_len"]
    origin = out["auc_origin_confidence"].astype(float)
    dele = np.array([auc(s) for s in trunc["maskout_conf"]])
    ins = np.array([auc(s) if s else np.nan for s in trunc["suff_conf"]])
    final = np.array([s[-1] if s else o for s, o in zip(trunc["suff_conf"], origin)])
    out["auc_deletion_auc"] = dele
    out["auc_insertion_auc"] = ins
    out["auc_deletion_aac"] = origin - dele
    out["auc_normalised_deletion_auc"] = np.where(origin != 0, dele / origin, np.nan)
    out["auc_normalised_insertion_auc"] = np.where((final != 0) & ~np.isnan(final), ins / final, np.nan)
    for k in PROGRESSION_TOP_K:
        out[f"progression_maskout_drop_k{k}"] = [at(s, k) for s in trunc["maskout_drop"]]
        out[f"progression_sufficiency_drop_k{k}"] = [at(s, k) for s in trunc["suff_drop"]]
        out[f"progression_concentration_top{k}"] = [conc(s, k) for s in trunc["maskout_drop"]]
    return out, changed


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--module-root", type=Path, default=Path("outputs/use_case/module_datasets"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--methods", nargs="+", default=["subgraphx"],
                    help="explainers whose stored progressions cover the whole input (the list length is the node count); "
                         "GraphSVX and TokenSHAP already store the budget only and must not be touched")
    a = ap.parse_args(argv)
    backup = a.module_root.parent / (a.module_root.name + "_fullspan")
    for f in sorted(a.module_root.glob("*/module_dataset_*.csv")):
        if not any(f"_{m}_" in f.name for m in a.methods):
            continue
        df = pd.read_csv(f, low_memory=False)
        if not all(c in df.columns for c in LISTS.values()) or "sparsity" not in df.columns:
            print(f"skip {f.name}: no progression columns")
            continue
        out, changed = process(df)
        d_old, d_new = df["auc_deletion_auc"].mean(), out["auc_deletion_auc"].mean()
        print(f"{f.name}: {changed}/{len(df)} rows truncated | mean deletion AUC {d_old:.3f} -> {d_new:.3f} | mean insertion AUC {df['auc_insertion_auc'].mean():.3f} -> {out['auc_insertion_auc'].mean():.3f}")
        if a.dry_run:
            continue
        dest = backup / f.parent.name / f.name
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dest)
        out.to_csv(f, index=False)


if __name__ == "__main__":
    main()
