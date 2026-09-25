#!/usr/bin/env python3
"""Run the stratified logistic regressions one module file per process (they are slow).

Usage (cwd must contain outputs/):
  python3 scripts/_lr_module.py --count            -> number of module files
  python3 scripts/_lr_module.py --index I          -> fit module I, summary to lr_parts/summary_I.csv
  python3 scripts/_lr_module.py --merge            -> concatenate the parts into the summary CSVs
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from use_case import save_logistic_coefficients as slc  # noqa: E402

ROOT = Path("outputs/use_case/module_datasets")


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--count", action="store_true")
    g.add_argument("--index", type=int)
    g.add_argument("--merge", action="store_true")
    ap.add_argument("--bootstrap-reps", type=int, default=200)
    ap.add_argument("--cv-splits", type=int, default=10)
    a = ap.parse_args()
    parts = ROOT / "lr_parts"
    files = sorted(slc._discover_module_files(), key=lambda e: str(e[1]))
    if a.count:
        print(len(files))
        return
    if a.merge:
        summaries = [pd.read_csv(p) for p in sorted(parts.glob("summary_*.csv"))]
        full = pd.concat(summaries, ignore_index=True)
        for name in ("error_signal_classification_summary.csv", "error_signal_classification_summary_final_boot200.csv"):
            full.to_csv(ROOT / name, index=False)
        print("LR summary modules:", full.groupby(["dataset", "method", "graph"]).ngroups, "rows", len(full))
        return
    entry = files[a.index]
    parts.mkdir(exist_ok=True)
    slc._discover_module_files = lambda: [entry]
    slc.SUMMARY_PATH = parts / f"summary_{a.index}.csv"
    _, summary = slc.process_modules(ROOT / "coefficients", max_splits=a.cv_splits, bootstrap_reps=a.bootstrap_reps)
    print("done", entry[0], entry[2], entry[3], len(summary))


if __name__ == "__main__":
    main()
