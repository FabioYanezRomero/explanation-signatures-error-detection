"""Concatenate generated table bodies into the merged supplementary tables (S4, S5a, S12-S14).

LaTeX cannot take \\midrule/\\multicolumn right after \\input, so the block titles and rules that
join two bodies are written into one file per merged table.

Usage::

    python src/use_case/merge_supplementary_tables.py --ft revision/manuscript_v2/tables_ft --base revision/manuscript_v2/tables_base
"""
from __future__ import annotations

import argparse
from pathlib import Path


def blocks(ft: Path, base: Path, name: str, ncols: int) -> str:
    empty = " &" * (ncols - 1)
    return "\n".join([
        rf"\emph{{Fine-tuned node features}}{empty} \\",
        r"\midrule",
        (ft / name).read_text().rstrip(),
        r"\midrule",
        rf"\multicolumn{{{ncols}}}{{@{{}}l}}{{\emph{{Pre-trained node features}}}} \\",
        r"\midrule",
        (base / name).read_text().rstrip(),
        r"\bottomrule",
    ]) + "\n"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ft", type=Path, required=True)
    ap.add_argument("--base", type=Path, required=True)
    a = ap.parse_args(argv)
    ft, base = a.ft, a.base
    # S4: coefficients, one block per dataset (each body carries its own header and \bottomrule)
    (ft / "merged_coefficients.tex").write_text("\n".join([
        r"\textbf{AG News}" + " &" * 9 + r" \\",
        (ft / "supp_coefficients_agnews.tex").read_text().rstrip(),
        r"\addlinespace[0.6em]",
        r"\multicolumn{10}{@{}l}{\textbf{SST-2}} \\",
        (ft / "supp_coefficients_sst2.tex").read_text().rstrip(),
    ]) + "\n")
    # S5(a): teacher-target detection body with its bottom rule
    (ft / "merged_error_detection_teacher.tex").write_text(
        (ft / "tab_error_detection_teacher.tex").read_text().rstrip() + "\n" + r"\bottomrule" + "\n")
    # S12-S14: fine-tuned and pre-trained blocks
    for name, ncols in (("tab_error_detection_full.tex", 12), ("tab_calibration_full.tex", 10), ("tab_control_full.tex", 7)):
        (ft / ("merged_" + name)).write_text(blocks(ft, base, name, ncols))
    print("merged tables written to", ft)


if __name__ == "__main__":
    main()
