"""Do the distilled (D) and the gold-trained (G) surrogates err on the same instances?

For every graph type, compares the gold-label error sets of the distilled
surrogate (module datasets), of its gold-trained twin (control run,
``<control-root>/true_label/...``) and of BERT: Jaccard overlaps, the share of
D's errors that G also makes, and the Jaccard expected if D and G erred
independently at their observed error rates.

Usage (cwd must contain outputs/)::

    python src/use_case/surrogate_error_overlap_dg.py --output revision/results_ft/surrogate_error_overlap_dg.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from control_error_overlap import DS, GRAPHS, KEY  # noqa: E402


def jaccard(a, b) -> float:
    return float((a & b).sum() / max(1, (a | b).sum()))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--module-root", type=Path, default=Path("outputs/use_case/module_datasets"))
    ap.add_argument("--control-root", type=Path, default=Path("outputs/gnn_models_control"))
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args(argv)
    rows = []
    for ds, (backbone, name) in DS.items():
        b = pd.read_csv(a.module_root / ds / f"module_dataset_{ds}_token_shap_llm_tokens.csv", usecols=[KEY, "label_id", "prediction_class"]).sort_values(KEY).reset_index(drop=True)
        gold = b.label_id.to_numpy()
        eB = b.prediction_class.to_numpy() != gold
        for g in GRAPHS:
            d = pd.read_csv(a.module_root / ds / f"module_dataset_{ds}_graphsvx_{g}.csv", usecols=[KEY, "prediction_class"]).sort_values(KEY).reset_index(drop=True)
            eD = d.prediction_class.to_numpy() != gold
            f = a.control_root / "true_label" / backbone / name / g / "test_predictions.csv"
            if not f.exists():
                print("missing", f, file=sys.stderr)
                continue
            p = pd.read_csv(f).sort_values("position").reset_index(drop=True)
            assert (p.true_label.to_numpy() == gold).all()
            eG = p.prediction_class.to_numpy() != gold
            pD, pG = eD.mean(), eG.mean()
            rows.append(
                dict(
                    dataset=ds, graph=g, n=len(gold), errors_bert=int(eB.sum()), errors_distilled=int(eD.sum()), errors_gold_trained=int(eG.sum()),
                    jaccard_D_G=jaccard(eD, eG), share_of_D_errors_made_by_G=float((eD & eG).sum() / max(1, eD.sum())),
                    jaccard_D_G_if_independent=float(pD * pG / (pD + pG - pD * pG)),
                    jaccard_D_bert=jaccard(eD, eB), jaccard_G_bert=jaccard(eG, eB),
                    bert_errors_reproduced_D=float((eD & eB).sum() / eB.sum()), bert_errors_reproduced_G=float((eG & eB).sum() / eB.sum()),
                )
            )
    out = pd.DataFrame(rows)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.output, index=False)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
