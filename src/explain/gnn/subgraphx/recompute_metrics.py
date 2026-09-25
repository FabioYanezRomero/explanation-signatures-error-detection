#!/usr/bin/env python3
"""Recompute SubgraphX post-hoc metrics from stored MCTS results (no new search).

DIG's ``SubgraphX.explain`` returns the explored MCTS nodes sorted by score, so the
first entry is (almost) the full graph.  The original pipeline took that entry as the
explanation; the explanation is the highest-scoring coalition with at most
``max_nodes`` nodes (DIG's ``find_closest_node_result``).  This script re-derives, for
every stored graph:

* the explanation coalition (<= max_nodes) and its MCTS score;
* a node ranking from the search: MCTS survival credit
  ``credit(i) = sum_{C explored, i in C} P(C) / |C|`` (coalition nodes first);
* masked / mask-out distributions and margins on the coalition (zero-filling);
* mask-out and sufficiency progressions over the ranking (batched);
* sparsity as the included fraction (|coalition| / N), matching GraphSVX/TokenSHAP,
  plus DIG's excluded-fraction convention.

Everything else (origin distribution, predicted class, teacher label, target class,
hyper-parameters) is carried over from the stored record.  The output is a compact
``results.pkl`` (no graph tensors) plus ``summary.json`` in the target directory.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import types
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch

from src.explain.common.batching import build_masked_batch
from src.explain.gnn.config import ExplainerRequest
from src.explain.gnn.model_loader import load_gnn_model


def _install_stub() -> None:
    """Allow unpickling SubgraphXResult without importing the DIG-dependent module."""
    name = "src.explain.gnn.subgraphx.main"
    if name in sys.modules and hasattr(sys.modules[name], "SubgraphXResult"):
        return
    module = types.ModuleType(name)

    class SubgraphXResult:  # noqa: D401
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    module.SubgraphXResult = SubgraphXResult
    sys.modules[name] = module
    setattr(sys.modules["__main__"], "SubgraphXResult", SubgraphXResult)


def find_closest_node_result(entries: Sequence[Dict[str, Any]], max_nodes: int) -> Dict[str, Any]:
    """DIG's selection rule on info dicts: highest P among coalitions with <= max_nodes."""
    ordered = sorted(entries, key=lambda e: len(e["coalition"]))
    best = ordered[0]
    for entry in ordered:
        if len(entry["coalition"]) <= max_nodes and entry["P"] > best["P"]:
            best = entry
    return best


def survival_ranking(entries: Sequence[Dict[str, Any]], coalition: Sequence[int], num_nodes: int) -> List[int]:
    credit = [0.0] * num_nodes
    for entry in entries:
        size = len(entry["coalition"])
        if size == 0:
            continue
        share = float(entry["P"]) / size
        for node in entry["coalition"]:
            if 0 <= node < num_nodes:
                credit[node] += share
    coalition_set = set(int(n) for n in coalition)
    first = sorted(coalition_set, key=lambda n: -credit[n])
    rest = sorted((n for n in range(num_nodes) if n not in coalition_set), key=lambda n: -credit[n])
    return first + rest


def contrastive_stats(distribution: Sequence[float], target: int):
    values = [float(v) for v in distribution]
    others = [(i, v) for i, v in enumerate(values) if i != target]
    if not others:
        return None, None, None
    second_idx, second_val = max(others, key=lambda t: t[1])
    return second_idx, second_val, values[target] - second_val


@torch.no_grad()
def probs_for_masks(model, x, edge_index, masks: torch.Tensor, chunk: int = 256) -> torch.Tensor:
    out = []
    for part in masks.split(chunk):
        logits = model(data=build_masked_batch(x, edge_index, part))
        out.append(torch.softmax(logits, dim=-1))
    return torch.cat(out, dim=0)


def request_from_run_dir(run_dir: Path, gnn_root: Path) -> ExplainerRequest:
    # .../gnn_models/<backbone>/<dataset>/<graph_type>/explanations/subgraphx*/<slug>
    parts = run_dir.resolve().parts
    idx = parts.index("gnn_models")
    backbone, dataset, graph_type = parts[idx + 1], parts[idx + 2], parts[idx + 3]
    slug = run_dir.name
    split = "test" if "_test" in slug else "validation" if "_validation" in slug else "test"
    return ExplainerRequest(dataset=dataset, graph_type=graph_type, backbone=backbone, split=split, method="subgraphx", gnn_root=gnn_root)


def recompute_run(run_dir: Path, output_dir: Path, *, gnn_root: Path, device: torch.device) -> int:
    _install_stub()
    with (run_dir / "results.pkl").open("rb") as fh:
        results = pickle.load(fh)
    if not results:
        return 0
    first = results[0].__dict__ if hasattr(results[0], "__dict__") else results[0]
    num_classes = len(first["related_prediction"]["origin_distribution"])
    stub = types.SimpleNamespace(num_node_features=int(first["explanation"][0]["data"].x.size(1)), num_classes=num_classes)
    request = request_from_run_dir(run_dir, gnn_root)
    request.device = str(device)
    model, _, _ = load_gnn_model(request, dataset=stub)
    model.eval()

    summary_old = json.loads((run_dir / "summary.json").read_text()) if (run_dir / "summary.json").exists() else {}
    new_results: List[Dict[str, Any]] = []
    for item in results:
        rec = item.__dict__ if hasattr(item, "__dict__") else dict(item)
        entries = rec["explanation"]
        if isinstance(entries, list) and entries and isinstance(entries[0], list):
            entries = entries[0]
        rp = dict(rec["related_prediction"])
        max_nodes = int(rec["hyperparams"]["max_nodes"])
        data = entries[0]["data"]
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        num_nodes = int(x.size(0))
        best = find_closest_node_result(entries, max_nodes)
        coalition = [int(n) for n in best["coalition"]]
        ranking = survival_ranking(entries, coalition, num_nodes)
        target = int(rp.get("target_class", rp.get("predicted_class")))

        keep = torch.zeros(num_nodes, dtype=torch.float32, device=device)
        keep[coalition] = 1.0
        drop = 1.0 - keep
        pm = probs_for_masks(model, x, edge_index, torch.stack([keep, drop]))
        masked_dist, maskout_dist = pm[0].tolist(), pm[1].tolist()
        origin_dist = [float(v) for v in rp["origin_distribution"]]

        # progressions over the ranking: remove top-1..k (mask-out) / keep only top-1..k (sufficiency)
        k = len(ranking)
        position = torch.empty(num_nodes, dtype=torch.long, device=device)
        position[torch.tensor(ranking, device=device)] = torch.arange(k, device=device)
        # order_mask[s] has ones exactly on ranking[: s + 1]
        order_mask = (position.unsqueeze(0) <= torch.arange(k, device=device).unsqueeze(1)).to(torch.float32)
        maskout_conf = probs_for_masks(model, x, edge_index, 1.0 - order_mask)[:, target].tolist()
        suff_conf = probs_for_masks(model, x, edge_index, order_mask)[:, target].tolist()
        origin = float(origin_dist[target])

        rp.update({
            "top_nodes": ranking,
            "ranked_nodes": ranking,
            "explanation_nodes": coalition,
            "explanation_score": float(best["P"]),
            "ranking_method": "mcts_survival_credit",
            "masked": float(masked_dist[target]),
            "maskout": float(maskout_dist[target]),
            "origin": origin,
            "masked_distribution": masked_dist,
            "maskout_distribution": maskout_dist,
            "sparsity": len(coalition) / max(num_nodes, 1),
            "sparsity_dig_excluded": 1.0 - len(coalition) / max(num_nodes, 1),
            "maskout_progression_confidence": maskout_conf,
            "maskout_progression_drop": [origin - v for v in maskout_conf],
            "sufficiency_progression_confidence": suff_conf,
            "sufficiency_progression_drop": [origin - v for v in suff_conf],
        })
        for key, dist in (("origin", origin_dist), ("masked", masked_dist), ("maskout", maskout_dist)):
            second_idx, second_val, contrast = contrastive_stats(dist, target)
            if key == "origin":
                rp["origin_second_class"] = second_idx
            rp[f"{key}_second_confidence"] = second_val
            rp[f"{key}_contrastivity"] = contrast
        for key in ("node_tokens", "ranked_tokens", "top_token_text"):
            rp.pop(key, None)

        new_results.append({
            "graph_index": rec["graph_index"],
            "label": rec.get("label"),
            "explanation": {
                "coalition": coalition,
                "coalition_score": float(best["P"]),
                "max_nodes": max_nodes,
                "explored": [(len(e["coalition"]), float(e["P"])) for e in entries],
                "ranking": ranking,
            },
            "related_prediction": rp,
            "num_nodes": num_nodes,
            "num_edges": rec.get("num_edges"),
            "hyperparams": rec.get("hyperparams"),
            "prediction": rec.get("prediction"),
            "is_correct": rec.get("is_correct"),
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "results.pkl").open("wb") as fh:
        pickle.dump(new_results, fh, protocol=pickle.HIGHEST_PROTOCOL)
    summary = dict(summary_old)
    summary.update({
        "num_graphs": len(new_results),
        "recomputed_from": str(run_dir),
        "coalition_selection": "find_closest_node_result(max_nodes)",
        "node_ranking": "mcts_survival_credit",
        "graphs": [
            {"graph_index": r["graph_index"], "label": r["label"], "num_nodes": r["num_nodes"], "num_edges": r["num_edges"],
             "related_prediction": r["related_prediction"], "hyperparams": r["hyperparams"], "prediction": r["prediction"],
             "prediction_class": (r["prediction"] or {}).get("class"), "prediction_confidence": (r["prediction"] or {}).get("confidence"),
             "is_correct": r["is_correct"]}
            for r in new_results
        ],
    })
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    return len(new_results)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-root", type=Path, default=Path("/app/outputs/gnn_models"), help="Root scanned for raw runs.")
    parser.add_argument("--raw-subdir", default="subgraphx_v2_rawmcts", help="explanations/<raw-subdir>/<slug>/results.pkl inputs.")
    parser.add_argument("--out-subdir", default="subgraphx", help="explanations/<out-subdir>/<slug>/ outputs.")
    parser.add_argument("--gnn-root", type=Path, default=Path("/app/outputs/gnn_models"))
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--only", nargs="*", default=None, help="Optional substrings a run dir must contain.")
    args = parser.parse_args(argv)

    device = torch.device(args.device)
    runs = sorted(p.parent for p in args.raw_root.glob(f"*/*/*/explanations/{args.raw_subdir}/*/results.pkl"))
    if args.only:
        runs = [r for r in runs if any(s in str(r) for s in args.only)]
    for run_dir in runs:
        out_dir = run_dir.parent.parent / args.out_subdir / run_dir.name
        n = recompute_run(run_dir, out_dir, gnn_root=args.gnn_root, device=device)
        print(f"✓ {run_dir.name}: {n} graphs -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
