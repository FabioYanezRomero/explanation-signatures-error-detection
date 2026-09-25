import argparse
import pickle
import sys
import types
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
from tqdm import tqdm


def ensure_subgraphx_stub() -> None:
    """Provide a lightweight SubgraphXResult to load pickles without heavy deps."""
    module_name = "src.explain.gnn.subgraphx.main"
    module = sys.modules.get(module_name)
    if module is None:
        module = types.ModuleType(module_name)
        sys.modules[module_name] = module

    if hasattr(module, "SubgraphXResult"):
        return

    class SubgraphXResult:  # type: ignore[too-many-instance-attributes]
        def __init__(
            self,
            graph_index: int,
            label: Optional[int],
            explanation: Any,
            related_prediction: Dict[str, Any],
            num_nodes: int,
            num_edges: int,
            hyperparams: Dict[str, Any],
        ) -> None:
            self.graph_index = graph_index
            self.label = label
            self.explanation = explanation
            self.related_prediction = related_prediction
            self.num_nodes = num_nodes
            self.num_edges = num_edges
            self.hyperparams = hyperparams

    module.SubgraphXResult = SubgraphXResult  # type: ignore[attr-defined]
    main_module = sys.modules.get("__main__")
    if main_module is None:
        main_module = types.ModuleType("__main__")
        sys.modules["__main__"] = main_module
    setattr(main_module, "SubgraphXResult", SubgraphXResult)


def slugify(value: str) -> str:
    safe = value.replace("/", "_").replace(" ", "_").replace("-", "_")
    return "".join(char for char in safe if char.isalnum() or char == "_").lower()


def normalize_dataset_name(raw: str) -> str:
    name = raw.replace("_", "-")
    if name.lower() == "sst2":
        return "sst-2"
    return name


def infer_split_from_run_id(run_id: str) -> Optional[str]:
    tokens = run_id.lower()
    for candidate in ("train", "test", "validation", "val", "dev"):
        if candidate in tokens:
            if candidate == "val":
                return "validation"
            if candidate == "dev":
                return "development"
            return candidate
    return None


def parse_metadata_from_path(path: Path) -> Dict[str, str]:
    parts = path.resolve().parts
    if "gnn_models" not in parts:
        raise ValueError(f"Unexpected path layout for {path}")
    idx = parts.index("gnn_models")
    try:
        backbone = parts[idx + 1]
        dataset = parts[idx + 2]
        graph_type = parts[idx + 3]
        method = parts[idx + 5]
        run_id = parts[idx + 6]
    except IndexError as exc:
        raise ValueError(f"Path too short to extract metadata: {path}") from exc
    dataset_backbone = f"{backbone}/{dataset}"
    dataset_normalised = normalize_dataset_name(dataset)
    split = infer_split_from_run_id(run_id) or ""
    return {
        "backbone": backbone,
        "dataset": dataset_normalised,
        "dataset_raw": dataset,
        "dataset_backbone": dataset_backbone,
        "graph_type": graph_type,
        "method": method,
        "run_id": run_id,
        "split": split,
    }


def load_payload(path: Path, *, method: str) -> Dict[str, Any]:
    if method == "subgraphx":
        ensure_subgraphx_stub()
    with path.open("rb") as handler:
        payload = pickle.load(handler)
    if isinstance(payload, dict):
        return payload
    if hasattr(payload, "__dict__"):
        return dict(payload.__dict__)
    raise ValueError(f"Unsupported pickle payload in {path}")


def _coerce_sequence(raw: Optional[Sequence[Any]]) -> List[float]:
    if raw is None:
        return []
    if not isinstance(raw, (list, tuple)):
        return []
    result: List[float] = []
    for value in raw:
        try:
            result.append(float(value))
        except Exception:
            continue
    return result


def _distribution_stats(distribution):
    """Label-free confidence baselines from the original class distribution."""
    try:
        values = [float(v) for v in (distribution or [])]
    except (TypeError, ValueError):
        return None, None, None
    if not values:
        return None, None, None
    import math
    ordered = sorted(values, reverse=True)
    msp = ordered[0]
    margin = ordered[0] - ordered[1] if len(ordered) > 1 else ordered[0]
    entropy = -sum(p * math.log(p) for p in values if p > 0)
    return msp, margin, entropy


def _compute_auc(values: List[float]) -> Optional[float]:
    n = len(values)
    if n == 0:
        return None
    if n == 1:
        return float(values[0])
    width = 1.0 / (n - 1)
    area = 0.0
    for i in range(n - 1):
        area += (values[i] + values[i + 1]) * 0.5 * width
    return area


def extract_progressions(payload: Mapping[str, Any]) -> Tuple[List[float], List[float]]:
    related = payload.get("related_prediction") or {}
    if not isinstance(related, Mapping):
        related = {}
    maskout_conf = payload.get("maskout_progression_confidence")
    if maskout_conf is None:
        maskout_conf = related.get("maskout_progression_confidence")
    suff_conf = payload.get("sufficiency_progression_confidence")
    if suff_conf is None:
        suff_conf = related.get("sufficiency_progression_confidence")
    return _coerce_sequence(maskout_conf), _coerce_sequence(suff_conf)


def _origin_confidence(payload: Mapping[str, Any], related: Mapping[str, Any]) -> Optional[float]:
    origin = payload.get("origin_confidence")
    if origin is None:
        origin = related.get("origin")
    if origin is None:
        pred = payload.get("prediction")
        if isinstance(pred, Mapping):
            origin = pred.get("confidence")
    try:
        return float(origin) if origin is not None else None
    except Exception:
        return None


def build_record(payload: Dict[str, Any], metadata: Dict[str, str]) -> Dict[str, Any]:
    related = payload.get("related_prediction") or {}
    if not isinstance(related, Mapping):
        related = {}
    maskout_conf, suff_conf = extract_progressions(payload)
    deletion_auc = _compute_auc(maskout_conf)
    insertion_auc = _compute_auc(suff_conf) if suff_conf else None

    origin_conf = _origin_confidence(payload, related)
    final_conf = None
    if suff_conf:
        final_conf = suff_conf[-1]
    elif origin_conf is not None:
        final_conf = origin_conf

    deletion_aac = None
    if deletion_auc is not None and origin_conf is not None:
        deletion_aac = origin_conf - deletion_auc

    normalised_deletion_auc = None
    if deletion_auc is not None and origin_conf:
        normalised_deletion_auc = deletion_auc / origin_conf if origin_conf else None

    normalised_insertion_auc = None
    if insertion_auc is not None and final_conf not in (None, 0.0):
        normalised_insertion_auc = insertion_auc / final_conf

    distribution = related.get("origin_distribution")
    if distribution is None and isinstance(payload.get("prediction"), Mapping):
        distribution = payload["prediction"].get("distribution")
    origin_msp, origin_margin, origin_entropy = _distribution_stats(distribution)

    label = payload.get("label")
    prediction_class = payload.get("prediction_class")
    is_correct = payload.get("is_correct")
    if is_correct is None and label is not None and prediction_class is not None:
        try:
            is_correct = int(label) == int(prediction_class)
        except Exception:
            is_correct = None

    record: Dict[str, Any] = {
        "method": metadata["method"],
        "backbone": metadata["backbone"],
        "dataset": metadata["dataset"],
        "dataset_raw": metadata["dataset_raw"],
        "dataset_backbone": metadata["dataset_backbone"],
        "graph_type": metadata["graph_type"],
        "run_id": metadata["run_id"],
        "split": metadata["split"],
        "graph_index": payload.get("graph_index"),
        "global_graph_index": payload.get("global_graph_index"),
        "label": label,
        "prediction_class": prediction_class,
        "prediction_confidence": payload.get("prediction_confidence"),
        "is_correct": is_correct,
        "origin_confidence": origin_conf,
        "origin_msp": origin_msp,
        "origin_margin_top2": origin_margin,
        "origin_entropy": origin_entropy,
        "deletion_auc": deletion_auc,
        "insertion_auc": insertion_auc,
        "deletion_aac": deletion_aac,
        "normalised_deletion_auc": normalised_deletion_auc,
        "normalised_insertion_auc": normalised_insertion_auc,
        "maskout_progression_len": len(maskout_conf),
        "sufficiency_progression_len": len(suff_conf),
    }
    return record


def discover_pickles(base_dir: Path, method: str) -> Sequence[Path]:
    pattern = f"**/explanations/{method}/**/results_split_pickle/graph_*.pkl"
    return sorted(base_dir.glob(pattern))


def process_method(
    method: str,
    base_dir: Path,
    output_dir: Path,
    limit: Optional[int] = None,
) -> Sequence[Path]:
    pickle_paths = discover_pickles(base_dir, method)
    if not pickle_paths:
        print(f"– No {method} pickles discovered under {base_dir}")
        return []
    if limit is not None:
        pickle_paths = pickle_paths[:limit]

    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for path in tqdm(pickle_paths, desc=f"Scanning {method} pickles", leave=False, colour="green"):
        try:
            metadata = parse_metadata_from_path(path)
            payload = load_payload(path, method=method)
            record = build_record(payload, metadata)
        except Exception as exc:
            print(f"! Skipping {path} ({exc})")
            continue
        key = (record["dataset_backbone"], record["graph_type"])
        grouped[key].append(record)

    written_paths: List[Path] = []
    for (dataset_backbone, graph_type), rows in tqdm(
        sorted(grouped.items()),
        desc=f"{method} datasets",
        leave=False,
        colour="cyan",
    ):
        if not rows:
            continue
        dataset_slug = slugify(dataset_backbone)
        graph_slug = slugify(graph_type)
        target_dir = output_dir / method / dataset_slug
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / f"{graph_slug}.csv"

        df = pd.DataFrame(rows)
        sort_columns = [column for column in ("split", "run_id", "graph_index") if column in df.columns]
        if sort_columns:
            df.sort_values(sort_columns, inplace=True, kind="mergesort")
        df.to_csv(target_path, index=False)
        written_paths.append(target_path)
        print(f"✓ {method} | {dataset_backbone} | {graph_type} → {target_path} ({len(df)} rows)")

    return written_paths


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate GNN AUC metrics into CSV summaries.")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("outputs/gnn_models"),
        help="Root directory containing GNN explanation outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/analytics/auc"),
        help="Directory where AUC CSV files will be written.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["subgraphx", "graphsvx"],
        help="Explainability methods to process.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on the number of pickles processed per method.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    base_dir: Path = args.base_dir
    output_dir: Path = args.output_dir
    methods = [method.lower() for method in args.methods]

    if not base_dir.exists():
        raise FileNotFoundError(f"Base directory not found: {base_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    all_written: List[Path] = []
    for method in tqdm(methods, desc="Processing methods", leave=False, colour="blue"):
        all_written.extend(process_method(method, base_dir, output_dir, args.limit))

    if all_written:
        total_rows = 0
        for path in all_written:
            try:
                df = pd.read_csv(path)
                total_rows += len(df)
            except Exception:
                continue
        print(f"\nCompleted AUC aggregation for {len(all_written)} CSV file(s) ({total_rows} total rows).")
    else:
        print("\nNo AUC CSV files were generated.")


if __name__ == "__main__":
    main()
