import argparse
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
from tqdm import tqdm


def slugify(value: str) -> str:
    safe = value.replace("/", "_").replace(" ", "_").replace("-", "_")
    return "".join(char for char in safe if char.isalnum() or char == "_").lower()


def normalize_dataset_name(raw: str) -> str:
    name = raw.replace("_", "-")
    if name.lower() == "sst2":
        return "sst-2"
    return name


def _parse_dataset_fields(dataset_backbone: Optional[str]) -> Tuple[str, str, str]:
    if not dataset_backbone:
        return "", "", ""
    parts = dataset_backbone.split("/")
    if len(parts) >= 2:
        backbone = parts[0]
        dataset_raw = parts[-1]
    else:
        backbone = ""
        dataset_raw = parts[0]
    dataset = normalize_dataset_name(dataset_raw)
    return backbone, dataset_raw, dataset


def discover_pickles(base_dir: Path) -> Sequence[Path]:
    pattern = "**/*split_pickle/graph_*.pkl"
    return sorted(base_dir.glob(pattern))


def load_payload(path: Path) -> Dict[str, Any]:
    with path.open("rb") as handler:
        payload = pickle.load(handler)
    if isinstance(payload, dict):
        return payload
    if hasattr(payload, "__dict__"):
        return dict(payload.__dict__)
    raise ValueError(f"Unsupported pickle payload at {path}")


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
    try:
        return float(origin) if origin is not None else None
    except Exception:
        return None


def build_record(payload: Dict[str, Any]) -> Dict[str, Any]:
    dataset_backbone = payload.get("dataset") or ""
    backbone, dataset_raw, dataset = _parse_dataset_fields(dataset_backbone)
    graph_type = payload.get("graph_type") or "tokens"
    method = payload.get("method") or "token_shap_llm"

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
    if deletion_auc is not None and origin_conf not in (None, 0.0):
        normalised_deletion_auc = deletion_auc / origin_conf

    normalised_insertion_auc = None
    if insertion_auc is not None and final_conf not in (None, 0.0):
        normalised_insertion_auc = insertion_auc / final_conf

    distribution = payload.get("origin_distribution")
    if distribution is None:
        distribution = related.get("origin_distribution")
    origin_msp, origin_margin, origin_entropy = _distribution_stats(distribution)

    record: Dict[str, Any] = {
        "method": method,
        "backbone": backbone,
        "dataset": dataset,
        "dataset_raw": dataset_raw,
        "dataset_backbone": dataset_backbone,
        "graph_type": graph_type,
        "run_id": payload.get("run_id"),
        "graph_index": payload.get("graph_index"),
        "global_graph_index": payload.get("global_graph_index"),
        "label": payload.get("label"),
        "prediction_class": payload.get("prediction_class"),
        "prediction_confidence": payload.get("prediction_confidence"),
        "is_correct": payload.get("is_correct"),
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


def process_pickles(
    base_dir: Path,
    output_dir: Path,
    *,
    limit: Optional[int] = None,
) -> Sequence[Path]:
    pickle_paths = discover_pickles(base_dir)
    if not pickle_paths:
        raise FileNotFoundError(f"No TokenSHAP pickles found under {base_dir}")
    if limit is not None:
        pickle_paths = pickle_paths[:limit]

    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for path in tqdm(pickle_paths, desc="Scanning TokenSHAP pickles", leave=False, colour="green"):
        try:
            payload = load_payload(path)
            record = build_record(payload)
        except Exception as exc:
            print(f"! Skipping {path} ({exc})")
            continue
        key = (record["dataset_backbone"] or record["dataset"], record["graph_type"])
        grouped[key].append(record)

    written_paths: List[Path] = []
    for (dataset_backbone, graph_type), rows in tqdm(
        sorted(grouped.items()),
        desc="Writing CSVs",
        leave=False,
        colour="cyan",
    ):
        if not rows:
            continue
        method = rows[0].get("method") or "token_shap_llm"
        dataset_slug = slugify(dataset_backbone)
        graph_slug = slugify(graph_type)
        target_dir = output_dir / method / dataset_slug
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / f"{graph_slug}.csv"

        df = pd.DataFrame(rows)
        sort_columns = [column for column in ("run_id", "graph_index") if column in df.columns]
        if sort_columns:
            df.sort_values(sort_columns, inplace=True, kind="mergesort")
        df.to_csv(target_path, index=False)
        written_paths.append(target_path)
        print(f"✓ {method} | {dataset_backbone} | {graph_type} → {target_path} ({len(df)} rows)")

    return written_paths


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate TokenSHAP AUC metrics into CSV summaries.")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("outputs/insights/news/LLM"),
        help="Root directory containing TokenSHAP outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/analytics/auc"),
        help="Directory where AUC CSV files will be written.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on the number of pickles processed.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    base_dir: Path = args.base_dir
    output_dir: Path = args.output_dir

    if not base_dir.exists():
        raise FileNotFoundError(f"Base directory not found: {base_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    written = process_pickles(base_dir, output_dir, limit=args.limit)

    if written:
        total_rows = 0
        for path in written:
            try:
                df = pd.read_csv(path)
                total_rows += len(df)
            except Exception:
                continue
        print(f"\nCompleted TokenSHAP AUC aggregation for {len(written)} CSV file(s) ({total_rows} total rows).")
    else:
        print("\nNo TokenSHAP AUC CSV files were generated.")


if __name__ == "__main__":
    main()
