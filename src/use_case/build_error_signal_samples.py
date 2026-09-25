#!/usr/bin/env python3
"""Create balanced error-signal datasets combining multiple analytics."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd
from datasets import load_dataset


ANALYTIC_ROOTS: Mapping[str, Path] = {
    "auc": Path("outputs/analytics/auc"),
    "consistency": Path("outputs/analytics/consistency"),
    "progression": Path("outputs/analytics/progression"),
    "fidelity": Path("outputs/analytics/fidelity"),
}

METHOD_GRAPHS: Mapping[str, Sequence[str]] = {
    # GraphSVX is topology-agnostic: it is also run on the hierarchical graphs so that
    # topology and explainer are not confounded (shared explainer across topologies).
    "graphsvx": ("skipgrams", "window", "constituency", "syntactic"),
    "subgraphx": ("constituency", "syntactic", "window", "skipgrams"),
    "token_shap_llm": ("tokens",),
}
METHOD_ORDER: Sequence[str] = ("graphsvx", "subgraphx", "token_shap_llm")

BASE_DROP_COLUMNS = {
    "method",
    "backbone",
    "dataset",
    "dataset_raw",
    "dataset_backbone",
    "graph_type",
    "run_id",
    "split",
    "graph_index",
    "label",
    "prediction_class",
    "prediction_confidence",
    "is_correct",
}

MODULE_DROP_COLUMNS = {
    "method",
    "backbone",
    "dataset",
    "dataset_raw",
    "dataset_backbone",
    "graph_type",
    "run_id",
    "split",
    "graph_index",
}

MODULE_DROP_COLUMNS = {
    "method",
    "backbone",
    "dataset",
    "dataset_raw",
    "dataset_backbone",
    "graph_type",
    "run_id",
    "split",
    "graph_index",
}


@dataclass(frozen=True)
class DatasetSpec:
    slug: str
    label: str
    hf_id: str
    split: str
    text_field: str
    label_field: str
    label_text_field: str | None = None
    label_names: Mapping[int, str] | None = None


@dataclass(frozen=True)
class ModuleAnalytics:
    key: str
    method: str
    graph: str
    frame: pd.DataFrame


DATASETS: Mapping[str, DatasetSpec] = {
    "setfit_ag_news": DatasetSpec(
        slug="setfit_ag_news",
        label="AG News (SetFit)",
        hf_id="SetFit/ag_news",
        split="test",
        text_field="text",
        label_field="label",
        label_text_field="label_text",
    ),
    "stanfordnlp_sst2": DatasetSpec(
        slug="stanfordnlp_sst2",
        label="SST-2 (StanfordNLP)",
        hf_id="stanfordnlp/sst2",
        split="validation",
        text_field="sentence",
        label_field="label",
        label_text_field=None,
        label_names={0: "negative", 1: "positive"},
    ),
}


def dataset_text_lookup(spec: DatasetSpec) -> Dict[int, Dict[str, str | int | None]]:
    dataset = load_dataset(spec.hf_id, split=spec.split)
    lookup: Dict[int, Dict[str, str | int | None]] = {}
    for idx in range(len(dataset)):
        record = dataset[idx]
        text = record.get(spec.text_field)
        label_value = record.get(spec.label_field)
        label_text = None
        if spec.label_text_field and spec.label_text_field in record:
            label_text = record[spec.label_text_field]
        elif spec.label_names is not None and label_value in spec.label_names:
            label_text = spec.label_names[label_value]
        lookup[idx] = {
            "text": text,
            "label_id": int(label_value) if label_value is not None else None,
            "label_text": label_text,
        }
    return lookup


def analytic_path(kind: str, method: str, dataset: str, graph: str) -> Path:
    path = ANALYTIC_ROOTS[kind] / method / dataset / f"{graph}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {kind} data for {dataset} · {method} · {graph}: {path}")
    return path


def prefix_columns(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    keep_cols = {"global_graph_index"}
    drop_cols = [col for col in BASE_DROP_COLUMNS if col in df.columns]
    trimmed = df.drop(columns=drop_cols, errors="ignore").copy()
    rename_map = {col: f"{prefix}__{col}" for col in trimmed.columns if col not in keep_cols}
    return trimmed.rename(columns=rename_map)


def to_bool(series: pd.Series) -> pd.Series:
    def cast(value):
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, np.integer)):
            return bool(value)
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "1", "yes"}:
                return True
            if lowered in {"false", "0", "no"}:
                return False
        return bool(value)

    return series.apply(cast)


def merge_analytics(dataset: str, method: str, graph: str) -> pd.DataFrame:
    fidelity_df = pd.read_csv(analytic_path("fidelity", method, dataset, graph))
    fidelity_df = fidelity_df.copy()
    fidelity_df["global_graph_index"] = fidelity_df["global_graph_index"].astype(int)

    merged = fidelity_df
    for kind in ("consistency", "progression", "auc"):
        df = pd.read_csv(analytic_path(kind, method, dataset, graph))
        df = df.copy()
        df["global_graph_index"] = df["global_graph_index"].astype(int)
        prefixed = prefix_columns(df, kind)
        merged = merged.merge(prefixed, on="global_graph_index", how="inner", validate="one_to_one")

    merged["is_correct"] = to_bool(merged["is_correct"])
    return merged


def collect_module_frames(dataset_slug: str) -> List[ModuleAnalytics]:
    frames: List[ModuleAnalytics] = []
    for method in METHOD_ORDER:
        graphs = METHOD_GRAPHS.get(method, ())
        for graph in graphs:
            fidelity_path = ANALYTIC_ROOTS["fidelity"] / method / dataset_slug / f"{graph}.csv"
            if not fidelity_path.exists():
                continue
            merged = merge_analytics(dataset_slug, method, graph)
            merged = merged.drop(columns=list(MODULE_DROP_COLUMNS & set(merged.columns)), errors="ignore")
            module_key = f"{method}_{graph}"
            prefixed = merged.rename(
                columns={col: f"{module_key}__{col}" for col in merged.columns if col != "global_graph_index"}
            )
            frames.append(ModuleAnalytics(module_key, method, graph, prefixed))
    return frames


def combine_modules(frames: List[ModuleAnalytics]) -> pd.DataFrame:
    if not frames:
        raise RuntimeError("No analytics modules available to combine.")
    combined = frames[0].frame
    for module in frames[1:]:
        combined = combined.merge(module.frame, on="global_graph_index", how="inner")
    return combined


def add_dataset_metadata(
    df: pd.DataFrame,
    spec: DatasetSpec,
    lookup: Mapping[int, Mapping[str, str | int | None]],
) -> pd.DataFrame:
    df = df.copy()
    df["dataset_slug"] = spec.slug
    df["dataset_label"] = spec.label
    df["global_graph_index"] = df["global_graph_index"].astype(int)
    df["text"] = df["global_graph_index"].map(lambda idx: lookup.get(int(idx), {}).get("text"))
    df["label_text"] = df["global_graph_index"].map(lambda idx: lookup.get(int(idx), {}).get("label_text"))
    df["label_id"] = df["global_graph_index"].map(lambda idx: lookup.get(int(idx), {}).get("label_id"))
    return df


def derive_sample_groups(df: pd.DataFrame, dataset_slug: str, seed: int) -> pd.DataFrame:
    is_correct_cols = [col for col in df.columns if col.endswith("__is_correct")]
    if not is_correct_cols:
        raise RuntimeError(f"No correctness columns present for dataset '{dataset_slug}'.")

    correctness_frames = [to_bool(df[col]) for col in is_correct_cols]
    all_correct_mask = np.logical_and.reduce(correctness_frames)
    all_wrong_mask = np.logical_and.reduce([~col for col in correctness_frames])

    wrong = df[all_wrong_mask].copy()
    wrong["sample_group"] = "all_modules_wrong"

    if wrong.empty:
        return wrong

    correct_pool = df[all_correct_mask].copy()
    if len(correct_pool) < len(wrong):
        raise RuntimeError(
            f"Dataset '{dataset_slug}': only {len(correct_pool)} all-correct sentences for {len(wrong)} all-wrong cases."
        )

    rng_seed = abs(hash((dataset_slug, seed))) % (2**32)
    correct_sample = correct_pool.sample(n=len(wrong), random_state=rng_seed, replace=False)
    correct_sample["sample_group"] = "all_modules_correct"

    return pd.concat([wrong, correct_sample], ignore_index=True)


def expand_modules(df: pd.DataFrame, modules: List[ModuleAnalytics]) -> pd.DataFrame:
    base_cols = [
        "dataset_slug",
        "dataset_label",
        "global_graph_index",
        "label_id",
        "label_text",
        "text",
        "sample_group",
    ]
    expanded: List[pd.DataFrame] = []
    for module in modules:
        module_cols = [col for col in df.columns if col.startswith(f"{module.key}__")]
        if not module_cols:
            continue
        module_df = df[base_cols].copy()
        metrics = df[module_cols].copy()
        metrics = metrics.rename(columns={col: col.split("__", 1)[1] for col in module_cols})
        module_df["method"] = module.method
        module_df["graph_type"] = module.graph
        module_df["module_id"] = f"{module.method}:{module.graph}"
        module_df = pd.concat([module_df.reset_index(drop=True), metrics.reset_index(drop=True)], axis=1)
        expanded.append(module_df)
    if not expanded:
        raise RuntimeError("No module metrics available to expand.")
    return pd.concat(expanded, ignore_index=True)


def process_dataset(dataset_slug: str, output_dir: Path, seed: int) -> Path:
    if dataset_slug not in DATASETS:
        raise KeyError(f"Unknown dataset slug: {dataset_slug}")

    spec = DATASETS[dataset_slug]
    lookup = dataset_text_lookup(spec)

    module_frames = collect_module_frames(dataset_slug)
    combined_modules = combine_modules(module_frames)
    combined = add_dataset_metadata(combined_modules, spec, lookup)
    combined = derive_sample_groups(combined, dataset_slug, seed)
    combined = expand_modules(combined, module_frames)

    front_cols = [
        "dataset_slug",
        "dataset_label",
        "global_graph_index",
        "label_id",
        "label_text",
        "text",
        "sample_group",
        "method",
        "graph_type",
        "module_id",
    ]
    ordered_cols = front_cols + [col for col in combined.columns if col not in front_cols]
    combined = combined.loc[:, ordered_cols]

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"error_signal_samples_{dataset_slug}.csv"
    combined.to_csv(output_path, index=False)
    print(f"✓ {dataset_slug}: wrote {len(combined)} rows -> {output_path}")
    return output_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build balanced error-signal datasets from analytics outputs.")
    parser.add_argument(
        "--dataset",
        action="append",
        choices=sorted(DATASETS.keys()),
        help="Dataset slug(s) to process (default: all).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/use_case"),
        help="Directory where the aggregated CSVs will be stored.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for correct-sample selection.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    datasets = args.dataset or sorted(DATASETS.keys())
    for dataset_slug in datasets:
        process_dataset(dataset_slug, args.output_dir, args.seed)


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv[1:])
