#!/usr/bin/env python3
"""Fit logistic models on top-performing modules and persist their coefficients."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from use_case.feature_config import (  # type: ignore
    filter_allowed_features,
    dimension_for_feature as configured_dimension,
)


SUMMARY_PATH = Path("outputs/use_case/module_datasets/error_signal_classification_summary.csv")
DATASET_ROOT = Path("outputs/use_case/module_datasets")
OUTPUT_DIR = DATASET_ROOT / "coefficients"


def load_summary(summary_path: Path) -> pd.DataFrame | None:
    if not summary_path.exists():
        return None
    df = pd.read_csv(summary_path)
    if df.empty:
        return None
    return df


def _filter_feature_columns(columns: List[str]) -> List[str]:
    return filter_allowed_features(columns)


def feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, List[str], np.ndarray]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    drop_cols = {
        "global_graph_index",
        "graph_index",
        "label",
        "label_id",
        "prediction_class",
        "prediction_confidence",
        "is_correct",
    }
    feature_cols = [col for col in numeric_cols if col not in drop_cols]
    feature_cols = _filter_feature_columns(feature_cols)
    if not feature_cols:
        raise ValueError("No numeric analytic features available.")
    X = df[feature_cols].to_numpy(copy=True)
    y = df["is_correct"].astype(int).to_numpy(copy=True)
    if len(np.unique(y)) < 2:
        raise ValueError("Target variable has a single class; cannot fit logistic regression.")
    return X, feature_cols, y


def fit_logistic(X: np.ndarray, y: np.ndarray) -> LogisticRegression:
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, class_weight="balanced"))
    pipe.fit(X, y)
    return pipe


def extract_coefficients(pipe, feature_names: List[str]) -> pd.DataFrame:
    lr: LogisticRegression = pipe.named_steps["logisticregression"]
    coef = lr.coef_.ravel()
    intercept = lr.intercept_[0]
    coeff_df = pd.DataFrame({"feature": feature_names, "coefficient": coef})
    coeff_df = coeff_df.sort_values("coefficient", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)
    coeff_df.attrs["intercept"] = intercept
    return coeff_df


def save_coefficients(coeff_df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    coeff_df.to_csv(output_path, index=False)
    intercept_path = output_path.with_suffix(".intercept.txt")
    intercept_path.write_text(f"intercept,{coeff_df.attrs.get('intercept', 0.0):.10f}\n")


def evaluate_accuracy(X: np.ndarray, y: np.ndarray, max_splits: int) -> tuple[float, float, int]:
    class_counts = np.bincount(y)
    min_class = class_counts[class_counts > 0].min() if class_counts.size else 0
    n_splits = min(max_splits, int(min_class))
    if n_splits < 2:
        raise ValueError("Not enough samples per class for cross-validation.")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, class_weight="balanced"))
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    return scores.mean(), scores.std(), n_splits


def bootstrap_accuracy(
    X: np.ndarray,
    y: np.ndarray,
    *,
    repetitions: int,
    max_splits: int,
    rng_seed: int = 42,
    desc: str = "bootstrap",
) -> tuple[float, float, int]:
    if repetitions <= 0:
        return float("nan"), float("nan"), 0
    rng = np.random.default_rng(rng_seed)
    scores: List[float] = []
    iterator = tqdm(range(repetitions), desc=desc, leave=False)
    for _ in iterator:
        indices = rng.integers(0, len(y), len(y))
        sample_y = y[indices]
        if len(np.unique(sample_y)) < 2:
            continue
        sample_X = X[indices]
        try:
            score, _, _ = evaluate_accuracy(sample_X, sample_y, max_splits)
        except ValueError:
            continue
        scores.append(score)
    if not scores:
        return float("nan"), float("nan"), 0
    return float(np.mean(scores)), float(np.std(scores, ddof=1) if len(scores) > 1 else 0.0), len(scores)


def bootstrap_coefficients(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    *,
    repetitions: int,
    rng_seed: int = 42,
    desc: str = "bootstrap coefficients",
) -> pd.DataFrame:
    if repetitions <= 0:
        return pd.DataFrame()

    rng = np.random.default_rng(rng_seed)
    coef_matrix: List[np.ndarray] = []
    intercepts: List[float] = []
    iterator = tqdm(range(repetitions), desc=desc, leave=False)
    for _ in iterator:
        indices = rng.integers(0, len(y), len(y))
        sample_y = y[indices]
        if len(np.unique(sample_y)) < 2:
            continue
        sample_X = X[indices]
        try:
            pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, class_weight="balanced"))
            pipe.fit(sample_X, sample_y)
            lr = pipe.named_steps["logisticregression"]
            coef_matrix.append(lr.coef_.ravel())
            intercepts.append(lr.intercept_[0])
        except Exception:
            continue

    if not coef_matrix:
        return pd.DataFrame()

    coef_array = np.asarray(coef_matrix)
    n_boot = len(coef_matrix)
    coef_means = coef_array.mean(axis=0)
    coef_stds = coef_array.std(axis=0, ddof=1) if n_boot > 1 else np.zeros_like(coef_means)
    coef_se = coef_stds / np.sqrt(n_boot) if n_boot > 0 else np.zeros_like(coef_means)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_stats = np.where(coef_se != 0, coef_means / coef_se, np.nan)
    ci_lower = np.percentile(coef_array, 2.5, axis=0)
    ci_upper = np.percentile(coef_array, 97.5, axis=0)
    significant = ~((ci_lower <= 0) & (ci_upper >= 0))

    results = pd.DataFrame(
        {
            "feature": feature_names,
            "coef_mean": coef_means,
            "coef_std": coef_stds,
            "coef_se": coef_se,
            "t_statistic": t_stats,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "significant_95": significant,
            "n_bootstraps": n_boot,
        }
    )
    results = results.iloc[np.abs(results["t_statistic"]).argsort(kind="mergesort")[::-1]].reset_index(drop=True)

    if intercepts:
        intercept_mean = float(np.mean(intercepts))
        intercept_std = float(np.std(intercepts, ddof=1) if len(intercepts) > 1 else 0.0)
        intercept_se = intercept_std / np.sqrt(len(intercepts)) if len(intercepts) > 0 else float("nan")
        intercept_t = intercept_mean / intercept_se if intercept_se not in (0.0, float("nan")) else float("nan")
        intercept_ci_lower = float(np.percentile(intercepts, 2.5))
        intercept_ci_upper = float(np.percentile(intercepts, 97.5))
        results.attrs["intercept_mean"] = intercept_mean
        results.attrs["intercept_std"] = intercept_std
        results.attrs["intercept_se"] = intercept_se
        results.attrs["intercept_t_stat"] = intercept_t
        results.attrs["intercept_ci_lower"] = intercept_ci_lower
        results.attrs["intercept_ci_upper"] = intercept_ci_upper

    return results


def _discover_module_files() -> List[tuple[str, Path, str, str]]:
    entries: List[tuple[str, Path, str, str]] = []
    for dataset_dir in DATASET_ROOT.iterdir():
        if not dataset_dir.is_dir() or dataset_dir.name == "coefficients":
            continue
        dataset = dataset_dir.name
        prefix = f"module_dataset_{dataset}_"
        for csv_path in dataset_dir.glob("module_dataset_*.csv"):
            stem = csv_path.stem
            if not stem.startswith(prefix):
                continue
            remainder = stem.replace(prefix, "", 1)
            if "_" not in remainder:
                continue
            method, graph = remainder.rsplit("_", 1)
            entries.append((dataset, csv_path, method, graph))
    return entries


def _process_subset(
    dataset: str,
    method: str,
    graph: str,
    label_name: str,
    subset: pd.DataFrame,
    output_dir: Path,
    *,
    max_splits: int,
    bootstrap_reps: int,
) -> tuple[dict, pd.DataFrame, Path, pd.DataFrame]:
    X, feature_cols, y = feature_matrix(subset)
    try:
        acc_mean, acc_std, splits = evaluate_accuracy(X, y, max_splits)
    except ValueError as exc:
        print(f"Skipping CV for {dataset}-{method}-{graph}-{label_name}: {exc}")
        acc_mean = acc_std = 0.0
        splits = 0
    boot_mean = float("nan")
    boot_std = float("nan")
    boot_samples = 0
    if bootstrap_reps > 0:
        boot_mean, boot_std, boot_samples = bootstrap_accuracy(
            X,
            y,
            repetitions=bootstrap_reps,
            max_splits=max_splits,
            desc=f"bootstrap {dataset}-{method}-{graph}-{label_name}",
        )
    bootstrap_coef_df: pd.DataFrame | None = None
    if bootstrap_reps > 0:
        bootstrap_coef_df = bootstrap_coefficients(
            X,
            y,
            feature_cols,
            repetitions=bootstrap_reps,
            desc=f"bootstrap coefficients {dataset}-{method}-{graph}-{label_name}",
        )
    pipeline = fit_logistic(X, y)
    coeff_df = extract_coefficients(pipeline, feature_cols)
    coeff_path = (
        output_dir
        / dataset
        / f"logistic_coefficients_{dataset}_{method}_{graph}_label{label_name}.csv"
    )
    save_coefficients(coeff_df, coeff_path)
    if bootstrap_coef_df is not None and not bootstrap_coef_df.empty:
        bootstrap_path = (
            output_dir
            / dataset
            / f"bootstrap_coefficients_{dataset}_{method}_{graph}_label{label_name}.csv"
        )
        bootstrap_path.parent.mkdir(parents=True, exist_ok=True)
        bootstrap_coef_df.to_csv(bootstrap_path, index=False)
        intercept_boot_path = bootstrap_path.with_suffix(".intercept_bootstrap.txt")
        if bootstrap_coef_df.attrs:
            intercept_boot_path.write_text(
                "metric,value\n"
                + "\n".join(
                    f"{key},{bootstrap_coef_df.attrs.get(key, float('nan')):.10f}"
                    for key in [
                        "intercept_mean",
                        "intercept_std",
                        "intercept_se",
                        "intercept_t_stat",
                        "intercept_ci_lower",
                        "intercept_ci_upper",
                    ]
                    if key in bootstrap_coef_df.attrs
                )
                + "\n"
            )
    summary_row = {
        "dataset": dataset,
        "method": method,
        "graph": graph,
        "label": label_name,
        "n": len(subset),
        "accuracy_mean": acc_mean,
        "accuracy_std": acc_std,
        "splits": splits,
        "bootstrap_mean": boot_mean,
        "bootstrap_std": boot_std,
        "bootstrap_samples": boot_samples,
    }
    coeff_df["dimension"] = coeff_df["feature"].map(configured_dimension)
    agg = coeff_df.groupby("dimension")["coefficient"].apply(lambda s: s.abs().sum()).reset_index(
        name="abs_weight_sum"
    )
    total = agg["abs_weight_sum"].sum()
    if total == 0:
        agg["weight_pct"] = 0.0
    else:
        agg["weight_pct"] = 100 * agg["abs_weight_sum"] / total
    agg["dataset"] = dataset
    agg["method"] = method
    agg["graph"] = graph
    agg["label"] = label_name
    coeff_with_meta = coeff_df.rename(columns={"coefficient": "coefficient_point"}).copy()
    coeff_with_meta.insert(0, "label", label_name)
    coeff_with_meta.insert(0, "graph", graph)
    coeff_with_meta.insert(0, "method", method)
    coeff_with_meta.insert(0, "dataset", dataset)
    if bootstrap_coef_df is not None and not bootstrap_coef_df.empty:
        bootstrap_meta = bootstrap_coef_df.copy()
        bootstrap_meta.insert(0, "label", label_name)
        bootstrap_meta.insert(0, "graph", graph)
        bootstrap_meta.insert(0, "method", method)
        bootstrap_meta.insert(0, "dataset", dataset)
        coeff_with_meta = coeff_with_meta.merge(
            bootstrap_meta,
            on=["dataset", "method", "graph", "label", "feature"],
            how="left",
        )
    return summary_row, agg, coeff_path, coeff_with_meta


def process_modules(
    output_dir: Path,
    *,
    max_splits: int,
    bootstrap_reps: int,
    stratify_by: str = "prediction_class",
) -> tuple[Dict[Tuple[str, str, str, str], Path], pd.DataFrame]:
    """Fit one logistic model per (dataset, module, graph, stratum) plus a pooled model.

    ``stratify_by`` must be a quantity available at inference time.  The original
    submission stratified by ``label`` (the true class); for error detection that leaks
    the target, because within true class *c* an error is simply "predicted another
    class", which class-dependent explanation features reveal.  The default therefore
    stratifies by the predicted class.
    """
    results: Dict[Tuple[str, str, str, str], Path] = {}
    summary_rows: List[dict] = []
    dimension_records: List[pd.DataFrame] = []
    dataset_coefficients: Dict[str, List[pd.DataFrame]] = defaultdict(list)

    module_files = _discover_module_files()
    if not module_files:
        raise RuntimeError("No module datasets found.")

    label_map: Dict[tuple[str, Path, str, str], List[float]] = {}
    total_tasks = 0
    for entry in module_files:
        dataset, csv_path, method, graph = entry
        try:
            label_series = pd.read_csv(csv_path, usecols=[stratify_by])[stratify_by]
        except ValueError:
            df_labels = pd.read_csv(csv_path)
            if stratify_by not in df_labels.columns:
                label_values: List[float] = []
            else:
                label_values = df_labels[stratify_by].dropna().unique().tolist()
        else:
            label_values = label_series.dropna().unique().tolist()
        label_values = sorted(label_values)
        label_map[entry] = label_values
        if label_values:
            total_tasks += len(label_values) + 1

    progress = tqdm(total=total_tasks, desc="Modules", leave=True) if total_tasks else None

    for entry in module_files:
        dataset, csv_path, method, graph = entry
        label_values = label_map.get(entry, [])
        df = pd.read_csv(csv_path)
        if df.empty or not label_values:
            continue
        label_rows: List[dict] = []
        processed_overall = False
        for label_value in label_values:
            subset = df[df[stratify_by] == label_value]
            if subset.empty:
                if progress:
                    progress.update(1)
                continue
            try:
                summary_row, agg, coeff_path, coeff_meta = _process_subset(
                    dataset,
                    method,
                    graph,
                    str(int(label_value)),
                    subset,
                    output_dir,
                    max_splits=max_splits,
                    bootstrap_reps=bootstrap_reps,
                )
            except ValueError as exc:
                print(f"Skipping {dataset}-{method}-{graph}-label{label_value}: {exc}")
            else:
                key = (dataset, method, graph, str(int(label_value)))
                results[key] = coeff_path
                summary_rows.append(summary_row)
                label_rows.append(summary_row)
                dimension_records.append(agg)
                dataset_coefficients[dataset].append(coeff_meta)
                processed_overall = True
            finally:
                if progress:
                    progress.update(1)

        if processed_overall:
            try:
                summary_row, agg, coeff_path, coeff_meta = _process_subset(
                    dataset,
                    method,
                    graph,
                    "all",
                    df,
                    output_dir,
                    max_splits=max_splits,
                    bootstrap_reps=bootstrap_reps,
                )
                key = (dataset, method, graph, "all")
                results[key] = coeff_path
                if label_rows:
                    total_n = sum(row["n"] for row in label_rows if row["n"] is not None)
                    summary_row["pooled_model_accuracy_mean"] = summary_row["accuracy_mean"]
                    summary_row["pooled_model_accuracy_std"] = summary_row["accuracy_std"]
                    if total_n:
                        weighted_acc = sum(row["n"] * row["accuracy_mean"] for row in label_rows) / total_n
                        weighted_var = (
                            sum(
                                row["n"]
                                * (
                                    (row["accuracy_std"] if not pd.isna(row["accuracy_std"]) else 0.0) ** 2
                                    + (row["accuracy_mean"] - weighted_acc) ** 2
                                )
                                for row in label_rows
                                if row["n"]
                            )
                            / total_n
                        )
                        summary_row["accuracy_mean"] = weighted_acc
                        summary_row["accuracy_std"] = math.sqrt(max(weighted_var, 0.0))
                        summary_row["n"] = total_n
                    valid_splits = [row["splits"] for row in label_rows if row["splits"] > 0]
                    if valid_splits:
                        summary_row["splits"] = min(valid_splits)
                    bootstrap_rows = [
                        row for row in label_rows if not math.isnan(row.get("bootstrap_mean", float("nan")))
                    ]
                    if bootstrap_rows:
                        total_bootstrap_n = sum(row["n"] for row in bootstrap_rows if row["n"])
                        if total_bootstrap_n:
                            weighted_boot = (
                                sum(row["n"] * row["bootstrap_mean"] for row in bootstrap_rows) / total_bootstrap_n
                            )
                            weighted_boot_var = (
                                sum(
                                    row["n"]
                                    * (
                                        (row["bootstrap_std"] if not pd.isna(row["bootstrap_std"]) else 0.0) ** 2
                                        + (row["bootstrap_mean"] - weighted_boot) ** 2
                                    )
                                    for row in bootstrap_rows
                                    if row["n"]
                                )
                                / total_bootstrap_n
                            )
                            summary_row["bootstrap_mean"] = weighted_boot
                            summary_row["bootstrap_std"] = math.sqrt(max(weighted_boot_var, 0.0))
                            summary_row["bootstrap_samples"] = sum(
                                row.get("bootstrap_samples", 0) for row in bootstrap_rows
                            )
                summary_rows.append(summary_row)
                dimension_records.append(agg)
                dataset_coefficients[dataset].append(coeff_meta)
            except ValueError as exc:
                print(f"Skipping overall {dataset}-{method}-{graph}: {exc}")
            finally:
                if progress:
                    progress.update(1)

    if progress:
        progress.close()

    if not summary_rows:
        raise RuntimeError("No module datasets processed; summary empty.")

    summary_df = pd.DataFrame(summary_rows)
    summary_df["stratify_by"] = stratify_by
    summary_df.to_csv(SUMMARY_PATH, index=False)

    if dimension_records:
        summary_dim = pd.concat(dimension_records, ignore_index=True)
        summary_path = output_dir / "dimension_weight_summary.csv"
        summary_dim.to_csv(summary_path, index=False)
    else:
        summary_path = output_dir / "dimension_weight_summary.csv"
        summary_path.write_text("dimension,abs_weight_sum,weight_pct,dataset,method,graph,label\n")

    for dataset, frames in dataset_coefficients.items():
        if not frames:
            continue
        combined = pd.concat(frames, ignore_index=True)
        combined_path = output_dir / dataset / f"all_logistic_coefficients_{dataset}.csv"
        combined_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(combined_path, index=False)

    return results, summary_df
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Save logistic coefficients for best-performing modules per dataset.")
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=SUMMARY_PATH,
        help="Path to the summary CSV containing accuracy results.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory where coefficient files will be written.",
    )
    parser.add_argument(
        "--cv-splits",
        type=int,
        default=10,
        help="Maximum number of stratified CV folds to use (default: 10).",
    )
    parser.add_argument(
        "--bootstrap-reps",
        type=int,
        default=200,
        help="Number of bootstrap repetitions per module subset (default: 200).",
    )
    parser.add_argument(
        "--stratify-by",
        choices=("prediction_class", "label"),
        default="prediction_class",
        help="Column defining the per-class models. 'label' (true class) reproduces the original submission but leaks the target; default: predicted class.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results, summary_df = process_modules(
        args.output_dir,
        max_splits=args.cv_splits,
        bootstrap_reps=args.bootstrap_reps,
        stratify_by=args.stratify_by,
    )
    if not results:
        raise SystemExit("No coefficient files were produced.")
    print("Saved coefficient tables:")
    for (dataset, method, graph, label), path in results.items():
        print(f"  - {dataset}:{method}:{graph}:label{label} -> {path}")
    print(f"Classification summary -> {SUMMARY_PATH}")
    print(f"Dimension weight summary -> {args.output_dir / 'dimension_weight_summary.csv'}")


if __name__ == "__main__":  # pragma: no cover
    main()
