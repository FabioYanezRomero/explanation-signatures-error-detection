import argparse
import pickle
import sys
import types
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


def slugify(value: str) -> str:
    safe = value.replace("/", "_").replace(" ", "_").replace("-", "_")
    return "".join(char for char in safe if char.isalnum() or char == "_").lower()


def normalize_dataset_name(raw: str) -> str:
    name = raw.replace("_", "-")
    if name.lower() == "sst2":
        return "sst-2"
    return name


def infer_split_from_run_id(run_id: Optional[str]) -> Optional[str]:
    if not run_id:
        return None
    tokens = run_id.lower()
    for candidate in ("train", "test", "validation", "val", "dev"):
        if candidate in tokens:
            if candidate == "val":
                return "validation"
            if candidate == "dev":
                return "development"
            return candidate
    return None


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


def format_sequence(values: Sequence[Any]) -> str:
    if not values:
        return ""
    return ",".join(str(value) for value in values)


def format_float_sequence(values: Sequence[float]) -> str:
    if not values:
        return ""
    return ",".join(f"{value:.6f}" for value in values)


def format_token_sequence(tokens: Sequence[str], indices: Sequence[int]) -> str:
    if not tokens or not indices:
        return ""
    chosen: List[str] = []
    token_count = len(tokens)
    for index in indices:
        try:
            idx = int(index)
        except Exception:
            continue
        if 0 <= idx < token_count:
            chosen.append(str(tokens[idx]))
    return ",".join(chosen)


def parse_metadata_from_path(path: Path) -> Dict[str, str]:
    parts = path.resolve().parts
    if "gnn_models" not in parts:
        raise ValueError(f"Unexpected path layout for {path}")
    idx = parts.index("gnn_models")
    try:
        backbone = parts[idx + 1]
        dataset_raw = parts[idx + 2]
        graph_type = parts[idx + 3]
        method = parts[idx + 5]
        run_id = parts[idx + 6]
    except IndexError as exc:
        raise ValueError(f"Path too short to extract metadata: {path}") from exc
    dataset = normalize_dataset_name(dataset_raw)
    dataset_backbone = f"{backbone}/{dataset_raw}"
    split = infer_split_from_run_id(run_id) or ""
    return {
        "backbone": backbone,
        "dataset_raw": dataset_raw,
        "dataset": dataset,
        "dataset_backbone": dataset_backbone,
        "graph_type": graph_type,
        "method": method,
        "run_id": run_id,
        "split": split,
    }


def _coerce_float_list(values: Optional[Iterable[Any]]) -> List[float]:
    if values is None:
        return []
    if isinstance(values, np.ndarray):
        values = values.tolist()
    floats: List[float] = []
    for value in values:
        try:
            floats.append(float(value))
        except Exception:
            continue
    return floats


def _extract_ranked_nodes(payload: Dict[str, Any]) -> List[int]:
    ranked: Iterable[Any] = ()
    related_prediction = payload.get("related_prediction") or {}
    ranked = related_prediction.get("ranked_nodes") or ()

    explanation = payload.get("explanation")
    if not ranked and isinstance(explanation, dict):
        ranked = explanation.get("node_ranking") or explanation.get("ranked_nodes") or ()
        if not ranked:
            raw_scores = explanation.get("node_importance")
            if isinstance(raw_scores, np.ndarray):
                scores = raw_scores.tolist()
                ranked = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)
            elif isinstance(raw_scores, (list, tuple)):
                scores = list(raw_scores)
                ranked = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)
    if not ranked and isinstance(explanation, list):
        scored_nodes: Dict[int, float] = {}
        for entry in explanation:
            if not isinstance(entry, dict):
                continue
            coalition = entry.get("coalition") or []
            weight = entry.get("P")
            if weight is None:
                weight = entry.get("W")
            if weight is None:
                continue
            try:
                weight_val = float(weight)
            except Exception:
                continue
            for node in coalition:
                try:
                    node_id = int(node)
                except Exception:
                    continue
                current = scored_nodes.get(node_id)
                if current is None or weight_val > current:
                    scored_nodes[node_id] = weight_val
        if scored_nodes:
            ranked = sorted(scored_nodes.keys(), key=lambda node: scored_nodes[node], reverse=True)

    indices: List[int] = []
    seen: set[int] = set()
    for value in ranked:
        try:
            idx = int(value)
        except Exception:
            continue
        if idx in seen:
            continue
        seen.add(idx)
        indices.append(idx)
    return indices


def _extract_node_scores(payload: Dict[str, Any], method: str, ranked_nodes: Sequence[int]) -> List[float]:
    explanation = payload.get("explanation")
    scores: Dict[int, float] = {}

    if isinstance(explanation, dict):
        raw_scores = explanation.get("node_importance")
        if raw_scores is not None:
            values = _coerce_float_list(raw_scores)
            return [values[idx] if 0 <= idx < len(values) else 0.0 for idx in ranked_nodes]

    if isinstance(explanation, list):
        for entry in explanation:
            if not isinstance(entry, dict):
                continue
            coalition = entry.get("coalition") or []
            weight = entry.get("P")
            if weight is None:
                weight = entry.get("W")
            if weight is None:
                continue
            try:
                weight_val = float(weight)
            except Exception:
                continue
            for node in coalition:
                try:
                    node_id = int(node)
                except Exception:
                    continue
                current = scores.get(node_id)
                if current is None or weight_val > current:
                    scores[node_id] = weight_val

    related_prediction = payload.get("related_prediction") or {}
    if not scores and isinstance(related_prediction, dict):
        ranked_values = related_prediction.get("ranked_scores")
        if isinstance(ranked_values, (list, tuple)):
            try:
                scores = {int(idx): float(val) for idx, val in enumerate(ranked_values)}
            except Exception:
                scores = {}

    node_scores = [scores.get(idx, 0.0) for idx in ranked_nodes]
    unique_scores = {round(score, 6) for score in node_scores}
    if node_scores and len(unique_scores) <= 1:
        total = len(node_scores)
        if total:
            node_scores = [(total - idx) / float(total) for idx in range(total)]
    return node_scores


def _sanitize_indices(values: Optional[Iterable[Any]]) -> List[int]:
    if values is None:
        return []
    indices: List[int] = []
    seen: set[int] = set()
    for value in values:
        try:
            idx = int(value)
        except Exception:
            continue
        if idx in seen:
            continue
        seen.add(idx)
        indices.append(idx)
    return indices


def discover_pickles(base_dir: Path, method: str) -> Sequence[Path]:
    pattern = f"**/explanations/{method}/**/results_split_pickle/graph_*.pkl"
    return sorted(base_dir.glob(pattern))


def load_payload(path: Path, method: str) -> Dict[str, Any]:
    if method == "subgraphx":
        ensure_subgraphx_stub()
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if isinstance(payload, dict):
        return payload
    if hasattr(payload, "__dict__"):
        return dict(payload.__dict__)
    raise ValueError(f"Unsupported {method} payload at {path}")


@dataclass
class NodeTextBundle:
    node_names: List[Any]
    node_text: List[str]


class GraphTokenResolver:
    def __init__(self, graph_root: Path, pyg_root: Path) -> None:
        self.graph_root = Path(graph_root)
        self.pyg_root = Path(pyg_root)
        self._nx_chunk_size: Dict[Path, int] = {}
        self._nx_cache: Dict[Tuple[Path, int], List[nx.DiGraph]] = {}
        self._pyg_chunk_size: Dict[Path, int] = {}
        self._pyg_cache: Dict[Tuple[Path, int], List] = {}

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def resolve(
        self,
        backbone: str,
        dataset_raw: str,
        split: str,
        graph_type: str,
        graph_index: int,
    ) -> NodeTextBundle:
        graph = self._load_networkx_graph(backbone, dataset_raw, split, graph_type, graph_index)
        node_names = self._load_node_names(backbone, dataset_raw, split, graph_type, graph_index)
        node_text = self._extract_node_text(graph, node_names)
        return NodeTextBundle(node_names=node_names, node_text=node_text)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _resolve_graph_variant(self, base: Path, graph_type: str) -> Path:
        direct = base / graph_type
        if direct.exists():
            return direct

        candidates = [
            path for path in base.iterdir() if path.is_dir() and path.name.startswith(graph_type)
        ]
        if not candidates:
            raise FileNotFoundError(f"No graph variant matching '{graph_type}' under {base}")
        candidates.sort(key=lambda p: (len(p.name), p.name))
        return candidates[0]

    def _load_networkx_graph(
        self,
        backbone: str,
        dataset_raw: str,
        split: str,
        graph_type: str,
        graph_index: int,
    ) -> nx.DiGraph:
        base = self.graph_root / backbone / dataset_raw / split
        variant_dir = self._resolve_graph_variant(base, graph_type)
        chunk_size = self._nx_chunk_size.get(variant_dir)
        if chunk_size is None:
            chunk_size = self._infer_nx_chunk_size(variant_dir)
            self._nx_chunk_size[variant_dir] = chunk_size
        chunk_idx = graph_index // chunk_size
        offset = graph_index % chunk_size
        graphs = self._nx_chunk(variant_dir, chunk_idx)
        try:
            return graphs[offset]
        except IndexError as exc:  # pragma: no cover - defensive guard
            raise IndexError(f"Graph index {graph_index} out of bounds for {variant_dir}") from exc

    def _infer_nx_chunk_size(self, directory: Path) -> int:
        first = self._first_file(directory, "*.pkl")
        graphs = self._extract_graphs(first)
        if not graphs:
            raise ValueError(f"No graphs stored in {first}")
        return len(graphs)

    def _nx_chunk(self, directory: Path, chunk_idx: int) -> List[nx.DiGraph]:
        key = (directory, chunk_idx)
        cached = self._nx_cache.get(key)
        if cached is not None:
            return cached
        file_path = directory / f"{chunk_idx}.pkl"
        graphs = self._extract_graphs(file_path)
        self._nx_cache[key] = graphs
        if len(self._nx_cache) > 8:  # Simple LRU to cap memory
            self._nx_cache.pop(next(iter(self._nx_cache)))
        return graphs

    def _extract_graphs(self, file_path: Path) -> List[nx.DiGraph]:
        with file_path.open("rb") as handle:
            payload = pickle.load(handle)

        graphs: List[nx.DiGraph] = []
        if isinstance(payload, list):
            for item in payload:
                if isinstance(item, tuple) and item:
                    maybe_graphs = item[0]
                    if isinstance(maybe_graphs, list):
                        graphs.extend(maybe_graphs)
                elif isinstance(item, list):
                    graphs.extend(item)
        elif isinstance(payload, tuple):
            first = payload[0]
            if isinstance(first, list):
                graphs.extend(first)
        else:  # pragma: no cover - unexpected payload
            raise TypeError(f"Unexpected graph payload in {file_path}: {type(payload)}")
        return graphs

    def _load_node_names(
        self,
        backbone: str,
        dataset_raw: str,
        split: str,
        graph_type: str,
        graph_index: int,
    ) -> List[Any]:
        base = self.pyg_root / backbone / dataset_raw / split / graph_type
        if not base.exists():
            raise FileNotFoundError(f"PyG directory not found: {base}")
        chunk_size = self._pyg_chunk_size.get(base)
        if chunk_size is None:
            chunk_size = self._infer_pyg_chunk_size(base)
            self._pyg_chunk_size[base] = chunk_size
        chunk_idx = graph_index // chunk_size
        offset = graph_index % chunk_size
        data_list = self._pyg_chunk(base, chunk_idx)
        try:
            item = data_list[offset]
        except IndexError as exc:  # pragma: no cover - defensive guard
            raise IndexError(f"Graph index {graph_index} out of bounds for {base}") from exc
        return list(getattr(item, "nx_node_names", []))

    def _infer_pyg_chunk_size(self, directory: Path) -> int:
        first = self._first_file(directory, "*.pt")
        data_list = torch.load(first)
        if not data_list:
            raise ValueError(f"No PyG entries in {first}")
        return len(data_list)

    def _pyg_chunk(self, directory: Path, chunk_idx: int):
        key = (directory, chunk_idx)
        cached = self._pyg_cache.get(key)
        if cached is not None:
            return cached
        file_path = directory / f"{chunk_idx:05d}.pt"
        data_list = torch.load(file_path)
        self._pyg_cache[key] = data_list
        if len(self._pyg_cache) > 8:
            self._pyg_cache.pop(next(iter(self._pyg_cache)))
        return data_list

    def _first_file(self, directory: Path, pattern: str) -> Path:
        matches = sorted(directory.glob(pattern))
        if not matches:
            raise FileNotFoundError(f"No files matching {pattern} under {directory}")
        return matches[0]

    def _extract_node_text(self, graph: nx.DiGraph, node_names: Sequence[Any]) -> List[str]:
        text: List[str] = []
        for name in node_names:
            attrs = graph.nodes.get(name, {})
            token = (
                attrs.get("text")
                or attrs.get("label")
                or attrs.get("token")
                or attrs.get("word")
                or name
            )
            text.append(str(token))
        return text


def build_record(
    payload: Dict[str, Any],
    metadata: Dict[str, str],
    resolver: GraphTokenResolver,
) -> Dict[str, Any]:
    graph_index = payload.get("graph_index")
    if graph_index is None:
        raise ValueError("Missing graph_index in payload")

    ranked_nodes = _extract_ranked_nodes(payload)
    # Node names must be looked up with the dataset-level index: sharded runs store a
    # shard-local ``graph_index`` and the corrected splitter adds ``global_graph_index``.
    lookup_index = payload.get("global_graph_index")
    if lookup_index is None:
        lookup_index = graph_index
    node_text_bundle = resolver.resolve(
        backbone=metadata["backbone"],
        dataset_raw=metadata["dataset_raw"],
        split=metadata["split"] or "test",
        graph_type=metadata["graph_type"],
        graph_index=int(lookup_index),
    )

    node_text = node_text_bundle.node_text
    ranked_tokens = format_token_sequence(node_text, ranked_nodes)

    node_scores = _extract_node_scores(payload, metadata["method"], ranked_nodes)

    num_nodes = payload.get("num_nodes")
    if num_nodes is None and isinstance(payload.get("explanation"), dict):
        num_nodes = payload["explanation"].get("num_nodes")
    if num_nodes is None:
        num_nodes = len(node_text)

    num_edges = payload.get("num_edges")
    if num_edges is None and isinstance(payload.get("explanation"), dict):
        num_edges = payload["explanation"].get("num_edges")

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
        "label": payload.get("label"),
        "prediction_class": payload.get("prediction_class"),
        "prediction_confidence": payload.get("prediction_confidence"),
        "is_correct": payload.get("is_correct"),
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "ranked_nodes": format_sequence(ranked_nodes),
        "ranked_tokens": ranked_tokens,
        "ranked_scores": format_float_sequence(node_scores),
    }
    return record


def process_method(
    method: str,
    base_dir: Path,
    output_dir: Path,
    resolver: GraphTokenResolver,
    *,
    limit: Optional[int] = None,
) -> List[Path]:
    pickle_paths = discover_pickles(base_dir, method)
    if limit is not None:
        pickle_paths = pickle_paths[:limit]

    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)

    for path in tqdm(pickle_paths, desc=f"Scanning {method} pickles", leave=False, colour="green"):
        try:
            metadata = parse_metadata_from_path(path)
            payload = load_payload(path, method=method)
            record = build_record(payload, metadata, resolver)
        except Exception as exc:
            print(f"! Skipping {path} ({exc})")
            continue
        key = (metadata["dataset_backbone"], metadata["graph_type"])
        grouped[key].append(record)

    written: List[Path] = []
    for (dataset_backbone, graph_type), rows in tqdm(
        sorted(grouped.items()),
        desc=f"Writing {method} CSVs",
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
        written.append(target_path)
        print(f"✓ {method} | {dataset_backbone} | {graph_type} → {target_path} ({len(df)} rows)")
    return written


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate GNN explanation pickles into token ranking CSVs.",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("outputs/gnn_models"),
        help="Root directory containing GNN explanation outputs.",
    )
    parser.add_argument(
        "--graph-root",
        type=Path,
        default=Path("outputs/graphs"),
        help="Directory containing NetworkX graph artefacts.",
    )
    parser.add_argument(
        "--pyg-root",
        type=Path,
        default=Path("outputs/pyg_graphs"),
        help="Directory containing PyG graph artefacts.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/analytics/tokens"),
        help="Destination directory for token analytics CSV files.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["subgraphx", "graphsvx"],
        help="Explainability methods to process (e.g., subgraphx graphsvx).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on the number of pickles processed (for debugging).",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    base_dir = args.base_dir
    graph_root = args.graph_root
    pyg_root = args.pyg_root
    output_dir = args.output_dir
    methods = [method.lower() for method in args.methods]
    limit = args.limit

    if not base_dir.exists():
        raise FileNotFoundError(f"Base directory not found: {base_dir}")
    if not graph_root.exists():
        raise FileNotFoundError(f"Graph root directory not found: {graph_root}")
    if not pyg_root.exists():
        raise FileNotFoundError(f"PyG root directory not found: {pyg_root}")

    resolver = GraphTokenResolver(graph_root=graph_root, pyg_root=pyg_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    written: List[Path] = []
    for method in tqdm(methods, desc="Processing methods", leave=False, colour="blue"):
        written.extend(process_method(method, base_dir, output_dir, resolver, limit=limit))

    if written:
        total_rows = 0
        for path in written:
            try:
                df = pd.read_csv(path)
                total_rows += len(df)
            except Exception:
                continue
        print(f"\nCompleted token aggregation for {len(written)} CSV file(s) ({total_rows} total rows).")
    else:
        print("\nNo GNN token CSV files were generated.")


if __name__ == "__main__":
    main()
