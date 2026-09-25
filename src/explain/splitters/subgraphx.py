from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import networkx as nx
import numpy as np

# Normalise tempfile location for environments with restricted /tmp.
_TMP_CANDIDATE = Path("outputs/tmp_dir").resolve()
if not _TMP_CANDIDATE.exists():
    _TMP_CANDIDATE.mkdir(parents=True, exist_ok=True)
tempfile.tempdir = str(_TMP_CANDIDATE)
if hasattr(tempfile, "_bin_openflags"):
    tempfile._bin_openflags = os.O_RDWR | os.O_CREAT | os.O_EXCL

try:  # Optional torch dependency
    import torch
except ImportError:  # pragma: no cover - torch might not be installed for quick tooling
    torch = None  # type: ignore[assignment]

try:  # Optional psutil dependency for diagnostics
    import psutil  # type: ignore[assignment]
except Exception:  # pragma: no cover - psutil is optional
    psutil = None  # type: ignore[assignment]


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Ensure pickle has the result classes registered ahead of loading
try:
    from src.explain.gnn.subgraphx.main import SubgraphXResult  # noqa: F401
except ModuleNotFoundError:
    SubgraphXResult = None  # type: ignore[assignment]

try:
    from src.explain.gnn.graphsvx.main import GraphSVXResult  # noqa: F401
except ModuleNotFoundError:
    GraphSVXResult = None  # type: ignore[assignment]


def _get_field(result: Any, attr: str, default: Any = None) -> Any:
    """Retrieve an attribute or key from either an object or dictionary."""

    if isinstance(result, dict):
        return result.get(attr, default)
    return getattr(result, attr, default)


SHARD_DIR_PATTERN = re.compile(r"^(?P<prefix>.+)_shard(?P<index>\d+)of(?P<total>\d+)$")

# SPLIT_DROP_EXPLANATION=1 keeps per-graph records small (see serialise_result).
DROP_EXPLANATION = os.environ.get("SPLIT_DROP_EXPLANATION", "1") not in ("0", "false", "False")

_DEBUG_FLAG = os.environ.get("SPLIT_SUBGRAPHX_DEBUG")
_DEBUG_ENABLED = _DEBUG_FLAG not in (None, "", "0", "false", "False")

OUTPUT_FORMAT_JSON = "json"
OUTPUT_FORMAT_PICKLE = "pickle"
OUTPUT_FORMAT_PICKLE_RAW = "pickle-raw"
VALID_OUTPUT_FORMATS = {
    OUTPUT_FORMAT_JSON,
    OUTPUT_FORMAT_PICKLE,
    OUTPUT_FORMAT_PICKLE_RAW,
}


def _debug(message: str) -> None:
    if _DEBUG_ENABLED:
        print(message, file=sys.stderr, flush=True)


class _StreamingResultWriter:
    __slots__ = (
        "output_dir",
        "overwrite",
        "written",
        "processed",
        "offset",
        "stride",
        "output_format",
        "extension",
        "make_serialisable",
        "skip_indices",
    )

    def __init__(
        self,
        output_dir: Path,
        overwrite: bool,
        *,
        offset: int = 0,
        stride: int = 1,
        output_format: str = OUTPUT_FORMAT_JSON,
        skip_indices: Optional[Iterable[int]] = None,
    ) -> None:
        self.output_dir = output_dir
        self.overwrite = overwrite
        self.written = 0
        self.processed = 0
        self.offset = offset
        self.stride = max(1, int(stride))
        if output_format not in VALID_OUTPUT_FORMATS:
            raise ValueError(f"Unsupported output format: {output_format}")
        self.output_format = output_format
        self.make_serialisable = output_format != OUTPUT_FORMAT_PICKLE_RAW
        self.extension = ".pkl" if output_format in {OUTPUT_FORMAT_PICKLE, OUTPUT_FORMAT_PICKLE_RAW} else ".json"
        self.skip_indices = set(skip_indices or [])

    def __call__(self, result: Any) -> None:
        self.processed += 1
        graph_index_raw = _get_field(result, "graph_index")
        if graph_index_raw is None:
            raise KeyError("Result is missing required 'graph_index' field")
        try:
            graph_index = int(graph_index_raw)
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive guard
            raise ValueError(f"Invalid graph_index value: {graph_index_raw!r}") from exc

        if graph_index in self.skip_indices:
            self._release(result)
            return

        destination = self.output_dir / f"graph_{graph_index:05d}{self.extension}"
        if destination.exists() and not self.overwrite:
            self._release(result)
            return

        record = serialise_result(result, make_serialisable=self.make_serialisable)
        # Shards are strided (loader keeps graphs with i % num_shards == shard_index),
        # so the dataset-level index is shard_index + local_index * num_shards.
        record["global_graph_index"] = graph_index * self.stride + self.offset

        if self.output_format in {OUTPUT_FORMAT_PICKLE, OUTPUT_FORMAT_PICKLE_RAW}:
            with destination.open("wb") as fh:
                pickle.dump(record, fh, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with destination.open("w", encoding="utf-8") as fh:
                json.dump(record, fh, ensure_ascii=False)
        self.written += 1
        self._release(result)
        if self.processed % 25 == 0:
            _trigger_gc()
            _debug(
                f"Streaming progress: processed={self.processed} written={self.written}"
            )

    @staticmethod
    def _release(result: Any) -> None:
        if isinstance(result, dict):
            result.clear()
        elif isinstance(result, list):
            result.clear()
        elif hasattr(result, "__dict__"):
            for attr in list(vars(result)):
                try:
                    setattr(result, attr, None)
                except Exception:
                    pass


class _StreamingResultStub:
    __slots__ = ()

    def __new__(cls, *args: Any, **kwargs: Any) -> "_StreamingResultStub":
        return object.__new__(cls)

    def __setstate__(self, state: Any) -> None:
        writer = _CURRENT_RESULT_WRITER
        if writer is None:
            raise RuntimeError("Streaming result stub invoked without active writer")
        try:
            writer(state)
        finally:
            if isinstance(state, dict):
                state.clear()
            elif isinstance(state, list):
                state.clear()


_STREAMING_CLASS_ALIASES: Dict[Tuple[str, str], Any] = {
    ("__main__", "SubgraphXResult"): _StreamingResultStub,
    ("src.explain.gnn.subgraphx.main", "SubgraphXResult"): _StreamingResultStub,
    ("explain.gnn.subgraphx.main", "SubgraphXResult"): _StreamingResultStub,
    ("__main__", "GraphSVXResult"): _StreamingResultStub,
    ("src.explain.gnn.graphsvx.main", "GraphSVXResult"): _StreamingResultStub,
    ("explain.gnn.graphsvx.main", "GraphSVXResult"): _StreamingResultStub,
}


class _StreamingUnpickler(pickle.Unpickler):
    def __init__(self, file_obj, *, replacements: Dict[Tuple[str, str], Any]) -> None:
        super().__init__(file_obj)
        self._replacements = replacements

    def find_class(self, module: str, name: str) -> Any:  # pragma: no cover - delegated to pickle
        replacement = self._replacements.get((module, name))
        if replacement is not None:
            return replacement
        return super().find_class(module, name)


_CURRENT_RESULT_WRITER: Optional[_StreamingResultWriter] = None


class _CountingSink:
    __slots__ = ("processed",)

    def __init__(self) -> None:
        self.processed = 0

    def __call__(self, state: Any) -> None:
        self.processed += 1


def _load_results_count(results_path: Path) -> int:
    if not results_path.exists():
        return 0
    sink = _CountingSink()
    streamed = False
    with results_path.open("rb") as handle:
        try:
            streamed = _load_results_streaming(handle, sink)
        except Exception:
            pass

    if streamed and sink.processed:
        return sink.processed

    try:
        with results_path.open("rb") as handle:
            results = pickle.load(handle)
    except Exception:
        return sink.processed

    if isinstance(results, list):
        return len(results)
    return sink.processed


def _load_results_streaming(file_obj, writer: _StreamingResultWriter) -> bool:
    """Stream results from pickle via custom unpickler, writing as we unpickle."""

    global _CURRENT_RESULT_WRITER
    unpickler = _StreamingUnpickler(file_obj, replacements=_STREAMING_CLASS_ALIASES)
    _CURRENT_RESULT_WRITER = writer
    _debug("Streaming unpickler: begin load")
    try:
        results = unpickler.load()
    finally:
        _CURRENT_RESULT_WRITER = None

    _debug(f"Streaming unpickler: processed={writer.processed}")
    streaming_used = writer.processed > 0
    if streaming_used and isinstance(results, list):
        results.clear()
        _debug("Streaming unpickler: cleared materialised list")
    elif not streaming_used:
        _debug("Streaming unpickler: no streaming replacements encountered")
    return streaming_used


def _trigger_gc() -> None:
    try:
        import gc

        gc.collect()
    except Exception:  # pragma: no cover - GC is a best-effort cleanup
        pass


def _determine_graph_index_offset(results_path: Path) -> Tuple[int, int]:
    """Return ``(offset, stride)`` mapping a shard-local graph index to the dataset index.

    ``src.explain.gnn.model_loader.load_graph_split`` assigns graph ``i`` to shard
    ``i % num_shards``; hence ``global = shard_index0 + local * num_shards``.
    """
    parent = results_path.parent
    match = SHARD_DIR_PATTERN.match(parent.name)
    if not match:
        return 0, 1

    shard_index = int(match.group("index"))  # 1-based in directory names
    total_shards = int(match.group("total"))
    if total_shards <= 1:
        return 0, 1
    return shard_index - 1, total_shards


def serialise_graph(graph: nx.Graph | nx.DiGraph | None) -> Dict[str, Any]:
    if graph is None:
        return {}
    nodes: List[Tuple[Any, Dict[str, Any]]] = list(graph.nodes(data=True))
    edges: List[Tuple[Any, Any, Dict[str, Any]]] = list(graph.edges(data=True))
    return {
        "directed": graph.is_directed(),
        "nodes": [
            {"id": to_serialisable(node), "attributes": to_serialisable(attrs)}
            for node, attrs in nodes
        ],
        "edges": [
            {
                "source": to_serialisable(src),
                "target": to_serialisable(dst),
                "attributes": to_serialisable(attrs),
            }
            for src, dst, attrs in edges
        ],
    }


def to_serialisable(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if torch is not None and isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if hasattr(obj, "to_dict"):  # e.g., torch_geometric.data.Data
        try:
            return {str(k): to_serialisable(v) for k, v in obj.to_dict().items()}
        except Exception:
            return str(obj)
    if isinstance(obj, nx.Graph):
        return serialise_graph(obj)
    if isinstance(obj, dict):
        return {str(key): to_serialisable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_serialisable(item) for item in obj]
    if hasattr(obj, "__dict__"):
        return {key: to_serialisable(value) for key, value in obj.__dict__.items()}
    return str(obj)


def serialise_explanation(explanation: Any) -> Any:
    if isinstance(explanation, list):
        cleaned: List[Any] = []
        for entry in explanation:
            if isinstance(entry, dict):
                cleaned.append(to_serialisable(entry))
            else:
                cleaned.append(to_serialisable(entry))
        return cleaned
    if isinstance(explanation, dict):
        return to_serialisable(explanation)
    return to_serialisable(explanation)


def extract_prediction_fields(record: Dict[str, Any]) -> Tuple[Any, Any]:
    prediction = record.get("prediction")
    related_pred = record.get("related_prediction") or {}
    prediction_class = None
    prediction_confidence = None

    if isinstance(prediction, dict):
        prediction_class = prediction.get("class")
        prediction_confidence = prediction.get("confidence")

    if prediction_class is None and isinstance(related_pred, dict):
        prediction_class = related_pred.get("predicted_class")
    if prediction_confidence is None and isinstance(related_pred, dict):
        prediction_confidence = related_pred.get("predicted_confidence")

    return prediction_class, prediction_confidence


def serialise_result(result: Any, *, make_serialisable: bool = True) -> Dict[str, Any]:
    if isinstance(result, dict):
        state = dict(result)
    else:
        state = {
            "graph_index": _get_field(result, "graph_index"),
            "label": _get_field(result, "label", None),
            "hyperparams": _get_field(result, "hyperparams", {}),
            "source": _get_field(result, "source", None),
            "related_prediction": _get_field(result, "related_prediction", {}),
            "explanation": _get_field(result, "explanation", {}),
            "num_nodes": _get_field(result, "num_nodes", None),
            "num_edges": _get_field(result, "num_edges", None),
            "is_correct": _get_field(result, "is_correct", None),
            "prediction": _get_field(result, "prediction", None),
        }

    graph_index = state.get("graph_index")
    if graph_index is None:
        raise KeyError("Result is missing required 'graph_index' field")
    try:
        state["graph_index"] = int(graph_index)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive guard
        raise ValueError(f"Invalid graph_index value: {graph_index!r}") from exc

    explanation_obj = state.get("explanation", {})

    if make_serialisable:
        state["label"] = to_serialisable(state.get("label"))
        state["hyperparams"] = to_serialisable(state.get("hyperparams", {}))
        state["source"] = to_serialisable(state.get("source"))
        state["related_prediction"] = to_serialisable(
            state.get("related_prediction", {})
        )
        if DROP_EXPLANATION:
            # The raw DIG explanation embeds one graph copy per explored MCTS node
            # (tens of MB per record); the analytics only read related_prediction.
            state["explanation"] = {"dropped": True, "top_nodes": to_serialisable(
                (state.get("related_prediction") or {}).get("top_nodes"))}
        else:
            state["explanation"] = serialise_explanation(explanation_obj)
    else:
        state["label"] = state.get("label")
        state["hyperparams"] = state.get("hyperparams", {})
        state["source"] = state.get("source")
        state["related_prediction"] = state.get("related_prediction", {})
        state["explanation"] = explanation_obj

    num_nodes = state.get("num_nodes")
    num_edges = state.get("num_edges")
    if num_nodes is None and isinstance(explanation_obj, dict):
        num_nodes = explanation_obj.get("num_nodes")
    if num_edges is None and isinstance(explanation_obj, dict):
        num_edges = explanation_obj.get("num_edges")
    if make_serialisable:
        state["num_nodes"] = to_serialisable(num_nodes)
        state["num_edges"] = to_serialisable(num_edges)
    else:
        state["num_nodes"] = num_nodes
        state["num_edges"] = num_edges

    is_correct = state.get("is_correct")
    if is_correct is None and isinstance(explanation_obj, dict):
        is_correct = explanation_obj.get("is_correct")
    state["is_correct"] = to_serialisable(is_correct) if make_serialisable else is_correct

    prediction = state.get("prediction")
    if prediction is None and isinstance(explanation_obj, dict):
        prediction = explanation_obj.get("original_prediction")
    state["prediction"] = to_serialisable(prediction) if make_serialisable else prediction

    prediction_class, prediction_confidence = extract_prediction_fields(state)
    if make_serialisable:
        state["prediction_class"] = to_serialisable(prediction_class)
        state["prediction_confidence"] = to_serialisable(prediction_confidence)
    else:
        state["prediction_class"] = prediction_class
        state["prediction_confidence"] = prediction_confidence

    keep_keys = {
        "graph_index",
        "label",
        "hyperparams",
        "source",
        "related_prediction",
        "explanation",
        "num_nodes",
        "num_edges",
        "is_correct",
        "prediction",
        "prediction_class",
        "prediction_confidence",
    }
    for key in list(state.keys()):
        if key not in keep_keys:
            state.pop(key, None)

    return state


def split_results(
    input_path: Path,
    output_dir: Path,
    overwrite: bool = False,
    *,
    output_format: str = OUTPUT_FORMAT_JSON,
    skip_indices: Optional[Iterable[int]] = None,
) -> int:
    if not input_path.exists():
        raise FileNotFoundError(f"Input pickle does not exist: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    file_size_mb = input_path.stat().st_size / (1024 * 1024)
    _debug(f"split_results: processing '{input_path}' ({file_size_mb:.1f} MiB)")
    if psutil is not None:
        try:
            available_gb = psutil.virtual_memory().available / (1024 ** 3)
            _debug(f"split_results: available memory {available_gb:.1f} GiB")
        except Exception:
            pass

    offset, stride = _determine_graph_index_offset(input_path)
    if offset or stride != 1:
        _debug(f"split_results: detected shard offset {offset}, stride {stride}")

    streaming_writer = _StreamingResultWriter(
        output_dir,
        overwrite,
        offset=offset,
        stride=stride,
        output_format=output_format,
        skip_indices=skip_indices,
    )
    streaming_used = False

    try:
        with input_path.open("rb") as handle:
            streaming_used = _load_results_streaming(handle, streaming_writer)
    except Exception:  # pragma: no cover - defensive fallback path
        if streaming_writer.processed:
            raise

    if streaming_used:
        _debug(
            f"split_results: streaming path wrote {streaming_writer.written} records"
        )
        _trigger_gc()
        return streaming_writer.written

    # Streaming either was not required for this pickle or is unsupported; fall back to
    # materialising the list in memory (suitable for smaller artefacts).
    _debug("split_results: falling back to materialised pickle load")
    process = None
    mem_before = None
    if psutil is not None:
        try:
            process = psutil.Process()
            mem_before = process.memory_info().rss
        except Exception:
            process = None

    with input_path.open("rb") as handle:
        results = pickle.load(handle)

    if not isinstance(results, list):
        raise TypeError(f"Expected the pickle to contain a list, got {type(results)}")

    fallback_writer = _StreamingResultWriter(
        output_dir,
        overwrite,
        offset=offset,
        stride=stride,
        output_format=output_format,
        skip_indices=skip_indices,
    )
    for idx, result in enumerate(results):
        fallback_writer(result)
        results[idx] = None

    del results
    _trigger_gc()
    if process is not None:
        try:
            mem_after = process.memory_info().rss
            if mem_before is not None:
                delta_mb = (mem_after - mem_before) / (1024 * 1024)
                _debug(f"split_results: fallback memory delta {delta_mb:.1f} MiB")
        except Exception:
            pass
    _debug(f"split_results: fallback path wrote {fallback_writer.written} records")
    return fallback_writer.written


def iter_results_pickles(root: Path) -> Iterable[Path]:
    for path in root.rglob("results.pkl"):
        yield path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split explanation results.pkl files into per-graph artefacts."
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Path to a specific results.pkl file to split. Omit when using --all.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Destination directory for the per-graph outputs. Defaults to <input_stem>_split_<format> alongside the pickle.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing per-graph outputs if they already exist.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Search /app/outputs/gnn_models/ for every results.pkl and split them.",
    )
    parser.add_argument(
        "--format",
        choices=sorted(VALID_OUTPUT_FORMATS),
        default=OUTPUT_FORMAT_JSON,
        help="Per-graph output format (json or pickle).",
    )
    parser.add_argument(
        "--skip",
        type=int,
        action="append",
        default=[],
        help="Graph indices to skip when splitting (can be specified multiple times).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.all:
        root = Path("/app/outputs/gnn_models")
        successes: List[Tuple[Path, int]] = []
        skipped: List[Path] = []
        suffix = "json" if args.format == OUTPUT_FORMAT_JSON else "pickle"
        for results_path in iter_results_pickles(root):
            output_dir = results_path.with_name(f"{results_path.stem}_split_{suffix}")
            if output_dir.exists() and not args.overwrite and any(output_dir.iterdir()):
                skipped.append(results_path)
                continue
            count = split_results(
                results_path,
                output_dir,
                overwrite=args.overwrite,
                output_format=args.format,
                skip_indices=args.skip,
            )
            successes.append((results_path, count))
            print(f"Wrote {count:4d} files to {output_dir}")
        if skipped:
            print(f"Skipped {len(skipped)} pickles that already had output (use --overwrite to regenerate).")
        print(f"Processed {len(successes)} pickles in total.")
        return

    if not args.input:
        raise ValueError("Either specify --input or use --all.")

    input_path: Path = args.input
    if args.output:
        output_dir = args.output
    else:
        suffix = "json" if args.format == OUTPUT_FORMAT_JSON else "pickle"
        output_dir = input_path.with_name(f"{input_path.stem}_split_{suffix}")
    count = split_results(
        input_path,
        output_dir,
        overwrite=args.overwrite,
        output_format=args.format,
        skip_indices=args.skip,
    )
    print(f"Wrote {count} per-graph files to {output_dir}")


if __name__ == "__main__":
    main()
