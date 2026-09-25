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

try:
    from src.explain.gnn.graphsvx.main import GraphSVXResult  # noqa: F401
except ModuleNotFoundError:
    GraphSVXResult = None  # type: ignore[assignment]

try:
    from src.explain.gnn.subgraphx.main import SubgraphXResult  # noqa: F401
except ModuleNotFoundError:
    SubgraphXResult = None  # type: ignore[assignment]

# Normalise tempfile usage for restricted environments
_TMP_CANDIDATE = Path("outputs/tmp_dir").resolve()
if not _TMP_CANDIDATE.exists():
    _TMP_CANDIDATE.mkdir(parents=True, exist_ok=True)
tempfile.tempdir = str(_TMP_CANDIDATE)
if hasattr(tempfile, "_bin_openflags"):
    tempfile._bin_openflags = os.O_RDWR | os.O_CREAT | os.O_EXCL


def _get_field(result: Any, attr: str, default: Any = None) -> Any:
    if isinstance(result, dict):
        return result.get(attr, default)
    return getattr(result, attr, default)


SHARD_DIR_PATTERN = re.compile(r"^(?P<prefix>.+)_shard(?P<index>\d+)of(?P<total>\d+)$")

_DEBUG_FLAG = os.environ.get("SPLIT_GRAPHSVX_DEBUG") or os.environ.get(
    "SPLIT_SUBGRAPHX_DEBUG"
)
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
    )

    def __init__(
        self,
        output_dir: Path,
        overwrite: bool,
        *,
        offset: int = 0,
        stride: int = 1,
        output_format: str = OUTPUT_FORMAT_JSON,
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
        self.extension = (
            ".pkl"
            if output_format in {OUTPUT_FORMAT_PICKLE, OUTPUT_FORMAT_PICKLE_RAW}
            else ".json"
        )

    def __call__(self, result: Any) -> None:
        self.processed += 1
        graph_index_raw = _get_field(result, "graph_index")
        if graph_index_raw is None:
            raise KeyError("Result is missing required 'graph_index' field")
        try:
            graph_index = int(graph_index_raw)
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive guard
            raise ValueError(f"Invalid graph_index value: {graph_index_raw!r}") from exc

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
    ("__main__", "GraphSVXResult"): _StreamingResultStub,
    ("src.explain.gnn.graphsvx.main", "GraphSVXResult"): _StreamingResultStub,
    ("explain.gnn.graphsvx.main", "GraphSVXResult"): _StreamingResultStub,
    ("__main__", "SubgraphXResult"): _StreamingResultStub,
    ("src.explain.gnn.subgraphx.main", "SubgraphXResult"): _StreamingResultStub,
    ("explain.gnn.subgraphx.main", "SubgraphXResult"): _StreamingResultStub,
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


def _load_results_streaming(file_obj, writer: _StreamingResultWriter) -> bool:
    global _CURRENT_RESULT_WRITER
    unpickler = _StreamingUnpickler(file_obj, replacements=_STREAMING_CLASS_ALIASES)
    _CURRENT_RESULT_WRITER = writer
    _debug("Streaming unpickler: begin load")
    try:
        results = unpickler.load()
    finally:
        _CURRENT_RESULT_WRITER = None

    _debug(f"Streaming unpickler: processed={writer.processed}")
    streamed = writer.processed > 0
    if streamed and isinstance(results, list):
        results.clear()
        _debug("Streaming unpickler: cleared materialised list")
    elif not streamed:
        _debug("Streaming unpickler: no streaming replacements encountered")
    return streamed


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


def serialise_prediction_metrics(payload: Any) -> Any:
    if not isinstance(payload, dict):
        return to_serialisable(payload) if payload is not None else {}

    return {
        str(key): to_serialisable(value)
        for key, value in payload.items()
    }


def serialise_graphsvx_explanation(explanation: Any) -> Any:
    if isinstance(explanation, dict):
        return {str(key): to_serialisable(value) for key, value in explanation.items()}
    if isinstance(explanation, list):
        converted: List[Any] = []
        for item in explanation:
            if isinstance(item, dict):
                converted.append({str(k): to_serialisable(v) for k, v in item.items()})
            else:
                converted.append(to_serialisable(item))
        return converted
    return to_serialisable(explanation)


def serialise_hyperparams(hyperparams: Any) -> Any:
    if not isinstance(hyperparams, dict):
        return to_serialisable(hyperparams)
    return {str(key): to_serialisable(value) for key, value in hyperparams.items()}


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
        state: Dict[str, Any] = dict(result)
    elif hasattr(result, "__dict__"):
        state = dict(result.__dict__)
    else:
        state = {"value": result}

    graph_index = state.get("graph_index")
    if graph_index is None:
        graph_index = _get_field(result, "graph_index")
    if graph_index is None:
        raise KeyError("Result is missing required 'graph_index' field")
    try:
        state["graph_index"] = int(graph_index)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid graph_index value: {graph_index!r}") from exc

    explanation_obj = state.get("explanation")
    hyperparams_obj = state.get("hyperparams")
    related_prediction_obj = state.get("related_prediction")
    prediction_obj = state.get("prediction")

    state["label"] = to_serialisable(state.get("label"))
    state["source"] = to_serialisable(state.get("source"))
    if make_serialisable:
        state["explanation"] = serialise_graphsvx_explanation(explanation_obj)
        state["hyperparams"] = serialise_hyperparams(hyperparams_obj)
        state["related_prediction"] = serialise_prediction_metrics(
            related_prediction_obj
        )
    else:
        state["explanation"] = explanation_obj
        state["hyperparams"] = hyperparams_obj
        state["related_prediction"] = related_prediction_obj
    if prediction_obj is None and isinstance(explanation_obj, dict):
        prediction_obj = explanation_obj.get("original_prediction")
    state["prediction"] = (
        to_serialisable(prediction_obj) if make_serialisable else prediction_obj
    )

    prediction_class, prediction_confidence = extract_prediction_fields(state)
    if make_serialisable:
        state["prediction_class"] = to_serialisable(prediction_class)
        state["prediction_confidence"] = to_serialisable(prediction_confidence)
    else:
        state["prediction_class"] = prediction_class
        state["prediction_confidence"] = prediction_confidence

    if make_serialisable:
        return {str(key): to_serialisable(value) for key, value in state.items()}
    return state


def split_results(
    input_path: Path,
    output_dir: Path,
    overwrite: bool = False,
    *,
    output_format: str = OUTPUT_FORMAT_JSON,
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
    )
    streamed = False

    try:
        with input_path.open("rb") as handle:
            streamed = _load_results_streaming(handle, streaming_writer)
    except Exception:  # pragma: no cover - defensive fallback path
        if streaming_writer.processed:
            raise

    if streamed:
        _debug(f"split_results: streaming path wrote {streaming_writer.written} records")
        _trigger_gc()
        return streaming_writer.written

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
        description="Split GraphSVX results.pkl files into per-graph artefacts.")
    parser.add_argument(
        "--input",
        type=Path,
        help="Path to a specific results.pkl file to split. Omit when using --all.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Destination directory for per-graph outputs. Defaults to <input_stem>_split_<format> alongside the pickle.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing per-graph outputs if they already exist.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Search for every GraphSVX results.pkl under the default outputs directory and split them.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/app/outputs/gnn_models"),
        help="Root directory for --all scanning. Defaults to /app/outputs/gnn_models.",
    )
    parser.add_argument(
        "--format",
        choices=sorted(VALID_OUTPUT_FORMATS),
        default=OUTPUT_FORMAT_JSON,
        help="Per-graph output format (json or pickle).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.all:
        successes: List[Tuple[Path, int]] = []
        skipped: List[Path] = []
        suffix = "json" if args.format == OUTPUT_FORMAT_JSON else "pickle"
        for results_path in iter_results_pickles(args.root):
            if "graphsvx" not in results_path.parts:
                continue
            output_dir = results_path.with_name(f"{results_path.stem}_split_{suffix}")
            if output_dir.exists() and not args.overwrite and any(output_dir.iterdir()):
                skipped.append(results_path)
                continue
            count = split_results(
                results_path,
                output_dir,
                overwrite=args.overwrite,
                output_format=args.format,
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
    )
    print(f"Wrote {count} per-graph files to {output_dir}")


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
    if hasattr(obj, "to_dict"):
        try:
            return {str(k): to_serialisable(v) for k, v in obj.to_dict().items()}
        except Exception:
            return str(obj)
    if isinstance(obj, nx.Graph):
        return {
            "directed": obj.is_directed(),
            "nodes": [
                {"id": to_serialisable(node), "attributes": to_serialisable(attrs)}
                for node, attrs in obj.nodes(data=True)
            ],
            "edges": [
                {
                    "source": to_serialisable(src),
                    "target": to_serialisable(dst),
                    "attributes": to_serialisable(attrs),
                }
                for src, dst, attrs in obj.edges(data=True)
            ],
        }
    if isinstance(obj, dict):
        return {str(key): to_serialisable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_serialisable(item) for item in obj]
    if hasattr(obj, "__dict__"):
        return {key: to_serialisable(value) for key, value in obj.__dict__.items()}
    return str(obj)


if __name__ == "__main__":
    main()
