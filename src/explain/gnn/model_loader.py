from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple
import warnings

import torch

from src.gnn_training.training import GNNClassifier, load_graph_data

from .config import ExplainerRequest
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader as PyGDataLoader


def _candidate_run_dirs(base_dir: Path) -> Tuple[Path, ...]:
    if not base_dir.exists():
        raise FileNotFoundError(f"GNN directory not found: {base_dir}")
    runs = [p for p in base_dir.iterdir() if p.is_dir()]
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    # Flattened layouts keep the checkpoint directly under the graph-type directory.
    if any(base_dir.glob("*.pt")) or any(base_dir.glob("*.pth")) or any(base_dir.glob("*.ckpt")):
        runs.insert(0, base_dir)
    if not runs:
        raise FileNotFoundError(f"No training runs detected under {base_dir}")
    return tuple(runs)


def resolve_checkpoint(request: ExplainerRequest) -> Tuple[Path, Optional[Path], Path]:
    """Locate the most recent checkpoint and its config for the given request."""
    base_dir = Path(request.gnn_root) / request.backbone / request.dataset_subpath / request.graph_type
    for run_dir in _candidate_run_dirs(base_dir):
        stem = Path(request.checkpoint_name)
        candidate_names = [request.checkpoint_name]
        if not stem.suffix:
            candidate_names.extend([f"{stem}.pt", f"{stem}.pth"])
        else:
            candidate_names.extend([
                stem.with_suffix(ext).name for ext in (".pt", ".pth", ".ckpt")
            ])
        candidate_names.extend(
            [
                "best_model.pt",
                "best_model.pth",
                "model.pt",
                "model.pth",
                "checkpoint.pt",
                "checkpoint.pth",
            ]
        )

        checkpoint_candidate: Optional[Path] = None
        for name in candidate_names:
            candidate = run_dir / name
            if candidate.exists():
                checkpoint_candidate = candidate
                break

        if checkpoint_candidate is None:
            for pattern in ("*.pt", "*.pth", "*.ckpt"):
                matches = sorted(run_dir.glob(pattern))
                if matches:
                    checkpoint_candidate = matches[0]
                    break

        if checkpoint_candidate is not None:
            args_path: Optional[Path] = None
            for config_name in ("args.json", "config.json", "hparams.json"):
                candidate = run_dir / config_name
                if candidate.exists():
                    args_path = candidate
                    break
            return checkpoint_candidate, args_path, run_dir
    raise FileNotFoundError(
        f"Unable to locate a checkpoint (*.pt) for {request.dataset}/{request.graph_type}"
    )


def _extract_model_kwargs(args: Dict[str, object]) -> Dict[str, object]:
    """Convert the serialized args into constructor kwargs for ``GNNClassifier``."""
    hidden_dim = (
        args.get("hidden_dim")
        or args.get("gnn_hidden_dim")
        or args.get("emb_dim")
    )
    if hidden_dim is None:
        raise ValueError("Training args must define 'hidden_dim' (or gnn_hidden_dim/emb_dim).")

    num_layers = args.get("num_layers") or args.get("gnn_layers") or 2

    kwargs: Dict[str, object] = {
        "input_dim": args.get("input_dim", 768),
        "hidden_dim": hidden_dim,
        "output_dim": args.get("num_classes") or args.get("output_dim", 2),
        "num_layers": num_layers,
        "dropout": args.get("dropout", 0.5),
        "module": args.get("module", "GCNConv"),
        "layer_norm": args.get("layer_norm", False),
        "residual": args.get("residual", False),
        "pooling": args.get("pooling", "mean"),
        "heads": args.get("heads") or args.get("attention_heads", 1),
    }
    return kwargs


def load_gnn_model(
    request: ExplainerRequest,
    dataset=None,
) -> Tuple[torch.nn.Module, Dict[str, object], Path]:
    """Instantiate and load the GNN trained for the provided dataset/graph type."""
    checkpoint_path, args_path, run_dir = resolve_checkpoint(request)
    args: Dict[str, object] = {}
    if args_path:
        with open(args_path, "r") as handle:
            args = json.load(handle)

    kwargs = _extract_model_kwargs(args)
    if dataset is not None:
        if hasattr(dataset, "num_node_features"):
            kwargs["input_dim"] = getattr(dataset, "num_node_features")
        if hasattr(dataset, "num_classes"):
            kwargs["output_dim"] = getattr(dataset, "num_classes")

    model = GNNClassifier(**kwargs)

    state_dict = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    model.load_state_dict(state_dict)

    device_str = request.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device_str)
    model.eval()
    return model, args, run_dir


def load_graph_split(request: ExplainerRequest, **loader_kwargs):
    """Load the PyG dataset/dataloader for the requested split and graph type."""
    graph_dir = request.graph_split_root
    if not graph_dir.exists():
        raise FileNotFoundError(f"Graph data split not found: {graph_dir}")

    dataset, loader = load_graph_data(
        data_dir=str(graph_dir),
        batch_size=loader_kwargs.pop("batch_size", 1),
        shuffle=loader_kwargs.pop("shuffle", False),
        num_workers=loader_kwargs.pop("num_workers", 0),
        **loader_kwargs,
    )
    if request.num_shards > 1:
        total = len(dataset)
        if request.shard_index < 0 or request.shard_index >= request.num_shards:
            raise ValueError(
                f"Shard index {request.shard_index} out of range for num_shards={request.num_shards}"
            )
        indices = [i for i in range(total) if i % request.num_shards == request.shard_index]
        if not indices:
            warnings.warn(
                f"Shard {request.shard_index}/{request.num_shards} has no graphs in {graph_dir}."
            )
        subset = Subset(dataset, indices)
        # Propagate metadata needed by downstream components
        for attr in ("num_node_features", "num_classes"):
            if hasattr(dataset, attr):
                setattr(subset, attr, getattr(dataset, attr))

        batch_size = getattr(loader, "batch_size", 1)
        num_workers = getattr(loader, "num_workers", 0)
        pin_memory = getattr(loader, "pin_memory", torch.cuda.is_available())

        loader = PyGDataLoader(
            subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )
        dataset = subset

    return dataset, loader
