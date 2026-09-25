from __future__ import annotations

import argparse
import json
import os
import pickle
import warnings
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from tqdm import tqdm

from src.explain.common.fairness import FairMultimodalHyperparameterAdvisor, FairnessConfig
from src.explain.gnn.config import (
    DEFAULT_GNN_ROOT,
    DEFAULT_GRAPH_DATA_ROOT,
    ExplainerRequest,
    SUBGRAPHX_DEFAULTS,
    SUBGRAPHX_PROFILES,
)
from src.explain.gnn.model_loader import load_gnn_model, load_graph_split
from src.utils.energy import EnergyMonitor
from .hyperparam_advisor import (
    ArchitectureSpec,
    SubgraphXHyperparameterAdvisor,
    FLOAT_PARAM_KEYS,
    INT_PARAM_KEYS,
)


class MarginalSubgraphDataset(Dataset):
    """Compatibility shim for the DIG SubgraphX sampler."""

    def __init__(self, data: Data, exclude_mask, include_mask, subgraph_build_func):
        self.num_nodes = data.num_nodes
        self.X = data.x
        self.edge_index = data.edge_index
        self.device = self.X.device
        self.label = data.y
        self.exclude_mask = torch.tensor(exclude_mask, dtype=torch.float32, device=self.device)
        self.include_mask = torch.tensor(include_mask, dtype=torch.float32, device=self.device)
        self.subgraph_build_func = subgraph_build_func

    def __len__(self) -> int:  # pragma: no cover - simple proxy method
        return self.exclude_mask.shape[0]

    def __getitem__(self, idx):  # pragma: no cover - DIG internal usage
        exclude_graph_X, exclude_graph_edge_index = self.subgraph_build_func(
            self.X, self.edge_index, self.exclude_mask[idx]
        )
        include_graph_X, include_graph_edge_index = self.subgraph_build_func(
            self.X, self.edge_index, self.include_mask[idx]
        )
        exclude_data = Data(x=exclude_graph_X, edge_index=exclude_graph_edge_index)
        include_data = Data(x=include_graph_X, edge_index=include_graph_edge_index)
        return exclude_data, include_data


# Monkey patch DIG so our dataset is used when constructing coalitions.
import dig.xgraph.method.shapley  # type: ignore  # pylint: disable=import-error

dig.xgraph.method.shapley.MarginalSubgraphDataset = MarginalSubgraphDataset  # type: ignore[attr-defined]


def _contrastive_stats(
    distribution: Optional[Sequence[float]],
    target_class: Optional[int],
) -> Tuple[Optional[int], Optional[float], Optional[float]]:
    if distribution is None:
        return None, None, None
    try:
        values = [float(v) for v in distribution]
    except (TypeError, ValueError):
        return None, None, None
    if not values:
        return None, None, None

    if target_class is not None and 0 <= int(target_class) < len(values):
        target_idx = int(target_class)
        target_conf = values[target_idx]
        others = [(idx, val) for idx, val in enumerate(values) if idx != target_idx]
        if not others:
            return None, None, None
        second_idx, second_val = max(others, key=lambda item: item[1])
        contrast = target_conf - second_val if target_conf is not None else None
        return second_idx, second_val, contrast

    ordered = sorted(enumerate(values), key=lambda item: item[1], reverse=True)
    if len(ordered) < 2:
        return None, None, None
    _, best_val = ordered[0]
    second_idx, second_val = ordered[1]
    contrast = best_val - second_val if best_val is not None else None
    return second_idx, second_val, contrast


class UniversalDataModelWrapper(torch.nn.Module):
    """Normalise the model forward signature for DIG's expectations."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, *args, **kwargs):  # pragma: no cover - thin wrapper
        if len(args) == 1 and isinstance(args[0], Data):
            data = args[0]
            return self.model(data=data)
        if len(args) >= 2 and isinstance(args[0], torch.Tensor) and isinstance(args[1], torch.Tensor):
            x = args[0]
            edge_index = args[1]
            batch = None
            if len(args) >= 3:
                batch = args[2]
            elif "batch" in kwargs:
                batch = kwargs["batch"]
            return self.model(x=x, edge_index=edge_index, batch=batch)
        if "data" in kwargs:
            return self.model(data=kwargs["data"])
        return self.model(*args, **kwargs)

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device


@dataclass
class SubgraphXResult:
    """Holds per-graph explanation artefacts."""

    graph_index: int
    label: Optional[int]
    explanation: object
    related_prediction: Dict[str, float]
    num_nodes: int
    num_edges: int
    hyperparams: Dict[str, float]
    prediction: Optional[Dict[str, object]] = None
    is_correct: Optional[bool] = None

    def to_json(self) -> Dict[str, object]:  # pragma: no cover - serialisation helper
        return {
            "graph_index": self.graph_index,
            "label": self.label,
            "num_nodes": self.num_nodes,
            "num_edges": self.num_edges,
            "related_prediction": self.related_prediction,
            "hyperparams": dict(self.hyperparams),
            "prediction": dict(self.prediction) if self.prediction else None,
            "prediction_class": self.prediction.get("class") if self.prediction else None,
            "prediction_confidence": self.prediction.get("confidence") if self.prediction else None,
            "is_correct": self.is_correct,
        }


def _merge_hyperparams(overrides: Optional[Dict[str, float]]) -> Dict[str, float]:
    params = dict(SUBGRAPHX_DEFAULTS)
    if overrides:
        params.update({k: v for k, v in overrides.items() if v is not None})
    # Cast to expected types for DIG
    int_keys = {"num_hops", "rollout", "min_atoms", "expand_atoms", "local_radius", "sample_num", "max_nodes"}
    for key in int_keys:
        params[key] = int(params[key])
    params["c_puct"] = float(params["c_puct"])
    return params


def _make_slug(request: ExplainerRequest, run_tag: Optional[str] = None) -> str:
    dataset = str(request.dataset_subpath) if request.dataset_subpath != Path('.') else request.dataset
    parts = [request.backbone, dataset, request.graph_type, request.split]
    if run_tag:
        parts.append(run_tag)
    if getattr(request, "num_shards", 1) > 1:
        shard_label = f"shard{request.shard_index + 1}of{request.num_shards}"
        parts.append(shard_label)
    safe = [p.replace("/", "-") for p in parts if p]
    return "_".join(safe)


def _prepare_explainer(
    wrapper: UniversalDataModelWrapper,
    args: Dict[str, object],
    save_dir: Path,
    hyperparams: Dict[str, float],
) -> CustomSubgraphX:
    from .custom_subgraphx import CustomSubgraphX

    num_classes = int(args.get("num_classes") or args.get("output_dim", 2))
    save_dir.mkdir(parents=True, exist_ok=True)

    return CustomSubgraphX(
        model=wrapper,
        num_classes=num_classes,
        device=wrapper.device,
        num_hops=hyperparams["num_hops"],
        rollout=hyperparams["rollout"],
        min_atoms=hyperparams["min_atoms"],
        c_puct=hyperparams["c_puct"],
        expand_atoms=hyperparams["expand_atoms"],
        local_radius=hyperparams["local_radius"],
        sample_num=hyperparams["sample_num"],
        save_dir=str(save_dir),
    )


def _extract_node_tokens(data: Data) -> Optional[List[str]]:
    token_keys = ("node_tokens", "tokens", "token_text", "words", "token_strings")
    for key in token_keys:
        value = getattr(data, key, None)
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            return [str(item) for item in value]
        if torch.is_tensor(value):
            flat = value.detach().cpu().tolist()
            if isinstance(flat, list):
                return [str(item) for item in flat]
    mapping = getattr(data, "token_map", None)
    if isinstance(mapping, dict):
        num_nodes = int(getattr(data, "num_nodes", len(mapping)))
        return [str(mapping.get(idx)) for idx in range(num_nodes)]
    return None


def _extract_architecture_spec(
    model: torch.nn.Module, args: Dict[str, object]
) -> ArchitectureSpec:
    """Normalise architecture metadata for hyperparameter tuning heuristics."""

    def _coerce_int(value: object, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    num_layers = _coerce_int(
        args.get("num_layers") or args.get("gnn_layers") or getattr(model, "num_layers", None),
        2,
    )
    module = str(args.get("module") or getattr(model, "module", "GCNConv") or "GCNConv")
    heads = _coerce_int(
        args.get("heads")
        or args.get("attention_heads")
        or getattr(model, "heads", None),
        1,
    )
    return ArchitectureSpec(num_layers=num_layers, module=module, heads=heads)


def collect_hyperparams(
    request: ExplainerRequest,
    *,
    progress: bool = True,
    output_dir: Optional[Path] = None,
    output_path: Optional[Path] = None,
    max_graphs: Optional[int] = None,
    fairness_config: Optional[FairnessConfig] = None,
) -> Path:
    """Collect advisor-suggested hyperparameters for every graph in the split."""

    device = torch.device(request.device) if request.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    updated_request = ExplainerRequest(
        dataset=request.dataset,
        graph_type=request.graph_type,
        backbone=request.backbone,
        split=request.split,
        method="subgraphx",
        device=str(device),
        checkpoint_name=request.checkpoint_name,
        gnn_root=request.gnn_root,
        graph_data_root=request.graph_data_root,
        hyperparams=request.hyperparams,
        profile=request.profile,
        num_shards=request.num_shards,
        shard_index=request.shard_index,
        fair_comparison=request.fair_comparison,
    )

    dataset, loader = load_graph_split(updated_request, batch_size=1, shuffle=False)
    model, train_args, run_dir = load_gnn_model(updated_request, dataset=dataset)

    profile_overrides = SUBGRAPHX_PROFILES.get((updated_request.profile or "").lower(), {})
    locked_overrides: Dict[str, float] = {}
    locked_overrides.update(profile_overrides)
    locked_overrides.update(updated_request.hyperparams or {})

    architecture_spec = _extract_architecture_spec(model, train_args)
    fair_mode = bool(updated_request.fair_comparison)
    fairness_advisor = (
        FairMultimodalHyperparameterAdvisor(fairness_config)
        if fair_mode
        else None
    )
    advisor: Optional[SubgraphXHyperparameterAdvisor] = None
    if not fair_mode:
        advisor = SubgraphXHyperparameterAdvisor(
            architecture=architecture_spec,
            locked_params=locked_overrides,
        )

    if output_path is not None:
        artifact_dir = output_path.parent
    else:
        base_dir = output_dir or run_dir / "explanations" / "subgraphx"
        artifact_dir = base_dir / (_make_slug(updated_request) + "_hyperparams")
        output_path = artifact_dir / "hyperparams.json"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    total = len(loader)
    if max_graphs is not None and max_graphs > 0:
        total = min(total, max_graphs)
    if progress:
        desc = f"CollectHParams[{updated_request.shard_index + 1}/{updated_request.num_shards}]"
        iterable = tqdm(
            loader,
            desc=desc,
            leave=False,
            position=updated_request.shard_index,
            dynamic_ncols=True,
            total=total,
        )
    else:
        iterable = loader

    per_graph: List[Dict[str, object]] = []
    for index, batch in enumerate(iterable):
        if max_graphs is not None and index >= max_graphs:
            break
        data: Data = batch
        if fairness_advisor is not None:
            params = fairness_advisor.subgraphx(
                num_nodes=int(getattr(data, "num_nodes", 0)),
                num_layers=architecture_spec.num_layers,
            )
        else:
            assert advisor is not None
            params = advisor.suggest(data)
        params["max_nodes"] = max(2, min(int(params["max_nodes"]), int(getattr(data, "num_nodes", 0) or params["max_nodes"])))
        per_graph.append(
            {
                "graph_index": index,
                "num_nodes": int(getattr(data, "num_nodes", 0)),
                "num_edges": int(getattr(data, "num_edges", getattr(data, "edge_index", torch.empty(2, 0)).size(1) if hasattr(data, "edge_index") and data.edge_index is not None else 0)),
                "hyperparams": dict(params),
            }
        )

    payload = {
        "method": "subgraphx",
        "dataset": updated_request.dataset,
        "graph_type": updated_request.graph_type,
        "split": updated_request.split,
        "backbone": updated_request.backbone,
        "num_shards": updated_request.num_shards,
        "shard_index": updated_request.shard_index,
        "base_defaults": fairness_advisor.describe() if fairness_advisor else dict(advisor.base_defaults),
        "locked_overrides": {} if fairness_advisor else dict(advisor.locked_params),
        "per_graph": per_graph,
    }

    output_path.write_text(json.dumps(payload, indent=2))
    return output_path


def _load_precomputed_hparams(path: Path) -> Tuple[Dict[int, Dict[str, float]], Dict[str, object]]:
    payload = json.loads(Path(path).read_text())
    graph_entries = payload.get("per_graph")
    if graph_entries is None:
        raise ValueError("Precomputed hyperparameter file missing 'per_graph' key")

    mapping: Dict[int, Dict[str, float]] = {}
    for entry in graph_entries:
        if "graph_index" not in entry:
            raise ValueError("Each entry in 'per_graph' must include 'graph_index'")
        index = int(entry["graph_index"])
        params = entry.get("hyperparams")
        if params is None:
            params = {k: v for k, v in entry.items() if k != "graph_index"}
        mapping[index] = {k: float(v) if k in FLOAT_PARAM_KEYS else int(v) if k in INT_PARAM_KEYS else v for k, v in params.items()}
    return mapping, payload


def explain_request(
    request: ExplainerRequest,
    *,
    progress: bool = True,
    hyperparams: Optional[Dict[str, float]] = None,
    precomputed_hparams: Optional[Dict[int, Dict[str, float]]] = None,
    precomputed_source: Optional[Path] = None,
    max_graphs: Optional[int] = None,
    fairness_config: Optional[FairnessConfig] = None,
    target_mode: str = "predicted",
    run_tag: Optional[str] = None,
) -> Tuple[List[SubgraphXResult], Path, Path, Optional[Path]]:
    """Run SubgraphX on the dataset implied by the request.

    ``target_mode`` selects the class the explainer targets: ``"predicted"`` (the
    model's argmax, the only choice that is label-free at inference time) or
    ``"label"`` (the legacy behaviour, which leaks the target into every
    confidence-based metric for misclassified instances).
    """
    if target_mode not in {"predicted", "label"}:
        raise ValueError(f"Unsupported target_mode: {target_mode}")

    device = torch.device(request.device) if request.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    updated_request = ExplainerRequest(
        dataset=request.dataset,
        graph_type=request.graph_type,
        backbone=request.backbone,
        split=request.split,
        method="subgraphx",
        device=str(device),
        checkpoint_name=request.checkpoint_name,
        gnn_root=request.gnn_root,
        graph_data_root=request.graph_data_root,
        hyperparams=request.hyperparams,
        profile=request.profile,
        num_shards=request.num_shards,
        shard_index=request.shard_index,
        fair_comparison=request.fair_comparison,
    )

    dataset, loader = load_graph_split(updated_request, batch_size=1, shuffle=False)
    model, train_args, run_dir = load_gnn_model(updated_request, dataset=dataset)

    wrapper = UniversalDataModelWrapper(model)
    profile_overrides = SUBGRAPHX_PROFILES.get((updated_request.profile or "").lower(), {})
    combined_overrides: Dict[str, float] = {}
    combined_overrides.update(profile_overrides)
    combined_overrides.update(updated_request.hyperparams or {})
    if hyperparams:
        combined_overrides.update(hyperparams)

    architecture_spec = _extract_architecture_spec(model, train_args)
    fair_mode = bool(updated_request.fair_comparison)
    fairness_advisor = (
        FairMultimodalHyperparameterAdvisor(fairness_config)
        if fair_mode
        else None
    )
    advisor: Optional[SubgraphXHyperparameterAdvisor] = None
    if not fair_mode:
        advisor = SubgraphXHyperparameterAdvisor(
            architecture=architecture_spec,
            locked_params=combined_overrides,
        )

    artifact_dir = run_dir / "explanations" / "subgraphx" / _make_slug(updated_request, run_tag)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    precomputed_lookup = precomputed_hparams or {}

    results: List[SubgraphXResult] = []
    per_graph_hparams: List[Dict[str, float]] = []
    energy_data: Dict[str, object] = {}
    total = len(loader)
    if max_graphs is not None and max_graphs > 0:
        total = min(total, max_graphs)
    if progress:
        desc = f"SubgraphX[{updated_request.shard_index + 1}/{updated_request.num_shards}]"
        iterable = tqdm(
            loader,
            desc=desc,
            leave=False,
            position=updated_request.shard_index,
            dynamic_ncols=True,
            total=total,
        )
    else:
        iterable = loader

    with EnergyMonitor("SubgraphX", output_dir=artifact_dir) as energy_monitor:
        for index, batch in enumerate(iterable):
            if max_graphs is not None and index >= max_graphs:
                break
            data: Data = batch.to(wrapper.device)
            label = int(data.y.item()) if hasattr(data, "y") and data.y is not None else None
            source = "advisor"
            candidate = precomputed_lookup.get(index)
            if fairness_advisor is not None:
                if candidate is not None:
                    warnings.warn(
                        "Ignoring precomputed hyperparameters when fair comparison mode is enabled.",
                    )
                graph_params = fairness_advisor.subgraphx(
                    num_nodes=data.num_nodes,
                    num_layers=architecture_spec.num_layers,
                )
                source = "fair_advisor"
            else:
                if candidate is not None:
                    graph_params = advisor.sanitise_for_graph(candidate, data)  # type: ignore[union-attr]
                    source = "precomputed"
                else:
                    graph_params = advisor.suggest(data)  # type: ignore[union-attr]
            graph_params["max_nodes"] = max(2, min(graph_params["max_nodes"], data.num_nodes))
            graph_dir = artifact_dir / f"graph_{index:05d}"
            explainer = _prepare_explainer(wrapper, train_args, graph_dir, graph_params)
            # The explained class must never depend on the label: use the model's own
            # prediction so that every downstream metric is computable without labels.
            origin_probs = explainer._predict_probs_from_inputs(data.x, data.edge_index)
            predicted_class: Optional[int] = int(torch.argmax(origin_probs).item()) if origin_probs.numel() else None
            predicted_confidence: Optional[float] = (
                float(origin_probs[predicted_class]) if predicted_class is not None else None
            )
            target_class = predicted_class if target_mode == "predicted" else label
            explanation, related_pred = explainer.explain(
                x=data.x,
                edge_index=data.edge_index,
                label=target_class,
                max_nodes=graph_params["max_nodes"],
            )
            final_params = dict(graph_params)
            related_pred["target_class"] = target_class
            related_pred["target_mode"] = target_mode
            related_pred["teacher_label"] = label
            origin_distribution = related_pred.get("origin_distribution")
            if origin_distribution and predicted_class is None:
                try:
                    max_index = max(range(len(origin_distribution)), key=lambda idx: origin_distribution[idx])
                    predicted_class = int(max_index)
                    predicted_confidence = float(origin_distribution[max_index])
                except (ValueError, TypeError):
                    predicted_class = None
                    predicted_confidence = None

            is_correct: Optional[bool] = None
            if label is not None and predicted_class is not None:
                try:
                    is_correct = int(label) == predicted_class
                except (TypeError, ValueError):
                    is_correct = label == predicted_class

            prediction_payload: Optional[Dict[str, object]] = None
            if predicted_class is not None:
                prediction_payload = {
                    "class": predicted_class,
                    "confidence": predicted_confidence,
                    "distribution": list(origin_distribution) if origin_distribution is not None else None,
                }

            if predicted_class is not None and "predicted_class" not in related_pred:
                related_pred["predicted_class"] = predicted_class
            if predicted_confidence is not None and "predicted_confidence" not in related_pred:
                related_pred["predicted_confidence"] = predicted_confidence

            if "ranked_nodes" not in related_pred:
                related_pred["ranked_nodes"] = list(related_pred.get("top_nodes", []))

            second_idx, second_conf, contrast = _contrastive_stats(origin_distribution, predicted_class)
            related_pred["origin_second_class"] = second_idx
            related_pred["origin_second_confidence"] = second_conf
            related_pred["origin_contrastivity"] = contrast

            masked_distribution = related_pred.get("masked_distribution")
            if masked_distribution is not None:
                _, masked_second_conf, masked_contrast = _contrastive_stats(masked_distribution, predicted_class)
                related_pred["masked_second_confidence"] = masked_second_conf
                related_pred["masked_contrastivity"] = masked_contrast
            else:
                related_pred.setdefault("masked_second_confidence", None)
                related_pred.setdefault("masked_contrastivity", None)

            maskout_distribution = related_pred.get("maskout_distribution")
            if maskout_distribution is not None:
                _, maskout_second_conf, maskout_contrast = _contrastive_stats(maskout_distribution, predicted_class)
                related_pred["maskout_second_confidence"] = maskout_second_conf
                related_pred["maskout_contrastivity"] = maskout_contrast
            else:
                related_pred.setdefault("maskout_second_confidence", None)
                related_pred.setdefault("maskout_contrastivity", None)

            # The progressions must span the explanation budget only (as GraphSVX and
            # TokenSHAP do): the ranking is restricted to the first |explanation_nodes|
            # entries, i.e. the selected coalition ordered by search credit. Runs before
            # 2026-09-23 stored the full ranking; see use_case/truncate_progressions_to_budget.py.
            top_nodes_sequence = list(related_pred.get("top_nodes") or [])
            explanation_nodes = related_pred.get("explanation_nodes") or []
            if explanation_nodes and len(top_nodes_sequence) > len(explanation_nodes):
                top_nodes_sequence = top_nodes_sequence[: len(explanation_nodes)]
            origin_confidence = related_pred.get("origin")
            if origin_confidence is not None and predicted_class is not None and top_nodes_sequence:
                progression_conf = explainer.cumulative_maskout_confidence(data, top_nodes_sequence, int(predicted_class))
                related_pred["maskout_progression_confidence"] = progression_conf
                related_pred["maskout_progression_drop"] = [origin_confidence - val for val in progression_conf]
                suff_conf = explainer.cumulative_sufficiency_confidence(data, top_nodes_sequence, int(predicted_class))
                related_pred["sufficiency_progression_confidence"] = suff_conf
                related_pred["sufficiency_progression_drop"] = [origin_confidence - val for val in suff_conf]

            node_tokens = _extract_node_tokens(data)
            if node_tokens:
                related_pred["node_tokens"] = node_tokens
                ranked_nodes = related_pred.get("ranked_nodes", [])
                related_pred["ranked_tokens"] = [
                    node_tokens[idx] for idx in ranked_nodes if isinstance(idx, int) and 0 <= idx < len(node_tokens)
                ]
                related_pred["top_token_text"] = [
                    node_tokens[idx] for idx in related_pred.get("top_nodes", [])
                    if isinstance(idx, int) and 0 <= idx < len(node_tokens)
                ]

            results.append(
                SubgraphXResult(
                    graph_index=index,
                    label=label,
                    explanation=explanation,
                    related_prediction=related_pred,
                    num_nodes=data.num_nodes,
                    num_edges=data.num_edges,
                    hyperparams=final_params,
                    prediction=prediction_payload,
                    is_correct=is_correct,
                )
            )
            per_graph_hparams.append({"graph_index": index, "source": source, **final_params})

        energy_data = getattr(energy_monitor, "result", None) or energy_monitor.get_stats()

    summary = {
        "method": "subgraphx",
        "dataset": updated_request.dataset,
        "graph_type": updated_request.graph_type,
        "split": updated_request.split,
        "backbone": updated_request.backbone,
        "num_shards": updated_request.num_shards,
        "shard_index": updated_request.shard_index,
        "num_graphs": len(results),
        "target_mode": target_mode,
        "run_tag": run_tag,
        "hyperparams": {
            "base_defaults": fairness_advisor.describe() if fairness_advisor else dict(advisor.base_defaults),
            "locked_overrides": {} if fairness_advisor else dict(advisor.locked_params),
            "precomputed_source": str(precomputed_source) if precomputed_source else None,
            "per_graph": per_graph_hparams,
        },
        "energy": energy_data,
        "graphs": [entry.to_json() for entry in results],
    }

    summary_path = artifact_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    if energy_data:
        energy_path = artifact_dir / "energy_metrics.json"
        energy_path.write_text(json.dumps(energy_data, indent=2))

    raw_path: Optional[Path] = artifact_dir / "results.pkl"
    try:
        with raw_path.open("wb") as handle:
            pickle.dump(results, handle)
    except Exception as exc:  # pragma: no cover - safeguard for unexpected objects
        warnings.warn(f"Could not pickle SubgraphX results: {exc}")
        raw_path = None

    return results, artifact_dir, summary_path, raw_path


def _env_request() -> ExplainerRequest:
    dataset = os.getenv("GRAPHTEXT_DATASET", "sst2")
    graph_type = os.getenv("GRAPHTEXT_GRAPH_TYPE", "syntactic")
    backbone = os.getenv("GRAPHTEXT_BACKBONE", "stanfordnlp")
    split = os.getenv("GRAPHTEXT_SPLIT", "validation")
    device = os.getenv("GRAPHTEXT_DEVICE")
    gnn_root = Path(os.getenv("GRAPHTEXT_GNN_ROOT", str(DEFAULT_GNN_ROOT)))
    data_root = Path(os.getenv("GRAPHTEXT_GRAPH_ROOT", str(DEFAULT_GRAPH_DATA_ROOT)))
    fair_flag = os.getenv("GRAPHTEXT_FAIR_COMPARISON")

    return ExplainerRequest(
        dataset=dataset,
        graph_type=graph_type,
        backbone=backbone,
        split=split,
        device=device,
        gnn_root=gnn_root,
        graph_data_root=data_root,
        fair_comparison=bool(int(fair_flag)) if fair_flag is not None else False,
    )


def main(argv: Optional[List[str]] = None) -> None:  # pragma: no cover - CLI helper
    parser = argparse.ArgumentParser(description="SubgraphX Explainability Runner")
    parser.add_argument(
        "--collect-only",
        action="store_true",
        help="Only collect advisor-suggested hyperparameters and exit.",
    )
    parser.add_argument(
        "--hyperparams-out",
        type=Path,
        help="Destination JSON file when collecting hyperparameters.",
    )
    parser.add_argument(
        "--hyperparams-in",
        type=Path,
        help="Use previously collected hyperparameters from JSON when running explanations.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars.",
    )
    parser.add_argument("--dataset", type=str, help="Dataset identifier passed to the explainer.")
    parser.add_argument("--graph-type", type=str, help="Graph type within the dataset.")
    parser.add_argument("--backbone", type=str, help="Backbone name (e.g., SetFit).")
    parser.add_argument("--split", type=str, help="Data split to use (train/validation/test).")
    parser.add_argument("--device", type=str, help="Explicit torch device (cpu/cuda:0).")
    parser.add_argument("--gnn-root", type=Path, help="Override path to trained GNN checkpoints.")
    parser.add_argument("--graph-data-root", type=Path, help="Override path to precomputed graphs.")
    parser.add_argument("--checkpoint-name", type=str, help="Checkpoint filename to load.")
    parser.add_argument("--profile", type=str, help="SubgraphX profile key (fast/quality).")
    parser.add_argument("--num-shards", type=int, help="Total shards when running in parallel.")
    parser.add_argument("--shard-index", type=int, help="Index of this shard (0-based).")
    parser.add_argument("--batch-size", type=int, help="Override batch size for data loader.")
    parser.add_argument("--max-graphs", type=int, help="Limit the number of graphs processed (for smoke tests).")
    parser.add_argument("--fair", action="store_true", help="Enable fair multimodal advisor alignment.")
    parser.add_argument(
        "--target-forward-passes",
        type=int,
        default=400,
        help="Target forward pass budget when using --fair (default: 400).",
    )
    parser.add_argument(
        "--target-class",
        choices=("predicted", "label"),
        default="predicted",
        help="Class explained by SubgraphX: the model prediction (default, label-free) or the dataset label (legacy, leaks the target).",
    )
    parser.add_argument(
        "--run-tag",
        type=str,
        default=None,
        help="Optional tag inserted in the artifact directory name (before the shard suffix).",
    )
    args = parser.parse_args(argv)

    request = _env_request()

    overrides: Dict[str, object] = {}
    if args.dataset:
        overrides["dataset"] = args.dataset
    if args.graph_type:
        overrides["graph_type"] = args.graph_type
    if args.backbone:
        overrides["backbone"] = args.backbone
    if args.split:
        overrides["split"] = args.split
    if args.device:
        overrides["device"] = args.device
    if args.gnn_root:
        overrides["gnn_root"] = args.gnn_root
    if args.graph_data_root:
        overrides["graph_data_root"] = args.graph_data_root
    if args.checkpoint_name:
        overrides["checkpoint_name"] = args.checkpoint_name
    if args.profile:
        overrides["profile"] = args.profile
    if args.num_shards is not None:
        overrides["num_shards"] = args.num_shards
    if args.shard_index is not None:
        overrides["shard_index"] = args.shard_index
    if args.fair:
        overrides["fair_comparison"] = True

    if overrides:
        request = replace(request, **overrides)

    target_budget = args.target_forward_passes if args.target_forward_passes and args.target_forward_passes > 0 else 400
    fairness_config = FairnessConfig(compute_budget=int(target_budget))

    max_graphs = args.max_graphs if args.max_graphs and args.max_graphs > 0 else None

    if args.collect_only:
        output_path = collect_hyperparams(
            request,
            progress=not args.no_progress,
            output_path=args.hyperparams_out,
            max_graphs=max_graphs,
            fairness_config=fairness_config,
        )
        print(f"Saved SubgraphX hyperparameters to {output_path}")
        return

    precomputed_lookup: Optional[Dict[int, Dict[str, float]]] = None
    precomputed_source: Optional[Path] = None
    if args.hyperparams_in:
        precomputed_lookup, payload = _load_precomputed_hparams(args.hyperparams_in)
        precomputed_source = Path(args.hyperparams_in).resolve()
        mismatches: List[str] = []
        for key in ("dataset", "graph_type", "split", "backbone"):
            expected = getattr(request, key)
            observed = payload.get(key)
            if observed is not None and str(observed) != str(expected):
                mismatches.append(f"{key}={observed} (expected {expected})")
        if mismatches:
            warnings.warn(
                "Precomputed hyperparameters metadata differs from request: "
                + "; ".join(mismatches)
            )

    results, artifact_dir, summary_path, _ = explain_request(
        request,
        progress=not args.no_progress,
        precomputed_hparams=precomputed_lookup,
        precomputed_source=precomputed_source,
        max_graphs=max_graphs,
        fairness_config=fairness_config,
        target_mode=args.target_class,
        run_tag=args.run_tag,
    )

    output_path = Path("subgraphx_results.json")
    output_path.write_text(json.dumps([entry.to_json() for entry in results], indent=2))
    print(f"Saved SubgraphX explanations to {output_path}")
    print(f"Artifacts: {artifact_dir}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":  # pragma: no cover - CLI invocation
    main()
