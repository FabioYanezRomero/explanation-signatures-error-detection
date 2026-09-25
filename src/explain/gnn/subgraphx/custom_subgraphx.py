import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch_geometric.data import Batch, Data

from dig.xgraph.method import SubgraphX

from .fast_shapley import build_masked_batch, install as _install_fast_shapley

FAST_SHAPLEY_ENABLED = _install_fast_shapley()


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


class CustomSubgraphX(SubgraphX):
    """Augmented SubgraphX explainer that captures probability distributions."""

    def __init__(self, *args, value_func=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._custom_value_func = value_func
        if value_func is not None:
            warnings.warn(
                "Custom value_func injected into SubgraphX. This will override internal model calls."
            )

    def explain(  # type: ignore[override]
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        label: Optional[int],
        max_nodes: int = 5,
        node_idx: Optional[int] = None,
        saved_MCTSInfo_list: Optional[List[List]] = None,
        **kwargs,
    ):
        call_kwargs: Dict[str, object] = {
            "x": x,
            "edge_index": edge_index,
            "label": label,
            "max_nodes": max_nodes,
            "node_idx": node_idx,
            "saved_MCTSInfo_list": saved_MCTSInfo_list,
        }
        call_kwargs.update(kwargs)
        if self._custom_value_func is not None:
            call_kwargs["value_func"] = self._custom_value_func

        def _filtered(include_value_func: bool) -> Dict[str, object]:
            filtered: Dict[str, object] = {}
            for key, value in call_kwargs.items():
                if key == "value_func" and not include_value_func:
                    continue
                if key in {"x", "edge_index", "label", "value_func"} or value is not None:
                    filtered[key] = value
            return filtered

        try:
            results, related_pred = super().explain(**_filtered(include_value_func=True))
        except TypeError as exc:
            if "value_func" in call_kwargs and "unexpected keyword argument 'value_func'" in str(exc):
                results, related_pred = super().explain(**_filtered(include_value_func=False))
            else:
                raise

        try:
            self._augment_related_prediction(results, related_pred, x, edge_index, label, max_nodes=max_nodes)
        except Exception as exc:  # pragma: no cover - defensive guard
            warnings.warn(f"Failed to augment SubgraphX probabilities: {exc}")
        return results, related_pred

    def _augment_related_prediction(
        self,
        results,
        related_pred: Dict[str, object],
        x: torch.Tensor,
        edge_index: torch.Tensor,
        label: Optional[int],
        max_nodes: Optional[int] = None,
    ) -> None:
        if not isinstance(related_pred, dict):
            return

        self._last_max_nodes = max_nodes
        base_data, coalition = self._extract_primary_data(results, max_nodes=max_nodes)
        if isinstance(coalition, list):
            num_nodes = int(x.size(0))
            related_pred["explanation_nodes"] = [int(idx) for idx in coalition]
            related_pred["sparsity_included"] = len(coalition) / max(num_nodes, 1)
            related_pred["top_nodes"] = self.survival_ranking(getattr(self, "_last_entries", []), coalition, num_nodes)
            related_pred["ranking_method"] = "mcts_survival_credit"

        origin_probs = self._predict_probs_from_inputs(x, edge_index)
        origin_distribution = origin_probs.detach().cpu().tolist()
        if origin_distribution:
            related_pred["origin_distribution"] = [float(v) for v in origin_distribution]

        target_index = self._resolve_target_index(label, origin_probs)
        if target_index is not None and 0 <= target_index < len(origin_distribution):
            related_pred["origin"] = float(origin_distribution[target_index])

        masked_distribution, maskout_distribution = self._compute_masked_distributions(results)

        if masked_distribution is not None:
            related_pred["masked_distribution"] = [float(v) for v in masked_distribution]
            if target_index is not None and 0 <= target_index < len(masked_distribution):
                related_pred["masked"] = float(masked_distribution[target_index])
        else:
            related_pred.setdefault("masked_distribution", None)

        if maskout_distribution is not None:
            related_pred["maskout_distribution"] = [float(v) for v in maskout_distribution]
            if target_index is not None and 0 <= target_index < len(maskout_distribution):
                related_pred["maskout"] = float(maskout_distribution[target_index])
        else:
            related_pred.setdefault("maskout_distribution", None)

        second_idx, second_conf, contrast = _contrastive_stats(origin_distribution, target_index)
        related_pred["origin_second_class"] = second_idx
        related_pred["origin_second_confidence"] = second_conf
        related_pred["origin_contrastivity"] = contrast

        if masked_distribution is not None:
            _, masked_second_conf, masked_contrast = _contrastive_stats(masked_distribution, target_index)
            related_pred["masked_second_confidence"] = masked_second_conf
            related_pred["masked_contrastivity"] = masked_contrast
        else:
            related_pred.setdefault("masked_second_confidence", None)
            related_pred.setdefault("masked_contrastivity", None)

        if maskout_distribution is not None:
            _, maskout_second_conf, maskout_contrast = _contrastive_stats(maskout_distribution, target_index)
            related_pred["maskout_second_confidence"] = maskout_second_conf
            related_pred["maskout_contrastivity"] = maskout_contrast
        else:
            related_pred.setdefault("maskout_second_confidence", None)
            related_pred.setdefault("maskout_contrastivity", None)

        node_tokens = None
        if base_data is not None:
            node_tokens = _extract_node_tokens(base_data)
        if node_tokens:
            related_pred["node_tokens"] = node_tokens
            ranked_nodes = related_pred.get("top_nodes", [])
            related_pred["ranked_tokens"] = [
                node_tokens[idx] for idx in ranked_nodes if isinstance(idx, int) and 0 <= idx < len(node_tokens)
            ]
            related_pred["top_token_text"] = [
                node_tokens[idx] for idx in related_pred.get("top_nodes", [])
                if isinstance(idx, int) and 0 <= idx < len(node_tokens)
            ]

        related_pred.setdefault("ranked_nodes", related_pred.get("top_nodes", []))

    def _resolve_target_index(
        self, label: Optional[int], origin_probs: torch.Tensor
    ) -> Optional[int]:
        if label is not None:
            try:
                idx = int(label)
                if 0 <= idx < origin_probs.numel():
                    return idx
            except (TypeError, ValueError):
                pass
        if origin_probs.numel() == 0:
            return None
        return int(torch.argmax(origin_probs).item())

    def _compute_masked_distributions(
        self, results
    ) -> Tuple[Optional[List[float]], Optional[List[float]]]:
        base_data, coalition = self._extract_primary_data(results, max_nodes=getattr(self, "_last_max_nodes", None))
        if base_data is None or not coalition:
            return None, None

        num_nodes = base_data.num_nodes or base_data.x.size(0)
        if num_nodes <= 0:
            return None, None

        valid_nodes = sorted({idx for idx in coalition if 0 <= idx < num_nodes})
        if not valid_nodes:
            return None, None

        mask_keep = torch.zeros(num_nodes, dtype=torch.float32)
        mask_keep[valid_nodes] = 1.0
        mask_drop = torch.ones(num_nodes, dtype=torch.float32)
        mask_drop[valid_nodes] = 0.0

        masked_batch = self._build_batch(base_data, mask_keep)
        maskout_batch = self._build_batch(base_data, mask_drop)

        masked_probs = self._predict_probs_from_inputs(masked_batch)
        maskout_probs = self._predict_probs_from_inputs(maskout_batch)

        return (
            masked_probs.detach().cpu().tolist(),
            maskout_probs.detach().cpu().tolist(),
        )

    def _extract_primary_data(
        self, results, max_nodes: Optional[int] = None
    ) -> Tuple[Optional[Data], Optional[Sequence[int]]]:
        """Return the graph and DIG's explanation coalition (highest score with <= max_nodes).

        DIG sorts the explored MCTS nodes by score, so ``results[0]`` is typically the
        (almost) full graph; the explanation is the best coalition within the node budget.
        """
        if not results:
            return None, None
        entries = results[0] if (isinstance(results[0], list) and results[0]) else results
        entries = [e for e in entries if isinstance(e, dict)]
        if not entries:
            return None, None
        budget = int(max_nodes) if max_nodes is not None else len(entries[0].get("coalition") or [])
        ordered = sorted(entries, key=lambda e: len(e.get("coalition") or []))
        best = ordered[0]
        for entry in ordered:
            if len(entry.get("coalition") or []) <= budget and float(entry.get("P", 0.0)) > float(best.get("P", 0.0)):
                best = entry
        data_obj = best.get("data")
        if isinstance(data_obj, Batch):
            data_list = data_obj.to_data_list()
            base_data = data_list[0] if data_list else None
        elif isinstance(data_obj, Data):
            base_data = data_obj
        else:
            return None, None
        coalition = [int(v) for v in (best.get("coalition") or [])]
        self._last_entries = entries
        return base_data, coalition

    @staticmethod
    def survival_ranking(entries, coalition, num_nodes: int) -> List[int]:
        """Node ranking from the MCTS search: sum over explored coalitions of P/|C|; coalition first."""
        credit = [0.0] * num_nodes
        for entry in entries:
            members = entry.get("coalition") or []
            if not members:
                continue
            share = float(entry.get("P", 0.0)) / len(members)
            for node in members:
                if 0 <= int(node) < num_nodes:
                    credit[int(node)] += share
        inside = set(int(n) for n in coalition)
        return sorted(inside, key=lambda n: -credit[n]) + sorted((n for n in range(num_nodes) if n not in inside), key=lambda n: -credit[n])

    def _build_batch(self, data: Data, mask: torch.Tensor) -> Batch:
        masked = data.clone().cpu()
        mask = mask.to(masked.x.device, dtype=masked.x.dtype)
        masked.x = masked.x * mask.unsqueeze(1)

        if self.subgraph_building_method == "split":
            node_mask = mask > 0.5
            row, col = masked.edge_index
            edge_mask = node_mask[row] & node_mask[col]
            masked.edge_index = masked.edge_index[:, edge_mask]
            if getattr(masked, "edge_attr", None) is not None:
                masked.edge_attr = masked.edge_attr[edge_mask]

        batch = Batch.from_data_list([masked])
        return batch.to(self.device)

    def _batched_target_probs(
        self, data: Data, masks: torch.Tensor, target_index: int
    ) -> List[float]:
        """Probability of ``target_index`` for every node mask, evaluated in batches."""
        if masks.numel() == 0:
            return []
        x = data.x.to(self.device)
        edge_index = data.edge_index.to(self.device)
        split = self.subgraph_building_method == "split"
        confidences: List[float] = []
        with torch.no_grad():
            for chunk in masks.split(256):
                batch = build_masked_batch(x, edge_index, chunk, split=split)
                logits = self.model(data=batch)
                if isinstance(logits, tuple):
                    logits = logits[0]
                probs = torch.softmax(logits, dim=-1)
                if probs.size(-1) <= target_index:
                    continue
                confidences.extend(probs[:, target_index].detach().cpu().tolist())
        return confidences

    @staticmethod
    def _valid_node_sequence(data: Data, ordered_nodes: Sequence[int]) -> Tuple[int, List[int]]:
        num_nodes = getattr(data, "num_nodes", None) or data.x.size(0)
        valid: List[int] = []
        for node_idx in ordered_nodes:
            if node_idx is None:
                continue
            try:
                node_int = int(node_idx)
            except (TypeError, ValueError):
                continue
            if 0 <= node_int < num_nodes:
                valid.append(node_int)
        return num_nodes, valid

    def cumulative_maskout_confidence(
        self,
        data: Data,
        ordered_nodes: Sequence[int],
        target_index: Optional[int],
    ) -> List[float]:
        """Confidence after removing the top-1, top-2, ... ranked nodes (necessity curve)."""
        if target_index is None:
            return []
        num_nodes, nodes = self._valid_node_sequence(data, ordered_nodes)
        if num_nodes <= 0 or not nodes:
            return []
        masks = torch.ones(len(nodes), num_nodes, dtype=torch.float32, device=self.device)
        for step, node_int in enumerate(nodes):
            masks[step:, node_int] = 0.0
        return self._batched_target_probs(data, masks, int(target_index))

    def cumulative_sufficiency_confidence(
        self,
        data: Data,
        ordered_nodes: Sequence[int],
        target_index: Optional[int],
    ) -> List[float]:
        """Confidence when keeping only the top-1, top-2, ... ranked nodes (sufficiency curve)."""
        if target_index is None:
            return []
        num_nodes, nodes = self._valid_node_sequence(data, ordered_nodes)
        if num_nodes <= 0 or not nodes:
            return []
        masks = torch.zeros(len(nodes), num_nodes, dtype=torch.float32, device=self.device)
        for step, node_int in enumerate(nodes):
            masks[step:, node_int] = 1.0
        return self._batched_target_probs(data, masks, int(target_index))

    def _predict_probs_from_inputs(self, *model_args, **model_kwargs) -> torch.Tensor:
        with torch.no_grad():
            logits = self.model(*model_args, **model_kwargs)
        if isinstance(logits, tuple):
            logits = logits[0]
        logits = logits.squeeze()
        if logits.dim() == 0:
            logits = logits.unsqueeze(0)
        probs = torch.softmax(logits, dim=-1)
        if probs.dim() > 1:
            probs = probs[0]
        return probs
