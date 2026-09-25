"""Vectorised coalition evaluation for DIG's SubgraphX.

DIG evaluates every Monte-Carlo Shapley sample by materialising one ``Data`` object
per mask on the GPU and collating them through a ``DataLoader``.  Profiling shows
that ~70% of SubgraphX wall time is spent inside that collate (hundreds of
thousands of tiny ``Tensor.to`` calls).  The functions below build the very same
disjoint-union batch with a handful of tensor ops instead, so the model sees
exactly the same inputs (same masked features, same edges, same graph ordering)
and the returned marginal contributions are numerically equivalent up to
floating-point summation order.

Set ``SUBGRAPHX_FAST_SHAPLEY=0`` to fall back to the reference DIG implementation.
"""
from __future__ import annotations

import os
from typing import Callable, Optional, Sequence

import numpy as np
import torch
from torch_geometric.data import Data

from src.explain.common.batching import CHUNK_SIZE, build_masked_batch, evaluate_masks  # noqa: F401


def fast_marginal_contribution(
    data: Data,
    exclude_mask: np.ndarray,
    include_mask: np.ndarray,
    value_func: Callable[[Data], torch.Tensor],
    subgraph_build_func: Callable,
) -> torch.Tensor:
    """Drop-in replacement for ``dig.xgraph.method.shapley.marginal_contribution``."""
    split = getattr(subgraph_build_func, "__name__", "") == "graph_build_split"
    x, edge_index = data.x, data.edge_index
    exclude = torch.as_tensor(np.asarray(exclude_mask), dtype=torch.float32, device=x.device)
    include = torch.as_tensor(np.asarray(include_mask), dtype=torch.float32, device=x.device)
    with torch.no_grad():
        include_values = evaluate_masks(value_func, x, edge_index, include, split=split)
        exclude_values = evaluate_masks(value_func, x, edge_index, exclude, split=split)
    return include_values - exclude_values


def install(force: Optional[bool] = None) -> bool:
    """Monkey-patch DIG unless ``SUBGRAPHX_FAST_SHAPLEY`` disables it."""
    enabled = force if force is not None else os.environ.get("SUBGRAPHX_FAST_SHAPLEY", "1") not in {"0", "false", "False"}
    if not enabled:
        return False
    import dig.xgraph.method.shapley as shapley_mod  # type: ignore

    if getattr(shapley_mod.marginal_contribution, "__name__", "") != "fast_marginal_contribution":
        shapley_mod._reference_marginal_contribution = shapley_mod.marginal_contribution
        shapley_mod.marginal_contribution = fast_marginal_contribution
    return True
