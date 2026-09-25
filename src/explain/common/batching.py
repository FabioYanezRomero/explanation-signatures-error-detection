"""Batched evaluation of node-masked copies of a graph.

Builds a disjoint-union batch of masked copies with a handful of tensor ops
(instead of one ``Data`` object per mask plus a ``DataLoader`` collate), so that a
GNN can score hundreds of coalitions in a single forward pass.  Used by both the
SubgraphX Shapley sampler and the GraphSVX-style explainer.
"""
from __future__ import annotations

import os
from typing import Callable

import torch
from torch_geometric.data import Data

CHUNK_SIZE = int(os.environ.get("SUBGRAPHX_SHAPLEY_CHUNK", "512"))


def build_masked_batch(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    masks: torch.Tensor,
    *,
    split: bool = False,
) -> Data:
    """Return a disjoint-union batch of ``masks.shape[0]`` copies of the graph.

    ``split=False`` reproduces DIG's ``graph_build_zero_filling`` (features of
    unselected nodes zeroed, edges untouched); ``split=True`` reproduces
    ``graph_build_split`` (features untouched, edges touching unselected nodes
    removed).
    """
    num_masks, num_nodes = masks.shape
    device = x.device
    masks = masks.to(device=device, dtype=x.dtype)
    if split:
        x_batch = x.repeat(num_masks, 1)
    else:
        x_batch = (x.unsqueeze(0) * masks.unsqueeze(-1)).reshape(num_masks * num_nodes, x.size(1))

    offsets = (torch.arange(num_masks, device=device) * num_nodes).view(num_masks, 1, 1)
    edge_batch = edge_index.unsqueeze(0) + offsets  # (B, 2, E)
    if split:
        keep = (masks[:, edge_index[0]] > 0.5) & (masks[:, edge_index[1]] > 0.5)  # (B, E)
        edge_batch = edge_batch.permute(1, 0, 2)[:, keep]
    else:
        edge_batch = edge_batch.permute(1, 0, 2).reshape(2, -1)

    batch_vector = torch.arange(num_masks, device=device).repeat_interleave(num_nodes)
    data = Data(x=x_batch, edge_index=edge_batch, batch=batch_vector)
    data.num_graphs = num_masks
    return data


def evaluate_masks(
    value_func: Callable[[Data], torch.Tensor],
    x: torch.Tensor,
    edge_index: torch.Tensor,
    masks: torch.Tensor,
    *,
    split: bool = False,
    chunk_size: int = CHUNK_SIZE,
) -> torch.Tensor:
    """Apply ``value_func`` to every mask, chunking to bound GPU memory."""
    outputs = []
    for chunk in masks.split(max(1, chunk_size)):
        outputs.append(value_func(build_masked_batch(x, edge_index, chunk, split=split)))
    return torch.cat(outputs, dim=0)
