from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
from torch import nn


@dataclass
class BipartiteGraphBatch:
    """Lightweight container for a batch of bipartite graphs.

    All tensors live on the same device. Shapes:
      - object_feats:  [B, num_objects, in_dim]
      - attr_feats:    [B, num_attrs, attr_dim]
      - edge_index:    [2, num_edges] with edges from object -> attribute
      - edge_weight:   [num_edges]
    """

    object_feats: torch.Tensor
    attr_feats: torch.Tensor
    edge_index: torch.Tensor
    edge_weight: torch.Tensor


class BipartiteMessagePassingLayer(nn.Module):
    """Single message passing layer for object <-> attribute bipartite graphs.

    For smoke training we keep a simple formulation:
      1) Aggregate attribute messages into each object using weighted mean.
      2) Project and combine with previous object features via residual MLP.
    """

    def __init__(self, in_dim: int, out_dim: int, attr_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.attr_to_obj = nn.Linear(attr_dim, in_dim)
        self.proj = nn.Linear(in_dim, out_dim)
        self.update = nn.Sequential(
            nn.Linear(in_dim + out_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        object_feats: torch.Tensor,
        attr_feats: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # object_feats: [B, O, Din], attr_feats: [B, A, Din_attr]
        bsz, num_objects, _ = object_feats.shape
        device = object_feats.device

        if edge_weight is None:
            edge_weight = torch.ones(edge_index.shape[1], device=device)

        src_obj = edge_index[0]  # indices in [0, B*O)
        src_attr = edge_index[1]  # indices in [0, B*A)

        # Flatten batch/object and batch/attr dimensions for gathering.
        flat_objects = object_feats.reshape(bsz * num_objects, -1)
        flat_attrs = attr_feats.reshape(bsz * attr_feats.shape[1], -1)

        # Messages from attributes to objects.
        attr_msgs = flat_attrs.index_select(0, src_attr)  # [E, Din_attr]
        attr_msgs = self.attr_to_obj(attr_msgs)  # [E, Din]
        w = edge_weight.view(-1, 1)
        weighted_msgs = attr_msgs * w

        # Aggregate messages per object index.
        agg = torch.zeros_like(flat_objects, device=device)
        agg.index_add_(0, src_obj, weighted_msgs)

        # Normalize by total incoming weight per object to compute mean.
        weight_sums = torch.zeros(flat_objects.shape[0], device=device)
        weight_sums.index_add_(0, src_obj, edge_weight)
        weight_sums = weight_sums.clamp_min(1e-6).view(-1, 1)
        agg = agg / weight_sums

        # Project aggregated messages and combine with original object features.
        proj_msgs = self.proj(agg)
        combined = torch.cat([flat_objects, proj_msgs], dim=-1)
        updated = self.update(combined)

        return updated.view(bsz, num_objects, self.out_dim)


class NativeGNNClassifier(nn.Module):
    """Simple bipartite GNN classifier for multi-label attribute prediction."""

    def __init__(
        self,
        in_dim: int,
        hidden_dims: List[int],
        num_attributes: int,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        dims = [in_dim] + hidden_dims
        attr_dim = in_dim
        for dim_in, dim_out in zip(dims[:-1], dims[1:]):
            layers.append(BipartiteMessagePassingLayer(dim_in, dim_out, attr_dim=attr_dim, dropout=dropout))
        self.layers = nn.ModuleList(layers)
        self.classifier = nn.Linear(dims[-1], num_attributes)

    def forward(
        self,
        graph: BipartiteGraphBatch,
    ) -> torch.Tensor:
        x = graph.object_feats
        for layer in self.layers:
            x = layer(
                object_feats=x,
                attr_feats=graph.attr_feats,
                edge_index=graph.edge_index,
                edge_weight=graph.edge_weight,
            )
        # Predict attributes for each object, then average over objects in batch.
        logits_per_object = self.classifier(x)  # [B, O, num_attributes]
        return logits_per_object.mean(dim=1)

