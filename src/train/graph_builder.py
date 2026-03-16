from __future__ import annotations

from typing import Dict, List, Tuple

import torch

from .graph_models import BipartiteGraphBatch


def build_bipartite_batch(
    feats: torch.Tensor,
    targets: torch.Tensor,
) -> BipartiteGraphBatch:
    """Construct a simple bipartite batch graph from features and multi-hot targets.

    Args:
        feats:   [B, O, D] fused object features (one object per image for now).
        targets: [B, A]   multi-hot attribute labels.

    Returns:
        BipartiteGraphBatch with:
          - object_feats: feats
          - attr_feats:   trainable attribute embeddings will be attached later
          - edge_index:   edges from object indices to attribute indices
          - edge_weight:  all ones
    """
    device = feats.device
    bsz, num_objects, _ = feats.shape
    num_attrs = targets.shape[1]

    # For smoke training we assume one object per image (num_objects == 1).
    # We still keep the general formulation for clarity.
    object_indices: List[int] = []
    attr_indices: List[int] = []

    # Flatten batch/object into a single dimension of size B * O.
    for b in range(bsz):
        for o in range(num_objects):
            flat_obj_idx = b * num_objects + o
            active_attrs = (targets[b] > 0.5).nonzero(as_tuple=False).view(-1)
            for a in active_attrs.tolist():
                object_indices.append(flat_obj_idx)
                attr_indices.append(b * num_attrs + a)

    if not object_indices:
        # No positive labels in batch; create a dummy self-loop to keep shapes valid.
        object_indices = [0]
        attr_indices = [0]

    edge_index = torch.tensor(
        [object_indices, attr_indices],
        dtype=torch.long,
        device=device,
    )
    edge_weight = torch.ones(edge_index.shape[1], device=device)

    # Attribute features are left as zeros here; they will be replaced by the model
    # with either CLIP text or trainable embeddings. For smoke we just keep zeros.
    attr_feats = torch.zeros(bsz, num_attrs, feats.shape[-1], device=device)

    return BipartiteGraphBatch(
        object_feats=feats,
        attr_feats=attr_feats,
        edge_index=edge_index,
        edge_weight=edge_weight,
    )

