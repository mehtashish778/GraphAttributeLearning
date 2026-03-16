from __future__ import annotations

import torch


def bce_logits_loss(logits: torch.Tensor, targets: torch.Tensor, pos_weight: torch.Tensor) -> torch.Tensor:
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    return loss_fn(logits, targets)
