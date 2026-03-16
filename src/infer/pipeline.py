from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import torch
from PIL import Image

from train.dataset import load_json
from .model_adapters import BaselineAdapter, GNNAdapter, InferenceResult


@dataclass
class LoadedModel:
    adapter: Any
    label_vocab: Dict[str, int]


def _detect_model_type(state: Dict[str, Any]) -> str:
    if "model_state" in state:
        return "baseline"
    if "gnn_state" in state and "backbone_state" in state:
        return "gnn"
    raise RuntimeError("Unrecognized checkpoint format (expected baseline or GNN keys).")


def load_checkpoint(path: Path, device: torch.device | None = None) -> LoadedModel:
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    state: Dict[str, Any] = torch.load(path, map_location=device)
    model_type = _detect_model_type(state)
    label_vocab = state["label_vocab"]
    if model_type == "baseline":
        adapter = BaselineAdapter(state, device=device)
    else:
        adapter = GNNAdapter(state, device=device)
    return LoadedModel(adapter=adapter, label_vocab=label_vocab)


def predict_image(
    image_path: Path,
    loaded: LoadedModel,
    top_k: int = 5,
    threshold: float = 0.5,
) -> InferenceResult:
    image = Image.open(image_path).convert("RGB")
    return loaded.adapter.predict(
        image=image,
        label_vocab=loaded.label_vocab,
        top_k=top_k,
        threshold=threshold,
    )

