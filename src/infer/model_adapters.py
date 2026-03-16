from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch
from PIL import Image

from train.encoders import build_backbones
from train.graph_builder import build_bipartite_batch
from train.graph_models import NativeGNNClassifier
from train.models import BaselineClassifier


@dataclass
class InferenceResult:
    labels: List[str]
    scores: List[float]
    positives: List[str]


def _sorted_labels(
    scores: torch.Tensor,
    label_vocab: Dict[str, int],
    top_k: int,
    threshold: float,
) -> InferenceResult:
    inv_vocab = {idx: label for label, idx in label_vocab.items()}
    probs = scores.sigmoid().detach().cpu().view(-1)
    values, indices = torch.topk(probs, k=min(top_k, probs.numel()))
    labels = [inv_vocab[int(i)] for i in indices]
    positives = [label for label, score in zip(labels, values) if float(score) >= threshold]
    return InferenceResult(
        labels=labels,
        scores=[float(v) for v in values],
        positives=positives,
    )


class BaseAdapter:
    def __init__(self, device: torch.device) -> None:
        self.device = device

    def predict(
        self,
        image: Image.Image,
        label_vocab: Dict[str, int],
        top_k: int = 5,
        threshold: float = 0.5,
    ) -> InferenceResult:
        raise NotImplementedError


class BaselineAdapter(BaseAdapter):
    def __init__(self, checkpoint: Dict[str, Any], device: torch.device) -> None:
        super().__init__(device)
        cfg = checkpoint["config"]
        num_labels = len(checkpoint["label_vocab"])
        use_clip = bool(cfg["encoders"]["clip"]["enabled"])
        use_dino = bool(cfg["encoders"]["dino"]["enabled"])
        clip_name = str(cfg["encoders"]["clip"].get("model_name", "vit_base_patch16_clip_224.openai"))
        dino_name = str(cfg["encoders"]["dino"].get("model_name", "vit_base_patch14_dinov2.lvd142m"))
        if clip_name == "ViT-B-32":
            clip_name = "vit_base_patch16_clip_224.openai"
        if dino_name == "vit_base_patch14_dinov2":
            dino_name = "vit_base_patch14_dinov2.lvd142m"

        clip_backbone, dino_backbone, feature_dim = build_backbones(
            use_clip=use_clip,
            use_dino=use_dino,
            clip_model_name=clip_name,
            dino_model_name=dino_name,
            clip_pretrained=bool(cfg["encoders"]["clip"].get("pretrained", True)),
            dino_pretrained=bool(cfg["encoders"]["dino"].get("pretrained", True)),
        )
        hidden_dims = list(cfg["baseline_heads"]["mlp"].get("hidden_dims", [512, 256]))
        dropout = float(cfg["baseline_heads"]["mlp"].get("dropout", 0.2))
        model = BaselineClassifier(
            clip_backbone=clip_backbone,
            dino_backbone=dino_backbone,
            feature_dim=feature_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            num_labels=num_labels,
        )
        model.load_state_dict(checkpoint["model_state"])
        model.eval().to(device)
        self.model = model

    def predict(
        self,
        image: Image.Image,
        label_vocab: Dict[str, int],
        top_k: int = 5,
        threshold: float = 0.5,
    ) -> InferenceResult:
        with torch.no_grad():
            logits = self.model([image], device=self.device)[0]
        return _sorted_labels(logits, label_vocab, top_k, threshold)


class GNNAdapter(BaseAdapter):
    def __init__(self, checkpoint: Dict[str, Any], device: torch.device) -> None:
        super().__init__(device)
        cfg = checkpoint["config"]
        num_labels = len(checkpoint["label_vocab"])

        use_clip = bool(cfg["encoders"]["clip"]["enabled"])
        use_dino = bool(cfg["encoders"]["dino"]["enabled"])
        clip_name = str(cfg["encoders"]["clip"].get("model_name", "vit_base_patch16_clip_224.openai"))
        dino_name = str(cfg["encoders"]["dino"].get("model_name", "vit_base_patch14_dinov2.lvd142m"))
        if clip_name == "ViT-B-32":
            clip_name = "vit_base_patch16_clip_224.openai"
        if dino_name == "vit_base_patch14_dinov2":
            dino_name = "vit_base_patch14_dinov2.lvd142m"

        clip_backbone, dino_backbone, feature_dim = build_backbones(
            use_clip=use_clip,
            use_dino=use_dino,
            clip_model_name=clip_name,
            dino_model_name=dino_name,
            clip_pretrained=bool(cfg["encoders"]["clip"].get("pretrained", True)),
            dino_pretrained=bool(cfg["encoders"]["dino"].get("pretrained", True)),
        )

        class BackboneWrapper(torch.nn.Module):
            def __init__(self, clip_module: torch.nn.Module | None, dino_module: torch.nn.Module | None) -> None:
                super().__init__()
                self.clip_module = clip_module
                self.dino_module = dino_module

            def encode(self, images: List[Image.Image], device: torch.device) -> torch.Tensor:
                feats: List[torch.Tensor] = []
                if self.clip_module is not None:
                    feats.append(self.clip_module.forward_pil(images, device=device))
                if self.dino_module is not None:
                    feats.append(self.dino_module.forward_pil(images, device=device))
                if len(feats) == 1:
                    return feats[0]
                return torch.cat(feats, dim=1)

        backbone = BackboneWrapper(clip_backbone, dino_backbone).to(device)
        self.backbone = backbone

        hidden_dims = [layer_cfg["out_dim"] for layer_cfg in cfg["gnn"]["layers"]]
        dropout = float(cfg["gnn"].get("dropout", 0.2))
        gnn_model = NativeGNNClassifier(
            in_dim=feature_dim,
            hidden_dims=hidden_dims,
            num_attributes=num_labels,
            dropout=dropout,
        )
        gnn_model.load_state_dict(checkpoint["gnn_state"])
        gnn_model.eval().to(device)
        self.gnn_model = gnn_model

    def predict(
        self,
        image: Image.Image,
        label_vocab: Dict[str, int],
        top_k: int = 5,
        threshold: float = 0.5,
    ) -> InferenceResult:
        with torch.no_grad():
            feats = self.backbone.encode([image], device=self.device).unsqueeze(1)
            # Single-sample multi-hot target is unknown at inference; use zeros to build graph shape.
            dummy_targets = torch.zeros(1, len(label_vocab), device=self.device)
            graph = build_bipartite_batch(feats=feats, targets=dummy_targets)
            logits = self.gnn_model(graph)[0]
        return _sorted_labels(logits, label_vocab, top_k, threshold)

