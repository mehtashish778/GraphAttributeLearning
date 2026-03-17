from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data.io_utils import write_json  # noqa: E402
from train.config_utils import (  # noqa: E402
    apply_dot_overrides,
    deep_merge,
    load_yaml,
    parse_overrides_json,
)
from train.dataset import (  # noqa: E402
    ChairAttributeDataset,
    build_dataloader,
    class_pos_weights,
    load_json,
    split_paths,
)
from train.encoders import build_backbones  # noqa: E402
from train.graph_builder import build_bipartite_batch  # noqa: E402
from train.graph_models import NativeGNNClassifier  # noqa: E402
from train.losses import bce_logits_loss  # noqa: E402
from train.metrics import compute_metrics  # noqa: E402


def log(message: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [train_gnn] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GNN model (bipartite graph).")
    parser.add_argument("--dataset-config", type=Path, default=REPO_ROOT / "configs" / "dataset.yaml")
    parser.add_argument("--model-config", type=Path, default=REPO_ROOT / "configs" / "model.yaml")
    parser.add_argument("--train-config", type=Path, default=REPO_ROOT / "configs" / "train.yaml")
    parser.add_argument("--eval-config", type=Path, default=REPO_ROOT / "configs" / "eval.yaml")
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--overrides-json", type=str, default="")
    parser.add_argument("--mode", type=str, default="full", choices=["smoke", "full"])
    parser.add_argument("--device", type=str, default="auto",
                    help="Device: auto, cuda, cpu, or cuda:N (e.g. cuda:1)")
    return parser.parse_args()


def set_seed(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def load_combined_config(args: argparse.Namespace) -> Dict[str, Any]:
    dataset_cfg = load_yaml(args.dataset_config)
    model_cfg = load_yaml(args.model_config)
    train_cfg = load_yaml(args.train_config)
    eval_cfg = load_yaml(args.eval_config)
    combined = deep_merge(dataset_cfg, model_cfg)
    combined = deep_merge(combined, train_cfg)
    combined = deep_merge(combined, eval_cfg)
    overrides = parse_overrides_json(args.overrides_json)
    if overrides:
        combined = apply_dot_overrides(combined, overrides)
    return combined


def build_loaders(cfg: Dict[str, Any], batch_size: int, eval_batch_size: int, smoke: bool) -> Tuple[Any, Any, Any, Dict[str, int]]:
    processed_root = REPO_ROOT / cfg["dataset"]["processed_dir"]
    label_vocab_path = processed_root / cfg["processing"]["label_vocab_file"]
    label_freq_path = processed_root / cfg["processing"]["label_frequency_file"]
    cache_dir = REPO_ROOT / cfg["dataset"]["root_dir"] / "images"
    splits = split_paths(processed_root=processed_root, split_dir_name=cfg["processing"]["split_dir"])

    label_vocab = load_json(label_vocab_path)
    label_freq = load_json(label_freq_path)

    train_ds = ChairAttributeDataset(splits["train"], label_vocab_path, REPO_ROOT, cache_dir)
    val_ds = ChairAttributeDataset(splits["val"], label_vocab_path, REPO_ROOT, cache_dir)
    test_ds = ChairAttributeDataset(splits["test"], label_vocab_path, REPO_ROOT, cache_dir)

    if smoke:
        # Cap samples for faster smoke training.
        train_ds.samples = train_ds.samples[:512]
        val_ds.samples = val_ds.samples[:128]
        test_ds.samples = test_ds.samples[:128]

    num_workers = int(cfg["dataset"].get("num_workers", 0))
    if smoke:
        num_workers = min(num_workers, 8)

    train_loader = build_dataloader(
        dataset=train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=bool(cfg["dataset"].get("pin_memory", True)),
    )
    val_loader = build_dataloader(
        dataset=val_ds,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=bool(cfg["dataset"].get("pin_memory", True)),
    )
    test_loader = build_dataloader(
        dataset=test_ds,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=bool(cfg["dataset"].get("pin_memory", True)),
    )
    return train_loader, val_loader, test_loader, {"vocab": label_vocab, "freq": label_freq}


def build_backbone_and_gnn(
    cfg: Dict[str, Any],
    num_labels: int,
    device: torch.device,
) -> Tuple[torch.nn.Module, NativeGNNClassifier, int]:
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

        def encode(self, images: List[Any], device: torch.device) -> torch.Tensor:
            feats: List[torch.Tensor] = []
            if self.clip_module is not None:
                feats.append(self.clip_module.forward_pil(images, device=device))
            if self.dino_module is not None:
                feats.append(self.dino_module.forward_pil(images, device=device))
            if len(feats) == 1:
                return feats[0]
            return torch.cat(feats, dim=1)

    backbone = BackboneWrapper(clip_backbone, dino_backbone).to(device)

    hidden_dims = [layer_cfg["out_dim"] for layer_cfg in cfg["gnn"]["layers"]]
    dropout = float(cfg["gnn"].get("dropout", 0.2))
    gnn_model = NativeGNNClassifier(
        in_dim=feature_dim,
        hidden_dims=hidden_dims,
        num_attributes=num_labels,
        dropout=dropout,
    ).to(device)

    return backbone, gnn_model, feature_dim


def run_eval(
    backbone: torch.nn.Module,
    gnn_model: NativeGNNClassifier,
    loader: Any,
    device: torch.device,
    pos_weight: torch.Tensor,
) -> Dict[str, Any]:
    backbone.eval()
    gnn_model.eval()
    all_logits: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []
    losses: List[float] = []
    with torch.no_grad():
        for batch in loader:
            targets = batch["targets"].to(device)
            images = batch["images"]
            feats = backbone.encode(images=images, device=device).unsqueeze(1)
            graph = build_bipartite_batch(feats=feats, targets=targets)
            logits = gnn_model(graph)
            loss = bce_logits_loss(logits, targets, pos_weight=pos_weight)
            losses.append(float(loss.item()))
            all_logits.append(logits.detach().cpu().numpy())
            all_targets.append(targets.detach().cpu().numpy())
    logits_np = np.concatenate(all_logits, axis=0) if all_logits else np.zeros((0, 1))
    targets_np = np.concatenate(all_targets, axis=0) if all_targets else np.zeros((0, 1))
    metrics = compute_metrics(targets=targets_np, logits=logits_np)
    metrics["loss"] = float(np.mean(losses)) if losses else 0.0
    return metrics


def train() -> None:
    args = parse_args()
    log(f"Starting run_name={args.run_name} mode={args.mode} device_request={args.device}")
    log(f"dataset_config={args.dataset_config}")
    log(f"model_config={args.model_config}")
    log(f"train_config={args.train_config}")
    cfg = load_combined_config(args)
    run_cfg = cfg.get("run", {})
    seed = int(run_cfg.get("seed", 42))
    deterministic = bool(run_cfg.get("deterministic", True))
    set_seed(seed=seed, deterministic=deterministic)
    log(f"Seed set to {seed}, deterministic={deterministic}")

    smoke = args.mode == "smoke"
    train_epochs = 1 if smoke else int(cfg["training"]["stage1"].get("epochs", 20))
    batch_size = 4 if smoke else int(cfg["batching"].get("batch_size", 32))
    eval_batch_size = 4 if smoke else int(cfg["batching"].get("eval_batch_size", 64))

    cuda_available = torch.cuda.is_available()
    selected_device = "cuda" if (args.device == "auto" and cuda_available) else args.device
    if args.device == "auto" and not cuda_available:
        selected_device = "cpu"
    if selected_device == "cuda" and not cuda_available:
        raise RuntimeError(
            "CUDA requested but not available in torch build/environment. "
            "Install CUDA-enabled PyTorch in the active virtualenv."
        )
    device = torch.device(selected_device)
    log(
        f"Resolved device={device.type}, torch_cuda_available={cuda_available}, "
        f"torch_version={torch.__version__}, torch_cuda={torch.version.cuda}"
    )
    log("Building dataloaders from processed manifests...")
    train_loader, val_loader, test_loader, label_info = build_loaders(
        cfg=cfg,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        smoke=smoke,
    )
    log(
        f"Dataloader sizes -> train_batches={len(train_loader)} val_batches={len(val_loader)} "
        f"test_batches={len(test_loader)}"
    )
    label_vocab = label_info["vocab"]
    label_freq = label_info["freq"]
    num_labels = len(label_vocab)
    log(f"Label space size: {num_labels}")
    log("Building encoder backbones and GNN head (first run may download pretrained weights)...")
    backbone, gnn_model, _ = build_backbone_and_gnn(cfg=cfg, num_labels=num_labels, device=device)
    log("Model construction complete")
    pos_weight = class_pos_weights(label_frequencies=label_freq, label_vocab=label_vocab).to(device)

    lr = float(cfg["training"]["stage1"].get("lr_head", 1e-3))
    weight_decay = float(cfg["optimization"].get("weight_decay", 1e-4))
    optimizer = torch.optim.AdamW(
        list(gnn_model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )

    grad_accum_steps = int(cfg["optimization"].get("gradient_accumulation_steps", 1))
    use_amp = bool(cfg["optimization"].get("mixed_precision", False)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    output_root = REPO_ROOT / run_cfg.get("output_dir", "outputs")
    run_dir = output_root / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    history: List[Dict[str, Any]] = []
    best_map = -1.0
    best_path = run_dir / "best.pt"
    log(
        f"Training for epochs={train_epochs}, batch_size={batch_size}, "
        f"eval_batch_size={eval_batch_size}, grad_accum_steps={grad_accum_steps}, amp={use_amp}"
    )
    for epoch in range(1, train_epochs + 1):
        backbone.eval()
        gnn_model.train()
        train_losses: List[float] = []
        step_times: List[float] = []
        start_epoch = time.time()
        for step_idx, batch in enumerate(
            tqdm(train_loader, desc=f"{args.run_name} epoch {epoch}/{train_epochs}")
        ):
            batch_start = time.time()
            targets = batch["targets"].to(device)
            images = batch["images"]
            feats = backbone.encode(images=images, device=device).unsqueeze(1)
            graph = build_bipartite_batch(feats=feats, targets=targets)

            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = gnn_model(graph)
                loss = bce_logits_loss(logits, targets, pos_weight=pos_weight)
                loss = loss / grad_accum_steps

            scaler.scale(loss).backward()
            if (step_idx + 1) % grad_accum_steps == 0 or (step_idx + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(gnn_model.parameters(), cfg["optimization"].get("grad_clip_norm", 1.0))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            train_losses.append(float(loss.item()) * grad_accum_steps)
            step_times.append(time.time() - batch_start)

        epoch_time = time.time() - start_epoch
        avg_step = float(np.mean(step_times)) if step_times else 0.0
        log(
            f"Epoch {epoch} finished in {epoch_time:.1f}s, "
            f"avg_step_time={avg_step:.3f}s, steps={len(step_times)}"
        )

        val_metrics = run_eval(
            backbone=backbone,
            gnn_model=gnn_model,
            loader=val_loader,
            device=device,
            pos_weight=pos_weight,
        )
        epoch_row = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)) if train_losses else 0.0,
            "val_loss": val_metrics["loss"],
            "val_map": val_metrics["map"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_micro_f1": val_metrics["micro_f1"],
        }
        history.append(epoch_row)
        log(
            f"Epoch {epoch}/{train_epochs} -> train_loss={epoch_row['train_loss']:.4f} "
            f"val_map={epoch_row['val_map']:.4f} val_macro_f1={epoch_row['val_macro_f1']:.4f}"
        )
        if val_metrics["map"] > best_map:
            best_map = val_metrics["map"]
            torch.save(
                {
                    "backbone_state": backbone.state_dict(),
                    "gnn_state": gnn_model.state_dict(),
                    "label_vocab": label_vocab,
                    "config": cfg,
                },
                best_path,
            )
            log(f"Saved new best checkpoint at epoch {epoch}: {best_path}")

    log("Running final test evaluation using best checkpoint...")
    checkpoint = torch.load(best_path, map_location=device)
    backbone.load_state_dict(checkpoint["backbone_state"])
    gnn_model.load_state_dict(checkpoint["gnn_state"])
    test_metrics = run_eval(
        backbone=backbone,
        gnn_model=gnn_model,
        loader=test_loader,
        device=device,
        pos_weight=pos_weight,
    )
    payload = {
        "run_name": args.run_name,
        "mode": args.mode,
        "num_labels": num_labels,
        "best_val_map": best_map,
        "history": history,
        "test_metrics": test_metrics,
    }
    write_json(run_dir / "metrics.json", payload)
    write_json(run_dir / "label_vocab.json", label_vocab)
    with (run_dir / "overrides.json").open("w", encoding="utf-8") as handle:
        json.dump(parse_overrides_json(args.overrides_json), handle, indent=2, ensure_ascii=True)
    log(f"Training complete for {args.run_name}")
    print(json.dumps({"run_name": args.run_name, "test_map": test_metrics["map"]}, indent=2), flush=True)


if __name__ == "__main__":
    train()

