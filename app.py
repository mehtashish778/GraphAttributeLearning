from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import gradio as gr
import torch
from PIL import Image
from huggingface_hub import hf_hub_download

import sys
from pathlib import Path as _Path

ROOT = _Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from infer.pipeline import LoadedModel, load_checkpoint, predict_image


MODEL_REPO_ID = "alphamike/GraphAttributeLearning_Model"
MODEL_FILENAMES = {
    "baseline": "baseline/best.pt",
    "gnn1": "gnn1/best.pt",
    "gnn2": "gnn2/best.pt",
}


class ModelRegistry:
    def __init__(self) -> None:
        self._cache: Dict[str, LoadedModel] = {}

    def get(self, key: str, path: Path, device: torch.device) -> LoadedModel:
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        loaded = load_checkpoint(path, device=device)
        self._cache[key] = loaded
        return loaded


registry = ModelRegistry()


def infer_gradio(
    image: Image.Image,
    model_type: str,
    checkpoint_path: str,
    top_k: int,
    threshold: float,
) -> Any:
    if image is None:
        return [], "No image provided."
    # Resolve checkpoint: either use the textbox value as filename within the model repo
    # or fall back to the default mapping for the selected model_type.
    filename = (checkpoint_path or "").strip() or MODEL_FILENAMES.get(model_type, "")
    if not filename:
        return [], f"Unknown model_type '{model_type}'."
    try:
        local_ckpt_path = hf_hub_download(
            repo_id=MODEL_REPO_ID,
            filename=filename,
            repo_type="model",
        )
    except Exception as exc:  # noqa: BLE001
        return [], f"Error downloading checkpoint from Hub: {exc}"

    path = Path(local_ckpt_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        key = f"{model_type}:{path}"
        loaded = registry.get(key, path, device)
        # Save temp image to disk-agnostic path is unnecessary; use pipeline logic directly.
        from infer.pipeline import predict_image as predict_from_loaded
        from infer.pipeline import LoadedModel as _LM  # type: ignore

        assert isinstance(loaded, _LM)
        # Wrap PIL image call using temporary in-memory path emulation.
        # For simplicity, reuse adapter directly.
        result = loaded.adapter.predict(
            image=image,
            label_vocab=loaded.label_vocab,
            top_k=top_k,
            threshold=threshold,
        )
        rows = [
            {"label": label, "score": round(score, 4), "positive@thr": label in result.positives}
            for label, score in zip(result.labels, result.scores)
        ]
        message = f"Positives (>= {threshold:.2f}): {', '.join(result.positives) if result.positives else 'none'}"
        return rows, message
    except Exception as exc:  # noqa: BLE001
        return [], f"Error: {exc}"


with gr.Blocks() as demo:
    gr.Markdown(
        "# Adjective-Aware Chair Attributes\n"
        "Select baseline, GNN 1, or GNN 2, upload an image, and view attribute scores.\n\n"
        f"Models are loaded from Hugging Face model repo: `{MODEL_REPO_ID}`."
    )

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="Chair image")
            model_type = gr.Radio(
                choices=["baseline", "gnn1", "gnn2"],
                value="baseline",
                label="Model type",
            )
            checkpoint = gr.Textbox(
                value=MODEL_FILENAMES["baseline"],
                label="Checkpoint filename (within Hub repo)",
            )

            def _sync_ckpt(choice: str) -> str:
                return MODEL_FILENAMES.get(choice, MODEL_FILENAMES["baseline"])

            model_type.change(_sync_ckpt, inputs=model_type, outputs=checkpoint)

            top_k = gr.Slider(1, 20, value=5, step=1, label="Top-K")
            threshold = gr.Slider(0.0, 1.0, value=0.5, step=0.01, label="Threshold")
            run_btn = gr.Button("Run inference")

        with gr.Column(scale=2):
            table = gr.Dataframe(
                headers=["label", "score", "positive@thr"],
                datatype=["str", "number", "bool"],
                label="Attribute scores",
            )
            message = gr.Markdown()

    run_btn.click(
        infer_gradio,
        inputs=[image_input, model_type, checkpoint, top_k, threshold],
        outputs=[table, message],
    )


if __name__ == "__main__":
    demo.launch()

