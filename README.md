# Adjective-Aware Object Embedding

This project builds adjective-aware object embeddings using:

- CLIP image features
- DINOv2 image features
- Graph Neural Network (GNN) reasoning over object-attribute graphs

Current scope starts with `chair` and multi-label attributes like `broken`, `blue`, `wooden`, and `plastic`.

## Project Structure

- `plan/plan.md` - detailed execution strategy and technical plan
- `configs/dataset.yaml` - dataset, attributes, splits, normalization
- `configs/model.yaml` - encoder, fusion, graph, and head settings
- `configs/train.yaml` - optimization, staging, losses, checkpointing
- `configs/eval.yaml` - metrics, visualization, error analysis
- `configs/experiment.yaml` - baseline/main/ablation run definitions
- `scripts/data/` - data download, extraction, processing, split pipeline
- `src/data/` - reusable normalization and IO utilities for data pipeline

## Config Overview

### `dataset.yaml`

Defines:

- dataset root/cache/processed paths
- attribute taxonomy and synonym normalization
- split strategy and filtering rules
- basic augmentation

### `model.yaml`

Defines:

- CLIP + DINO encoder settings
- fusion type (`concat`, optional gated/projected ablations)
- graph variant (`bipartite` / `enriched`)
- attribute node initialization (`clip_text` / `trainable`)
- GNN layers and baseline MLP head

### `train.yaml`

Defines:

- two-stage training (frozen then partial fine-tuning)
- optimizer, scheduler, AMP, grad clipping
- weighted BCE + optional contrastive loss
- early stopping and checkpoint monitoring

### `eval.yaml`

Defines:

- classification metrics (mAP, macro/micro F1, per-attribute metrics)
- embedding metrics (retrieval@K, silhouette)
- UMAP/t-SNE options
- error analysis outputs

### `experiment.yaml`

Defines:

- baseline runs (A/B/C)
- main proposed run (CLIP+DINO+GNN)
- ablation runs (graph type, node init, loss, fine-tuning)

## Execution Strategy (No Timeline)

Follow this order:

1. Data prep + label normalization
2. Baselines (A/B/C)
3. Graph model training
4. Ablations and tuning
5. Robustness checks + packaging
6. Build a minimal Gradio demo interface (final step, after model is fully trained and selected)

Use `plan/plan.md` as the source of truth for quality gates and deliverables.

## How To Run

### 1) Run Chair-Only Data Pipeline (Visual Genome)

Default full pipeline:

`python scripts/data/run_data_pipeline.py --config configs/dataset.yaml`

Run individual stages:

- `python scripts/data/run_data_pipeline.py --download-only --config configs/dataset.yaml`
- `python scripts/data/run_data_pipeline.py --extract-only --config configs/dataset.yaml`
- `python scripts/data/run_data_pipeline.py --process-only --config configs/dataset.yaml`
- `python scripts/data/run_data_pipeline.py --split-only --config configs/dataset.yaml`

Direct scripts (if needed):

- `python scripts/data/download_visual_genome.py --config configs/dataset.yaml`
- `python scripts/data/extract_visual_genome.py --config configs/dataset.yaml`
- `python scripts/data/process_visual_genome.py --config configs/dataset.yaml`
- `python scripts/data/build_splits.py --config configs/dataset.yaml`

Important: the processing pipeline is strict chair-only (`primary_object_class: chair`).

### 2) Train/Evaluate Models

This repo provides dataset + experiment configs. Connect model training to your entrypoints (for example `scripts/train.py` and `scripts/eval.py`).

Typical flow:

1. Run baseline experiments from `configs/experiment.yaml`.
2. Run the proposed GNN experiment.
3. Run ablation experiments.
4. Compare all runs in one metrics table.

## Expected Outputs

Data artifacts:

- `data/processed/visual_genome/samples.parquet` (or `samples.csv` fallback)
- `data/processed/visual_genome/label_vocab.json`
- `data/processed/visual_genome/label_frequencies.json`
- `data/processed/visual_genome/processing_report.json`
- `data/processed/visual_genome/splits/train.json`
- `data/processed/visual_genome/splits/val.json`
- `data/processed/visual_genome/splits/test.json`
- `data/processed/visual_genome/splits/split_report.json`

Training/eval artifacts:

- model checkpoints
- evaluation reports (mAP/F1/per-attribute)
- embedding analysis artifacts (UMAP/retrieval results)
- error analysis summaries
- minimal Gradio demo for interactive predictions

## Notes

- Keep experiments reproducible with fixed seeds.
- Track all runs with consistent config snapshots.
- Measure gains against **Baseline C** to isolate graph contribution.

