# Adjective-Aware Object Embedding

## 1) Goal and Scope

Build an adjective-aware representation pipeline for object images (starting with `chair`) that:

- predicts multi-label attributes (for example `broken`, `blue`, `wooden`, `plastic`)
- reshapes embedding space so items cluster by adjectives, not only by object identity
- supports both offline analysis (embedding quality) and online inference (attribute scoring)

Initial scope is single-object class (`chair`) to de-risk data and modeling, then generalize to multi-class.

---

## 2) Target Outcomes (Definition of Done)

The project is complete when all conditions below are met:

- **Model quality**
  - mAP improves over baseline by >= 8-12% relative.
  - Macro-F1 improves over baseline by >= 5% relative.
  - At least 4 key attributes (`broken`, `blue`, `wooden`, `plastic`) achieve per-class AP >= 0.70 on validation (adjustable by data quality).
- **Embedding quality**
  - UMAP/t-SNE shows tighter adjective clusters than baseline.
  - Retrieval@K for same-adjective neighbors is measurably improved.
- **Engineering quality**
  - Reproducible training with fixed seeds and config-driven runs.
  - End-to-end training + inference scripts run without manual edits.
  - Experiment logs and checkpoints are versioned and comparable.

---

## 3) Technical Strategy

### 3.1 Core Idea

Use a dual-encoder visual backbone:

- CLIP image features (semantic alignment)
- DINOv2 image features (visual structure)

Fuse both features and perform message passing over an object-attribute graph, where:

- object nodes use fused visual features
- attribute nodes use CLIP text embeddings (or learnable vectors as ablation)
- edges encode object-attribute membership (+ optional attribute-attribute priors)

### 3.2 Modeling Choice Rationale

- CLIP alone is strong semantically but can miss fine visual states.
- DINOv2 adds complementary visual detail.
- GNN introduces relational inductive bias, useful for correlated attributes (`plastic` with `material`, `broken` with `damage`).

### 3.3 Baselines (Mandatory)

Implement before full GNN rollout:

1. **Baseline A:** CLIP-only + MLP multi-label head
2. **Baseline B:** DINO-only + MLP multi-label head
3. **Baseline C:** CLIP+DINO fusion + MLP head (no graph)
4. **Proposed:** CLIP+DINO + GNN

All gains must be reported against Baseline C to isolate graph contribution.

---

## 4) Data Strategy

## 4.1 Dataset Selection (Phase-1 Priority)

Use this order:

1. **Visual Genome** (object + attributes, fastest path for graph supervision)
2. **MIT States / SUN Attributes** (state-centric signals, useful for adjective robustness)
3. **Open Images V7** (scale-up if labels fit target adjectives)

Avoid LAION in Phase-1 training due to noisy weak labels; use only for optional pretraining or pseudo-label mining.

## 4.2 Label Taxonomy

Normalize attributes into controlled groups:

- **Color:** blue, red, white, black, ...
- **Material:** wooden, plastic, metal, ...
- **Condition/State:** broken, clean, dirty, old, ...

Rules:

- lowercase, lemmatized labels
- synonym merge (`wood` -> `wooden`, `damaged` -> `broken` when valid)
- keep only attributes with minimum support threshold (for example >= 200 samples)

## 4.3 Data Splits

- train/val/test by image IDs (no leakage)
- stratified split by high-level attribute groups
- maintain long-tail report for rare attributes

## 4.4 Data Quality Gates

Before model training, enforce:

- duplicate image check
- invalid/empty label filtering
- per-attribute frequency report
- noisy label audit on random sample (manual spot check)

---

## 5) Graph Construction Plan

## 5.1 Graph Types

Build two graph variants for experiments:

- **Variant 1 (Bipartite):** object <-> attribute
- **Variant 2 (Enriched):** object <-> attribute + attribute <-> attribute priors

Start with Variant 1 for stability, then ablate Variant 2.

## 5.2 Node Features

- **Object node:** fused visual embedding (`[clip || dino]`)
- **Attribute node (default):** CLIP text embedding from prompt template:
  - `a {attribute} chair`
- **Attribute node (ablation):** trainable vectors

## 5.3 Edge Weights

Use weighted edges for confidence:

- labeled positive edge: weight = 1.0
- weak/pseudo edge (if used): weight in `[0.3, 0.8]`
- optional attribute prior edge from co-occurrence PMI or normalized co-frequency

---

## 6) Model Architecture

## 6.1 Feature Fusion

Start with concatenation:

- `z = concat(z_clip, z_dino)`

Then ablate:

- gated fusion (`z = g * z_clip + (1 - g) * z_dino`)
- projected fusion (linear layers to shared dimension before concat)

## 6.2 GNN Head

Recommended start:

- `GATConv(input_dim -> 512, heads=4)`
- `ReLU + Dropout(0.2)`
- `GATConv(512 -> 256, heads=2)`
- `Linear(256 -> num_attributes)` for object-node logits

Alternative for stability if needed: GraphSAGE.

## 6.3 Loss Design

Primary:

- weighted BCE with class-frequency weights

Secondary:

- contrastive / InfoNCE on object embeddings:
  - positives: same adjective group
  - negatives: conflicting adjective groups

Total:

- `L = L_bce + lambda_contrast * L_contrast`

Tune `lambda_contrast` in `{0.05, 0.1, 0.2}`.

---

## 7) Training Strategy

## 7.1 Two-Stage Optimization

Stage 1:

- freeze CLIP and DINO encoders
- train fusion + GNN head

Stage 2:

- unfreeze top N layers of encoders (N configurable)
- fine-tune with lower LR for encoders

This reduces instability and GPU cost early.

## 7.2 Hyperparameter Defaults

- optimizer: AdamW
- LR head: `1e-3`
- LR encoder: `1e-5` (stage 2)
- weight decay: `1e-4`
- batch size: maximize stable GPU usage (target 32+ effective via grad accumulation)
- epochs: 30-60 with early stopping on val mAP

## 7.3 Training Controls

- deterministic seed support
- mixed precision enabled
- gradient clipping (`1.0`)
- checkpoint by best val mAP
- scheduler: cosine decay with warmup

---

## 8) Evaluation Strategy

## 8.1 Core Metrics

- mAP (primary)
- Macro-F1 and Micro-F1
- per-attribute AP and F1
- calibration metrics (ECE / reliability bins) if serving probabilities

## 8.2 Embedding Metrics

- retrieval precision@K for same-adjective neighbors
- cluster compactness/separation (silhouette on adjective labels)
- UMAP/t-SNE visual audits (fixed random seed)

## 8.3 Error Analysis

Every run should include:

- top false positives and false negatives per attribute
- confusion among correlated materials/colors
- performance by attribute frequency bucket (head/mid/tail)

---

## 9) Ablation Matrix (Required)

Minimum experiment grid:

1. CLIP-only vs DINO-only vs fused
2. no-graph vs bipartite graph vs enriched graph
3. text-initialized attribute nodes vs trainable attribute nodes
4. BCE only vs BCE + contrastive
5. frozen encoders vs partial fine-tuning

Keep one variable changed per ablation for clean conclusions.

---

## 10) Execution Sequence (No Timeline)

### Phase A: Data and Baselines

- finalize attribute taxonomy and label normalization
- build data loader + split pipeline
- train Baseline A/B/C
- output: baseline report and dataset diagnostics

### Phase B: Graph Pipeline

- implement graph builder (Variant 1)
- train GNN model with frozen encoders
- evaluate against Baseline C
- output: first graph-vs-no-graph comparison

### Phase C: Optimization and Ablations

- run stage-2 fine-tuning
- run required ablations
- tune `lambda_contrast`, fusion, graph depth
- output: ablation table + selected final config

### Phase D: Robustness and Packaging

- run long-tail and error analysis
- package inference script, checkpoints, and reproducibility docs
- add minimal Gradio demo interface for inference
- generate final report with metrics and visualizations
- output: final model + execution docs + demo outputs

Gradio demo is an end-stage task and starts only after training, ablations, and final model selection are complete.

---

## 11) Repository Execution Structure

Suggested structure:

- `configs/` (dataset/model/train/eval yaml)
- `data/` (raw + processed metadata, ignore large binaries in git)
- `src/encoders/` (clip, dino wrappers)
- `src/graph/` (graph builder, priors, batching)
- `src/models/` (fusion, gnn heads, mlp baselines)
- `src/train/` (trainer, losses, schedulers)
- `src/eval/` (metrics, retrieval, visualization)
- `scripts/` (train, eval, infer)
- `reports/` (experiment tables, plots, analysis)

---

## 12) Risks and Mitigations

- **Risk:** label noise in attributes  
  **Mitigation:** filtering, confidence weights, manual audit sample.

- **Risk:** class imbalance hurts rare adjectives  
  **Mitigation:** weighted BCE, focal loss fallback, frequency-aware sampling.

- **Risk:** overfitting with small attribute subsets  
  **Mitigation:** stronger augmentation, early stopping, regularization.

- **Risk:** graph complexity without measurable gain  
  **Mitigation:** enforce baseline comparison and remove graph if gain is negligible.

---

## 13) Final Deliverables

- trained adjective-aware embedding model
- reproducible training and inference pipeline
- ablation and baseline comparison report
- embedding visualization and retrieval analysis
- concise model card (dataset, metrics, limits, known failure cases)
- minimal Gradio demo app for interactive attribute prediction

---

## 14) Immediate Next Actions

1. Lock Phase-1 dataset (`Visual Genome`) and attribute list.
2. Implement data normalization and split scripts.
3. Train Baseline C (`CLIP+DINO+MLP`) as reference.
4. Build bipartite graph constructor and run first GNN experiment.
5. Record metrics in a single comparison table from day one.
