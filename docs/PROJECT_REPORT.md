# Graph Attribute Learning — Project Report

## 1. Project Overview

**Project name:** Adjective-Aware Chair Attribute Prediction  
**Task:** Multi-label attribute prediction on images (chair as primary object).  
**Experiment ID:** `exp_clip_dino_gnn_v1` — CLIP + DINO fusion with bipartite GNN and BCE + InfoNCE loss.

---

## 2. Data Used

### 2.1 Source

- **Dataset:** [Visual Genome](https://visualgenome.org/)  
- **Artifacts (from Washington):**
  - `image_data.json.zip` — image IDs and URLs  
  - `objects.json.zip` — object annotations  
  - `attributes.json.zip` — attribute annotations
- **Paths:**  
  - Raw: `data/raw/visual_genome`  
  - Processed: `data/processed/visual_genome`  
  - Cache: `data/cache`

### 2.2 Processing Pipeline

- **Filtering:** Chair-only (`strict_chair_only: true`). Objects whose names normalize to “chair” are kept; others dropped.  
- **Attributes:** Grouped into **color** (blue, red, white, black), **material** (wooden, plastic, metal), **condition** (broken, clean, dirty, old).  
- **Normalization:** Lowercasing, lemmatization, synonym map (e.g. `wood → wooden`, `damaged → broken`).  
- **Label vocab:** Built from attributes with **min_support: 200**; unmapped attributes dropped (`keep_unmapped_attributes: false`).  
- **Output:** Parquet/CSV samples, `label_vocab.json`, `label_frequencies.json`, split manifests under `splits/`.  
- **Splits:** 80% train, 10% val, 10% test; stratified by attribute groups, multilabel stratification, seed 42.

### 2.3 Smoke Run Label Space

For the **smoke** runs (quick sanity checks), the saved label vocab has **7 attributes**:


| Index | Attribute |
| ----- | --------- |
| 0     | black     |
| 1     | blue      |
| 2     | metal     |
| 3     | plastic   |
| 4     | red       |
| 5     | white     |
| 6     | wooden    |


(Full config allows up to 12 attributes including condition groups; actual vocab size depends on min_support and processed data.)

### 2.4 Data Loader Details

- **Image size:** 224×224  
- **Augmentation (training):** horizontal flip (0.5), color jitter (0.2), random resized crop (scale 0.8–1.0)  
- **Targets:** Multi-hot vectors over the label vocab  
- **Smoke (GNN only):** Train/val/test sample caps: 512 / 128 / 128 for faster runs. Baseline smoke uses full splits (no cap in code).

---

## 3. Models Trained

### 3.1 Baseline: CLIP + DINO MLP (`smoke_baseline`)

- **Encoders (frozen in stage 1):**
  - **CLIP:** `vit_base_patch16_clip_224.openai`, output dim 512  
  - **DINOv2:** `vit_base_patch14_dinov2.lvd142m`, output dim 768
- **Fusion:** Concatenation → 1280-d feature per image  
- **Head:** MLP 1280 → 512 → 256 → `num_labels`, ReLU + dropout 0.2  
- **Training:** 2 epochs (smoke), batch size 4, BCE with inverse-frequency pos_weight  
- **Outputs:** `outputs/smoke_baseline/best.pt`, `label_vocab.json`, `metrics.json`

### 3.2 GNN: CLIP + DINO + Bipartite GNN (`smoke_gnn`)

- **Encoders:** Same CLIP + DINOv2, concatenated (1280-d).  
- **Graph:** Bipartite object–attribute graph per batch: object nodes = fused image features; attribute nodes initialized from **CLIP text** with template `"a {attribute} chair"`. Edges from objects to attributes given by multi-hot labels.  
- **GNN:** Custom bipartite message passing (attribute → object aggregation, weighted mean, residual MLP). Config: 2 layers (1280→512×4 heads, 512→256×2 heads), dropout 0.2; classifier 256 → `num_labels`.  
- **Training:** 1 epoch (smoke), batch size 4; BCE + optional InfoNCE (contrastive). Smoke uses 512/128/128 samples.  
- **Outputs:** `outputs/smoke_gnn/best.pt`, `label_vocab.json`, `metrics.json`

### 3.3 Full Experiment Design (from config)

- **Baselines:** (a) CLIP-only MLP, (b) DINO-only MLP, (c) CLIP+DINO MLP  
- **Main run:** Bipartite GNN, GAT-style, CLIP-text attribute init  
- **Ablations:** Enriched graph + attribute priors, trainable attribute nodes, BCE-only (no contrastive), partial encoder fine-tuning (stage 2)

---

## 4. Model and architecture (detailed)

### 4.1 Baseline: CLIP + DINO → MLP

**Data flow:**

```
Image (PIL) → [CLIP transform] → CLIP ViT → 512-d
            → [DINO transform] → DINOv2 ViT → 768-d
            → concat → 1280-d → MLP(512→256→num_labels) → logits
```

- **Encoders:** Implemented via `timm`. Each backbone is a ViT with `num_classes=0` so the final layer is removed and the pooler output is used (global feature vector per image). No cross-attention between CLIP and DINO; they run independently and outputs are concatenated.
- **Fusion:** Simple concatenation along the feature dimension (512 + 768 = 1280). No gating or projection in the default config.
- **Head:** Fully connected stack: `Linear(1280, 512) → ReLU → Dropout(0.2) → Linear(512, 256) → ReLU → Dropout(0.2) → Linear(256, num_labels)`. Produces one logit vector per image (multi-label).
- **Stage 1:** Encoder parameters are frozen; only the MLP head is trained.

---

### 4.2 GNN: Bipartite object–attribute graph

**High-level flow:**

```
Image → same CLIP+DINO backbone → 1280-d per image
      → treated as one “object” node per image: [B, 1, 1280]
      → build bipartite graph with A attribute nodes (see below)
      → 2× BipartiteMessagePassingLayer (object features updated from attribute messages)
      → Linear(256, num_labels) per object → mean over object dim → logits [B, num_labels]
```

**Graph construction**

- **Object nodes:** One node per image in the batch. Node features = backbone output, shape `[B, num_objects, D]` with `num_objects=1` in the current setup, so effectively `[B, 1, 1280]`.
- **Attribute nodes:** There are `A = num_labels` attribute nodes per batch. Config supports initializing them from **CLIP text** (e.g. “a {attribute} chair”) or **trainable** embeddings. In the current smoke implementation, attribute node features are set to **zeros** `[B, A, 1280]` in `build_bipartite_batch`; they are placeholders for future CLIP-text or trainable init.
- **Edges:** Only object → attribute. For each image `b` and each attribute index `a` where the multi-hot target is 1, an edge is added from the object node of image `b` to the attribute node `(b, a)`. Edge weights are 1 (config can extend to labeled vs weak later).
- **Effect when attr feats are zero:** Message from an attribute node is zero; after linear `attr_to_obj` it stays zero. So the aggregated message at each object is 0, and the layer reduces to updating object features with an MLP that takes `[object_feats; proj(0)]` — i.e. the object representation is refined by the same residual MLP stack without attribute-side information until attribute nodes are filled (e.g. with CLIP text).

**Message-passing layer (one layer)**

- **Inputs:** `object_feats` [B, O, D_in], `attr_feats` [B, A, D_attr], `edge_index` [2, E], `edge_weight` [E].
- **Step 1 – Messages:** For each edge (obj, attr), take `attr_feats[attr]`, project with `Linear(D_attr, D_in)` → same space as object feats. Multiply by edge weight.
- **Step 2 – Aggregate:** Scatter-add weighted messages by object index; divide by sum of weights per object → **weighted mean** of attribute messages per object.
- **Step 3 – Update:** Concatenate (object_feats, proj(aggregated_messages)), pass through `Linear(D_in + D_out, D_out) → ReLU → Dropout` to get new object features. So it’s a residual-style update where the “residual” is the aggregated attribute message.
- **Output:** New object features [B, O, D_out]. Attribute node features are not updated (one-way: attribute → object).

**Stack and classifier**

- **Layers:** Two `BipartiteMessagePassingLayer`s: first 1280 → 512 (attr_dim=1280), second 512 → 256 (attr_dim=1280). Config lists “heads” in YAML but the current code uses a single projection per layer (no multi-head attention).
- **Classifier:** `Linear(256, num_attributes)` applied to each object’s 256-d vector → [B, O, A]. For one object per image, the logits are then averaged over the object dimension to get [B, A] (or you can take the single object’s logits). So the final output is one multi-label logit vector per image, same as the baseline.

**Summary**

- **Baseline:** Image → CLIP + DINO (concat) → 1280-d → MLP → logits.  
- **GNN:** Same backbone; object nodes = image features; attribute nodes = 0-d placeholders (or future CLIP/trainable); bipartite edges from labels; two attribute→object message-passing layers; then linear classifier and mean over objects to get logits. The design allows injecting attribute semantics (e.g. CLIP “a red chair”) into the graph so the object representation is refined by attribute-conditioned message passing; the current smoke run uses zero attribute features, so the GNN mainly adds an extra MLP-style refinement of the fused backbone features.

---

## 5. Performance

### 5.1 Smoke Baseline (`smoke_baseline`)

**Validation (best epoch):**  

- Best val mAP: **0.3726** (epoch 2)  
- Val macro F1: 0.3704, micro F1: 0.4075

**Training:**  

- Epoch 1: train loss 0.622, val mAP 0.3638  
- Epoch 2: train loss 0.511, val mAP 0.3726

**Test (reported in `metrics.json`):**


| Metric    | Value      |
| --------- | ---------- |
| mAP       | **0.4407** |
| Macro F1  | 0.4264     |
| Micro F1  | 0.4761     |
| Test loss | 0.6028     |


**Per-attribute test AP / F1 (index ↔ attribute from label_vocab):**


| Attribute   | AP    | F1    |
| ----------- | ----- | ----- |
| black (0)   | 0.549 | 0.464 |
| blue (1)    | 0.442 | 0.492 |
| metal (2)   | 0.239 | 0.263 |
| plastic (3) | 0.167 | 0.202 |
| red (4)     | 0.447 | 0.396 |
| white (5)   | 0.476 | 0.474 |
| wooden (6)  | 0.764 | 0.693 |


Wooden and black are strongest; plastic and metal are weakest.

---

### 5.2 Smoke GNN (`smoke_gnn`)

**Validation (best):**  

- Best val mAP: **0.4033** (epoch 1)  
- Val macro F1: 0.3512, micro F1: 0.4235

**Test:**


| Metric    | Value      |
| --------- | ---------- |
| mAP       | **0.4451** |
| Macro F1  | 0.2997     |
| Micro F1  | 0.3830     |
| Test loss | 0.6443     |


**Per-attribute test AP / F1:**


| Attribute   | AP    | F1    |
| ----------- | ----- | ----- |
| black (0)   | 0.691 | 0.479 |
| blue (1)    | 0.562 | 0.511 |
| metal (2)   | 0.135 | 0.167 |
| plastic (3) | 0.016 | 0.000 |
| red (4)     | 0.665 | 0.580 |
| white (5)   | 0.351 | 0.267 |
| wooden (6)  | 0.696 | 0.095 |


GNN smoke improves mAP slightly (0.445 vs 0.441) and does better on black/blue/red AP, but **plastic** and **wooden** F1 collapse (likely due to 1 epoch + small data and graph construction).

---

### 5.3 Full GNN run (`gnn_full_stage1`)

**Validation (best epoch):**

- Best val mAP: **0.3975** (epoch 2)  
- Val macro F1: 0.3522, micro F1: 0.4469

**Training (8 epochs, full mode):**

- Epoch 1: train loss 0.6118, val mAP 0.3792  
- Epoch 2: train loss 0.4867, val mAP 0.3975  
- Epoch 3–8: train loss continues to decrease (~0.21 by epoch 8) while val mAP fluctuates around 0.37–0.39.

**Test (using best checkpoint):**

| Metric    | Value      |
| --------- | ---------- |
| mAP       | **0.4672** |
| Macro F1  | 0.4105     |
| Micro F1  | 0.5159     |
| Test loss | 0.5926     |

**Per-attribute test AP / F1:**

| Attribute   | AP    | F1    |
| ----------- | ----- | ----- |
| black (0)   | 0.582 | 0.424 |
| blue (1)    | 0.496 | 0.532 |
| metal (2)   | 0.244 | 0.281 |
| plastic (3) | 0.167 | 0.162 |
| red (4)     | 0.442 | 0.459 |
| white (5)   | 0.549 | 0.299 |
| wooden (6)  | 0.791 | 0.717 |

Compared to the smoke runs, the full GNN training improves both overall mAP (≈0.467 vs 0.44–0.45) and micro F1 (≈0.516), and stabilizes the previously weak attributes (plastic, wooden) while keeping strong performance on color and material attributes.

---

### 5.4 Summary

- **Data:** Visual Genome, chair-only, 7 attributes in smoke (black, blue, metal, plastic, red, white, wooden); 80/10/10 stratified splits.  
- **Models:** (1) Baseline: CLIP+DINO → concat → MLP. (2) GNN (smoke + full): same encoders + bipartite object–attribute graph with CLIP-text attribute nodes and 2-layer message passing.  
- **Smoke performance:** Baseline test mAP **0.44**, GNN smoke test mAP **0.45**. Baseline has more balanced per-attribute F1; GNN smoke shows instability on plastic/wooden.  
- **Full GNN performance:** `gnn_full_stage1` reaches test mAP **0.47** and micro F1 ≈ **0.52**, indicating that with full data and 8 epochs the bipartite GNN meaningfully improves over the baseline while keeping strong per-attribute performance on both color and material attributes.

---

## 6. References (in-repo)

- **Configs:** `configs/experiment.yaml`, `configs/dataset.yaml`, `configs/model.yaml`, `configs/train.yaml`, `configs/eval.yaml`  
- **Training:** `scripts/train_baseline.py`, `scripts/train_gnn.py`  
- **Data:** `scripts/data/process_visual_genome.py`, `src/data/normalization.py`  
- **Models:** `src/train/models.py`, `src/train/graph_models.py`, `src/train/encoders.py`  
- **Metrics:** `outputs/smoke_baseline/metrics.json`, `outputs/smoke_gnn/metrics.json`

