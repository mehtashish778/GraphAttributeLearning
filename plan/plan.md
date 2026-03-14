# Adjective-Aware Object Embedding using CLIP, DINOv2, and Graph Neural Networks

## 1. Project Objective

Build a system that learns **adjectives associated with objects** (e.g.,
chairs) and reshapes the embedding space so objects cluster based on
attributes like:

-   broken
-   blue
-   wooden
-   plastic
-   dirty
-   metal

Encoders used: - CLIP Vision Encoder - DINOv2 Vision Encoder - Graph
Neural Network layer for attribute reasoning

------------------------------------------------------------------------

## 2. System Architecture

Image │ ├── CLIP Encoder ├── DINOv2 Encoder │ └── Feature Fusion │ ▼
Object Embedding │ ▼ Graph Construction │ ▼ Graph Neural Network │ ▼
Adjective-aware embedding │ ▼ Attribute prediction

------------------------------------------------------------------------

## 3. Encoders

### CLIP Vision Encoder

Common models: - ViT-B/32 - ViT-L/14

Output embedding size: 512--768

Benefits: - aligned with language - semantic understanding

### DINOv2 Encoder

Common models: - vit-small - vit-base - vit-large

Output embedding size: 768--1024

Benefits: - strong visual structure representation - self-supervised
learning

------------------------------------------------------------------------

## 4. Feature Fusion

clip_embedding = CLIP(image)\
dino_embedding = DINO(image)

combined_embedding = concat(clip_embedding, dino_embedding)

Typical dimension:

1536

------------------------------------------------------------------------

## 5. Graph Design

Graph nodes:

Object nodes\
Attribute nodes

Example:

Objects: chair_1 chair_2 chair_3

Attributes: blue broken wooden plastic dirty metal

Edges:

chair → blue\
chair → broken\
chair → plastic

Optional attribute relations:

plastic ↔ material\
blue ↔ color\
broken ↔ damage

------------------------------------------------------------------------

## 6. Node Features

### Object nodes

Feature = fused embedding

dimension = 1536

### Attribute nodes

Option 1 (recommended)

Use CLIP text embeddings:

"a blue chair"\
"a broken chair"\
"a wooden chair"

Option 2

Learn attribute embeddings (trainable vectors)

------------------------------------------------------------------------

## 7. GNN Model

Suggested architecture:

GATConv(input_dim,512)\
ReLU\
GATConv(512,256)\
Linear(256,1)

Prediction is computed from attribute nodes.

------------------------------------------------------------------------

## 8. Training Objectives

### Attribute Prediction

Multi‑label classification

Loss:

Binary Cross Entropy

Example label:

chair_1: broken=1\
blue=0\
plastic=1

### Contrastive Embedding Loss

Encourage clustering by adjectives.

Example:

Positive pair: blue chair A blue chair B

Negative pair: blue chair vs broken chair

Loss options: - Triplet loss - Contrastive loss - InfoNCE

------------------------------------------------------------------------

## 9. Datasets

### Visual Genome

\~108k images with objects, attributes, and relationships.

Example annotation:

chair → wooden\
chair → old

Download: https://visualgenome.org

------------------------------------------------------------------------

### Open Images V7

\~9M images with detection and attribute labels.

Example:

chair → plastic\
chair → old

Download: https://storage.googleapis.com/openimages/web/index.html

------------------------------------------------------------------------

### MIT States Dataset

Object‑state dataset.

Examples:

broken chair\
painted chair\
clean chair

------------------------------------------------------------------------

### SUN Attribute Dataset

Dataset containing \~102 visual attributes.

Examples:

rusty\
shiny\
old\
colorful

------------------------------------------------------------------------

### COCO Dataset

Object detection dataset (\~330k images).

Attributes must be manually annotated.

Download: https://cocodataset.org

------------------------------------------------------------------------

### LAION‑400M

Large image‑text dataset with captions.

Example captions:

"a broken plastic chair"\
"a blue chair"

Attributes can be extracted using NLP.

Download: https://laion.ai

------------------------------------------------------------------------

### ImageNet‑Attribute variants

Researchers often annotate ImageNet objects with attributes such as:

metal\
wooden\
colorful

Useful for transfer learning.

------------------------------------------------------------------------

## 10. Data Preparation

Dataset pipeline:

images ↓ object detection (chair) ↓ extract attributes ↓ build
object‑attribute graph

Example dataset:

dataset/ chair_001.jpg chair_002.jpg

labels:

chair_001.jpg → \[blue,plastic\]\
chair_002.jpg → \[broken,wooden\]

------------------------------------------------------------------------

## 11. Training Pipeline

Image ↓ CLIP encoder ↓ DINOv2 encoder ↓ feature fusion ↓ graph
construction ↓ GNN forward pass ↓ attribute prediction ↓ loss
computation ↓ backpropagation

------------------------------------------------------------------------

## 12. Inference Pipeline

New Image ↓ CLIP + DINO ↓ embedding fusion ↓ graph node creation ↓ GNN
inference ↓ attribute scores

Example output:

broken: 0.91\
blue: 0.12\
plastic: 0.87\
wooden: 0.04

------------------------------------------------------------------------

## 13. Evaluation

Metrics:

-   mean average precision
-   F1 score
-   multi‑label accuracy

Embedding analysis:

-   t‑SNE
-   UMAP

Goal: observe clusters based on adjectives.

------------------------------------------------------------------------

## 14. Future Extensions

Graph Transformer instead of GNN.

Scene graph learning:

chair → broken → leg

LLM integration to generate descriptions:

"This is a broken blue plastic chair."

------------------------------------------------------------------------

## 15. Deliverables

-   adjective aware embedding model
-   attribute prediction model
-   graph dataset
-   clustering visualizations

------------------------------------------------------------------------

## 16. Summary

The system combines:

CLIP semantic features\
DINOv2 visual features\
Graph neural network reasoning

to produce an **embedding space structured by object adjectives**.
