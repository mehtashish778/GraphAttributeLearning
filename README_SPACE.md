## Hugging Face Space Setup

### 1. Prepare Repository

Push this project (including `app.py`, `requirements-space.txt`, and checkpoints) to a Git repository that you will connect to your Space.

Recommended checkpoint locations:

- Baseline: `outputs/smoke_baseline/best.pt`
- GNN: `outputs/smoke_gnn/best.pt`

### 2. Create a New Space

1. On Hugging Face, create a new **Space**.
2. Select **Gradio** as the SDK.
3. Point the Space to this repository (or upload the files manually).

### 3. Configure Runtime

- Hardware: start with **CPU**; upgrade to GPU if needed.
- Python dependencies:
  - Set the Space to use `requirements-space.txt` as the dependency file.

### 4. Local Test Before Pushing

Create and activate a clean virtualenv, then run:

```bash
pip install -r requirements-space.txt
python app.py
```

Open the printed URL in your browser, upload a chair image, and verify that:

- Baseline and GNN models both run.
- Switching model type updates which checkpoint is used.

### 5. Expected Behavior in the Space

- The app starts from `app.py`.
- Default checkpoint paths:
  - Baseline: `outputs/smoke_baseline/best.pt`
  - GNN: `outputs/smoke_gnn/best.pt`
- You can override checkpoint paths via the text box in the UI.

