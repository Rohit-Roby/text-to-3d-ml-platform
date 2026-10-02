# Text-to-3D ML Data Platform

A portfolio-grade **data engineering + machine learning** project that turns an academic text-to-3D notebook into a reproducible training and inference system.

The main production experiment is a conditional 3D GAN that generates a `32×32×32` voxel volume from a text embedding and random noise, then converts the generated volume into an OBJ mesh with marching cubes.

## Current runnable status

The repository now has two execution paths:

1. **Offline smoke path** — uses generated cube/sphere/cylinder voxel data plus a deterministic hash text encoder. It requires no external model download and exists so the full pipeline can be tested quickly in CI or on a laptop.
2. **Portfolio path** — uses Objaverse/LVIS assets, Open3D voxelisation and T5 embeddings. This is the path intended for real experiments and cloud scaling.

The offline path has been validated end to end:

```text
synthetic 3D data
    ↓
manifest
    ↓
text embeddings + voxel tensors
    ↓
conditional GAN training
    ↓
generator_checkpoint.pt
    ↓
checkpoint reload
    ↓
prompt inference
    ↓
32³ voxel prediction
    ↓
marching cubes
    ↓
OBJ mesh
```

Unit-test status at the time this package was produced: **6 passed**.

## Architecture

```text
Objaverse / LVIS
      │
      ├──────────────→ metadata manifest
      │
      ▼
 raw GLB assets
      │
      ▼
mesh validation + normalisation
      │
      ▼
Open3D voxelisation ─────────┐
                             │
text descriptions            │
      │                      │
      ▼                      │
T5 encoder                    │
      │                      │
      └──────────────┬───────┘
                     ▼
             ML-ready manifest
           voxels + embeddings
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
    BigQuery / dbt       PyTorch cGAN
     data quality          + MLflow
                                │
                                ▼
                      versioned checkpoint
                                │
                                ▼
                           FastAPI
                                │
                                ▼
                        prompt → voxel
                                │
                                ▼
                       marching cubes
                                │
                                ▼
                            OBJ mesh
```

## Why there is an offline smoke mode

A professional ML repository should be testable without requiring a reviewer to download a large dataset or multi-gigabyte model weights. Smoke mode tests the engineering contract of the whole project while the Objaverse/T5 path is used for meaningful experiments.

The hash encoder is **not** presented as the real ML text model. It is a deterministic test backend. Real experiments should use T5.

## Quick start

### 1. Create an environment

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Python 3.11 is recommended.

### 2. Prove the entire project works

```bash
make smoke
```

or:

```bash
python -m scripts.run_smoke --samples 9 --epochs 1
```

Expected final output includes:

```text
SMOKE TEST PASSED
manifest=...
checkpoint=...
mesh=...generated_cube.obj
```

### 3. Run unit tests

```bash
pytest -q
```

## Local demo workflow

Build a small local dataset without downloading Objaverse:

```bash
python pipelines/build_dataset.py --source synthetic --limit 24
```

Train a small model:

```bash
python -m src.training.train_cgan \
  --manifest data/processed/training_manifest.parquet \
  --epochs 5 \
  --encoder-backend hash \
  --no-mlflow
```

If PyArrow is not installed, dataset construction automatically writes `training_manifest.csv`; pass that path to the training command instead.

Start the API:

```bash
uvicorn src.inference.api:app --reload
```

Health check:

```text
GET /health
```

Generation request:

```json
POST /generate
{
  "prompt": "a simple cube shaped 3d object",
  "seed": 42,
  "threshold": 0.5
}
```

The API returns a `/meshes/<filename>.obj` download endpoint when a trained checkpoint is available.

## Real Objaverse + T5 workflow

Start small on a development machine:

```bash
python pipelines/build_dataset.py \
  --source objaverse \
  --category chair \
  --limit 50 \
  --encoder-backend t5
```

This performs:

- LVIS category lookup
- Objaverse annotation ingestion
- asset download
- text cleaning
- GLB mesh loading
- mesh normalisation
- Open3D voxelisation
- T5 embedding generation
- per-asset tensor persistence
- training-manifest creation
- failed-asset logging

Then train:

```bash
python -m src.training.train_cgan \
  --manifest data/processed/training_manifest.parquet \
  --epochs 20 \
  --encoder-backend t5
```

The resulting model is stored as:

```text
artifacts/generator_checkpoint.pt
```

The checkpoint contains both model weights and inference metadata such as text dimension, noise dimension and encoder backend, so inference does not rely on hidden notebook state.

## Repository structure

```text
configs/                 reproducible settings
pipelines/               dataset build entry points
scripts/                 end-to-end smoke workflow
src/
  features/              T5 + deterministic test encoder
  ingestion/             Objaverse + cloud ingestion
  inference/             runtime, API, voxel-to-mesh export
  models/                conditional 3D GAN
  preprocessing/         text, voxelisation, synthetic test data
  training/              Dataset + training loop
  utils/                 config, manifests, reproducibility
dbt/                     BigQuery staging and quality marts
tests/                   unit and inference tests
.github/workflows/        CI
```

## Data engineering features

- raw/interim/processed data-layer design
- reproducible dataset manifests
- Parquet with CSV fallback for lightweight environments
- isolated per-asset failure handling
- persistent voxel and embedding features
- Objaverse metadata ingestion
- GCS integration hooks
- BigQuery analytical metadata layer
- dbt staging and quality marts
- deterministic smoke dataset for CI
- testable modular transformations
- container and CI setup

## Machine learning features

- T5 text embeddings for the real experiment
- conditional 3D GAN
- text + latent-noise conditioning
- versioned self-describing PyTorch checkpoints
- MLflow experiment logging
- deterministic inference seeds
- voxel-to-mesh reconstruction with marching cubes
- FastAPI serving and mesh download

## Important correction from the master's notebook

The original conditional GAN experiment did not consistently use the semantic text representation as the generator condition. This refactor makes the conditioning explicit:

```text
T5 text embedding + random noise
              ↓
     ConditionalGenerator
              ↓
       generated voxel
```

This makes the implementation match the intended text-conditioned generation problem.

## What is deliberately not claimed

The original NeRF exploration is not treated as a finished production model. A defensible NeRF implementation requires multi-view images, camera poses, ray sampling and volume rendering. Until those elements are implemented and evaluated, NeRF remains a research extension rather than part of the main runnable path.

Likewise, a one-epoch smoke model proves **software integration**, not generation quality. Portfolio ML results should be based on a larger real dataset and quantitative evaluation.

## Evaluation roadmap

Before presenting model quality claims, add:

- train/validation/test splits
- voxel IoU
- Chamfer Distance between reconstructed point clouds
- prompt-to-shape retrieval consistency
- diversity across multiple random seeds
- qualitative rendered examples
- experiment comparison in MLflow

## GCP scale-up

The intended cloud version uses:

- **Cloud Storage** — GLB assets, voxels, embeddings and model artifacts
- **BigQuery** — asset metadata, pipeline runs, failures and evaluation metrics
- **dbt** — staging and data-quality marts
- **MLflow** — experiment tracking
- **Cloud Run** — FastAPI inference container
- **Artifact Registry** — Docker images
- **GitHub Actions** — CI/CD

## Portfolio positioning

This project should be presented as an **ML data platform**, not merely a generative-AI notebook.

It demonstrates unstructured 3D-data ingestion, feature pipelines, reproducible datasets, data-quality handling, transformer features, deep-learning training, experiment tracking, checkpoint management and model serving within one project.
