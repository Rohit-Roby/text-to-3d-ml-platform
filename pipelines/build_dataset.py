from __future__ import annotations

import argparse
import multiprocessing
from pathlib import Path

import pandas as pd
import torch

from src.features.text_encoder import create_text_encoder
from src.preprocessing.synthetic import SHAPES, make_voxel_shape
from src.preprocessing.text import clean_manifest
from src.utils.manifest import write_manifest


def _dirs(output_dir: Path) -> tuple[Path, Path]:
    voxel_dir = output_dir / "voxels"
    embed_dir = output_dir / "embeddings"
    voxel_dir.mkdir(parents=True, exist_ok=True)
    embed_dir.mkdir(parents=True, exist_ok=True)
    return voxel_dir, embed_dir


def build_synthetic_dataset(
    limit: int,
    output_dir: str | Path = "data/processed",
    encoder_backend: str = "hash",
    embedding_dim: int = 256,
) -> Path:
    output_dir = Path(output_dir)
    voxel_dir, embed_dir = _dirs(output_dir)
    encoder = create_text_encoder(backend=encoder_backend, embedding_dim=embedding_dim)
    rows: list[dict] = []

    for i in range(limit):
        shape = SHAPES[i % len(SHAPES)]
        uid = f"synthetic-{shape}-{i:04d}"
        description = f"a simple {shape} shaped 3d object"
        voxel = make_voxel_shape(shape, jitter=(i % 5) - 2)
        voxel_path = voxel_dir / f"{uid}.pt"
        embedding_path = embed_dir / f"{uid}.pt"
        torch.save(torch.from_numpy(voxel), voxel_path)
        torch.save(encoder.encode([description])[0], embedding_path)
        rows.append(
            {
                "uid": uid,
                "category": shape,
                "description": description,
                "voxel_path": str(voxel_path),
                "embedding_path": str(embedding_path),
                "source": "synthetic",
            }
        )

    manifest = pd.DataFrame(rows)
    return write_manifest(manifest, output_dir)


def build_objaverse_dataset(
    category: str,
    limit: int,
    output_dir: str | Path = "data/processed",
    encoder_backend: str = "t5",
    model_name: str = "google-t5/t5-base",
    embedding_dim: int = 256,
    download_processes: int | None = None,
) -> Path:
    from src.ingestion.objaverse_ingest import collect_category_metadata, download_objects
    from src.preprocessing.voxelize import mesh_to_voxel_array

    output_dir = Path(output_dir)
    voxel_dir, embed_dir = _dirs(output_dir)
    df = clean_manifest(collect_category_metadata(category, limit))
    if df.empty:
        raise RuntimeError(f"No usable metadata found for category={category!r}")

    processes = download_processes or max(1, multiprocessing.cpu_count() // 2)
    objects = download_objects(df["uid"].tolist(), processes=processes)
    encoder = create_text_encoder(
        backend=encoder_backend,
        model_name=model_name,
        embedding_dim=embedding_dim,
    )

    rows: list[dict] = []
    failures: list[dict] = []
    for row in df.itertuples(index=False):
        mesh_path = objects.get(row.uid)
        if not mesh_path:
            failures.append({"uid": row.uid, "reason": "download_missing"})
            continue
        try:
            voxel = mesh_to_voxel_array(mesh_path)
            voxel_path = voxel_dir / f"{row.uid}.pt"
            embedding_path = embed_dir / f"{row.uid}.pt"
            torch.save(torch.from_numpy(voxel), voxel_path)
            torch.save(encoder.encode([row.description])[0], embedding_path)
            rows.append(
                {
                    "uid": row.uid,
                    "category": row.category,
                    "description": row.description,
                    "voxel_path": str(voxel_path),
                    "embedding_path": str(embedding_path),
                    "source": "objaverse",
                }
            )
        except Exception as exc:  # noqa: BLE001 - one bad asset must not kill the entire build
            failures.append({"uid": row.uid, "reason": f"{type(exc).__name__}: {exc}"})

    manifest = pd.DataFrame(rows)
    if manifest.empty:
        raise RuntimeError("Dataset build produced zero valid assets")
    output = write_manifest(manifest, output_dir)
    pd.DataFrame(failures).to_csv(output_dir / "failed_assets.csv", index=False)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a text-to-3D training dataset")
    parser.add_argument("--source", choices=["synthetic", "objaverse"], default="synthetic")
    parser.add_argument("--category", default="chair")
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--encoder-backend", choices=["hash", "t5"], default=None)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--model-name", default="google-t5/t5-base")
    parser.add_argument("--download-processes", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    backend = args.encoder_backend or ("hash" if args.source == "synthetic" else "t5")
    if args.source == "synthetic":
        output = build_synthetic_dataset(
            limit=args.limit,
            output_dir=args.output_dir,
            encoder_backend=backend,
            embedding_dim=args.embedding_dim,
        )
    else:
        output = build_objaverse_dataset(
            category=args.category,
            limit=args.limit,
            output_dir=args.output_dir,
            encoder_backend=backend,
            model_name=args.model_name,
            embedding_dim=args.embedding_dim,
            download_processes=args.download_processes,
        )
    print(f"Built manifest: {output}")
