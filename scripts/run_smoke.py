from __future__ import annotations

import argparse
from pathlib import Path
import shutil

from pipelines.build_dataset import build_synthetic_dataset
from src.inference.runtime import GenerationRuntime
from src.training.train_cgan import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the entire project offline on tiny data")
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()

    root = Path("data/smoke")
    artifact_dir = Path("artifacts/smoke")
    shutil.rmtree(root, ignore_errors=True)
    shutil.rmtree(artifact_dir, ignore_errors=True)

    manifest = build_synthetic_dataset(
        limit=args.samples,
        output_dir=root,
        encoder_backend="hash",
        embedding_dim=64,
    )
    checkpoint = train(
        str(manifest),
        epochs=args.epochs,
        batch_size=min(3, args.samples),
        noise_dim=16,
        num_workers=0,
        artifact_dir=str(artifact_dir),
        encoder_backend="hash",
        encoder_model_name="deterministic-hash-v1",
        mlflow_enabled=False,
    )
    runtime = GenerationRuntime(checkpoint)
    output = runtime.generate_mesh(
        "a simple cube shaped 3d object",
        output_path=artifact_dir / "generated_cube.obj",
        seed=42,
    )
    print("SMOKE TEST PASSED")
    print(f"manifest={manifest}")
    print(f"checkpoint={checkpoint}")
    print(f"mesh={output}")


if __name__ == "__main__":
    main()
