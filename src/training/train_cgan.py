from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.models.cgan import ConditionalDiscriminator, ConditionalGenerator
from src.training.dataset import TextVoxelDataset
from src.utils.reproducibility import seed_everything


def _mlflow_module(enabled: bool):
    if not enabled:
        return None
    try:
        import mlflow

        return mlflow
    except ImportError:
        return None


def train(
    manifest_path: str,
    epochs: int = 20,
    batch_size: int = 16,
    lr: float = 2e-4,
    noise_dim: int = 128,
    num_workers: int = 0,
    artifact_dir: str = "artifacts",
    encoder_backend: str = "t5",
    encoder_model_name: str = "google-t5/t5-base",
    seed: int = 42,
    mlflow_enabled: bool = True,
) -> Path:
    seed_everything(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = TextVoxelDataset(manifest_path)
    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
    )

    _, sample_embedding, _ = dataset[0]
    text_dim = int(sample_embedding.numel())
    generator = ConditionalGenerator(text_dim=text_dim, noise_dim=noise_dim).to(device)
    discriminator = ConditionalDiscriminator(text_dim=text_dim).to(device)

    criterion = nn.BCELoss()
    opt_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    mlflow = _mlflow_module(mlflow_enabled)
    run = None
    if mlflow is not None:
        mlflow.set_experiment("text-to-3d-cgan")
        run = mlflow.start_run()
        mlflow.log_params(
            {
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "noise_dim": noise_dim,
                "text_dim": text_dim,
                "encoder_backend": encoder_backend,
            }
        )

    history: list[dict[str, float]] = []
    try:
        for epoch in range(epochs):
            g_total = d_total = 0.0
            batches = 0
            for voxels, text_embeddings, _ in loader:
                voxels = voxels.to(device)
                if voxels.ndim == 4:
                    voxels = voxels.unsqueeze(1)
                text_embeddings = text_embeddings.to(device).view(voxels.size(0), -1)
                bs = voxels.size(0)
                real = torch.ones(bs, 1, device=device)
                fake = torch.zeros(bs, 1, device=device)

                opt_d.zero_grad(set_to_none=True)
                real_loss = criterion(discriminator(voxels, text_embeddings), real)
                noise = torch.randn(bs, noise_dim, device=device)
                generated = generator(text_embeddings, noise)
                fake_loss = criterion(discriminator(generated.detach(), text_embeddings), fake)
                d_loss = (real_loss + fake_loss) / 2
                d_loss.backward()
                opt_d.step()

                opt_g.zero_grad(set_to_none=True)
                noise = torch.randn(bs, noise_dim, device=device)
                generated = generator(text_embeddings, noise)
                g_loss = criterion(discriminator(generated, text_embeddings), real)
                g_loss.backward()
                opt_g.step()

                g_total += float(g_loss.item())
                d_total += float(d_loss.item())
                batches += 1

            metrics = {
                "epoch": float(epoch + 1),
                "generator_loss": g_total / max(1, batches),
                "discriminator_loss": d_total / max(1, batches),
            }
            history.append(metrics)
            print(
                f"epoch={epoch + 1}/{epochs} "
                f"g_loss={metrics['generator_loss']:.4f} "
                f"d_loss={metrics['discriminator_loss']:.4f}"
            )
            if mlflow is not None:
                mlflow.log_metrics(
                    {
                        "generator_loss": metrics["generator_loss"],
                        "discriminator_loss": metrics["discriminator_loss"],
                    },
                    step=epoch,
                )

        artifact_path = Path(artifact_dir)
        artifact_path.mkdir(parents=True, exist_ok=True)
        checkpoint_path = artifact_path / "generator_checkpoint.pt"
        checkpoint = {
            "model_state_dict": generator.state_dict(),
            "text_dim": text_dim,
            "noise_dim": noise_dim,
            "encoder_backend": encoder_backend,
            "encoder_model_name": encoder_model_name,
            "voxel_shape": [32, 32, 32],
            "seed": seed,
        }
        torch.save(checkpoint, checkpoint_path)
        torch.save(discriminator.state_dict(), artifact_path / "discriminator.pt")
        (artifact_path / "training_history.json").write_text(json.dumps(history, indent=2))

        if mlflow is not None:
            mlflow.log_artifact(str(checkpoint_path))
            mlflow.log_artifact(str(artifact_path / "training_history.json"))
        return checkpoint_path
    finally:
        if mlflow is not None and run is not None:
            mlflow.end_run()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the conditional 3D GAN")
    parser.add_argument("--manifest", default="data/processed/training_manifest.parquet")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--noise-dim", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--encoder-backend", choices=["hash", "t5"], default="t5")
    parser.add_argument("--encoder-model-name", default="google-t5/t5-base")
    parser.add_argument("--no-mlflow", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    path = train(
        manifest_path=args.manifest,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        noise_dim=args.noise_dim,
        num_workers=args.num_workers,
        encoder_backend=args.encoder_backend,
        encoder_model_name=args.encoder_model_name,
        mlflow_enabled=not args.no_mlflow,
    )
    print(f"Saved checkpoint: {path}")
