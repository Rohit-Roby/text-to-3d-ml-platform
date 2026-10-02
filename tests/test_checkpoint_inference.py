from pathlib import Path

import torch

from src.inference.runtime import GenerationRuntime
from src.models.cgan import ConditionalGenerator


def test_runtime_generates_voxel(tmp_path: Path):
    model = ConditionalGenerator(text_dim=16, noise_dim=8)
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "text_dim": 16,
            "noise_dim": 8,
            "encoder_backend": "hash",
            "encoder_model_name": "deterministic-hash-v1",
            "voxel_shape": [32, 32, 32],
            "seed": 42,
        },
        checkpoint,
    )
    runtime = GenerationRuntime(checkpoint)
    voxel = runtime.generate_voxel("a cube", seed=1)
    assert voxel.shape == (32, 32, 32)
