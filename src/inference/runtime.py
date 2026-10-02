from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.features.text_encoder import create_text_encoder
from src.inference.mesh import export_mesh
from src.models.cgan import ConditionalGenerator


class GenerationRuntime:
    def __init__(self, checkpoint_path: str | Path = "artifacts/generator_checkpoint.pt"):
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found: {self.checkpoint_path}. Train the model first."
            )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=True)
        self.text_dim = int(checkpoint["text_dim"])
        self.noise_dim = int(checkpoint["noise_dim"])
        self.encoder_backend = str(checkpoint["encoder_backend"])
        self.encoder_model_name = str(checkpoint.get("encoder_model_name", "google-t5/t5-base"))
        self.encoder = create_text_encoder(
            backend=self.encoder_backend,
            model_name=self.encoder_model_name,
            embedding_dim=self.text_dim,
            device=self.device,
        )
        self.generator = ConditionalGenerator(
            text_dim=self.text_dim,
            noise_dim=self.noise_dim,
        ).to(self.device)
        self.generator.load_state_dict(checkpoint["model_state_dict"])
        self.generator.eval()

    @torch.inference_mode()
    def generate_voxel(self, prompt: str, seed: int | None = None) -> np.ndarray:
        prompt = (prompt or "").strip()
        if not prompt:
            raise ValueError("Prompt must not be empty")
        embedding = self.encoder.encode([prompt]).to(self.device).view(1, -1)
        if seed is None:
            noise = torch.randn(1, self.noise_dim, device=self.device)
        else:
            generator = torch.Generator(device=self.device).manual_seed(seed)
            noise = torch.randn(1, self.noise_dim, generator=generator, device=self.device)
        voxel = self.generator(embedding, noise)[0, 0]
        return voxel.detach().cpu().numpy()

    def generate_mesh(
        self,
        prompt: str,
        output_path: str | Path,
        seed: int | None = None,
        threshold: float = 0.5,
    ) -> Path:
        voxel = self.generate_voxel(prompt, seed=seed)
        return export_mesh(voxel, output_path=output_path, threshold=threshold)
