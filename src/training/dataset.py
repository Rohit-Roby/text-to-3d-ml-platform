from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from src.utils.manifest import read_manifest


class TextVoxelDataset(Dataset):
    def __init__(self, manifest_path: str | Path):
        self.manifest_path = Path(manifest_path)
        self.df = read_manifest(self.manifest_path).reset_index(drop=True)
        if self.df.empty:
            raise ValueError(f"Manifest contains no rows: {manifest_path}")
        required = {"uid", "voxel_path", "embedding_path"}
        missing = required.difference(self.df.columns)
        if missing:
            raise ValueError(f"Manifest is missing columns: {sorted(missing)}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        voxel = torch.load(row["voxel_path"], map_location="cpu", weights_only=True).float()
        embedding = torch.load(row["embedding_path"], map_location="cpu", weights_only=True).float()
        return voxel, embedding, row["uid"]
