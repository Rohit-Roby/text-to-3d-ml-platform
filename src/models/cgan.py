from __future__ import annotations

import torch
from torch import nn


class ConditionalGenerator(nn.Module):
    def __init__(self, text_dim: int = 768, noise_dim: int = 128):
        super().__init__()
        self.noise_dim = noise_dim
        self.fc = nn.Sequential(
            nn.Linear(text_dim + noise_dim, 512 * 4 * 4 * 4),
            nn.ReLU(True),
        )
        self.net = nn.Sequential(
            nn.ConvTranspose3d(512, 256, 4, 2, 1),
            nn.BatchNorm3d(256),
            nn.ReLU(True),
            nn.ConvTranspose3d(256, 128, 4, 2, 1),
            nn.BatchNorm3d(128),
            nn.ReLU(True),
            nn.ConvTranspose3d(128, 1, 4, 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, text_embedding: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        x = torch.cat([text_embedding, noise], dim=1)
        x = self.fc(x).view(-1, 512, 4, 4, 4)
        return self.net(x)


class ConditionalDiscriminator(nn.Module):
    def __init__(self, text_dim: int = 768):
        super().__init__()
        self.voxel_net = nn.Sequential(
            nn.Conv3d(1, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(64, 128, 4, 2, 1),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(128, 256, 4, 2, 1),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.text_net = nn.Sequential(nn.Linear(text_dim, 256), nn.LeakyReLU(0.2, inplace=True))
        self.head = nn.Sequential(nn.Linear(256 * 4 * 4 * 4 + 256, 1), nn.Sigmoid())

    def forward(self, voxels: torch.Tensor, text_embedding: torch.Tensor) -> torch.Tensor:
        voxel_features = self.voxel_net(voxels).flatten(1)
        text_features = self.text_net(text_embedding)
        return self.head(torch.cat([voxel_features, text_features], dim=1))
