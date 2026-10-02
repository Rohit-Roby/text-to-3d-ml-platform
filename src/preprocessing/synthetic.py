from __future__ import annotations

import numpy as np


SHAPES = ("cube", "sphere", "cylinder")


def make_voxel_shape(shape: str, size: int = 32, jitter: int = 0) -> np.ndarray:
    """Create simple binary 3D primitives for offline end-to-end smoke tests."""
    coords = np.indices((size, size, size), dtype=np.float32)
    x, y, z = coords
    c = (size - 1) / 2.0
    radius = size * 0.28 + jitter * 0.25

    if shape == "sphere":
        arr = ((x - c) ** 2 + (y - c) ** 2 + (z - c) ** 2 <= radius**2)
    elif shape == "cylinder":
        radial = (x - c) ** 2 + (y - c) ** 2 <= radius**2
        height = np.abs(z - c) <= size * 0.28
        arr = radial & height
    elif shape == "cube":
        half = max(3, int(size * 0.24 + jitter * 0.1))
        arr = (
            (np.abs(x - c) <= half)
            & (np.abs(y - c) <= half)
            & (np.abs(z - c) <= half)
        )
    else:
        raise ValueError(f"Unsupported synthetic shape: {shape}")

    return arr.astype(np.float32)
