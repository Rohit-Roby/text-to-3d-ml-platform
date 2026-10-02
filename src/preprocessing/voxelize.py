from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.ndimage import zoom


def mesh_to_voxel_array(
    mesh_path: str | Path,
    voxel_size: float = 0.05,
    target_shape: tuple[int, int, int] = (32, 32, 32),
) -> np.ndarray:
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if mesh.is_empty():
        raise ValueError(f"Empty mesh: {mesh_path}")

    extent = np.asarray(mesh.get_max_bound()) - np.asarray(mesh.get_min_bound())
    max_extent = float(np.max(extent))
    if max_extent <= 0:
        raise ValueError(f"Degenerate mesh: {mesh_path}")

    mesh.scale(1.0 / max_extent, center=mesh.get_center())
    grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh, voxel_size=voxel_size)
    voxels = grid.get_voxels()
    if not voxels:
        return np.zeros(target_shape, dtype=np.float32)

    positions = np.asarray([v.grid_index for v in voxels], dtype=np.int64)
    positions -= positions.min(axis=0)
    source_shape = tuple((positions.max(axis=0) + 1).tolist())
    arr = np.zeros(source_shape, dtype=np.float32)
    arr[tuple(positions.T)] = 1.0

    factors = [target_shape[i] / arr.shape[i] for i in range(3)]
    resized = zoom(arr, factors, order=0)
    resized = resized[: target_shape[0], : target_shape[1], : target_shape[2]]

    padded = np.zeros(target_shape, dtype=np.float32)
    slices = tuple(slice(0, min(resized.shape[i], target_shape[i])) for i in range(3))
    padded[slices] = resized[slices]
    return padded
