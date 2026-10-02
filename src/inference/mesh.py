from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh
from skimage.measure import marching_cubes


def voxel_to_mesh(voxel_grid: np.ndarray, threshold: float = 0.5) -> trimesh.Trimesh:
    grid = np.asarray(voxel_grid, dtype=np.float32).squeeze()
    if grid.ndim != 3:
        raise ValueError(f"Expected a 3D voxel grid, got shape {grid.shape}")
    if not np.isfinite(grid).all():
        raise ValueError("Voxel grid contains non-finite values")
    if float(grid.max()) <= threshold or float(grid.min()) >= threshold:
        # GANs can be poorly calibrated early in training. Adapt the level to the
        # prediction range so smoke-test models can still yield a valid surface.
        lo, hi = float(grid.min()), float(grid.max())
        if hi <= lo:
            raise ValueError("Voxel grid has no isosurface")
        threshold = (lo + hi) / 2.0

    vertices, faces, normals, _ = marching_cubes(grid, level=threshold)
    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        vertex_normals=normals,
        process=False,
    )
    mesh.vertices -= mesh.bounding_box.centroid
    scale = float(np.max(mesh.extents)) if mesh.vertices.size else 1.0
    if scale > 0:
        mesh.vertices /= scale
    return mesh


def export_mesh(voxel_grid: np.ndarray, output_path: str | Path, threshold: float = 0.5) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    mesh = voxel_to_mesh(voxel_grid, threshold=threshold)
    mesh.export(output)
    return output
