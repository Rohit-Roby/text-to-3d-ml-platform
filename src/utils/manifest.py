from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_manifest(
    df: pd.DataFrame, output_dir: str | Path, stem: str = "training_manifest"
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = output_dir / f"{stem}.parquet"
    try:
        df.to_parquet(parquet_path, index=False)
        return parquet_path
    except ImportError:
        csv_path = output_dir / f"{stem}.csv"
        df.to_csv(csv_path, index=False)
        return csv_path


def read_manifest(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported manifest format: {path.suffix}")
