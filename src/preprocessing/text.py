from __future__ import annotations

import re
import pandas as pd


def clean_description(text: str) -> str:
    """Light cleaning only; transformer tokenizers should retain useful semantics."""
    text = (text or "").lower().strip()
    text = re.sub(r"[^a-z0-9\s-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_manifest(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["description"] = out["description_raw"].map(clean_description)
    out = out[out["description"].str.len() > 0].drop_duplicates(subset=["uid"])
    return out.reset_index(drop=True)
