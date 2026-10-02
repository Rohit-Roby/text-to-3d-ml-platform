from __future__ import annotations

from pathlib import Path
from unicodedata import category
import pandas as pd
import objaverse

import re



IGNORED_TAGS = {
    "3d",
    "model",
    "3d model",
    "free",
    "free model",
    "free3dmodel",
    "download",
    "downloadable",
    "sketchfab",
    "blender",
    "maya",
    "3ds max",
    "3dsmax",
    "substance painter",
    "zbrush",
    "unity",
    "unreal",
}


JUNK_PHRASES = {
    "newfields design gallery",
    "design gallery",
}


def clean_text(value):
    if value is None:
        return ""

    value = str(value)

    value = value.replace("-", " ")
    value = value.replace("_", " ")

    value = re.sub(r"\s+", " ", value)

    return value.strip()

def sanitize_description(text, category, max_words=35):
    text = clean_text(text)

    if not text:
        return f"{category} furniture"

    # Remove URLs
    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text,
        flags=re.IGNORECASE,
    )

    # Remove obvious promotional / platform language
    promotional_patterns = [
        r"please\s+follow.*",
        r"follow\s+and\s+support.*",
        r"support\s+me.*",
        r"free\s+to\s+use.*",
        r"click\s+here.*",
        r"facebook.*",
        r"search\s+page.*",
        r"publisher.*",
        r"asset\s+store.*",
        r"download\s+link.*",
    ]

    for pattern in promotional_patterns:
        text = re.sub(
            pattern,
            " ",
            text,
            flags=re.IGNORECASE,
        )

    # Remove standalone IDs / hashes
    text = re.sub(
        r"\b[0-9a-f]{12,}\b",
        " ",
        text,
        flags=re.IGNORECASE,
    )

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Limit excessively long descriptions
    words = text.split()

    if len(words) > max_words:
        text = " ".join(words[:max_words])

    if not text:
        return f"{category} furniture"

    return text

def is_useful_text(value):
    text = clean_text(value)

    if not text:
        return False

    lower = text.lower().strip()

    # ------------------------------------------------
    # Known non-semantic source / collection labels
    # ------------------------------------------------
    junk_phrases = {
        "newfields design gallery",
        "design gallery",
    }

    if lower in junk_phrases:
        return False

    # ------------------------------------------------
    # UUID / hash / identifier-like strings
    # ------------------------------------------------
    compact = re.sub(r"[^0-9a-f]", "", lower)

    if (
        len(compact) >= 12
        and re.fullmatch(r"[0-9a-f]+", compact)
        and not re.search(r"\s", lower)
    ):
        return False

    # ------------------------------------------------
    # Date/time object names
    #
    # Handles:
    # Sat, 15 Jun 2019 18:32:10
    # sat 15 jun 2019 18 32 10
    # Wed, 17 Oct 2018 19:26:44
    # ------------------------------------------------
    date_pattern = (
        r"^(?:mon|tue|wed|thu|fri|sat|sun)"
        r"[\s,]+"
        r"\d{1,2}\s+"
        r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)"
        r"\s+\d{4}"
        r"(?:[\s,]+"
        r"\d{1,2}(?:[:\s]\d{2})(?:[:\s]\d{2})?"
        r")?$"
    )

    if re.fullmatch(date_pattern, lower):
        return False

    # ------------------------------------------------
    # Mostly numeric / meaningless strings
    # ------------------------------------------------
    alphabetic_chars = sum(
        char.isalpha()
        for char in text
    )

    if alphabetic_chars < 3:
        return False

    return True

def build_description(annotation, category):
    parts = []

    # ------------------------------------------------
    # 1. Name
    # ------------------------------------------------
    name = clean_text(annotation.get("name"))

    if is_useful_text(name):
        parts.append(name)

    # ------------------------------------------------
    # 2. Description
    # ------------------------------------------------
    raw_description = clean_text(
        annotation.get("description")
    )

    if is_useful_text(raw_description):
        parts.append(raw_description)

    # ------------------------------------------------
    # 3. Tags
    # ------------------------------------------------
    tags = annotation.get("tags") or []

    tag_names = []

    for tag in tags:

        if not isinstance(tag, dict):
            continue

        tag_name = clean_text(
            tag.get("name")
        )

        if (
            is_useful_text(tag_name)
            and tag_name.lower() not in IGNORED_TAGS
        ):
            tag_names.append(tag_name)

    tag_names = list(
        dict.fromkeys(tag_names)
    )

    parts.extend(tag_names)

    # ------------------------------------------------
    # 4. Categories
    # ------------------------------------------------
    categories = annotation.get(
        "categories"
    ) or []

    category_names = []

    for item in categories:

        if not isinstance(item, dict):
            continue

        category_name = clean_text(
            item.get("name")
        )

        if is_useful_text(category_name):
            category_names.append(
                category_name
            )

    category_names = list(
        dict.fromkeys(category_names)
    )

    parts.extend(category_names)

    # ------------------------------------------------
    # 5. Remove duplicate fragments
    # ------------------------------------------------
    parts = list(dict.fromkeys(parts))

    text = clean_text(
        " ".join(parts)
    )

    # ------------------------------------------------
    # 6. Guaranteed semantic fallback
    # ------------------------------------------------
    if not text:
        text = f"{category} furniture"

    return sanitize_description(
        text,
        category,
    )


def collect_category_metadata(category: str, limit: int | None = None) -> pd.DataFrame:
    """Collect UID and tag-derived text metadata for one Objaverse/LVIS category."""
    lvis = objaverse.load_lvis_annotations()
    if category not in lvis:
        raise ValueError(f"Unknown Objaverse LVIS category: {category}")

    uids = lvis[category][:limit] if limit else lvis[category]
    annotations = objaverse.load_annotations(uids)
    rows: list[dict] = []

    for uid in uids:
        annotation = annotations.get(uid) or {}
        tags = annotation.get("tags") or []

        description = build_description(annotation, category)

        # Fallback when Objaverse metadata has no usable tags.
        if not description:
            description = f"a 3d model of a {category}"

        rows.append(
            {
                "uid": uid,
                "category": category,
                "description_raw": description,
            }
        )

    return pd.DataFrame(rows)


def download_objects(uids: list[str], processes: int = 4) -> dict[str, str]:
    return objaverse.load_objects(uids=uids, download_processes=processes)


def save_manifest(df: pd.DataFrame, output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output, index=False)
    return output
