from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Dict, Iterable, Tuple

import pandas as pd


def normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def match_schema(columns: Iterable[str], canonical: Dict[str, Iterable[str]], threshold: float = 0.84) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for original in columns:
        norm = normalize_name(original)
        best_name, best_score = None, 0.0
        for target, aliases in canonical.items():
            candidates = [target, *aliases]
            score = max(SequenceMatcher(None, norm, normalize_name(x)).ratio() for x in candidates)
            if score > best_score:
                best_name, best_score = target, score
        if best_name and best_score >= threshold and best_name != original:
            mapping[original] = best_name
    return mapping


def apply_schema_mapping(df: pd.DataFrame, canonical: Dict[str, Iterable[str]], threshold: float = 0.84) -> Tuple[pd.DataFrame, Dict[str, str]]:
    mapping = match_schema(df.columns, canonical, threshold)
    return df.rename(columns=mapping), mapping
