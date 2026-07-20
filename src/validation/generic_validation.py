from __future__ import annotations

import re
from typing import Dict, Iterable, Optional

import pandas as pd


DEFAULT_PATTERNS = {
    "email": r"^[^@\s]+@[^@\s]+\.[^@\s]+$",
    "phone": r"^\+?[0-9][0-9\- ()]{6,19}$",
    "date": r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}$",
}


def infer_pattern(column: str) -> Optional[str]:
    name = column.lower()
    for token, pattern in DEFAULT_PATTERNS.items():
        if token in name:
            return pattern
    return None


def validate_dataframe(
    df: pd.DataFrame,
    required_columns: Optional[Iterable[str]] = None,
    unique_columns: Optional[Iterable[str]] = None,
    regex_rules: Optional[Dict[str, str]] = None,
) -> Dict:
    required_columns = list(required_columns or [])
    unique_columns = list(unique_columns or [])
    regex_rules = dict(regex_rules or {})
    rules = []

    for col in required_columns:
        exists = col in df.columns
        invalid = int(df[col].isna().sum()) if exists else len(df)
        rules.append({"rule": "not_null", "column": col, "invalid": invalid,
                      "total": len(df), "passed": exists and invalid == 0})

    for col in unique_columns:
        exists = col in df.columns
        invalid = int(df[col].duplicated(keep=False).sum()) if exists else len(df)
        rules.append({"rule": "unique", "column": col, "invalid": invalid,
                      "total": len(df), "passed": exists and invalid == 0})

    for col in df.columns:
        pattern = regex_rules.get(col) or infer_pattern(col)
        if not pattern or not (df[col].dtype == object or pd.api.types.is_string_dtype(df[col])):
            continue
        values = df[col].dropna().astype(str)
        invalid = int((~values.str.match(pattern)).sum())
        rules.append({"rule": "regex", "column": col, "pattern": pattern,
                      "invalid": invalid, "total": len(values), "passed": invalid == 0})

    total_checks = sum(r["total"] for r in rules)
    total_valid = sum(r["total"] - r["invalid"] for r in rules)
    consistency = 100.0 * total_valid / total_checks if total_checks else 100.0
    return {
        "consistency_score": round(consistency, 4),
        "rules_total": len(rules),
        "rules_passed": sum(1 for r in rules if r["passed"]),
        "rules": rules,
    }
