from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd


class DecisionAgent:
    name = "Engine Decision"

    def __init__(self, high_cardinality_ratio: float = 0.5,
                 high_missing_ratio: float = 0.5):
        self.high_cardinality_ratio = high_cardinality_ratio
        self.high_missing_ratio = high_missing_ratio

    def run(self, df: pd.DataFrame, profiler_report: Dict) -> Tuple[List[Dict], List[Dict]]:
        trace: List[Dict] = []
        decisions: List[Dict] = []

        flagged = profiler_report.get("_flagged_columns", [])
        trace.append({
            "agent": self.name,
            "thought": f"Route each flagged column to rule-based or semantic (LLM) "
                       f"handling based on dtype and cardinality. Flagged: {flagged or 'none'}.",
        })

        n_rows = len(df) or 1
        for col in flagged:
            missing = int(df[col].isna().sum())
            missing_ratio = missing / n_rows
            if pd.api.types.is_numeric_dtype(df[col]):
                route, action = "rule", "impute_median"
                reason = "numeric column -> deterministic median imputation"
            else:
                cardinality_ratio = df[col].nunique(dropna=True) / n_rows
                if missing_ratio >= self.high_missing_ratio:
                    route, action = "rule", "missing_indicator"
                    reason = (f"categorical, high missing ratio "
                              f"({missing_ratio:.3f} >= {self.high_missing_ratio}) "
                              f"-> preserve missingness as UNKNOWN + indicator")
                elif cardinality_ratio <= self.high_cardinality_ratio:
                    route, action = "llm", "semantic_impute"
                    reason = (f"categorical, low cardinality ratio "
                              f"({cardinality_ratio:.3f} <= {self.high_cardinality_ratio}) "
                              f"-> needs semantic reasoning")
                else:
                    route, action = "rule", "mode_or_manual_review"
                    reason = (f"categorical, high cardinality ratio "
                              f"({cardinality_ratio:.3f} > {self.high_cardinality_ratio}) "
                              f"-> identifier-like, not a good LLM target")

            decisions.append({
                "column": col,
                "missing_count": missing,
                "missing_ratio": missing_ratio,
                "route": route,
                "action": action,
                "reason": reason,
            })
            trace.append({
                "agent": self.name,
                "action": f"route('{col}') -> {route}:{action}",
                "observation": reason,
            })

        return decisions, trace
