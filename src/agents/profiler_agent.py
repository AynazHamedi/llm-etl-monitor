
from __future__ import annotations

import os
import sys
from typing import Dict, List, Tuple

import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "profiling"))
from profile_data import profile as compute_profile  # noqa: E402


class ProfilerAgent:
    name = "Agent Profiler"

    def run(self, df: pd.DataFrame, dataset_name: str) -> Tuple[Dict, List[Dict]]:
        trace: List[Dict] = []
        trace.append({
            "agent": self.name,
            "thought": f"Need a baseline data-quality report for '{dataset_name}' "
                       f"before any routing decision can be made.",
            "action": "profile_data.profile(df)",
        })

        report = compute_profile(df)

        flagged_cols = [c for c, info in report["columns"].items() if info["missing_count"] > 0]
        trace.append({
            "agent": self.name,
            "observation": (
                f"{report['n_rows']} rows x {report['n_columns']} cols, "
                f"{report['overall_missing_cells']} missing cells "
                f"({report['overall_missing_pct']}%), "
                f"{report['duplicate_rows']} duplicate rows. "
                f"Columns needing attention: {flagged_cols or 'none'}."
            ),
        })

        report["_flagged_columns"] = flagged_cols
        return report, trace
