import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.pipelines.run_pipeline import ingest, quality_metrics, rule_based_clean
from src.profiling.profile_data import profile
from src.transformation.schema_matching import apply_schema_mapping
from src.validation.generic_validation import validate_dataframe
from src.evaluation.ground_truth import verified_correction_rate
from src.agents.decision_agent import DecisionAgent


class CoreRequirementsTests(unittest.TestCase):
    def test_csv_ingestion_and_cleaning(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.csv"
            path.write_text("id,name,age\n1, Alice ,10\n1, Alice ,10\n2,?,\n", encoding="utf-8")
            frame = ingest(str(path), ["?"])
            cleaned, log = rule_based_clean(frame)
            self.assertEqual(log["duplicates_removed"], 1)
            self.assertFalse(cleaned.isna().any().any())

    def test_profile_statistics_outliers_and_fuzzy_candidates(self):
        frame = pd.DataFrame({"name": ["John Smith", " john  smith "], "x": [1, 1]})
        report = profile(frame)
        self.assertIn("std", report["columns"]["x"])
        self.assertIn("outlier_count_iqr", report["columns"]["x"])
        self.assertEqual(report["fuzzy_duplicate_candidate_rows"], 2)

    def test_generic_validation_and_consistency(self):
        frame = pd.DataFrame({"id": [1, 1], "email": ["valid@example.com", "bad"]})
        report = validate_dataframe(frame, required_columns=["id"], unique_columns=["id"])
        self.assertLess(report["consistency_score"], 100)
        self.assertEqual(report["rules_total"], 3)

    def test_schema_matching(self):
        frame = pd.DataFrame({"E-mail Address": ["a@b.com"], "cust id": [1]})
        mapped, mapping = apply_schema_mapping(
            frame, {"email": ["email_address", "e_mail_address"], "customer_id": ["cust_id"]}
        )
        self.assertIn("email", mapped.columns)
        self.assertIn("customer_id", mapped.columns)
        self.assertEqual(len(mapping), 2)

    def test_quality_metrics_include_consistency(self):
        frame = pd.DataFrame({"id": [1], "x": [None]})
        cleaned, log = rule_based_clean(frame)
        before = {"consistency_score": 50.0}
        after = {"consistency_score": 100.0}
        metrics = quality_metrics(frame, cleaned, log["duplicates_removed"], before, after)
        self.assertEqual(metrics["consistency_after"], 100.0)

    def test_verified_correction_requires_ground_truth(self):
        self.assertIsNone(verified_correction_rate([], None))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "truth.csv"
            path.write_text("row_index,column,expected_value\n1,status,OK\n", encoding="utf-8")
            reviews = [{"row_index": 1, "column": "status", "suggested_value": "OK", "accepted": True}]
            self.assertEqual(verified_correction_rate(reviews, str(path)), 100.0)

    def test_sparse_categorical_column_is_not_sent_to_llm(self):
        frame = pd.DataFrame({"cabin": [None, None, None, "C85"]})
        report = {"_flagged_columns": ["cabin"]}
        decisions, _ = DecisionAgent(high_missing_ratio=0.5).run(frame, report)
        self.assertEqual(decisions[0]["route"], "rule")
        self.assertEqual(decisions[0]["action"], "missing_indicator")


if __name__ == "__main__":
    unittest.main()
