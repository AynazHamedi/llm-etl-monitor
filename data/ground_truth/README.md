# Manual ground truth format

Create one CSV per dataset with these columns:

```csv
row_index,column,expected_value
61,embarked,S
829,embarked,S
```

Then add `ground_truth_path` to that dataset in `config/config.yaml`. The
orchestrator will compute `error_correction_rate_verified`. Without this file,
the verified metric intentionally remains `null`/`n/a`.
