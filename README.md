# LLM-based Intelligent ETL Monitor — Setup Guide

This repository is the project skeleton for "Design and Implementation of an
LLM-based Intelligent ETL Monitor for Tabular Data Cleaning and Integration".

## Folder structure

```
etl_llm_project/
├── config/                # Settings (config.yaml)
├── data/
│   ├── raw/                # Raw datasets (Titanic, Adult Income, OpenML Dirty, Synthetic)
│   └── processed/          # Cleaned output
├── src/
│   ├── ingestion/           # Layer 6.1 - Data ingestion
│   ├── profiling/            # Layer 6.2 - Data profiling
│   ├── validation/           # Layer 6.3 - Rule-based validation
│   ├── decision_engine/      # Layer 6.4 - Hybrid decision engine
│   ├── llm_layer/            # Layer 6.5 - Local large language model (Ollama)
│   ├── transformation/       # Layer 6.7 - Output merging and final transformation
│   ├── monitoring/           # Layer 6.8 - Monitoring and logging (MLflow)
│   ├── evaluation/           # Layer 6.9 - Data quality evaluation
│   └── agents/               # Profiler / Cleaner / Reviewer agents (ReAct)
├── scripts/
│   ├── setup_ollama.sh / .ps1   # Install Ollama + pull model
│   ├── setup_mlflow.sh / .ps1   # Start MLflow tracking server
│   └── test_infrastructure.py
├── notebooks/                # Exploratory analysis
├── reports/                  # Final evaluation reports
├── tests/
└── requirements.txt
```

## Installation steps

### 1. Create a Python virtual environment

```bash
cd etl_llm_project
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Install and start Ollama

**Windows (PowerShell):**

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\scripts\setup_ollama.ps1
```

**Linux/Mac:**

```bash
bash scripts/setup_ollama.sh
```

This installs Ollama, starts the service (port 11434), and pulls a local model.

> Hardware note: at least 8 GB of free RAM, ideally a GPU with 6+ GB VRAM.
> It also works without a GPU, just slower at inference.
>
> **Known issue (Windows + NVIDIA GPU):** if `ollama run` fails with
> `CUDA error: device kernel image is invalid` / `llama-server process has
terminated`, this is a GPU driver / CUDA build mismatch, not a code issue.
> Fix: fully quit Ollama from the system tray, then **restart the machine**
> (a full restart was required in our case — just restarting the Ollama
> process was not enough). After reboot, `ollama run mistral "hello"` works
> normally.

### 3. Start MLflow (in a separate terminal)

**Windows (PowerShell):**

```powershell
.\scripts\setup_mlflow.ps1
```

**Linux/Mac:**

```bash
bash scripts/setup_mlflow.sh
```

Once running, the UI is available at http://localhost:5000

### 4. Test the infrastructure

```bash
python scripts/test_infrastructure.py
```

If both checks pass, Phase 1 is complete and you're ready for Phase 2 (dataset preparation).

## Progress so far

### Phase 1 - Infrastructure (done)

- Ollama running locally with `mistral:latest` (7.2B params, Q4_K_M
  quantization, `http://localhost:11434`)
- MLflow Tracking Server running (`http://localhost:5000`)
- Verified via `scripts/test_infrastructure.py`

### Phase 2 - Dataset preparation (Titanic done, 3 datasets pending)

- `data/raw/titanic.csv` downloaded (891 rows, 12 columns)
- `src/ingestion/load_data.py` - Layer 6.1, loads + standardizes column names
- `src/profiling/profile_data.py` - Layer 6.2, computes missing values /
  duplicates / per-column stats, saved as baseline to
  `reports/titanic_profile_before.json`
  - Result: 866 missing cells (8.1%) - mostly in `Age` (19.87%),
    `Cabin` (77.1%), `Embarked` (0.22%). No duplicate rows.

### Phase 3 - Core ETL layer implementation (Titanic pipeline done end-to-end)

**Layer 6.3 — Rule-based validation** (`src/validation/rule_validation.py`)

- Originally attempted with `great_expectations`, but the installed version
  (1.18.x) has a completely different API than the tutorials/older docs
  (`context.sources`, `gx.from_pandas`, etc. don't exist anymore). Rather than
  fight GE's breaking API changes, we implemented the same rule set directly
  with pandas — functionally equivalent, more stable, easier to maintain.
- Rules implemented: `PassengerId` uniqueness, `Pclass`/`Sex` not-null,
  `Age` range check (0–120), `Age` missing-value detection.
- Output saved to `reports/titanic_validation_report.json`.
- Result: all structural rules passed; only `Age` missing values failed
  (177 rows).

**Layer 6.4 — Decision Engine** (`src/decision_engine/decision_engine.py`)

- Routes each failed rule to either the `rule` path (deterministic fix) or
  the `llm` path (semantic reasoning needed), based on a routing policy
  table (numeric columns → rule-based imputation, categorical columns →
  LLM-based imputation).
- Output saved to `reports/titanic_decisions.json`.
- Result: `Age` missing values routed to `rule` → `impute_median`.

**Layer 6.7 — Transformation (rule-based actions)**
(`src/transformation/apply_rule_actions.py`)

- Applies the rule-based decisions from the Decision Engine.
- Result: 177 missing `Age` values filled with median (28.0). Output saved to
  `data/processed/titanic_after_rules.csv`. 689 missing cells remain
  (`Cabin` ~687, `Embarked` 2) — these are intentionally left for the LLM
  layer / a later feature-engineering decision (`Cabin`'s 77% missing rate
  makes imputation unreliable; converting it to a binary `Has_Cabin` flag is
  a candidate for Phase 6).

**Layer 6.5 — Local LLM Layer** (`src/llm_layer/semantic_cleaner.py`)

- Connects to Ollama's local REST API (`http://localhost:11434/api/generate`)
  using the `mistral` model.
- Few-shot prompt built for predicting missing `Embarked` values from
  `Pclass`, `Fare`, and `Sex`.
- Applied to the 2 rows with missing `Embarked` (rows 61 and 829) — both
  predicted `S`, with a fallback to the column mode if the model ever
  returns an invalid value.
- Output saved to `data/processed/titanic_after_llm.csv`, decision log saved
  to `reports/titanic_llm_log.json`.

**Phase 4 — Agent Reviewer** (`src/agents/reviewer_agent.py`)

- Validates LLM outputs against ground rules (must be one of `S`/`C`/`Q`,
  response must be short/clean — a proxy for detecting hallucinated output).
- Result: 2/2 LLM decisions accepted. Output saved to
  `reports/titanic_reviewer_report.json`.
- _Profiler agent and full multi-agent ReAct orchestration (LangChain/CrewAI)
  not yet implemented — current pipeline runs these steps as sequential
  scripts rather than autonomous agents._

**Phase 5 — MLflow monitoring** (`src/monitoring/log_run.py`)

- Logs one full pipeline run (`titanic_run_1`) under experiment
  `titanic_etl_pipeline`.
- Metrics logged:

  | Metric                                        | Value   |
  | --------------------------------------------- | ------- |
  | `completeness_before`                         | 91.90%  |
  | `completeness_after`                          | 93.57%  |
  | `dqis_score` (Data Quality Improvement Score) | 1.82%   |
  | `duplicate_rate`                              | 0.00%   |
  | `error_correction_rate`                       | 100.00% |
  | `llm_total_decisions`                         | 2       |
  | `llm_accepted_decisions`                      | 2       |

- Artifacts logged: `titanic_validation_report.json`,
  `titanic_decisions.json`, `titanic_llm_log.json`,
  `titanic_reviewer_report.json`, `titanic_after_llm.csv`.
- Verified visually in the MLflow UI (`http://localhost:5000`).

> Note: `completeness_after` is not 100% because `Cabin` is still mostly
> missing — this is expected and by design, not a bug (see note above).

**Layer 8.3 — Data Drift Detection** (`src/monitoring/drift_detection.py`)

- Detects distribution drift between a **reference** batch and a **current**
  batch at the ETL layer, before data reaches any downstream model (per
  Section 8.3 of the proposal). Dataset-agnostic — works on any two DataFrames
  with the same schema, so it is reused across all datasets.
- Numeric columns: **Kolmogorov–Smirnov** test (+ **KL divergence** on
  histograms). A column is flagged when the KS p-value < 0.05.
- Categorical columns: **Population Stability Index (PSI)** + **Jensen–Shannon
  divergence** on category frequencies. Flagged when PSI ≥ 0.2.
- Logs drift metrics/artifacts to MLflow (experiment `etl_drift_detection`),
  degrading gracefully when no tracking server is running.
- Includes a `--demo` mode that simulates a drifted batch from a single
  dataset (the "Simulation Test" evaluation method from Section 10), so drift
  can be demonstrated without a second real batch.

```bash
# Compare two real batches:
python -m src.monitoring.drift_detection \
    --reference data/raw/titanic.csv --current data/processed/titanic_after_llm.csv

# Self-contained demo (simulates drift on one dataset):
python -m src.monitoring.drift_detection --demo data/raw/titanic.csv
```

- Verified: identical batches → 0 columns flagged (no false positives);
  simulated drift → numeric and categorical columns correctly flagged.

### Phase 2/3 — Second dataset: Adult Income (generalization)

To prove the pipeline is not Titanic-specific, a **config-driven, generic
runner** was added: `src/pipelines/run_pipeline.py`. It reads a dataset entry
from `config/config.yaml` and runs the whole flow (ingest → profile → rule-based
clean → metrics → drift detection → MLflow) for **any** dataset, so the same
code now serves Titanic and Adult Income.

```bash
python -m src.pipelines.run_pipeline --dataset adult_income
# add --no-mlflow to run without a tracking server
```

**Adult Income result** (UCI, 32,561 rows × 15 cols — main errors per the
proposal: inconsistency, noise, duplicates):

| Metric                 | Before   | After   |
| ---------------------- | -------- | ------- |
| Rows                   | 32,561   | 32,537  |
| Completeness           | 99.13%   | 100.0%  |
| Duplicate rate         | 0.074%   | 0.0%    |
| DQIS score             |          | +0.88%  |

- **24 exact duplicate rows removed** (matches the well-documented Adult
  duplicate count).
- Rule-based layer handled all three of Adult's error types deterministically:
  whitespace **noise** (`" Bachelors"` → `"Bachelors"`), **inconsistency** via
  `?` → NaN normalization, and **duplicates**.
- `?` missing markers (1,836 in `workclass`, 1,843 in `occupation`, etc.)
  imputed → 0 missing cells.
- Drift across a 50/50 batch split: 0/15 columns flagged (expected on an IID
  split — confirms no false positives on real data).

### Next steps (Phase 2 continued / Phase 6)

1. ~~Repeat the full pipeline for **Adult Income**~~ — **done** via the generic
   `src/pipelines/run_pipeline.py` (see above). Remaining datasets:
   **OpenML Dirty Tabular** and the **synthetic LLMClean-style dataset** — both
   run with the same `--dataset` runner once their CSVs are placed in
   `data/raw/` and given a config entry.
2. Decide on and implement a strategy for the `Cabin` column (impute vs.
   binary `Has_Cabin` feature).
3. ~~Implement **Drift Detection** (Layer 8.3 of the proposal)~~ — **done**,
   see `src/monitoring/drift_detection.py` above. Remaining: wire it into the
   per-batch pipeline runs for the other datasets.
4. Build the full multi-agent ReAct orchestration (Profiler/Cleaner/Reviewer
   as actual agents via LangChain or CrewAI) instead of sequential scripts.
5. Produce the final evaluation report comparing Completeness, Consistency,
   Duplicate Rate, and DQIS across all 4 datasets, per Section 10 of the
   proposal.

## Phase checklist

- [x] Phase 1: Infrastructure setup
- [~] Phase 2: Dataset preparation (Titanic + Adult Income done, 2 datasets
  pending: OpenML Dirty, synthetic LLMClean)
- [~] Phase 3: Core ETL layer implementation (Titanic per-dataset scripts +
  generic config-driven runner working for Titanic and Adult Income)
- [~] Phase 4: Multi-agent architecture (Cleaner + Reviewer logic implemented
  as scripts; full ReAct/LangChain orchestration pending)
- [~] Phase 5: Monitoring and evaluation (MLflow logging working end-to-end
  for Titanic; drift detection implemented — `src/monitoring/drift_detection.py`)
- [ ] Phase 6: Final evaluation and report
