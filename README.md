# LLM-based Intelligent ETL Monitor

This project implements a lightweight system for monitoring and cleaning tabular data in an ETL pipeline. Simple data-quality problems are handled with rule-based methods, while cases that require semantic reasoning are routed to a local language model.

Ollama is used to run the language model locally, and MLflow is used to record pipeline runs, metrics, and output artifacts. The system does not depend on commercial cloud APIs.

## Main Workflow

The pipeline follows these steps:

1. Load a CSV or Excel file.
2. Generate an initial data-quality profile.
3. Run rule-based validation checks.
4. Select either the rule-based or LLM processing path.
5. Apply data-cleaning operations.
6. Review LLM suggestions before applying them.
7. Calculate data-quality metrics.
8. Check the input data for drift.
9. Save reports and log the run in MLflow.

The agent-based part of the project contains the following components:

- `ProfilerAgent`: analyzes the input data and creates a quality report.
- `DecisionAgent`: selects the rule-based or LLM route for each detected problem.
- `CleanerAgent`: uses the local model to suggest semantic corrections.
- `ReviewerAgent`: checks the confidence and validity of each suggestion before applying it.

## Project Structure

```text
config/                 Project and dataset settings
data/raw/               Raw datasets
data/processed/         Cleaned output datasets
data/ground_truth/      Manual annotations for accuracy evaluation
reports/                Pipeline reports and evaluation results
scripts/                Setup, preparation, and execution scripts
src/agents/             Agents and the main orchestrator
src/ingestion/          CSV and Excel ingestion
src/profiling/          Data-quality profiling
src/validation/         Rule-based validation
src/transformation/     Cleaning and schema matching
src/monitoring/         MLflow logging and drift detection
src/evaluation/         Ground-truth evaluation
tests/                  Automated tests
```

## Installation

Create a virtual environment:

```bash
python -m venv venv
```

On Windows:

```powershell
.\venv\Scripts\activate
pip install -r requirements.txt
```

On Linux or macOS:

```bash
source venv/bin/activate
pip install -r requirements.txt
```

## Preparing the Datasets

The following command downloads the public Titanic and Adult Income datasets and generates the two controlled synthetic datasets:

```bash
python scripts/prepare_public_datasets.py
```

If the files are downloaded manually, place them in `data/raw` using the paths and filenames defined in `config/config.yaml`.

The project is configured for four datasets:

- Titanic
- Adult Income
- OpenML Dirty
- Synthetic LLMClean

## Setting Up Ollama

On Windows:

```powershell
.\scripts\setup_ollama.ps1
```

On Linux or macOS:

```bash
bash scripts/setup_ollama.sh
```

The default model is defined in `config/config.yaml`. The infrastructure can be checked with:

```bash
python scripts/test_infrastructure.py
```

## Setting Up MLflow

On Windows:

```powershell
.\scripts\setup_mlflow.ps1
```

On Linux or macOS:

```bash
bash scripts/setup_mlflow.sh
```

After the server starts, the MLflow interface is available at:

```text
http://localhost:5000
```

## Running the Rule-based Pipeline

Example for the Titanic dataset:

```bash
python -m src.pipelines.run_pipeline --dataset titanic
```

To run without MLflow:

```bash
python -m src.pipelines.run_pipeline --dataset titanic --no-mlflow
```

## Running the Agent-based Pipeline

Run the pipeline with the local Ollama model:

```bash
python -m src.agents.orchestrator --dataset titanic
```

The mock backend can be used to test the software flow without running a language model:

```bash
python -m src.agents.orchestrator --dataset titanic --mock-llm --no-mlflow
```

Mock runs only test the connection between pipeline components. They should not be used to report the actual accuracy or speed of the language model.

## Running the Complete Evaluation

After preparing the datasets and starting Ollama and MLflow, run:

```bash
python scripts/run_complete_evaluation.py
```

For an offline test run:

```bash
python scripts/run_complete_evaluation.py --mock-llm --no-mlflow
```

## Output Reports

The generated reports are saved in `reports/`. They include:

- Data profiles before and after cleaning
- Rule-validation results
- Rule-based and LLM routing decisions
- Cleaner suggestions and Reviewer decisions
- Agent execution traces
- Data-quality metrics
- Data-drift reports
- Final reports in JSON, Markdown, and HTML formats

The final report can be regenerated with:

```bash
python scripts/generate_final_report.py
```

## Evaluation Metrics

The main evaluation metrics are:

- Completeness
- Consistency Score
- Duplicate Rate
- Data Quality Improvement Score
- LLM latency per row
- Error Correction Rate
- Log Coverage
- Drift Detection Delay

A manually annotated ground-truth file is required to calculate a verified Error Correction Rate. The required CSV format is described in `data/ground_truth/README.md`. Until ground truth is provided, verified correction accuracy remains `n/a` in the final report.

## Running the Tests

```bash
python -m unittest discover -s tests -v
```

The tests cover data ingestion, rule-based cleaning, profiling, validation, schema matching, and ground-truth evaluation.

## Current Status

The main ETL pipeline, agent flow, drift detection, reporting, and MLflow integration have been implemented. The historical reports included in the repository were generated with the mock backend and are kept only as examples of the software flow.
