"""Run the complete reproducible evaluation for every configured dataset."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock-llm", action="store_true", help="CI/demo only; not valid for final accuracy claims")
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()
    cfg = yaml.safe_load((ROOT / "config/config.yaml").read_text(encoding="utf-8"))
    missing = [name for name, item in cfg["datasets"].items() if not (ROOT / item["path"]).exists()]
    if missing:
        raise SystemExit(f"Missing datasets: {missing}. Put the declared files under data/raw before evaluation.")

    for dataset in cfg["datasets"]:
        common = ["--dataset", dataset]
        if args.no_mlflow:
            common.append("--no-mlflow")
        subprocess.run([sys.executable, "-m", "src.pipelines.run_pipeline", *common], cwd=ROOT, check=True)
        agent = [sys.executable, "-m", "src.agents.orchestrator", *common]
        if args.mock_llm:
            agent.append("--mock-llm")
        subprocess.run(agent, cwd=ROOT, check=True)

    subprocess.run([sys.executable, "scripts/generate_final_report.py"], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
