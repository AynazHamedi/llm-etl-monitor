
from __future__ import annotations

import subprocess
import sys
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"
FILES = {
    "titanic.csv": "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv",
    "adult_income.csv": "https://raw.githubusercontent.com/jbrownlee/Datasets/master/adult-all.csv",
}
ADULT_COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education_num", "marital_status",
    "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss",
    "hours_per_week", "native_country", "income",
]


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        target = RAW / name
        if target.exists():
            print(f"exists: {target}")
            continue
        print(f"downloading: {name}")
        if name == "adult_income.csv":
            body = urllib.request.urlopen(url).read().decode("utf-8")
            target.write_text(",".join(ADULT_COLUMNS) + "\n" + body, encoding="utf-8")
        else:
            urllib.request.urlretrieve(url, target)
    subprocess.run([sys.executable, "scripts/generate_synthetic_datasets.py"], cwd=ROOT, check=True)
    print("Dataset preparation complete.")


if __name__ == "__main__":
    main()
