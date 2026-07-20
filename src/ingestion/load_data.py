
import pandas as pd
import os

RAW_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw", "titanic.csv")


def load_raw(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path)
    if ext in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported input format '{ext}'. Use CSV, XLSX, or XLS.")


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def ingest(path: str) -> pd.DataFrame:
    df = load_raw(path)
    df = standardize_columns(df)
    return df


if __name__ == "__main__":
    df = ingest(RAW_PATH)

    print("=== INGESTION REPORT ===")
    print(f"File: {RAW_PATH}")
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Columns: {list(df.columns)}")
    print("\nDtypes:")
    print(df.dtypes)
    print("\nFirst 5 rows:")
    print(df.head())
