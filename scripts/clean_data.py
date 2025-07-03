import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Load environment vars
load_dotenv()

RAW_DIR   = "data_ingestion/raw"
CLEAN_DIR = "data/clean"

def clean_file(filename):
    in_path  = os.path.join(RAW_DIR, filename)
    out_name = filename.replace(".csv", "_clean.csv")
    out_path = os.path.join(CLEAN_DIR, out_name)

    # 1. Load
df = pd.read_csv(in_path)

    # 2. Inspect (print stats)
    print(f"--- {filename} info() ---")
    print(df.info())
    print(df.isna().sum())

    # 3. Cleaning steps
    df = df.drop_duplicates()
    for col in df.select_dtypes(include=[np.number]):
        df[col] = df[col].fillna(df[col].median())
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # 4. Save
    print(f"Saving cleaned to {out_path}")
    df.to_csv(out_path, index=False)


def main():
    os.makedirs(CLEAN_DIR, exist_ok=True)
    for file in os.listdir(RAW_DIR):
        if file.lower().endswith(".csv"):
            clean_file(file)

if __name__ == "__main__":
    main()