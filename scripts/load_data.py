"""
load_data.py
Simple utility to load the sample CSV dataset.
"""
import pandas as pd
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "sample_posts.csv"

def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    return df

if __name__ == "__main__":
    df = load_data()
    print(f"Loaded {len(df)} rows")
    print(df.head())
