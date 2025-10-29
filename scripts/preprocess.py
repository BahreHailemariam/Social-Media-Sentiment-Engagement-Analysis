"""
preprocess.py
Text cleaning and basic feature engineering.
"""
import re
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "sample_posts.csv"

def clean_text(text):
    if pd.isna(text):
        return ""
    text = re.sub(r"http\S+", "", text)      # remove urls
    text = re.sub(r"@\S+", "", text)         # remove mentions
    text = re.sub(r"[^A-Za-z0-9\s]", " ", text)  # keep letters and numbers
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()

def load_and_preprocess():
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df['clean_text'] = df['text'].apply(clean_text)
    # map labels to numeric for modeling
    label_map = {"negative": 0, "neutral": 1, "positive": 2}
    df['label_num'] = df['label'].map(label_map)
    return df

if __name__ == "__main__":
    df = load_and_preprocess()
    print(df[['post_id','text','clean_text','label','label_num']])
