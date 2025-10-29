"""
visualize.py
Simple visualizations for sentiment distribution and engagement.
"""
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "sample_posts.csv"

def plot_sentiment_distribution():
    df = pd.read_csv(DATA_PATH)
    counts = df['label'].value_counts().reindex(['positive','neutral','negative']).fillna(0)
    counts.plot(kind='bar')
    plt.title("Sentiment Distribution")
    plt.ylabel("Number of posts")
    plt.xlabel("Sentiment")
    plt.tight_layout()
    plt.show()

def plot_engagement_by_sentiment():
    df = pd.read_csv(DATA_PATH)
    df['engagement'] = df['likes'] + df['comments'] + df['shares']
    agg = df.groupby('label')['engagement'].mean().reindex(['positive','neutral','negative']).fillna(0)
    agg.plot(kind='bar')
    plt.title("Average Engagement by Sentiment")
    plt.ylabel("Average Engagement (likes+comments+shares)")
    plt.xlabel("Sentiment")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_sentiment_distribution()
    plot_engagement_by_sentiment()
