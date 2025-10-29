"""
sentiment_model.py
Train a Logistic Regression classifier using TF-IDF features and save the trained model.
"""
import pickle
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "sample_posts.csv"
MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "sentiment_model.pkl"

def train_and_save_model():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=['text','label'])
    X = df['text']
    y = df['label']
    # convert labels to numeric if needed
    label_mapping = {'negative':0, 'neutral':1, 'positive':2}
    y_num = y.map(label_mapping)
    X_train, X_test, y_train, y_test = train_test_split(X, y_num, test_size=0.2, random_state=42, stratify=y_num)
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1,2), max_features=2000)),
        ('clf', LogisticRegression(max_iter=1000))
    ])
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds, target_names=['negative','neutral','positive']))
    # save model
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump({'model': pipeline, 'label_mapping': label_mapping}, f)
    print(f"Saved model to {MODEL_PATH}")

if __name__ == "__main__":
    train_and_save_model()
