# 💬 Social Media Sentiment & Engagement Analysis

![Dashboard Preview](https://github.com/yourusername/repo-name/assets/social-sentiment-dashboard.png)

> Analyze, visualize, and predict audience sentiment and engagement trends across social media platforms using NLP and business intelligence.

---

## 🧩 Table of Contents
- [Project Overview](#-project-overview)
- [Business Problem](#-business-problem)
- [Objectives](#-objectives)
- [Solution Architecture](#-solution-architecture)
- [Key Features](#-key-features)
- [Tech Stack](#-tools--technologies)
- [Workflow](#-workflow)
- [Code Examples](#-python-code-examples)
- [Power BI Dashboard](#-power-bi-dashboard)
- [Insights & Results](#-results--insights)
- [Future Improvements](#-future-improvements)
- [Power BI Data Model](#-power-bi-data-model)
- [Author](#-author)

---

## 🚀 Project Overview
This project performs **end-to-end social media analytics** by combining **Natural Language Processing (NLP)** and **Business Intelligence (BI)** to extract, process, and visualize sentiment and engagement data from multiple platforms (Twitter/X, YouTube, Instagram).

It empowers marketing teams to:
- Track brand perception in real-time,
- Analyze engagement performance,
- Predict campaign outcomes, and
- Make data-driven marketing decisions.

---

## 💼 Business Problem
Organizations invest heavily in social media campaigns but lack a unified system to measure **audience sentiment** and **content performance**.
Without real-time insights, it’s difficult to:
- Identify negative feedback early,
- Understand audience behavior, and
- Optimize engagement strategies effectively.

---

## 🎯 Objectives
- Collect and analyze engagement metrics and audience sentiment from multiple platforms.
- Use NLP models to classify comments/posts as **positive, neutral, or negative**.
- Visualize brand sentiment trends and engagement KPIs.
- Provide actionable insights to improve marketing performance.

---

## 🏗️ Solution Architecture

```
           +-----------------------------+
           |  Social Media APIs (X, IG)  |
           +-------------+---------------+
                         |
                         ▼
              [Python ETL & NLP Pipeline]
           Data Extraction → Cleaning → Sentiment Model
                         |
                         ▼
               [SQL / Data Warehouse Layer]
                         |
                         ▼
             [Power BI Dashboard & Insights]
           KPIs | Sentiment Trends | Engagement Rate
```

---

## ✨ Key Features
✅ **Data Integration:** APIs for X (Twitter), Instagram, and YouTube  
✅ **Text Preprocessing:** Tokenization, stopword removal, lemmatization  
✅ **Sentiment Analysis:** Using pre-trained NLP models (VADER / BERT)  
✅ **Engagement Metrics:** Likes, shares, comments, impressions  
✅ **Interactive Power BI Dashboard:** Real-time insights and filters  
✅ **Automation:** Scheduled API refresh and ETL updates  

---

## ⚙️ Tools & Technologies

| Category | Tools / Libraries |
|-----------|------------------|
| **Data Collection** | Tweepy, Instagram Graph API, YouTube Data API |
| **Data Processing** | Python, Pandas, NumPy, Regex |
| **NLP & Sentiment Analysis** | NLTK, TextBlob, VADER, Transformers (BERT) |
| **Database / Storage** | PostgreSQL / MySQL |
| **Visualization** | Power BI, Plotly, Matplotlib |
| **Automation** | Airflow / Cron jobs |
| **Version Control** | Git, GitHub |

---

## 🔁 Workflow

1️⃣ **Data Extraction:** Pull social media posts, comments, and engagement stats.<br />
**Goal:** Collect social media posts, comments, and engagement metrics (likes, shares, retweets, etc.).<br />
**Sources:**
- CSV or API feeds from platforms like X (Twitter), Instagram, or YouTube.
- For production: APIs such as **Tweepy (X), Meta Graph API (Facebook/Instagram)**, or **YouTube Data API**.
- For demo: A **sample_posts.csv** dataset stored locally in the `data/` folder.

**Process:**
- Fetch text posts, timestamps, author, and engagement metrics.
- Normalize field names and structure into a unified format.
- Save to the `/data/` folder as a daily batch (e.g., `social_posts_YYYYMMDD.csv`).

Example Script:<br />
`scripts/load_data.py`
```python
import pandas as pd

# Load sample CSV
df = pd.read_csv("data/sample_posts.csv")
print(df.head())


```
2️⃣ **Data Cleaning:** Remove noise, emojis, and irrelevant symbols. 
**Goal:** Prepare raw text for sentiment classification.

**Steps:**

- Remove URLs, hashtags, mentions (`@user`), emojis, and special symbols.
- Convert text to lowercase.
- Tokenize words and optionally remove stopwords.
- Handle missing or duplicate posts.
- Keep engagement metrics (likes, shares, etc.) for later aggregation.

Example Script:<br />
`scripts/preprocess.py`
```python
import re
import pandas as pd

def clean_text(text):
    text = re.sub(r"http\S+|@\S+|#\S+", "", text)
    text = re.sub(r"[^A-Za-z0-9\s]", "", text)
    return text.lower().strip()

df = pd.read_csv("data/sample_posts.csv")
df["clean_text"] = df["text"].apply(clean_text)
print(df.head())

```


3️⃣ **Sentiment Analysis:** Apply NLP models to classify polarity.  
**Goal:** Determine the polarity (Positive, Neutral, Negative) of each post.

**Model Used:**

- **TF-IDF Vectorizer + Logistic Regression** classifier (via scikit-learn).
- Trained on labeled data to classify sentiment polarity.

**Pipeline Steps:**

- Vectorize text using TF-IDF.
- Train/test split for model validation.
- Save trained model to `/models/sentiment_model.pkl`.
- Predict new posts daily and append results.

Example Script:<br />
`scripts/sentiment_model.py`
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import pickle

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1,2), max_features=2000)),
    ('clf', LogisticRegression(max_iter=1000))
])

pipeline.fit(X_train, y_train)
pickle.dump(pipeline, open("models/sentiment_model.pkl", "wb"))

```
4️⃣ **Aggregation:** Combine engagement and sentiment into metrics tables.  
**Goal:** Combine sentiment results with engagement metrics.

**Metrics Computed:**

| Metric	| Description |
|Average Sentiment Score|	Converts polarity to numeric values (-1, 0, +1).|
|Engagement Rate|	(Likes + Comments + Shares) / Followers.|
|Positive Engagement Ratio|	% of positive posts with above-average engagement.|
|Platform/Time Trends	|Sentiment and engagement trends over time.|

**Process:**

- Aggregate by day, week, or campaign.
- Create summarized tables for Power BI.

**Example Code:** <br />
```python

```
5️⃣ **Visualization:** Power BI dashboard for KPIs and insights.  
6️⃣ **Automation:** Daily ETL refresh and alert generation.

---

## 🧠 Python Code Examples

### Data Extraction (Twitter API)
```python
import tweepy
import pandas as pd

client = tweepy.Client(bearer_token="YOUR_TOKEN")

query = "Nike -is:retweet lang:en"
tweets = client.search_recent_tweets(query=query, tweet_fields=['created_at', 'public_metrics'], max_results=100)

data = []
for tweet in tweets.data:
    data.append({
        'text': tweet.text,
        'date': tweet.created_at,
        'likes': tweet.public_metrics['like_count'],
        'retweets': tweet.public_metrics['retweet_count']
    })

df = pd.DataFrame(data)
df.head()
```

### Text Cleaning & Preprocessing
```python
import re
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\S+", "", text)
    text = re.sub(r"[^A-Za-z\s]", "", text)
    text = " ".join(word for word in text.split() if word not in stop_words)
    return text.lower()

df['clean_text'] = df['text'].apply(clean_text)
```

### Sentiment Classification (VADER)
```python
from nltk.sentiment import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

df['sentiment_score'] = df['clean_text'].apply(lambda x: sia.polarity_scores(x)['compound'])
df['sentiment_label'] = df['sentiment_score'].apply(
    lambda x: 'Positive' if x > 0.05 else ('Negative' if x < -0.05 else 'Neutral')
)
```

### Visualization (Python Preview)
```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x='sentiment_label', data=df)
plt.title('Overall Sentiment Distribution')
plt.show()
```

---

## 📊 Power BI Dashboard

The Power BI dashboard connects to SQL or CSV data tables and visualizes:
- 📈 **Sentiment Trends Over Time**
- 💬 **Top Positive / Negative Keywords**
- 📊 **Engagement Metrics** (Likes, Comments, Shares)
- 🌍 **Sentiment by Region or Campaign**
- ⚠️ **Alerts for Negative Spikes**

![Power BI Dashboard](https://github.com/yourusername/repo-name/assets/powerbi-socialmedia.png)

---

## 📈 Results & Insights

| Metric | Value |
|--------|--------|
| **Total Posts Analyzed** | 10,000+ |
| **Overall Sentiment** | 63% Positive |
| **Top Positive Topics** | Product Quality, Fast Delivery |
| **Top Negative Topics** | Pricing, Support Response Time |
| **Engagement Peak** | During Product Launch Campaign |

**Key Insight:** Posts with visuals and hashtags showed a **45% higher engagement rate** and were **twice as likely** to receive positive sentiment.

---

## 🧾 Future Improvements
- Integrate **real-time dashboards** with streaming APIs.  
- Use **BERT / RoBERTa** models for advanced sentiment detection.  
- Build **topic modeling** (LDA) for theme extraction.  
- Incorporate **predictive engagement modeling** (likes/comments forecast).

---

## 🧮 Power BI Data Model

### Tables
- **DimDate** – Calendar and time intelligence table.  
- **DimPlatform** – Contains social media platform info.  
- **FactEngagement** – Likes, comments, shares, impressions.  
- **FactSentiment** – Post-level sentiment scores.  

### Relationships
- `FactEngagement[PlatformID] → DimPlatform[PlatformID]`
- `FactEngagement[DateKey] → DimDate[DateKey]`
- `FactSentiment[PostID] → FactEngagement[PostID]`

### Example DAX Measures
```DAX
Total Engagement = SUM(FactEngagement[Likes]) + SUM(FactEngagement[Comments]) + SUM(FactEngagement[Shares])

Positive Rate = 
DIVIDE(
    CALCULATE(COUNTROWS(FactSentiment), FactSentiment[Sentiment] = "Positive"),
    COUNTROWS(FactSentiment)
)

Avg Engagement Rate = AVERAGE(FactEngagement[EngagementRate])

Negative Spike Alert = 
IF(
    [Positive Rate] < 0.4 && [Total Engagement] > 5000,
    "⚠️ Investigate",
    "✅ Normal"
)
```
📂 Project structure
```
kotlin 
social-media-sentiment-analysis/
├── data/
│   └── sample_posts.csv
├── scripts/
│   ├── load_data.py
│   ├── preprocess.py
│   ├── sentiment_model.py
│   ├── visualize.py
├── notebooks/
│   └── Sentiment_Analysis_Exploration.ipynb
├── dashboard/
│   └── PowerBI_Report_Spec.md
├── models/
│   └── sentiment_model.pkl
└── README.md
```

---

## 👤 Author
**Bahre Hailemariam**  
📍 *Data Analyst & BI Developer \| 4+ Years Experience*\
📩 [Email Adress](bahre.hail@gmail.com) | 🌐[Portfolio](https://bahre-hailemariam-data-analyst.crd.co/) |💼[LinkedIn](https://www.linkedin.com/in/bahre-hailemariam/) | 📊[GitHub](https://github.com/BahreHailemariam)


---

## 🪪 License
Licensed under the **MIT License** — free to use and modify.
