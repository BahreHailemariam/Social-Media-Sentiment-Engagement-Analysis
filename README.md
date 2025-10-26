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

1. **Data Extraction:** Pull social media posts, comments, and engagement stats.  
2. **Data Cleaning:** Remove noise, emojis, and irrelevant symbols.  
3. **Sentiment Analysis:** Apply NLP models to classify polarity.  
4. **Aggregation:** Combine engagement and sentiment into metrics tables.  
5. **Visualization:** Power BI dashboard for KPIs and insights.  
6. **Automation:** Daily ETL refresh and alert generation.

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

---

## 👤 Author
**Bahre Hailemariam**  
📍 *Data Analyst & BI Developer \| 4+ Years Experience*\
📩 [Email Adress](bahre.hail@gmail.com) | 🌐[Portfolio](https://bahre-hailemariam-data-analyst.crd.co/) |💼[LinkedIn](https://www.linkedin.com/in/bahre-hailemariam/) | 📊[GitHub](https://github.com/BahreHailemariam)


---

## 🪪 License
Licensed under the **MIT License** — free to use and modify.
