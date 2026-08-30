# Sentiment Analysis on Amazon Product Reviews
### Multi-Class NLP Sentiment Classification using VADER and TF-IDF + Logistic Regression

---

## Problem Statement
E-commerce platforms and brand monitoring tools 
need to automatically detect customer sentiment 
across millions of reviews to identify dissatisfied 
customers before issues escalate. Manual review 
analysis is not scalable at 500,000+ records.

---

## Project Overview
This project performs multi-class sentiment 
classification (Positive / Neutral / Negative) on 
568,454 Amazon Fine Food reviews using two 
complementary NLP approaches:

1. **VADER** — Rule-based lexicon sentiment scorer
2. **TF-IDF + Logistic Regression** — Supervised ML classifier

Both approaches are compared on accuracy, F1-score, 
and real-world suitability.

---

## Dataset
- **Source:** Amazon Fine Food Reviews (Kaggle)
- **Size:** 568,454 reviews | 10 features
- **Target:** Star Rating (1–5) mapped to sentiment
- **Text Features:** Review Summary + Full Review Text
- **Working Sample:** 50,000 reviews (random_state=42)

---

## Methodology

### Step 1 — Data Cleaning & EDA
- Removed 16 missing ProfileName and 27 Summary rows
- Identified and removed duplicate reviews 
  (same user, timestamp, text)
- Filtered invalid helpfulness ratios
- Converted Unix timestamps to datetime for 
  time series analysis

### Step 2 — Sentiment Labelling
- Mapped 1–2 stars → Negative
- Mapped 3 stars → Neutral  
- Mapped 4–5 stars → Positive
- Class distribution: 78% Positive | 14.4% Negative 
  | 7.6% Neutral

### Step 3 — VADER (Lexicon Baseline)
- Applied VADER SentimentIntensityAnalyzer 
  on full review text
- Compound score thresholds: 
  ≥ 0.05 = Positive | ≤ -0.05 = Negative

### Step 4 — TF-IDF + Logistic Regression (ML Model)
- TF-IDF vectorisation: max_features=10,000, 
  ngram_range=(1,2), stop_words='english'
- Stratified 80/20 train-test split
- class_weight='balanced' to handle class imbalance
- Evaluated using classification report + 
  confusion matrix

### Step 5 — Word Cloud Visualisation
- Separate word clouds for Positive and Negative 
  review text
- Top 100 most frequent words per sentiment class

### Step 6 — Sentiment Trend Over Time
- Monthly average VADER compound score plotted 
  from 2000–2012
- Identifies sentiment stability and volatility 
  periods

---

## Results

| Metric | VADER (Lexicon) | TF-IDF + LogReg (ML) |
|--------|----------------|----------------------|
| Overall Accuracy | 79.5% | 77.9% |
| Negative F1-Score | 0.46 | 0.64 |
| Negative Recall | 40% | 70% |
| Neutral F1-Score | 0.06 | 0.32 |
| Training Required | None | Yes |

---

## Key Finding
**TF-IDF + Logistic Regression is the better 
production choice** despite lower overall accuracy.

Why: It detects **70% of negative reviews** vs 
VADER's 40%. For brand monitoring, missing a 
negative review is far more costly than 
misclassifying a neutral one.

---

## Confusion Matrix Insights
- 1,005 out of 1,436 negative reviews correctly 
  identified (70% recall)
- Neutral class remains hardest to classify due 
  to overlapping language patterns with both 
  Positive and Negative
- Positive class shows highest precision — 
  model rarely misclassifies positives

---

## Business Recommendations
1. **Deploy TF-IDF + LogReg for production** — 
   better at catching critical negative feedback
2. **Use VADER for real-time streaming** where 
   training data is unavailable
3. **Prioritise neutral class improvement** via 
   SMOTE or additional training data collection
4. **Monthly sentiment monitoring** can detect 
   product quality degradation before it impacts 
   sales volume

---

## Real-World Applications
- Brand monitoring (Brandwatch, Sprinklr)
- E-commerce review filtering (Amazon, Flipkart)
- Customer satisfaction tracking (HUL, Tata Consumer)
- Product launch feedback analysis

---

## Tech Stack
| Category | Tools |
|----------|-------|
| Language | Python 3 |
| NLP | NLTK, VADER, TF-IDF |
| ML | Scikit-learn, Logistic Regression |
| Data | Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn, WordCloud |
| Environment | Jupyter Notebook |

---

## Project Structure
Sentiment-Analysis-Product-Reviews/
├── SENTIMENT-ANALYSIS(PRODUCT-REVIEWS).ipynb
├── graphs/
│ ├── sentiment_distribution.png
│ ├── confusion_matrix.png
│ ├── wordcloud_positive.png
│ ├── wordcloud_negative.png
│ └── sentiment_trend.png
└── README.md

---

## How to Run
```bash
# Clone repository
git clone https://github.com/DP-0111/Sentiment-Analysis-Product-Reviews.git

# Install dependencies
pip install pandas numpy matplotlib seaborn 
pip install scikit-learn nltk wordcloud

# Download NLTK data
import nltk
nltk.download('vader_lexicon')

# Download dataset from Kaggle
# Search: "Amazon Fine Food Reviews"
# Place Reviews.csv in project folder

# Open notebook
jupyter notebook SENTIMENT-ANALYSIS_PRODUCT-REVIEWS_.ipynb
```

---

## Dataset Source
[Amazon Fine Food Reviews — Kaggle](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)

---

## Author
**Drashti Patil**  
Aspiring Data Analyst | M.Sc Physics  
[GitHub](https://github.com/DP-0111) | 
[LinkedIn](https://linkedin.com/in/drashti-patil-8ba4601aa)
