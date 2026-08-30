# Product Review Sentiment Analysis & Classification Pipeline

An end-to-end NLP pipeline evaluating customer sentiment on product reviews. This project compares rule-based lexicon models (VADER) against supervised machine learning algorithms (TF-IDF + Logistic Regression) to automate brand monitoring and flag negative feedback.

---

## 📌 Executive Summary

Understanding customer sentiment allows e-commerce platforms and brand managers to automatically detect issue trends, escalate critical negative reviews, and aggregate overall customer satisfaction. 

This repository demonstrates data cleaning, text preprocessing, handling severe class imbalance, and model comparison to establish a reliable baseline and production-ready machine learning pipeline.

---

## 📊 Key Results

| Model | Evaluation Metric | Precision | Recall | Macro F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **VADER (Rule-Based)** | Unsupervised Baseline | 0.68 | 0.54 | 0.58 |
| **TF-IDF + Logistic Regression** | Weighted Loss | 0.84 | 0.81 | **0.82** |

*Note: Models were evaluated primarily using **Macro F1-Score** due to severe dataset class imbalance (~78% positive, ~14% negative, ~8% neutral).*

---

## 📁 Repository Structure

```text
├── data/                  # Dataset directory (see Data Setup)
├── notebooks/             # Exploratory Data Analysis & Model Prototyping
│   └── Sentiment_Analysis_Pipeline.ipynb
├── src/                   # Modular Python Scripts
│   ├── preprocess.py      # Text cleaning & data validation
│   ├── train.py           # Model training and hyperparameter optimization
│   └── evaluate.py        # Metrics calculation & confusion matrix generation
├── .gitignore             # Standard gitignore for data & Python artifacts
├── README.md              # Project documentation
└── requirements.txt       # Python dependencies

## 🛠️ Installation & Setup
1. Clone the Repository
Bash
git clone [https://github.com/your-username/product-review-sentiment-analysis.git](https://github.com/your-username/product-review-sentiment-analysis.git)
cd product-review-sentiment-analysis
2. Set Up Virtual Environment
Bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
3. Download the Dataset
This project uses the Amazon Fine Food Reviews dataset from Kaggle.

Download Reviews.csv from Kaggle Amazon Fine Food Reviews Dataset.

Place Reviews.csv inside the data/ directory.

## 🚀 Usage
Run the Jupyter Notebook
To view the EDA, data cleaning steps, baseline comparisons, and model evaluations interactively:

Bash
jupyter notebook notebooks/Sentiment_Analysis_Pipeline.ipynb

## 🧪 Pipeline & Methodology
Data Cleaning & Deduplication:

Removed duplicate review entries sharing identical UserId, Time, and Text.

Filtered invalid rows where HelpfulnessNumerator > HelpfulnessDenominator.

Preprocessing:

Standardized text cleaning (lowercasing, HTML tag removal, punctuation removal).

Strict train-test splitting prior to vectorization to prevent data leakage.

Feature Engineering:

TF-IDF (Term Frequency-Inverse Document Frequency) vectorization using uni-grams and bi-grams (ngram_range=(1,2)).

Class Imbalance Handling:

Applied cost-sensitive learning (class_weight='balanced') to penalize misclassifications on minority (Negative/Neutral) classes.
