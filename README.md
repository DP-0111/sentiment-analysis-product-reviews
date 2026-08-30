Sentiment Analysis on Product Reviews
An end-to-end NLP pipeline evaluating 500k+ Amazon product reviews to compare a rule-based lexicon baseline (VADER) against a supervised machine learning model (TF-IDF + Logistic Regression). The project includes data deduplication, class imbalance handling, word cloud visualizations, temporal sentiment trend analysis, and model inference on custom edge cases.

📌 Key Features & Highlights
Data Cleaning & Deduplication: Identified and removed over 174,000 duplicate reviews across product variants using [UserId, Time, Text]. Filtered out invalid helpfulness ratios.

Lexicon Baseline (VADER): Evaluated rule-based polarity scoring using NLTK's VADER sentiment analyzer.

Supervised Machine Learning: Built a TF-IDF vectorizer paired with Logistic Regression, leveraging class_weight='balanced' to tackle class imbalance across Positive (78%), Negative (14.4%), and Neutral (7.6%) ratings.

Performance Boost: Improved Negative Recall from 40% (VADER) to 70% (Logistic Regression), ensuring critical customer complaints are captured effectively.

Visualizations & Trends: Rendered class distributions, confusion matrices, side-by-side word clouds, and historical VADER sentiment trends over time.

📊 Model Comparison & Metrics
Metric / Feature	Lexicon Baseline (VADER)	Supervised ML (TF-IDF + LogReg)
Accuracy	79.47%	77.68%
Negative F1-Score	0.46 (Recall: 40%)	0.64 (Recall: 70%)
Neutral F1-Score	0.06 (Fails to detect)	0.31 (Substantial improvement)
Positive Precision	0.84	0.95
Training Required	None (Pre-trained)	Supervised Training Required
Key Takeaway: While VADER yields a slightly higher overall accuracy due to the heavy positive class bias, Logistic Regression with balanced class weights performs significantly better at capturing minority classes (Negative & Neutral reviews), making it far more suitable for production e-commerce applications.

📁 Repository Structure
Plaintext
├── SENTIMENT-ANALYSIS(PRODUCT-REVIEWS).ipynb   # Main Jupyter Notebook
├── graphs/                                      # Exported visualization plots
│   ├── sentiment_distribution.png
│   ├── confusion_matrix.png
│   ├── wordclouds.png
│   └── sentiment_trend.png
└── README.md                                    # Project documentation
🚀 How to Run
Clone the repository:

Bash
git clone https://github.com/DP-0111/sentiment-analysis-product-reviews.git
cd sentiment-analysis-product-reviews
Install required dependencies:

Bash
pip install pandas numpy matplotlib seaborn scikit-learn nltk wordcloud
Open and execute the notebook:

Bash
jupyter notebook
💡 Business Recommendations
Prioritize Recall for Customer Support: Use the Logistic Regression model to route negative reviews automatically to support agents, reducing churn by catching 70% of dissatisfied customers.

Confidence Thresholding: Apply predict_proba thresholds before auto-flagging reviews to minimize false positives while maintaining high positive prediction precision (95%).
