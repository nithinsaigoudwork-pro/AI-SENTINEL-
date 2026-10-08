# 🛡️ Spam Email & SMS Classifier

An end-to-end Machine Learning college minor project that classifies text messages and emails as **Spam** or **Not Spam (Ham)**, featuring an interactive, futuristic Streamlit web interface.

Built using **scikit-learn**'s `MultinomialNB` with **TF-IDF Vectorization** wrapped in a unified pipeline, with zero deep-learning bloat or external API dependencies.

---

## 📌 Table of Contents
1. [Project Overview](#-project-overview)
2. [How It Works (Pipeline Architecture)](#-how-it-works-pipeline-architecture)
3. [Project Structure](#-project-structure)
4. [Dataset Details](#-dataset-details)
5. [Model Evaluation & Results](#-model-evaluation--results)
6. [Setup and Installation](#-setup-and-installation)
7. [Running the Project](#-running-the-project)
8. [Viva Q&A / Plain-English Explanations](#-viva-qa--plain-english-explanations)
9. [Limitations & Future Improvements](#-limitations--future-improvements)

---

## 🚀 Project Overview

Unsolicited spam messages and phishing emails pose privacy and security risks. This project demonstrates a production-style, lightweight machine learning solution:
- **Clean Text Processing**: Normalizes raw inputs by removing URLs, symbols, and irregular spacing without needing large NLP libraries like NLTK or spaCy.
- **TF-IDF Feature Extraction**: Converts words and word pairs (unigrams & bigrams) into statistical relevance scores.
- **Multinomial Naive Bayes Classifier**: Calculates probability of a message being spam based on Bayes' Theorem.
- **Evaluation Benchmark**: Evaluates accuracy, precision, recall, F1, 5-fold cross-validation, and benchmarks against Logistic Regression and LinearSVC.
- **Interactive Cyber Web UI**: A modern Streamlit application with live text metrics, preset test messages, and real-time probability visualization.

---

## 🧠 How It Works (Pipeline Architecture)

The system follows a strict 4-step pipeline designed to prevent **data leakage** (the test set is never seen by the vectorizer or model during training):

```
Raw Message ──► clean_text() ──► TfidfVectorizer ──► MultinomialNB ──► Prediction & Probability
```

1. **Text Normalization (`clean_text`)**:
   - Converts text to lowercase.
   - Strips URLs (`http://`, `https://`, `www.`).
   - Removes numbers and punctuation symbols (keeps only alphabetic characters `a-z` and spaces).
   - Collapses redundant whitespace.

2. **Feature Extraction (`TfidfVectorizer`)**:
   - Computes **Term Frequency - Inverse Document Frequency** (TF-IDF).
   - **N-gram Range `(1, 2)`**: Captures both single words (e.g., `"prize"`) and two-word combinations (e.g., `"cash prize"`, `"call now"`).
   - **Stop Words**: Uses built-in `"english"` stop words to ignore ubiquitous filler words (`"the"`, `"is"`, `"at"`).
   - **Minimum Document Frequency (`min_df=2`)**: Prunes typos or isolated words that appear fewer than twice across the dataset.

3. **Classification (`MultinomialNB`)**:
   - Uses **Multinomial Naive Bayes**, which assumes word frequencies follow a multinomial distribution.
   - Calculates the posterior probability:
     $$P(\text{Spam} \mid \text{Message}) \propto P(\text{Spam}) \prod_{i=1}^{m} P(w_i \mid \text{Spam})$$
   - Computes class probabilities using `.predict_proba()` to provide confidence percentages.

4. **Inference & UI (`app.py` & `predict.py`)**:
   - Pre-trained pipeline loaded via `@st.cache_resource` for sub-second responses.
   - Provides risk classification (`Spam` vs `Not Spam`) and probability breakdown.

---

## 📁 Project Structure

```text
Spam email classifier/
│
├── data/
│   └── spam.csv                    # Dataset (Kaggle SMS Spam Collection format)
│
├── models/                         # Saved ML models (git-ignored)
│   └── spam_classifier.joblib      # Trained end-to-end sklearn pipeline
│
├── reports/                        # Evaluation metrics and benchmark artifacts
│   ├── confusion_matrix.png        # Test set confusion matrix visualization
│   ├── metrics.json                # Summary of evaluation metrics & 5-fold CV
│   └── model_comparison.csv        # Benchmark vs Logistic Regression & LinearSVC
│
├── src/
│   ├── __init__.py                 # Package initializer
│   ├── config.py                   # Central paths, random seeds, hyperparameters
│   ├── data_loader.py              # Robust CSV loader, cleaner, and data summarizer
│   ├── preprocess.py               # Custom text normalization function
│   ├── train.py                    # Pipeline builder & stratified 80/20 trainer
│   ├── evaluate.py                 # Evaluation suite, cross-validation, and comparison
│   └── predict.py                  # Model loading and single-message inference
│
├── app.py                          # Modern, futuristic Streamlit web application
├── requirements.txt                # Fixed project dependencies
├── README.md                       # Comprehensive project documentation
└── .gitignore                      # Git ignore rules for virtualenvs and models
```

---

## 📊 Dataset Details

The dataset is based on the SMS Spam Collection dataset (`data/spam.csv`):
- **Total clean messages**: 5,158
- **Ham (Not Spam)**: 4,516 messages (87.55%)
- **Spam**: 642 messages (12.45%)
- **Duplicate handling**: 414 duplicate messages were dropped by `data_loader.py` to prevent data leakage between train and test splits.
- **Split Strategy**: 80/20 Stratified Split (`random_state=42`) ensuring identical class ratios in both partitions:
  - **Train**: 4,126 messages (3,612 Ham, 514 Spam)
  - **Test**: 1,032 messages (904 Ham, 128 Spam)

---

## 📈 Model Evaluation & Results

### 1. Test Set Performance (Multinomial Naive Bayes)

| Metric | Score | Explanation |
| :--- | :---: | :--- |
| **Accuracy** | **96.61%** | Overall correct classifications across all test samples |
| **Precision (Spam)** | **98.95%** | Only 1 false alarm out of 904 legitimate emails |
| **Recall (Spam)** | **73.44%** | Successfully catches ~74% of all spam messages |
| **F1-Score** | **84.30%** | Harmonic mean balancing precision and recall |

> **Why High Precision Matters Most**: In email spam filtering, a **False Positive** (marking a genuine job offer or college email as spam) is far more damaging than a **False Negative** (a spam email slipping into your inbox). Our model's **98.95% precision** ensures legitimate emails are safely preserved.

### 2. 5-Fold Stratified Cross-Validation (Training Set Only)
- **Mean CV Accuracy**: `96.41% (+/- 0.29%)`
- **Mean CV F1-Score**: `83.16% (+/- 1.62%)`
- Demonstrates consistent stability across distinct folds with near-zero variance.

### 3. Model Benchmark Comparison (`reports/model_comparison.csv`)

| Model | Accuracy | Precision | Recall | F1-Score | 5-Fold CV F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Multinomial Naive Bayes** *(Deployed)* | **0.9661** | **0.9895** | 0.7344 | 0.8430 | 0.8316 |
| **Logistic Regression** | 0.9603 | 0.9888 | 0.6875 | 0.8111 | 0.7691 |
| **Linear Support Vector (LinearSVC)** | 0.9835 | 0.9826 | 0.8828 | 0.9300 | 0.9169 |

### 4. Top 10 Spam Keywords (Learned Log Probabilities)
Extracted from $\log P(w \mid \text{Spam})$ in `predict.py`:
1. `free` (`-5.70`)
2. `txt` (`-6.09`)
3. `text` (`-6.10`)
4. `stop` (`-6.16`)
5. `mobile` (`-6.18`)
6. `claim` (`-6.24`)
7. `reply` (`-6.28`)
8. `ur` (`-6.33`)
9. `prize` (`-6.36`)
10. `won` (`-6.53`)

---

## ⚙️ Setup and Installation

### Prerequisites
- Python 3.10+ (Tested on Python 3.10 – 3.14)
- Git

### 1. Clone or Open the Repository
```bash
git clone <your-repository-url>
cd "Spam email classifier"
```

### 2. Create and Activate a Virtual Environment (Recommended)
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🏃 Running the Project

Run each step as a Python module directly from the project root:

### 1. Inspect Data & Class Distribution
```bash
python -m src.data_loader
```

### 2. Verify Preprocessing
```bash
python -m src.preprocess
```

### 3. Train and Save the Model Pipeline
```bash
python -m src.train
```
*(Saves the fitted pipeline to `models/spam_classifier.joblib`).*

### 4. Run Model Evaluation & Generate Reports
```bash
python -m src.evaluate
```
*(Saves `reports/confusion_matrix.png`, `reports/metrics.json`, and `reports/model_comparison.csv`).*

### 5. Test Inference in Terminal
```bash
python -m src.predict
```

### 6. Launch the Streamlit Web Application
```bash
python -m streamlit run app.py
```
Open your web browser at `http://localhost:8501`.

---

## 🎓 Viva Q&A / Plain-English Explanations

### Q1: What is Data Leakage and how did you prevent it?
> **Answer**: Data leakage happens when information from the test set inadvertently influences the training phase, giving falsely optimistic results. We prevented this by splitting our data **before** vectorization and calling `.fit()` strictly on the training set. The test set is only transformed during evaluation.

### Q2: Why choose Multinomial Naive Bayes over complex deep learning models?
> **Answer**: For small-to-medium text datasets (5,000 samples), Naive Bayes is exceptionally fast (trains in milliseconds), lightweight (< 300 KB model size), highly interpretable (we can inspect feature log probabilities), and achieves **98.95% Precision** without needing GPUs or heavy dependencies.

### Q3: What is TF-IDF?
> **Answer**: Term Frequency measures how often a word appears in a specific message. Inverse Document Frequency penalizes common words (like "the" or "is") that appear everywhere. High TF-IDF means a word is characteristic of that specific message.

### Q4: Why is Stratified Splitting important?
> **Answer**: Since spam messages only make up ~12.5% of the dataset (imbalanced), random splitting could accidentally create a test set with too few or too many spam samples. Stratified splitting preserves the exact 87.5% / 12.5% ratio across both train and test splits.

---

## 🔮 Limitations & Future Improvements

### Limitations
1. **Short Message Length**: Dataset is primarily SMS-based; lengthy email headers or multipart HTML layouts require additional parsing.
2. **Evolving Spam Vocabulary**: Spammers frequently obfuscate words (e.g., `"F-r-e-e"`, `"w1n"`), which exact n-gram matching may miss.
3. **Context Ignorance**: Naive Bayes treats words independently and cannot capture deep semantic negation (e.g., `"not free"` vs `"free"`).

### Future Improvements
1. **Character-level N-Grams**: Add sub-word character n-grams to detect misspelled words or leetspeak.
2. **Incremental Retraining**: Implement online learning with `partial_fit()` so users can flag false positives in the UI and improve the model over time.
3. **DKIM / SPF Header Analysis**: Extend the classifier to analyze raw email headers and domain reputation scores alongside message text.

---

## 📄 License
This project is open-source under the MIT License for educational and college project purposes.
