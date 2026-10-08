"""Configuration module for paths, constants, and hyperparameters.

This keeps all project settings in one place so nothing is hard-coded
in individual scripts.
"""

from pathlib import Path

# Base project directory (spam-email-classifier root)
BASE_DIR: Path = Path(__file__).resolve().parent.parent

# Data directory and paths
DATA_DIR: Path = BASE_DIR / "data"
DATA_FILE_PATH: Path = DATA_DIR / "spam.csv"

# Models directory and saved model artifact
MODELS_DIR: Path = BASE_DIR / "models"
MODEL_FILE_PATH: Path = MODELS_DIR / "spam_classifier.joblib"

# Reports directory and output artifacts
REPORTS_DIR: Path = BASE_DIR / "reports"
METRICS_FILE_PATH: Path = REPORTS_DIR / "metrics.json"
CONFUSION_MATRIX_PATH: Path = REPORTS_DIR / "confusion_matrix.png"
MODEL_COMPARISON_PATH: Path = REPORTS_DIR / "model_comparison.csv"

# Standard normalized column names across the project
LABEL_COLUMN: str = "label"
TEXT_COLUMN: str = "text"

# Kaggle SMS Spam Collection legacy column headers
KAGGLE_LABEL_COL: str = "v1"
KAGGLE_TEXT_COL: str = "v2"

# Label encoding: 0 for ham (not spam), 1 for spam
LABEL_MAPPING: dict[str, int] = {
    "ham": 0,
    "spam": 1,
}

INVERSE_LABEL_MAPPING: dict[int, str] = {
    0: "Not Spam (Ham)",
    1: "Spam",
}

# Machine Learning experiment constants
RANDOM_STATE: int = 42
TEST_SIZE: float = 0.2

# TF-IDF Vectorizer settings
TFIDF_NGRAM_RANGE: tuple[int, int] = (1, 2)  # Unigrams and bigrams
TFIDF_MIN_DF: int = 2                       # Ignore tokens appearing in < 2 documents
TFIDF_STOP_WORDS: str = "english"
