"""Model training module for Spam Email Classifier.

Builds a scikit-learn Pipeline with TfidfVectorizer and MultinomialNB,
splits the dataset using stratified 80/20 train/test split, fits ONLY
on the training data to prevent data leakage, and saves the trained pipeline.
"""

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from src.config import (
    LABEL_COLUMN,
    MODEL_FILE_PATH,
    MODELS_DIR,
    RANDOM_STATE,
    TEST_SIZE,
    TEXT_COLUMN,
    TFIDF_MIN_DF,
    TFIDF_NGRAM_RANGE,
    TFIDF_STOP_WORDS,
)
from src.data_loader import clean_and_encode_dataset, load_raw_data
from src.preprocess import clean_text


def build_pipeline() -> Pipeline:
    """Create the end-to-end Machine Learning pipeline.

    Combines TF-IDF feature extraction with a Multinomial Naive Bayes classifier.
    The vectorizer uses clean_text as its preprocessor, extracts unigrams and
    bigrams, removes English stop words, and ignores rare terms (min_df=2).

    Returns:
        Pipeline: Unfitted scikit-learn pipeline.
    """
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    preprocessor=clean_text,
                    ngram_range=TFIDF_NGRAM_RANGE,
                    min_df=TFIDF_MIN_DF,
                    stop_words=TFIDF_STOP_WORDS,
                ),
            ),
            ("nb", MultinomialNB()),
        ]
    )


def train_model() -> Pipeline:
    """Execute data loading, stratified train/test split, training, and artifact saving.

    Returns:
        Pipeline: Trained scikit-learn pipeline.
    """
    print("=" * 60)
    print("STARTING MODEL TRAINING")
    print("=" * 60)

    # Load and clean dataset
    raw_df = load_raw_data()
    df = clean_and_encode_dataset(raw_df)
    print(f"Loaded {len(df)} total clean messages.")

    X = df[TEXT_COLUMN]
    y = df[LABEL_COLUMN]

    # Stratified 80/20 split ensures both train and test sets have identical class ratios
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    train_ham = (y_train == 0).sum()
    train_spam = (y_train == 1).sum()
    test_ham = (y_test == 0).sum()
    test_spam = (y_test == 1).sum()

    print(f"\nTraining set size : {len(X_train)} messages (Ham: {train_ham}, Spam: {train_spam})")
    print(f"Testing set size  : {len(X_test)} messages (Ham: {test_ham}, Spam: {test_spam})")

    # Build pipeline
    pipeline = build_pipeline()

    # Fit ONLY on the training data to strictly prevent data leakage
    print("\nFitting TF-IDF Vectorizer and Multinomial Naive Bayes on training set...")
    pipeline.fit(X_train, y_train)

    # Ensure models directory exists
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Save the fitted pipeline
    joblib.dump(pipeline, MODEL_FILE_PATH)
    print(f"Successfully saved trained model pipeline to: {MODEL_FILE_PATH}")
    print("=" * 60)

    return pipeline


if __name__ == "__main__":
    train_model()
