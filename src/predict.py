"""Inference and prediction module for Spam Email Classifier.

Provides functions to load the trained pipeline, classify single messages,
compute spam probabilities, extract top predictive spam terms, and handle
empty or invalid inputs gracefully.
"""

from pathlib import Path
from typing import Any
import joblib
import numpy as np
from sklearn.pipeline import Pipeline

from src.config import MODEL_FILE_PATH


def load_model(model_path: Path = MODEL_FILE_PATH) -> Pipeline:
    """Load the trained machine learning pipeline from disk.

    Args:
        model_path: Path to the saved .joblib model file.

    Returns:
        Pipeline: Trained scikit-learn Pipeline.

    Raises:
        FileNotFoundError: If the model artifact has not been trained yet.
    """
    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found at: {model_path}\n"
            "Please run 'python -m src.train' first to train and save the model."
        )
    return joblib.load(model_path)


def predict_message(
    text: str, pipeline: Pipeline | None = None
) -> dict[str, Any]:
    """Classify a text message as 'Spam' or 'Not Spam'.

    Handles empty or whitespace-only inputs gracefully without throwing errors.

    Args:
        text: Input message string to classify.
        pipeline: Optional pre-loaded pipeline. If None, loads from disk.

    Returns:
        dict: Classification results in the format:
            {"label": "Spam" | "Not Spam", "spam_probability": float}
    """
    # Gracefully handle empty or whitespace-only input
    if text is None or not str(text).strip():
        return {
            "label": "Not Spam",
            "spam_probability": 0.0,
        }

    # Ensure pipeline is available
    if pipeline is None:
        pipeline = load_model()

    # Predict class probabilities: index 0 is Ham, index 1 is Spam
    probabilities = pipeline.predict_proba([text])[0]
    spam_prob = float(probabilities[1])

    # Classify as Spam if probability is at or above 50%
    label = "Spam" if spam_prob >= 0.5 else "Not Spam"

    return {
        "label": label,
        "spam_probability": round(spam_prob, 4),
    }


def top_spam_words(
    n: int = 15, pipeline: Pipeline | None = None
) -> list[tuple[str, float]]:
    """Retrieve the top words associated most with Spam based on feature log probabilities.

    In Multinomial Naive Bayes, feature_log_prob_[1] represents log P(word | Spam).
    Terms with higher (less negative) log probabilities occur most frequently and
    with highest TF-IDF weight in spam messages.

    Args:
        n: Number of top words to return (default: 15).
        pipeline: Optional pre-loaded pipeline. If None, loads from disk.

    Returns:
        list[tuple[str, float]]: List of (word, log_probability) pairs sorted descending.
    """
    if pipeline is None:
        pipeline = load_model()

    # Access the vectorizer and naive bayes steps from the pipeline
    tfidf = pipeline.named_steps["tfidf"]
    nb = pipeline.named_steps["nb"]

    feature_names = tfidf.get_feature_names_out()

    # Locate the spam class index (label 1)
    spam_class_idx = list(nb.classes_).index(1)
    spam_log_probs = nb.feature_log_prob_[spam_class_idx]

    # Get the indices of the highest log probabilities in descending order
    top_indices = np.argsort(spam_log_probs)[::-1][:n]

    return [
        (str(feature_names[idx]), round(float(spam_log_probs[idx]), 4))
        for idx in top_indices
    ]


if __name__ == "__main__":
    print("=" * 60)
    print("TESTING PREDICTION MODULE (src/predict.py)")
    print("=" * 60)

    clf = load_model()

    # 1. Test empty input handling
    print("Test 1: Empty string input:")
    print(" ", predict_message("", pipeline=clf))

    print("\nTest 2: Whitespace-only input:")
    print(" ", predict_message("    \n\t   ", pipeline=clf))

    # 2. Test sample spam message
    spam_sample = "WINNER! You have been selected for a free $1,000 gift card. Call 0800123 now!"
    print(f"\nTest 3: Sample Spam ('{spam_sample}'):")
    print(" ", predict_message(spam_sample, pipeline=clf))

    # 3. Test sample ham message
    ham_sample = "Hey, are you free to grab lunch together around noon?"
    print(f"\nTest 4: Sample Ham ('{ham_sample}'):")
    print(" ", predict_message(ham_sample, pipeline=clf))

    # 4. Display top 15 spam words
    print("\n" + "=" * 60)
    print(f"TOP 15 SPAM WORDS (Feature Log Probabilities log P(word|Spam)):")
    print("=" * 60)
    for rank, (word, score) in enumerate(top_spam_words(15, pipeline=clf), start=1):
        print(f"  {rank:2d}. {word:<15} (log prob: {score})")
    print("=" * 60)
