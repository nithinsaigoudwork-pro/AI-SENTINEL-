"""Model evaluation module for Spam Email Classifier.

Evaluates the trained pipeline on the unseen test set, performs 5-fold
stratified cross-validation on the training set, compares Naive Bayes against
Logistic Regression and LinearSVC, saves reports (confusion matrix plot,
comparison CSV, and metrics JSON), and runs a 6-message sanity test.
"""

import json
from pathlib import Path
import joblib
import matplotlib
matplotlib.use("Agg")  # Use headless backend for saving figures without a GUI window
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import (
    CONFUSION_MATRIX_PATH,
    LABEL_COLUMN,
    METRICS_FILE_PATH,
    MODEL_COMPARISON_PATH,
    MODEL_FILE_PATH,
    RANDOM_STATE,
    REPORTS_DIR,
    TEST_SIZE,
    TEXT_COLUMN,
    TFIDF_MIN_DF,
    TFIDF_NGRAM_RANGE,
    TFIDF_STOP_WORDS,
)
from src.data_loader import clean_and_encode_dataset, load_raw_data
from src.preprocess import clean_text


def load_data_splits() -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Load cleaned data and apply the exact stratified 80/20 train/test split.

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    raw_df = load_raw_data()
    df = clean_and_encode_dataset(raw_df)

    X = df[TEXT_COLUMN]
    y = df[LABEL_COLUMN]

    # Recreate the exact same train/test split using random_state=42 and stratify=y
    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def evaluate_test_set(
    pipeline: Pipeline, X_test: pd.Series, y_test: pd.Series
) -> dict[str, float]:
    """Evaluate pipeline on the unseen test set and save confusion matrix.

    Args:
        pipeline: Trained scikit-learn pipeline.
        X_test: Test messages.
        y_test: True test labels (0=ham, 1=spam).

    Returns:
        dict[str, float]: Test set metric scores.
    """
    y_pred = pipeline.predict(X_test)

    # Compute key metrics (pos_label=1 specifies spam as positive class)
    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, pos_label=1))
    rec = float(recall_score(y_test, y_pred, pos_label=1))
    f1 = float(f1_score(y_test, y_pred, pos_label=1))

    print("\n" + "=" * 60)
    print("TEST SET EVALUATION METRICS (Spam = Positive Class)")
    print("=" * 60)
    print(f"Accuracy  : {acc:.4f} ({acc * 100:.2f}%)")
    print(f"Precision : {prec:.4f} ({prec * 100:.2f}%)")
    print(f"Recall    : {rec:.4f} ({rec * 100:.2f}%)")
    print(f"F1-Score  : {f1:.4f} ({f1 * 100:.2f}%)")
    print("=" * 60)

    print("\nCLASSIFICATION REPORT:")
    print(classification_report(y_test, y_pred, target_names=["Ham", "Spam"]))

    # Generate and save confusion matrix plot
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Ham", "Spam"])

    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix - Multinomial Naive Bayes", fontsize=12, pad=12)
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=300)
    plt.close(fig)
    print(f"Saved confusion matrix plot to: {CONFUSION_MATRIX_PATH}")

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
    }


def run_cross_validation(
    pipeline: Pipeline, X_train: pd.Series, y_train: pd.Series
) -> dict[str, float]:
    """Perform 5-fold stratified cross-validation on the training set only.

    Args:
        pipeline: Scikit-learn pipeline.
        X_train: Training messages.
        y_train: Training labels.

    Returns:
        dict[str, float]: Mean and std of CV F1 and Accuracy scores.
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    cv_f1 = cross_val_score(pipeline, X_train, y_train, cv=skf, scoring="f1")
    cv_acc = cross_val_score(pipeline, X_train, y_train, cv=skf, scoring="accuracy")

    print("\n" + "=" * 60)
    print("5-FOLD STRATIFIED CROSS-VALIDATION (Training Set Only)")
    print("=" * 60)
    print(f"Mean CV Accuracy : {cv_acc.mean():.4f} (+/- {cv_acc.std():.4f})")
    print(f"Mean CV F1-Score : {cv_f1.mean():.4f} (+/- {cv_f1.std():.4f})")
    print("=" * 60)

    return {
        "cv_5fold_accuracy_mean": round(float(cv_acc.mean()), 4),
        "cv_5fold_accuracy_std": round(float(cv_acc.std()), 4),
        "cv_5fold_f1_mean": round(float(cv_f1.mean()), 4),
        "cv_5fold_f1_std": round(float(cv_f1.std()), 4),
    }


def compare_models(
    X_train: pd.Series, y_train: pd.Series, X_test: pd.Series, y_test: pd.Series
) -> pd.DataFrame:
    """Compare Naive Bayes vs Logistic Regression vs LinearSVC on identical data.

    Saves the resulting comparison table to reports/model_comparison.csv.

    Args:
        X_train: Training messages.
        y_train: Training labels.
        X_test: Test messages.
        y_test: Test labels.

    Returns:
        pd.DataFrame: Table comparing model performances.
    """
    classifiers = {
        "Multinomial Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(random_state=RANDOM_STATE),
        "Linear Support Vector (LinearSVC)": LinearSVC(
            random_state=RANDOM_STATE, dual="auto"
        ),
    }

    results = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    print("\n" + "=" * 60)
    print("RUNNING MODEL COMPARISON (Naive Bayes vs LogReg vs LinearSVC)")
    print("=" * 60)

    for name, clf in classifiers.items():
        # Build a fresh pipeline using the exact same TF-IDF preprocessor
        pipe = Pipeline(
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
                ("clf", clf),
            ]
        )

        # 5-fold CV F1 on training set
        cv_f1 = cross_val_score(pipe, X_train, y_train, cv=skf, scoring="f1").mean()

        # Fit on training set and evaluate on test set
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, pos_label=1)
        rec = recall_score(y_test, y_pred, pos_label=1)
        f1 = f1_score(y_test, y_pred, pos_label=1)

        results.append(
            {
                "Model": name,
                "Accuracy": round(acc, 4),
                "Precision": round(prec, 4),
                "Recall": round(rec, 4),
                "F1-Score": round(f1, 4),
                "CV_F1_Mean": round(cv_f1, 4),
            }
        )

    comparison_df = pd.DataFrame(results)
    comparison_df.to_csv(MODEL_COMPARISON_PATH, index=False)
    print(comparison_df.to_string(index=False))
    print(f"\nSaved model comparison table to: {MODEL_COMPARISON_PATH}")
    print("=" * 60)

    return comparison_df


def run_sanity_tests(pipeline: Pipeline) -> None:
    """Run sanity test predictions on 6 hand-written messages.

    Args:
        pipeline: Trained scikit-learn pipeline.
    """
    sanity_samples = [
        # 3 obvious spam messages
        (
            "Congratulations! You have won a $1,000 Walmart gift card. Click here to claim your prize now!",
            "Spam",
        ),
        (
            "URGENT: Your account has been suspended! Call 08000123456 immediately to verify credentials.",
            "Spam",
        ),
        (
            "Exclusive offer: Text WIN to 88990 to receive your free luxury holiday vouchers today!",
            "Spam",
        ),
        # 3 obvious ham messages
        (
            "Hey, are we still meeting for coffee at 3pm today?",
            "Ham",
        ),
        (
            "Could you please email me the class lecture notes whenever you get home?",
            "Ham",
        ),
        (
            "I'm on the bus heading back now, will call you in twenty minutes.",
            "Ham",
        ),
    ]

    print("\n" + "=" * 60)
    print("SANITY TEST ON 6 HAND-WRITTEN MESSAGES")
    print("=" * 60)

    for idx, (msg, expected) in enumerate(sanity_samples, start=1):
        pred_label_id = pipeline.predict([msg])[0]
        pred_label_str = "Spam" if pred_label_id == 1 else "Ham"
        prob = pipeline.predict_proba([msg])[0][1] if hasattr(pipeline, "predict_proba") else None
        prob_str = f" (Spam Probability: {prob:.4f})" if prob is not None else ""
        status = "PASS" if pred_label_str == expected else "FAIL"

        print(f"[{idx}] Text    : {msg}")
        print(f"    Expected: {expected} | Predicted: {pred_label_str}{prob_str} -> [{status}]\n")
    print("=" * 60)


def evaluate() -> None:
    """Execute the full evaluation workflow and save all metrics."""
    # Ensure trained model exists
    if not MODEL_FILE_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found at {MODEL_FILE_PATH}. "
            "Please run 'python -m src.train' first."
        )

    pipeline: Pipeline = joblib.load(MODEL_FILE_PATH)
    X_train, X_test, y_train, y_test = load_data_splits()

    # 1. Test set metrics and confusion matrix
    test_metrics = evaluate_test_set(pipeline, X_test, y_test)

    # 2. 5-fold CV on training set
    cv_metrics = run_cross_validation(pipeline, X_train, y_train)

    # 3. Model comparison table
    compare_models(X_train, y_train, X_test, y_test)

    # 4. Save combined metrics to reports/metrics.json
    all_metrics = {
        "model_name": "Multinomial Naive Bayes",
        **test_metrics,
        **cv_metrics,
    }
    with open(METRICS_FILE_PATH, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=4)
    print(f"Saved metrics summary to: {METRICS_FILE_PATH}")

    # 5. Sanity tests
    run_sanity_tests(pipeline)


if __name__ == "__main__":
    evaluate()
