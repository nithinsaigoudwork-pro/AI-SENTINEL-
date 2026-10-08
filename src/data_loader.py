"""Data loading and preprocessing module.

Loads the raw CSV dataset, normalizes column formats, cleans missing
or duplicate entries, maps string labels to binary integers, and prints
a summary of the dataset.
"""

from pathlib import Path
import pandas as pd

from src.config import (
    DATA_FILE_PATH,
    INVERSE_LABEL_MAPPING,
    KAGGLE_LABEL_COL,
    KAGGLE_TEXT_COL,
    LABEL_COLUMN,
    LABEL_MAPPING,
    TEXT_COLUMN,
)


def load_raw_data(file_path: Path = DATA_FILE_PATH) -> pd.DataFrame:
    """Load the raw spam dataset from a CSV file.

    Handles both Kaggle format (v1, v2 with extra unnamed columns and latin-1
    encoding) and standard clean CSV format (label, text).

    Args:
        file_path: Path to the CSV file.

    Returns:
        pd.DataFrame: Raw DataFrame with standardized 'label' and 'text' columns.

    Raises:
        FileNotFoundError: If the file does not exist at file_path.
        ValueError: If neither expected column format is detected.
    """
    # Verify that the dataset file actually exists before attempting to read
    if not file_path.exists():
        raise FileNotFoundError(
            f"\n[ERROR] Dataset file not found at: {file_path}\n"
            f"Please place your 'spam.csv' file inside the '{file_path.parent}' directory."
        )

    # Try standard UTF-8 first; fall back to latin-1 for Kaggle SMS spam datasets
    try:
        df = pd.read_csv(file_path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding="latin-1")

    # Clean leading/trailing spaces from column headers
    df.columns = [str(col).strip() for col in df.columns]

    # Detect Kaggle SMS Spam Collection format: 'v1' (label), 'v2' (text)
    if KAGGLE_LABEL_COL in df.columns and KAGGLE_TEXT_COL in df.columns:
        # Keep only the two relevant columns, dropping any extra unnamed columns
        df = df[[KAGGLE_LABEL_COL, KAGGLE_TEXT_COL]].copy()
        df = df.rename(
            columns={
                KAGGLE_LABEL_COL: LABEL_COLUMN,
                KAGGLE_TEXT_COL: TEXT_COLUMN,
            }
        )
    # Detect standard format: 'label', 'text' (case-insensitive check)
    elif LABEL_COLUMN in [c.lower() for c in df.columns] and TEXT_COLUMN in [
        c.lower() for c in df.columns
    ]:
        # Normalize column header casing
        column_rename = {c: c.lower() for c in df.columns}
        df = df.rename(columns=column_rename)
        df = df[[LABEL_COLUMN, TEXT_COLUMN]].copy()
    else:
        raise ValueError(
            f"Unrecognized CSV format in '{file_path}'. "
            f"Expected columns ('{KAGGLE_LABEL_COL}', '{KAGGLE_TEXT_COL}') or "
            f"('{LABEL_COLUMN}', '{TEXT_COLUMN}'). Found: {list(df.columns)}"
        )

    return df


def clean_and_encode_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Clean missing values, drop duplicates, and map labels to integers.

    Args:
        df: Raw DataFrame containing 'label' and 'text' columns.

    Returns:
        pd.DataFrame: Cleaned DataFrame with mapped binary labels (ham=0, spam=1).
    """
    df = df.copy()

    # Drop any rows with nulls in text or label
    df = df.dropna(subset=[LABEL_COLUMN, TEXT_COLUMN])

    # Convert text column to string and strip whitespace
    df[TEXT_COLUMN] = df[TEXT_COLUMN].astype(str).str.strip()

    # Standardize label strings (trim spaces, lowercase)
    df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(str).str.strip().str.lower()

    # Filter to only keep rows with valid labels defined in LABEL_MAPPING
    df = df[df[LABEL_COLUMN].isin(LABEL_MAPPING.keys())]

    # Remove duplicate messages to avoid data leakage between splits
    df = df.drop_duplicates(subset=[TEXT_COLUMN])

    # Map string labels to numeric integers (ham -> 0, spam -> 1)
    df[LABEL_COLUMN] = df[LABEL_COLUMN].map(LABEL_MAPPING).astype(int)

    # Reset index for clean row numbering
    df = df.reset_index(drop=True)

    return df


def print_data_summary(df: pd.DataFrame) -> None:
    """Print a clean summary of dataset size, class balance, and sample messages.

    Args:
        df: Cleaned DataFrame with 'label' and 'text' columns.
    """
    total_rows = len(df)
    ham_label_name = INVERSE_LABEL_MAPPING[LABEL_MAPPING["ham"]]
    spam_label_name = INVERSE_LABEL_MAPPING[LABEL_MAPPING["spam"]]
    ham_count = (df[LABEL_COLUMN] == LABEL_MAPPING["ham"]).sum()
    spam_count = (df[LABEL_COLUMN] == LABEL_MAPPING["spam"]).sum()
    spam_pct = (spam_count / total_rows) * 100 if total_rows > 0 else 0.0

    print("=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"Total messages (after cleaning) : {total_rows}")
    print(f"{ham_label_name} count (label 0) : {ham_count} ({100 - spam_pct:.2f}%)")
    print(f"{spam_label_name} count           (label 1) : {spam_count} ({spam_pct:.2f}%)")
    print("=" * 60)

    print(f"\n--- 3 SAMPLE {ham_label_name.upper()} MESSAGES ---")
    ham_samples = df[df[LABEL_COLUMN] == LABEL_MAPPING["ham"]][TEXT_COLUMN].head(3)
    for idx, sample in enumerate(ham_samples, 1):
        print(f"[{idx}] {sample}\n")

    print(f"--- 3 SAMPLE {spam_label_name.upper()} MESSAGES ---")
    spam_samples = df[df[LABEL_COLUMN] == LABEL_MAPPING["spam"]][TEXT_COLUMN].head(3)
    for idx, sample in enumerate(spam_samples, 1):
        print(f"[{idx}] {sample}\n")
    print("=" * 60)


def load_and_prepare_data(file_path: Path = DATA_FILE_PATH) -> pd.DataFrame:
    """Execute the full data loading, cleaning, and summary workflow.

    Args:
        file_path: Path to the raw CSV dataset file.

    Returns:
        pd.DataFrame: Cleaned, validated, and label-encoded dataset.
    """
    raw_df = load_raw_data(file_path=file_path)
    cleaned_df = clean_and_encode_dataset(raw_df)
    print_data_summary(cleaned_df)
    return cleaned_df


if __name__ == "__main__":
    try:
        load_and_prepare_data()
    except FileNotFoundError as err:
        print(err)
