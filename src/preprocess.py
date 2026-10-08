"""Text preprocessing module for cleaning email and SMS messages.

Performs text normalization: lowercasing, stripping URLs, removing
non-letter characters, and collapsing whitespace. Stop words are intentionally
left for TfidfVectorizer(stop_words="english") in the pipeline.
"""

import re
from src.config import DATA_FILE_PATH, TEXT_COLUMN
from src.data_loader import load_raw_data


def clean_text(text: str) -> str:
    """Clean and normalize a raw text message.

    Steps applied:
    1. Lowercase all characters.
    2. Remove web URLs (http, https, www).
    3. Remove all non-letter characters (numbers, punctuation, symbols).
    4. Collapse multiple spaces into a single space and strip edges.

    Args:
        text: Raw input string.

    Returns:
        str: Normalized, cleaned string containing only lowercase letters.
    """
    if not isinstance(text, str):
        text = str(text) if text is not None else ""

    # Convert all characters to lowercase for uniform tokenization
    cleaned = text.lower()

    # Remove URLs (e.g., http://example.com, www.example.com)
    cleaned = re.sub(r"https?://\S+|www\.\S+", " ", cleaned)

    # Remove non-letter characters (keep only lowercase a-z and whitespace)
    # This strips punctuation, numbers, and special symbols
    cleaned = re.sub(r"[^a-z\s]", " ", cleaned)

    # Collapse repeated whitespace characters into a single space and trim edges
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned


def show_preprocessing_examples(n_samples: int = 3) -> None:
    """Display before and after preprocessing comparisons on real messages.

    Args:
        n_samples: Number of sample messages to display.
    """
    print("=" * 70)
    print("TEXT PREPROCESSING DEMONSTRATION (BEFORE vs AFTER)")
    print("=" * 70)

    try:
        df = load_raw_data(DATA_FILE_PATH)
        samples = df[TEXT_COLUMN].head(n_samples).tolist()
    except Exception:
        # Fallback sample messages if file reading encounters issues
        samples = [
            "Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...",
            "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's",
            "XXXMobileMovieClub: To use your credit, click the WAP link in the next txt message or click here>> http://wap.xxxmobilemovieclub.com?n=QJKGIGHJJGCBL",
        ][:n_samples]

    for idx, raw_text in enumerate(samples, start=1):
        cleaned_text = clean_text(raw_text)
        print(f"\n[Example {idx}]")
        print(f"BEFORE:\n  {raw_text}")
        print(f"AFTER:\n  {cleaned_text}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    show_preprocessing_examples(3)
