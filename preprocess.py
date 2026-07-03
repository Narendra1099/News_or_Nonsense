"""
preprocess.py
--------------
Text cleaning utilities for the fake news detector.

These functions take raw article text/titles and clean them up so
they're ready for TF-IDF vectorization (or any other NLP pipeline).
"""

import re
import string

import nltk
from nltk.corpus import stopwords

# Download stopwords quietly the first time this runs
try:
    STOPWORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords", quiet=True)
    STOPWORDS = set(stopwords.words("english"))


def clean_text(text: str) -> str:
    """
    Lowercase, strip URLs/punctuation/numbers, and remove stopwords.

    Parameters
    ----------
    text : str
        Raw article title or body text.

    Returns
    -------
    str
        Cleaned text, ready for vectorization.
    """
    if not isinstance(text, str):
        return ""

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", " ", text)

    # Remove HTML tags (common in scraped news data)
    text = re.sub(r"<.*?>", " ", text)

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Remove numbers
    text = re.sub(r"\d+", " ", text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Remove stopwords
    words = [w for w in text.split() if w not in STOPWORDS]

    return " ".join(words)


def clean_dataframe(df, text_columns=("title", "text")):
    """
    Apply clean_text() to one or more columns of a DataFrame and
    combine them into a single 'clean_text' column.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain the columns listed in text_columns.
    text_columns : tuple[str]
        Which columns to clean and combine.

    Returns
    -------
    pandas.DataFrame
        Same DataFrame with a new 'clean_text' column added.
    """
    existing_cols = [c for c in text_columns if c in df.columns]
    if not existing_cols:
        raise ValueError(
            f"None of {text_columns} found in DataFrame columns: {list(df.columns)}"
        )

    combined = df[existing_cols].fillna("").agg(" ".join, axis=1)
    df["clean_text"] = combined.apply(clean_text)
    return df
