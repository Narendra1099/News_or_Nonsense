"""
train.py
--------
Trains a fake-news classifier and saves the model + vectorizer to disk.

Expected input data (see README for how to get it):
    data/Fake.csv   -> fake news articles
    data/True.csv   -> real news articles

Both files should have at least 'title' and 'text' columns
(this matches the popular Kaggle "Fake and Real News Dataset").

Usage:
    python src/train.py --model logistic
    python src/train.py --model random_forest
"""

import argparse
import os
import sys

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Allow running as `python src/train.py` from project root
sys.path.append(os.path.dirname(__file__))
from preprocess import clean_dataframe  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

MODEL_REGISTRY = {
    "logistic": LogisticRegression(max_iter=1000),
    "naive_bayes": MultinomialNB(),
    "random_forest": RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42),
}


def load_data():
    fake_path = os.path.join(DATA_DIR, "Fake.csv")
    true_path = os.path.join(DATA_DIR, "True.csv")

    if not (os.path.exists(fake_path) and os.path.exists(true_path)):
        raise FileNotFoundError(
            "Couldn't find data/Fake.csv and data/True.csv.\n"
            "Download the dataset and place both files in the data/ folder "
            "(see README.md for the link)."
        )

    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)

    fake_df["label"] = 0  # 0 = fake
    true_df["label"] = 1  # 1 = real

    df = pd.concat([fake_df, true_df], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
    return df


def train(model_name: str, test_size: float = 0.2):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'. Choose from {list(MODEL_REGISTRY)}")

    print(f"[1/5] Loading data...")
    df = load_data()
    print(f"      Loaded {len(df)} articles ({df['label'].value_counts().to_dict()})")

    print(f"[2/5] Cleaning text...")
    df = clean_dataframe(df, text_columns=("title", "text"))

    print(f"[3/5] Splitting train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"], df["label"], test_size=test_size, random_state=42, stratify=df["label"]
    )

    print(f"[4/5] Vectorizing (TF-IDF) and training '{model_name}'...")
    vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = MODEL_REGISTRY[model_name]
    model.fit(X_train_vec, y_train)

    print(f"[5/5] Evaluating...")
    preds = model.predict(X_test_vec)
    print("\nClassification report:")
    print(classification_report(y_test, preds, target_names=["fake", "real"]))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, preds))

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODEL_DIR, f"{model_name}.joblib"))
    joblib.dump(vectorizer, os.path.join(MODEL_DIR, "vectorizer.joblib"))
    print(f"\nSaved model to models/{model_name}.joblib")
    print(f"Saved vectorizer to models/vectorizer.joblib")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a fake news classifier.")
    parser.add_argument(
        "--model",
        choices=list(MODEL_REGISTRY.keys()),
        default="logistic",
        help="Which model to train (default: logistic)",
    )
    args = parser.parse_args()
    train(args.model)
