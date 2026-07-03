"""
explain.py
----------
Shows which words most influenced a prediction — makes the model's
decisions interpretable instead of a black box.
"""

import os
import sys

import joblib
import numpy as np

sys.path.append(os.path.dirname(__file__))
from preprocess import clean_text  # noqa: E402

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def explain_linear(text: str, model_name: str = "logistic", top_n: int = 10):
    """
    Fast explanation for linear models (Logistic Regression, Naive Bayes)
    using the model's own coefficients.
    """
    model = joblib.load(os.path.join(MODEL_DIR, f"{model_name}.joblib"))
    vectorizer = joblib.load(os.path.join(MODEL_DIR, "vectorizer.joblib"))

    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    feature_names = np.array(vectorizer.get_feature_names_out())

    present_idx = vec.nonzero()[1]
    if len(present_idx) == 0:
        return []

    if hasattr(model, "coef_"):
        weights = model.coef_[0][present_idx]
    elif hasattr(model, "feature_log_prob_"):
        weights = model.feature_log_prob_[1][present_idx] - model.feature_log_prob_[0][present_idx]
    else:
        raise ValueError(
            f"Model type {type(model)} has no coef_ or feature_log_prob_. "
            "Use explain_shap() instead for tree-based models."
        )

    words = feature_names[present_idx]
    pairs = sorted(zip(words, weights), key=lambda p: -abs(p[1]))
    return pairs[:top_n]


def explain_shap(text: str, model_name: str = "random_forest", top_n: int = 10):
    """
    General explanation using SHAP — works for any sklearn model,
    including tree-based ones like Random Forest.

    Requires: pip install shap
    """
    import shap

    model = joblib.load(os.path.join(MODEL_DIR, f"{model_name}.joblib"))
    vectorizer = joblib.load(os.path.join(MODEL_DIR, "vectorizer.joblib"))

    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    feature_names = np.array(vectorizer.get_feature_names_out())

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(vec.toarray())

    values = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]

    present_idx = vec.nonzero()[1]
    words = feature_names[present_idx]
    weights = values[present_idx]

    pairs = sorted(zip(words, weights), key=lambda p: -abs(p[1]))
    return pairs[:top_n]


def print_explanation(pairs):
    if not pairs:
        print("No recognized words to explain (try a longer excerpt).")
        return
    print(f"{'word':<20} {'influence':>10}  direction")
    print("-" * 45)
    for word, weight in pairs:
        direction = "→ real" if weight > 0 else "→ fake"
        print(f"{word:<20} {weight:>10.3f}  {direction}")


if __name__ == "__main__":
    sample = "Scientists confirm the earth is flat, NASA admits cover-up."
    print(f"Text: {sample}\n")

    pairs = explain_linear(sample, model_name="logistic")
    print_explanation(pairs)