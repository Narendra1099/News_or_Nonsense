"""
predict.py
----------
Load a trained model + vectorizer and classify new article text as
"real" or "fake", with a confidence score.
"""

import os

import joblib

from preprocess import clean_text

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


class FakeNewsPredictor:
    def __init__(self, model_name: str = "logistic"):
        model_path = os.path.join(MODEL_DIR, f"{model_name}.joblib")
        vectorizer_path = os.path.join(MODEL_DIR, "vectorizer.joblib")

        if not (os.path.exists(model_path) and os.path.exists(vectorizer_path)):
            raise FileNotFoundError(
                f"Model files not found in {MODEL_DIR}. "
                "Train a model first with: python src/train.py --model logistic"
            )

        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)

    def predict(self, text: str) -> dict:
        """
        Classify a single piece of text.

        Returns
        -------
        dict with keys: label ("real"/"fake"), confidence (0-1)
        """
        cleaned = clean_text(text)
        vec = self.vectorizer.transform([cleaned])

        pred = self.model.predict(vec)[0]
        label = "real" if pred == 1 else "fake"

        confidence = None
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(vec)[0]
            confidence = float(max(proba))

        return {"label": label, "confidence": confidence}


if __name__ == "__main__":
    predictor = FakeNewsPredictor(model_name="logistic")
    sample = "Scientists confirm the earth is flat, NASA admits cover-up."
    result = predictor.predict(sample)
    print(f"Text: {sample}")
    print(f"Prediction: {result['label']} (confidence: {result['confidence']:.2f})")
