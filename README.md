# 📰 Fake News Detector

A machine learning project that classifies news articles as **real** or **fake**
using TF-IDF text vectorization and classic ML models (Logistic Regression,
Naive Bayes, Random Forest), wrapped in a Streamlit web app.

## Project structure

```
fake-news-detector/
├── data/               # Put Fake.csv and True.csv here (see below)
├── notebooks/          # For exploration / experiments
├── src/
│   ├── preprocess.py   # Text cleaning utilities
│   ├── train.py        # Trains and saves a model
│   └── predict.py       # Loads a model and classifies new text
├── app.py               # Streamlit web app
├── requirements.txt
└── README.md
```

## 1. Setup

```bash
cd fake-news-detector
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Get the dataset

This project is built around the popular **"Fake and Real News Dataset"** on Kaggle:

👉 https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

1. Download it (you'll need a free Kaggle account).
2. Unzip it — you should get `Fake.csv` and `True.csv`.
3. Place both files in the `data/` folder.

(Any dataset with `title`/`text` columns and fake/real labels will work —
just adjust `load_data()` in `src/train.py` if column names differ.)

## 3. Train a model

```bash
python src/train.py --model logistic
```

Other options: `--model naive_bayes`, `--model random_forest`

This will:
- Load and clean the data
- Split into train/test sets
- Vectorize text with TF-IDF
- Train the chosen model
- Print a classification report + confusion matrix
- Save the model and vectorizer to `models/`

Try training all three and comparing their reports — this is a great thing
to include in a project writeup or portfolio README.

## 4. Run the web app

```bash
streamlit run app.py
```

Paste in some article text, pick a model, and hit **Analyze**.

## 5. Ideas to extend this project

- **Explainability**: show which words most influenced the prediction
  (e.g. `model.coef_` for Logistic Regression, or use SHAP).
- **Transformer model**: fine-tune `distilbert-base-uncased` with Hugging Face
  `transformers` for a stronger (if slower) model.
- **Source credibility check**: cross-reference the article's domain against
  a list of known reliable/unreliable sources.
- **Deploy it**: Streamlit Community Cloud (free) is the easiest way to put
  this online and share a live link.

## ⚠️ A note on limitations

This model detects *writing-style patterns* correlated with fake news in its
training data — it does not fact-check claims or verify sources. It can be
confidently wrong, especially on topics or writing styles not represented in
the training data. Frame it in your writeup as a statistical pattern-detector,
not a truth oracle — that framing is more accurate and, honestly, more
impressive to anyone reviewing the project.
