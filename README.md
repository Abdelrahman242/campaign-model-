# 📊 Ad Campaign Predictor

A Streamlit web application that uses a pre-trained **SVC (Support Vector Classifier)** pipeline to predict ad campaign outcomes based on five input features.

---

## Features

| Input | Type | Description |
|-------|------|-------------|
| `Channel` | Categorical | Marketing channel (e.g., Facebook, Google Ads, Instagram, TikTok, Email) |
| `Impressions` | Numeric | Total number of ad impressions |
| `Clicks` | Numeric | Total number of clicks on the ad |
| `Spend` | Numeric (float) | Total money spent on the campaign ($) |
| `Active` | Categorical | Whether the campaign is currently active |

**Output:** Binary classification — `Class 1` (Positive) or `Class 0` (Negative).

---

## Project Structure

```
209/
├── app.py            # Streamlit application
├── model.pkl         # Pre-trained sklearn Pipeline (SVC)
├── requirements.txt  # Python dependencies
└── README.md         # This file
```

---

## Setup & Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Launch the app

```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

---

## Model Details

- **Model type:** `sklearn.pipeline.Pipeline`
- **Classifier:** `SVC` (Support Vector Classifier)
- **Preprocessing:**
  - Numeric features (`Impressions`, `Clicks`, `Spend`): median imputation → standard scaling
  - Categorical features (`Channel`, `Active`): most-frequent imputation → one-hot encoding
- **Output classes:** `0` (negative), `1` (positive)

> ⚠️ `model.pkl` is **read-only**. The app only performs inference — it does not retrain or modify the model.
