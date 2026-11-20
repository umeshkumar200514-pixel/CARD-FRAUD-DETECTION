# 💳 Debit/Credit Card Fraud Detection using Machine Learning

A B.Tech final-year project that detects fraudulent card transactions with machine learning, explains every prediction with **SHAP (Explainable AI)**, and lets users ask an **AI assistant** why a transaction was flagged, all inside a **Streamlit** web app.

## Features

- **Feature engineering** from raw transactions: time of day, customer age, distance from home, "impossible travel" speed, amount compared with the card's usual spending, and transaction velocity (count in the last 1 h / 24 h). All history features use **past transactions only**, so there is no data leakage.
- **Class-imbalance handling with SMOTE**: compares no balancing, class weights, under-sampling, and under-sampling + SMOTENC. By default the final model uses **under-sampling + SMOTENC**, which creates about 24,000 synthetic fraud transactions for training; samples of them are shown in the app.
- **Five models compared**: Logistic Regression, Decision Tree, Random Forest, XGBoost, KNN.
- **Fair evaluation**: time-based validation split; the model and decision threshold are chosen on validation data, and the test set is used only to report results; metrics are PR-AUC, precision, recall and F1 (not accuracy).
- **Explainable AI**: SHAP shows which features pushed each prediction towards fraud or genuine.
- **Streamlit app with four pages**:
  1. **Check one transaction**: fraud score, risk level, warning signs and a SHAP chart. Includes **real fraud and genuine transactions from the test set** as one-click examples, each showing its true label
  2. **Batch check (CSV)**: upload many transactions (or use the built-in demo file), get predictions and metrics, download the results
  3. **AI assistant**: explains the last result in plain language (Gemini free tier, Claude, or offline mode)
  4. **Model & dashboard**: model comparison, SMOTE before/after counts with synthetic examples, confusion matrix, feature importance, SHAP summary, EDA

## Project structure

```
card-fraud-detection/
├── app.py                       # Streamlit web app (run this)
├── fraud_utils.py               # shared feature engineering, prediction and SHAP code
├── chatbot.py                   # AI assistant (Gemini / Claude / offline)
├── train_model.py               # trains the model from the terminal (main way)
├── check_model.py               # checks the trained model on fraudTest.csv
├── card_fraud_detection.ipynb   # same training pipeline as a notebook, step by step
├── requirements.txt             # Python packages
├── data/                        # put fraudTrain.csv and fraudTest.csv here
├── models/                      # best_model.pkl is created here by training
├── reports/                     # charts and tables created by training
└── .vscode/launch.json          # one-click run buttons for VS Code
```

### How the files connect

```
data/*.csv ──► train_model.py (or the notebook) ──► models/best_model.pkl + reports/
                    │ uses                                    │
                    ▼                                         ▼
              fraud_utils.py  ◄──────── uses ──────────  app.py ──► chatbot.py
```

`train_model.py` and `app.py` both use the feature code in `fraud_utils.py`, so the app always builds exactly the features the model was trained on.

## Setup

**Requirements:** Python 3.10, 3.11 or 3.12, and VS Code with the **Python** and **Jupyter** extensions.

### 1. Open the project
In VS Code: **File → Open Folder** → select the `card-fraud-detection` folder. Open a terminal with **Terminal → New Terminal**.

### 2. Create a virtual environment and install packages

**Windows (Command Prompt or PowerShell):**
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

**Mac / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

You should see `(venv)` at the start of the terminal line. Then press **Ctrl+Shift+P → "Python: Select Interpreter"** and pick the one inside `venv`.

> **PowerShell error "running scripts is disabled"?** Run this once, then try activating again:
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

### 3. Download the dataset
1. Go to <https://www.kaggle.com/datasets/kartik2112/fraud-detection> (free Kaggle account needed).
2. Download and unzip it.
3. Copy `fraudTrain.csv` and `fraudTest.csv` into the `data/` folder.

## Run the project

### Step 1: Train the model

Quick trial (a few minutes, good for checking that everything works):
```bash
python train_model.py --nrows 200000
```

Full training (best results, can take 20–60 minutes depending on your computer):
```bash
python train_model.py
```

Optional flags: `--no-knn` skips KNN, and `--data-dir path/to/folder` reads the CSVs from another folder.

Optional flags:
- `--balance auto` lets validation PR-AUC choose the balancing method instead of SMOTE.
- `--balance under` (or `none`, `class_weight`, `under+smote`) forces a method.

Training creates:
- `models/best_model.pkl`: the trained model, scaler, encoder and threshold
- `models/demo_examples.json`: 6 real fraud and 3 real genuine test transactions for the app, each re-checked through the app's own code
- `reports/`: comparison tables and charts, `balancing_summary.json`, `smote_synthetic_examples.csv` (50 synthetic fraud rows in real units) and `demo_batch.csv` (a small file for the Batch page)

To check the model afterwards:
```bash
python check_model.py
```

**Alternative:** open `card_fraud_detection.ipynb` in VS Code, select the `venv` kernel (top right), and click **Run All**. It trains the same models step by step, but it does not create the demo examples or SMOTE samples, so use `train_model.py` for the app.

### Step 2: Start the web app
```bash
streamlit run app.py
```
The app opens in your browser at <http://localhost:8501>. Press **Ctrl+C** in the terminal to stop it.

### Run with one click in VS Code
Open the **Run and Debug** panel (Ctrl+Shift+D), choose one of these, and press ▶ (or F5):
- **1. Train model (quick, 200k rows)**
- **2. Train model (full data)**
- **3. Run Streamlit app**

## AI assistant setup (optional)

The assistant works in three modes:

| Mode | Cost | How |
|---|---|---|
| **Gemini** | Free tier | Get a key at <https://aistudio.google.com> → *Get API key* |
| **Claude** | Paid (pay per use) | Get a key at <https://console.anthropic.com> |
| **Offline** | Free | No key needed; answers "why flagged", threshold, SHAP and balancing questions |

Paste the key into the sidebar on the **AI assistant** page, or set it before starting the app:

```bash
# Windows Command Prompt
set GEMINI_API_KEY=your-key-here
# Windows PowerShell
$env:GEMINI_API_KEY="your-key-here"
# Mac / Linux
export GEMINI_API_KEY=your-key-here
```
(For Claude, use `ANTHROPIC_API_KEY` instead.)

⚠️ Never put API keys inside code files, screenshots or GitHub.

## How to demo
1. **Check one transaction** → choose *"Real FRAUD #1 from test set ..."* → **Check transaction**. The form fills in every feature of a real fraud transaction the model never saw, and the app shows **⚠ FRAUDULENT** with its true label.
2. Look at the fraud score, warning signs and the SHAP chart, then open *"Features sent to the model"* to show all 20 features.
3. Choose *"Real GENUINE #1 ..."* to show a genuine result for comparison.
4. Open **AI assistant** → ask *"Why was this transaction flagged?"*
5. **Batch check (CSV)** → click *"Or use the demo file"* to check thousands of real transactions at once, with precision, recall and the confusion matrix.
6. **Model & dashboard** → show the model comparison, the SMOTE before/after table and the synthetic fraud examples.

## Troubleshooting

| Problem | Fix |
|---|---|
| `models/best_model.pkl not found` | Run `python train_model.py` first, from the project folder |
| `File not found: data/fraudTrain.csv` | Put both CSV files in the `data/` folder |
| `ModuleNotFoundError` | Activate the venv (`(venv)` must show) and run `pip install -r requirements.txt` |
| `'streamlit' is not recognized` | Activate the venv, or use `python -m streamlit run app.py` |
| Training runs out of memory | Use `--nrows 500000` (or smaller) |
| Errors when loading the model | Train and run the app in the same venv (package versions must match) |
| Gemini error 429 | Free-tier rate limit; wait a minute and try again |
| Gemini model not found | Copy a current free model name from AI Studio into the sidebar "Gemini model" box |

## Limitations
- The Sparkov dataset is **simulated**, so scores will be higher than on real bank data.
- A fraud flag is a **warning for review**, not proof of fraud.
- On the single-transaction page, card history (average amount, recent transactions) is typed by hand; a real system would fetch it from a database.
- The "Card type" field is for display only, because the dataset has no card-type column.

## Tech stack
Python · pandas · NumPy · scikit-learn · XGBoost · imbalanced-learn · SHAP · Matplotlib / Seaborn · Streamlit · Google Gemini API / Anthropic Claude API
