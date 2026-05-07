# 📈 Stock Price Prediction – CSE247 Applied Machine Learning

**Lovely Professional University | School of Computer Science and Engineering**

## Project Overview

A complete Machine Learning pipeline to predict next-day stock closing prices using historical OHLCV data and technical indicators.

## Requirements Met

| Requirement | Status |
|---|---|
| Problem Statement | ✅ |
| Data Preprocessing & EDA | ✅ |
| Minimum 3 ML Models | ✅ Linear Regression, Decision Tree, Random Forest |
| Model Evaluation & Comparison | ✅ MAE, RMSE, R² comparison table |
| Hyperparameter Tuning | ✅ GridSearchCV with 5-fold CV on Random Forest |
| Final Report / Notebook | ✅ |

## Installation

```bash
pip install -r requirements.txt
```

## Run the App

```bash
streamlit run app.py
```

## Project Structure

```
stock_prediction_project/
├── app.py                    # Streamlit web app
├── stock_prediction.ipynb    # Full ML pipeline notebook
├── requirements.txt
└── README.md
```

## Features Used

- Close, Open, High, Low prices
- SMA 10 & SMA 50 (Simple Moving Averages)
- Daily Return (%)
- 10-day Volatility

## Models

1. **Linear Regression** – baseline (with StandardScaler)
2. **Decision Tree Regressor** – max_depth=5
3. **Random Forest Regressor** – 100 estimators, max_depth=8
4. **Tuned Random Forest** – GridSearchCV optimal hyperparameters

## Disclaimer

Educational project for CSE247. Do NOT use for real financial trading.
