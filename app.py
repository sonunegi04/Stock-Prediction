import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Prediction App",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Beginner Stock Prediction App")
st.write(
    "This app allows you to explore historical stock data and predict the next day's closing price "
    "using multiple **Machine Learning models**. Compare models, tune hyperparameters, and make predictions!"
)

# ─── Sidebar ─────────────────────────────────────────────────────────────────
st.sidebar.header("User Input")
ticker_symbol = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL, GOOGL, MSFT, TSLA)", "AAPL")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Hyperparameter Tuning")
run_tuning = st.sidebar.checkbox("Enable Hyperparameter Tuning (GridSearchCV)", value=False)

# ─── Data Loading ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data(ticker, start, end):
    stock = yf.Ticker(ticker)
    data = stock.history(start=start, end=end)
    return data

data_load_state = st.text("Fetching data from Yahoo Finance...")
data = load_data(ticker_symbol, start_date, end_date)
data_load_state.empty()

if data.empty:
    st.error("❌ No data found. Please check the ticker symbol or date range.")
    st.stop()

# ─── Section 1: Raw Data & EDA ────────────────────────────────────────────────
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader(f"Raw Data for {ticker_symbol}")
    st.dataframe(data[["Open", "High", "Low", "Close", "Volume"]].tail(10))

with col2:
    st.subheader("Historical Close Price")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(data.index, data["Close"], label="Close Price", color="royalblue", linewidth=1.2)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    st.pyplot(fig)
    plt.close()

st.divider()

# ─── Section 2: EDA – Moving Averages & Volume ───────────────────────────────
st.header("🔍 Exploratory Data Analysis")

eda_col1, eda_col2 = st.columns(2)

with eda_col1:
    st.subheader("Close Price with Moving Averages")
    eda_df = data.copy()
    eda_df["SMA_10"] = eda_df["Close"].rolling(10).mean()
    eda_df["SMA_50"] = eda_df["Close"].rolling(50).mean()
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    ax3.plot(eda_df.index, eda_df["Close"], label="Close", color="royalblue", linewidth=1)
    ax3.plot(eda_df.index, eda_df["SMA_10"], label="SMA 10", color="orange", linewidth=1.2, linestyle="--")
    ax3.plot(eda_df.index, eda_df["SMA_50"], label="SMA 50", color="green", linewidth=1.2, linestyle="--")
    ax3.set_xlabel("Date"); ax3.set_ylabel("Price (USD)")
    ax3.legend(); ax3.grid(True, linestyle="--", alpha=0.5)
    st.pyplot(fig3); plt.close()

with eda_col2:
    st.subheader("Daily Trading Volume")
    fig4, ax4 = plt.subplots(figsize=(8, 4))
    ax4.bar(data.index, data["Volume"], color="steelblue", alpha=0.6, width=2)
    ax4.set_xlabel("Date"); ax4.set_ylabel("Volume")
    ax4.grid(True, linestyle="--", alpha=0.5)
    st.pyplot(fig4); plt.close()

# Daily returns distribution
st.subheader("Daily Return Distribution")
daily_returns = data["Close"].pct_change().dropna() * 100
fig5, ax5 = plt.subplots(figsize=(10, 3))
ax5.hist(daily_returns, bins=80, color="royalblue", alpha=0.7, edgecolor="white")
ax5.axvline(daily_returns.mean(), color="red", linestyle="--", linewidth=1.5, label=f"Mean: {daily_returns.mean():.2f}%")
ax5.set_xlabel("Daily Return (%)"); ax5.set_ylabel("Frequency")
ax5.legend(); ax5.grid(True, linestyle="--", alpha=0.5)
st.pyplot(fig5); plt.close()

st.divider()

# ─── Section 3: Feature Engineering ─────────────────────────────────────────
st.header("🛠️ Feature Engineering")
st.write("We engineer the following features for our ML models:")

feat_col1, feat_col2, feat_col3 = st.columns(3)
with feat_col1:
    st.info("**Price Features**\n\nClose, Open, High, Low prices as direct signals.")
with feat_col2:
    st.info("**Moving Averages**\n\nSMA 10 & SMA 50 – short and mid-term trend smoothing.")
with feat_col3:
    st.info("**Volatility & Returns**\n\nDaily % return and 10-day rolling standard deviation (volatility).")

df = data.copy()
df["SMA_10"]    = df["Close"].rolling(10).mean()
df["SMA_50"]    = df["Close"].rolling(50).mean()
df["Daily_Return"] = df["Close"].pct_change()
df["Volatility"] = df["Daily_Return"].rolling(10).std()
df["Target"]    = df["Close"].shift(-1)  # Next day's close
df_clean = df.dropna()

features = ["Close", "Open", "High", "Low", "SMA_10", "SMA_50", "Daily_Return", "Volatility"]
X = df_clean[features]
y = df_clean["Target"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Feature scaling
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

st.write(f"**Dataset size:** {len(df_clean)} samples | **Training:** {len(X_train)} | **Testing:** {len(X_test)}")

st.divider()

# ─── Section 4: Multiple ML Models ──────────────────────────────────────────
st.header("🤖 Predict Next Day's Price")
st.write(
    "We train and compare **3 Machine Learning models**: Linear Regression, "
    "Decision Tree, and Random Forest Regressor. "
    "Features used: **Close, Open, High, Low, SMA 10, SMA 50, Daily Return, Volatility**."
)

if len(df_clean) < 60:
    st.warning("⚠️ Not enough data to train. Please select a wider date range (at least a few months).")
    st.stop()

# ── Define models ────────────────────────────────────────────────────────────
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree":     DecisionTreeRegressor(max_depth=5, random_state=42),
    "Random Forest":     RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42),
}

results = {}
predictions_dict = {}

for name, model in models.items():
    # Use scaled features for Linear Regression; raw for tree models
    X_tr = X_train_sc if name == "Linear Regression" else X_train
    X_te = X_test_sc  if name == "Linear Regression" else X_test

    model.fit(X_tr, y_train)
    preds = model.predict(X_te)
    predictions_dict[name] = preds

    mae  = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2   = r2_score(y_test, preds)
    results[name] = {"MAE": mae, "RMSE": rmse, "R²": r2}

# ─── Model Evaluation Table ───────────────────────────────────────────────────
st.subheader("📊 Model Evaluation (Testing on last 20% of data)")

results_df = pd.DataFrame(results).T.reset_index().rename(columns={"index": "Model"})
results_df = results_df.round(4)

# Highlight best model
best_model_name = results_df.loc[results_df["R²"].idxmax(), "Model"]

def highlight_best(row):
    color = "background-color: #d4edda" if row["Model"] == best_model_name else ""
    return [color] * len(row)

st.dataframe(
    results_df.style.apply(highlight_best, axis=1),
    use_container_width=True
)
st.success(f"✅ **Best Model:** {best_model_name} (Highest R² Score = {results[best_model_name]['R²']:.4f})")

# ─── Prediction Charts for all models ─────────────────────────────────────────
st.subheader("📈 Predicted vs Actual Price – All Models")

fig2, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
colors = ["orange", "green", "red"]

for ax, (name, preds), color in zip(axes, predictions_dict.items(), colors):
    ax.plot(y_test.index, y_test.values, label="Actual", color="royalblue", linewidth=1.2)
    ax.plot(y_test.index, preds, label=f"Predicted", color=color, linestyle="--", linewidth=1.2)
    ax.set_title(f"{name}\nR²={results[name]['R²']:.4f}  MAE={results[name]['MAE']:.2f}")
    ax.set_xlabel("Date"); ax.set_ylabel("Price (USD)")
    ax.legend(); ax.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
st.pyplot(fig2)
plt.close()

st.divider()

# ─── Section 5: Hyperparameter Tuning ────────────────────────────────────────
st.header("🔧 Hyperparameter Tuning")
st.write(
    "We use **GridSearchCV** with 5-fold cross-validation to find the best hyperparameters "
    "for the Random Forest model."
)

if run_tuning:
    st.info("Running GridSearchCV… this may take a moment ⏳")
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth":    [5, 8, 12],
        "min_samples_split": [2, 5],
    }
    rf_base = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(rf_base, param_grid, cv=5, scoring="r2", n_jobs=-1, verbose=0)
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    best_rf     = grid_search.best_estimator_
    tuned_preds = best_rf.predict(X_test)
    tuned_r2    = r2_score(y_test, tuned_preds)
    tuned_mae   = mean_absolute_error(y_test, tuned_preds)
    tuned_rmse  = np.sqrt(mean_squared_error(y_test, tuned_preds))

    st.success(f"**Best Parameters Found:** {best_params}")
    col_t1, col_t2, col_t3 = st.columns(3)
    col_t1.metric("R² (Tuned RF)", f"{tuned_r2:.4f}", delta=f"{tuned_r2 - results['Random Forest']['R²']:+.4f} vs default")
    col_t2.metric("MAE (Tuned RF)", f"${tuned_mae:.2f}", delta=f"{tuned_mae - results['Random Forest']['MAE']:+.2f} vs default")
    col_t3.metric("RMSE (Tuned RF)", f"${tuned_rmse:.2f}", delta=f"{tuned_rmse - results['Random Forest']['RMSE']:+.2f} vs default")

    fig_t, ax_t = plt.subplots(figsize=(12, 4))
    ax_t.plot(y_test.index, y_test.values, label="Actual", color="royalblue", linewidth=1.2)
    ax_t.plot(y_test.index, tuned_preds, label="Tuned RF Predicted", color="purple", linestyle="--", linewidth=1.2)
    ax_t.set_xlabel("Date"); ax_t.set_ylabel("Price (USD)")
    ax_t.legend(); ax_t.grid(True, linestyle="--", alpha=0.5)
    ax_t.set_title("Tuned Random Forest – Predicted vs Actual")
    st.pyplot(fig_t); plt.close()

    # Update best model if tuned RF is better
    if tuned_r2 > results[best_model_name]["R²"]:
        best_model_name = "Tuned Random Forest"
        best_model_obj  = best_rf
        use_scaled      = False
    else:
        best_model_obj = models[best_model_name]
        use_scaled     = best_model_name == "Linear Regression"
else:
    st.info("Enable **Hyperparameter Tuning** in the sidebar to run GridSearchCV.")
    # CV scores on default models
    st.subheader("Cross-Validation Scores (5-Fold) – Default Models")
    cv_results = {}
    for name, model in models.items():
        X_cv = X_train_sc if name == "Linear Regression" else X_train
        cv_scores = cross_val_score(model, X_cv, y_train, cv=5, scoring="r2")
        cv_results[name] = {"Mean R²": cv_scores.mean().round(4), "Std R²": cv_scores.std().round(4)}
    cv_df = pd.DataFrame(cv_results).T.reset_index().rename(columns={"index": "Model"})
    st.dataframe(cv_df, use_container_width=True)

    best_model_obj = models[best_model_name]
    use_scaled     = best_model_name == "Linear Regression"

st.divider()

# ─── Section 6: Feature Importance ───────────────────────────────────────────
st.header("📌 Feature Importance (Random Forest)")
rf_model = models["Random Forest"]
importances = pd.Series(rf_model.feature_importances_, index=features).sort_values(ascending=True)

fig_fi, ax_fi = plt.subplots(figsize=(8, 4))
importances.plot(kind="barh", ax=ax_fi, color="steelblue", edgecolor="white")
ax_fi.set_xlabel("Importance"); ax_fi.set_title("Feature Importance – Random Forest")
ax_fi.grid(True, linestyle="--", alpha=0.5, axis="x")
st.pyplot(fig_fi); plt.close()

st.divider()

# ─── Section 7: Tomorrow's Prediction ────────────────────────────────────────
st.header("🔮 Prediction for Tomorrow")
st.write(f"Using the **{best_model_name}** model (best performer) to predict the next day's closing price.")

latest = data.copy()
latest["SMA_10"]      = latest["Close"].rolling(10).mean()
latest["SMA_50"]      = latest["Close"].rolling(50).mean()
latest["Daily_Return"] = latest["Close"].pct_change()
latest["Volatility"]   = latest["Daily_Return"].rolling(10).std()

current_row = latest[features].iloc[-1:]

if current_row.isnull().values.any():
    st.warning("⚠️ Latest data has missing values – using Random Forest as fallback.")
    next_pred = models["Random Forest"].predict(current_row.fillna(0))[0]
else:
    if use_scaled:
        next_pred = best_model_obj.predict(scaler.transform(current_row))[0]
    else:
        next_pred = best_model_obj.predict(current_row)[0]

last_close   = latest["Close"].iloc[-1]
diff         = next_pred - last_close
diff_percent = (diff / last_close) * 100

pred_col1, pred_col2, pred_col3 = st.columns(3)
pred_col1.metric(
    label=f"Predicted Closing Price for {ticker_symbol}",
    value=f"${next_pred:.2f}",
    delta=f"{diff:+.2f} ({diff_percent:+.2f}%) vs last close"
)
pred_col2.metric("Last Close", f"${last_close:.2f}")
pred_col3.metric("Model Used", best_model_name)

st.caption(
    "⚠️ **Disclaimer:** This is an educational ML project for CSE247 – Applied Machine Learning. "
    "It demonstrates data preprocessing, feature engineering, model comparison, and hyperparameter tuning. "
    "**Do not use this for actual financial trading decisions.**"
)
