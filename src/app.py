import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.stats.diagnostic import acorr_ljungbox
import warnings
import os

warnings.filterwarnings("ignore")

DATA_DIR = "data"

st.set_page_config(page_title="Demand Forecasting System", layout="wide")
st.title("Demand Forecasting System")
st.caption("Walmart M5 — CA_1 Store, FOODS Category | Statistical Validation of Model Assumptions")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv(f"{DATA_DIR}/ts_ca1_foods.csv", index_col="date", parse_dates=True)
    df = df.asfreq("D").bfill()
    return df

df = load_data()

# Sidebar
st.sidebar.header("Settings")
model_choice = st.sidebar.selectbox("Select Model", ["XGBoost", "ARIMA", "Prophet"])
test_days    = st.sidebar.slider("Test Period (days)", min_value=30, max_value=180, value=90, step=10)
show_decomp  = st.sidebar.checkbox("Show Decomposition", value=False)
show_resid   = st.sidebar.checkbox("Show Residual Analysis", value=False)

train = df.iloc[:-test_days]
test  = df.iloc[-test_days:]

# Raw data chart
st.subheader("Raw Sales Data")
st.line_chart(df["sales"])

# Decomposition
if show_decomp:
    st.subheader("STL Decomposition")
    result = seasonal_decompose(df["sales"], model="additive", period=7)
    fig, axes = plt.subplots(4, 1, figsize=(12, 8))
    axes[0].plot(df.index, result.observed,  color="steelblue",  linewidth=0.7); axes[0].set_title("Observed")
    axes[1].plot(df.index, result.trend,     color="darkorange", linewidth=1.0); axes[1].set_title("Trend")
    axes[2].plot(df.index, result.seasonal,  color="seagreen",   linewidth=0.7); axes[2].set_title("Seasonality")
    axes[3].plot(df.index, result.resid,     color="crimson",    linewidth=0.5); axes[3].axhline(0, color="black", linestyle="--"); axes[3].set_title("Residual")
    plt.tight_layout()
    st.pyplot(fig)

# Model fitting
st.subheader(f"{model_choice} Forecast")

with st.spinner(f"Fitting {model_choice} model..."):

    if model_choice == "ARIMA":
        model = SARIMAX(train["sales"], order=(1,1,1), seasonal_order=(1,1,0,7),
                        enforce_stationarity=False, enforce_invertibility=False)
        fitted = model.fit(disp=False)
        pred   = fitted.forecast(steps=test_days).values

    elif model_choice == "Prophet":
        prophet_train = train.reset_index().rename(columns={"date":"ds","sales":"y"})
        m = Prophet(weekly_seasonality=True, yearly_seasonality=True, seasonality_mode="additive")
        m.fit(prophet_train)
        future = m.make_future_dataframe(periods=test_days)
        pred   = m.predict(future).iloc[-test_days:]["yhat"].values

    elif model_choice == "XGBoost":
        def make_features(df):
            df = df.copy()
            df["lag_1"]      = df["sales"].shift(1)
            df["lag_7"]      = df["sales"].shift(7)
            df["lag_14"]     = df["sales"].shift(14)
            df["lag_28"]     = df["sales"].shift(28)
            df["rolling_7"]  = df["sales"].shift(1).rolling(7).mean()
            df["rolling_28"] = df["sales"].shift(1).rolling(28).mean()
            df["dayofweek"]  = df.index.dayofweek
            df["month"]      = df.index.month
            df["dayofmonth"] = df.index.day
            return df.dropna()

        features = ["lag_1","lag_7","lag_14","lag_28","rolling_7","rolling_28","dayofweek","month","dayofmonth"]
        full_featured = make_features(df)
        tr = full_featured.iloc[:-test_days]
        te = full_featured.iloc[-test_days:]
        xgb = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=5,
                           subsample=0.8, colsample_bytree=0.8, random_state=42)
        xgb.fit(tr[features], tr["sales"])
        pred = xgb.predict(te[features])
        test = te[["sales"]].rename(columns={"sales":"sales"})

actual = test["sales"].values[:len(pred)]

# Metrics
mae  = mean_absolute_error(actual, pred)
rmse = np.sqrt(mean_squared_error(actual, pred))
mape = np.mean(np.abs((actual - pred) / actual)) * 100

col1, col2, col3 = st.columns(3)
col1.metric("MAE",  round(mae, 2))
col2.metric("RMSE", round(rmse, 2))
col3.metric("MAPE", f"{round(mape, 2)}%")

# Forecast chart
fig2, ax = plt.subplots(figsize=(12, 4))
ax.plot(test.index[:len(pred)], actual, color="black",      label="Actual",   linewidth=1)
ax.plot(test.index[:len(pred)], pred,   color="darkorange", label=f"{model_choice} Forecast", linewidth=1.2, linestyle="--")
ax.set_title(f"{model_choice} Forecast vs Actual")
ax.set_xlabel("Date")
ax.set_ylabel("Units Sold")
ax.legend()
st.pyplot(fig2)

# Model comparison table
st.subheader("Model Comparison")
try:
    all_metrics = pd.read_csv(f"{DATA_DIR}/all_metrics.csv")
    st.dataframe(all_metrics.set_index("model").round(2), use_container_width=True)
except:
    st.info("Run model_comparison.py first to see full comparison table.")

# Residual analysis
if show_resid:
    st.subheader("Residual Analysis")
    residuals = pd.Series(actual - pred)

    lb = acorr_ljungbox(residuals, lags=[10, 20], return_df=True)
    lb_result = "WHITE NOISE — assumptions satisfied" if (lb["lb_pvalue"] > 0.05).all() else "Autocorrelation detected — model missed some pattern"

    st.write(f"**Ljung-Box Result:** {lb_result}")
    st.dataframe(lb.round(4))

    fig3, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(residuals.values, color="crimson", linewidth=0.7)
    axes[0].axhline(0, color="black", linestyle="--")
    axes[0].set_title("Residuals Over Time")

    axes[1].hist(residuals, bins=30, color="steelblue", edgecolor="white")
    axes[1].set_title("Residual Distribution")

    plt.tight_layout()
    st.pyplot(fig3)

st.sidebar.markdown("---")
st.sidebar.caption("Demand Forecasting System | Walmart M5 Dataset")