import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
import os

warnings.filterwarnings("ignore")

DATA_DIR = "data"
OUT_DIR  = "outputs/plots"
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(f"{DATA_DIR}/ts_ca1_foods.csv", index_col="date", parse_dates=True)
df = df.asfreq("D").fillna(method="ffill")

# Train/test split — last 90 days as test
train = df.iloc[:-90]
test  = df.iloc[-90:]

print(f"Train size : {len(train)}")
print(f"Test size  : {len(test)}")

# Fit SARIMA(1,1,1)(1,1,0)[7] — captures weekly seasonality
print("\nFitting SARIMA model...")
model = SARIMAX(
    train["sales"],
    order=(1, 1, 1),
    seasonal_order=(1, 1, 0, 7),
    enforce_stationarity=False,
    enforce_invertibility=False
)
fitted = model.fit(disp=False)
print("Done.")

# Forecast
forecast = fitted.forecast(steps=90)

# Metrics
mae  = mean_absolute_error(test["sales"], forecast)
rmse = np.sqrt(mean_squared_error(test["sales"], forecast))
mape = np.mean(np.abs((test["sales"].values - forecast.values) / test["sales"].values)) * 100

print(f"\nARIMA Results")
print(f"  MAE  : {round(mae, 2)}")
print(f"  RMSE : {round(rmse, 2)}")
print(f"  MAPE : {round(mape, 2)}%")

# Save metrics
metrics = {"model": "ARIMA", "MAE": mae, "RMSE": rmse, "MAPE": mape}
pd.DataFrame([metrics]).to_csv(f"{DATA_DIR}/arima_metrics.csv", index=False)

# Save predictions
pred_df = pd.DataFrame({"actual": test["sales"].values, "predicted": forecast.values}, index=test.index)
pred_df.to_csv(f"{DATA_DIR}/arima_predictions.csv")

# Plot
plt.figure(figsize=(14, 5))
plt.plot(train.index[-90:], train["sales"].iloc[-90:], color="steelblue", label="Train (last 90d)", linewidth=0.8)
plt.plot(test.index, test["sales"], color="black", label="Actual", linewidth=1)
plt.plot(test.index, forecast, color="crimson", label="ARIMA Forecast", linewidth=1.2, linestyle="--")
plt.title("ARIMA Forecast vs Actual — CA_1 FOODS")
plt.xlabel("Date")
plt.ylabel("Units Sold")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/arima_forecast.png", dpi=150)
plt.show()
print("Plot saved.")