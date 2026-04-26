import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os

DATA_DIR = "data"
OUT_DIR  = "outputs/plots"
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(f"{DATA_DIR}/ts_ca1_foods.csv", index_col="date", parse_dates=True)
df = df.asfreq("D").fillna(method="ffill")

# Prophet needs columns named ds and y
prophet_df = df.reset_index().rename(columns={"date": "ds", "sales": "y"})

# Train/test split — last 90 days
train = prophet_df.iloc[:-90]
test  = prophet_df.iloc[-90:]

print(f"Train size : {len(train)}")
print(f"Test size  : {len(test)}")

# Fit Prophet
print("\nFitting Prophet model...")
model = Prophet(
    weekly_seasonality=True,
    yearly_seasonality=True,
    daily_seasonality=False,
    seasonality_mode="additive"
)
model.fit(train)
print("Done.")

# Forecast
future   = model.make_future_dataframe(periods=90)
forecast = model.predict(future)
pred     = forecast.iloc[-90:]["yhat"].values

# Metrics
actual = test["y"].values
mae  = mean_absolute_error(actual, pred)
rmse = np.sqrt(mean_squared_error(actual, pred))
mape = np.mean(np.abs((actual - pred) / actual)) * 100

print(f"\nProphet Results")
print(f"  MAE  : {round(mae, 2)}")
print(f"  RMSE : {round(rmse, 2)}")
print(f"  MAPE : {round(mape, 2)}%")

# Save metrics and predictions
pd.DataFrame([{"model": "Prophet", "MAE": mae, "RMSE": rmse, "MAPE": mape}]).to_csv(f"{DATA_DIR}/prophet_metrics.csv", index=False)
pd.DataFrame({"actual": actual, "predicted": pred}, index=test["ds"].values).to_csv(f"{DATA_DIR}/prophet_predictions.csv")

# Plot
plt.figure(figsize=(14, 5))
plt.plot(test["ds"].values, actual, color="black", label="Actual", linewidth=1)
plt.plot(test["ds"].values, pred, color="seagreen", label="Prophet Forecast", linewidth=1.2, linestyle="--")
plt.title("Prophet Forecast vs Actual — CA_1 FOODS")
plt.xlabel("Date")
plt.ylabel("Units Sold")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/prophet_forecast.png", dpi=150)
plt.show()
print("Plot saved.")