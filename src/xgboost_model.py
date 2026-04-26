import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os

DATA_DIR = "data"
OUT_DIR  = "outputs/plots"
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(f"{DATA_DIR}/ts_ca1_foods.csv", index_col="date", parse_dates=True)
df = df.asfreq("D").bfill()

# Feature engineering
df["lag_1"]       = df["sales"].shift(1)
df["lag_7"]       = df["sales"].shift(7)
df["lag_14"]      = df["sales"].shift(14)
df["lag_28"]      = df["sales"].shift(28)
df["rolling_7"]   = df["sales"].shift(1).rolling(7).mean()
df["rolling_28"]  = df["sales"].shift(1).rolling(28).mean()
df["dayofweek"]   = df.index.dayofweek
df["month"]       = df.index.month
df["dayofmonth"]  = df.index.day

df = df.dropna()

features = ["lag_1", "lag_7", "lag_14", "lag_28", "rolling_7", "rolling_28", "dayofweek", "month", "dayofmonth"]
target   = "sales"

# Train/test split — last 90 days
train = df.iloc[:-90]
test  = df.iloc[-90:]

X_train, y_train = train[features], train[target]
X_test,  y_test  = test[features],  test[target]

print(f"Train size : {len(train)}")
print(f"Test size  : {len(test)}")

# Fit XGBoost
print("\nFitting XGBoost model...")
model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
model.fit(X_train, y_train)
print("Done.")

pred = model.predict(X_test)

# Metrics
mae  = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
mape = np.mean(np.abs((y_test.values - pred) / y_test.values)) * 100

print(f"\nXGBoost Results")
print(f"  MAE  : {round(mae, 2)}")
print(f"  RMSE : {round(rmse, 2)}")
print(f"  MAPE : {round(mape, 2)}%")

# Save
pd.DataFrame([{"model": "XGBoost", "MAE": mae, "RMSE": rmse, "MAPE": mape}]).to_csv(f"{DATA_DIR}/xgboost_metrics.csv", index=False)
pd.DataFrame({"actual": y_test.values, "predicted": pred}, index=test.index).to_csv(f"{DATA_DIR}/xgboost_predictions.csv")

# Feature importance
plt.figure(figsize=(8, 5))
importance = pd.Series(model.feature_importances_, index=features).sort_values()
importance.plot(kind="barh", color="steelblue")
plt.title("XGBoost Feature Importance")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/xgboost_importance.png", dpi=150)
plt.show()

# Forecast plot
plt.figure(figsize=(14, 5))
plt.plot(test.index, y_test.values, color="black", label="Actual", linewidth=1)
plt.plot(test.index, pred, color="darkorange", label="XGBoost Forecast", linewidth=1.2, linestyle="--")
plt.title("XGBoost Forecast vs Actual — CA_1 FOODS")
plt.xlabel("Date")
plt.ylabel("Units Sold")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/xgboost_forecast.png", dpi=150)
plt.show()
print("Plot saved.")