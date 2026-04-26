import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

DATA_DIR = "data"
OUT_DIR  = "outputs/plots"
os.makedirs(OUT_DIR, exist_ok=True)

# Load all metrics
arima   = pd.read_csv(f"{DATA_DIR}/arima_metrics.csv")
prophet = pd.read_csv(f"{DATA_DIR}/prophet_metrics.csv")
xgb     = pd.read_csv(f"{DATA_DIR}/xgboost_metrics.csv")

metrics = pd.concat([arima, prophet, xgb], ignore_index=True)
metrics = metrics.set_index("model")
print(metrics.round(2))
metrics.to_csv(f"{DATA_DIR}/all_metrics.csv")

# Load predictions
arima_pred   = pd.read_csv(f"{DATA_DIR}/arima_predictions.csv",   index_col=0, parse_dates=True)
prophet_pred = pd.read_csv(f"{DATA_DIR}/prophet_predictions.csv", index_col=0, parse_dates=True)
xgb_pred     = pd.read_csv(f"{DATA_DIR}/xgboost_predictions.csv", index_col=0, parse_dates=True)

# Bar chart — metric comparison
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
colors = ["steelblue", "seagreen", "darkorange"]

for i, metric in enumerate(["MAE", "RMSE", "MAPE"]):
    axes[i].bar(metrics.index, metrics[metric], color=colors)
    axes[i].set_title(metric)
    axes[i].set_ylabel(metric)
    for j, val in enumerate(metrics[metric]):
        axes[i].text(j, val + 1, str(round(val, 2)), ha="center", fontsize=9)

fig.suptitle("Model Comparison — ARIMA vs Prophet vs XGBoost", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/model_comparison_bar.png", dpi=150)
plt.show()

# Overlay forecast plot
plt.figure(figsize=(14, 5))
plt.plot(arima_pred.index, arima_pred["actual"], color="black", label="Actual", linewidth=1)
plt.plot(arima_pred.index, arima_pred["predicted"], color="crimson", label="ARIMA", linewidth=1, linestyle="--")
plt.plot(prophet_pred.index, prophet_pred["predicted"], color="seagreen", label="Prophet", linewidth=1, linestyle="--")
plt.plot(xgb_pred.index, xgb_pred["predicted"], color="darkorange", label="XGBoost", linewidth=1, linestyle="--")
plt.title("All Models vs Actual — CA_1 FOODS (Test Period)")
plt.xlabel("Date")
plt.ylabel("Units Sold")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/model_comparison_overlay.png", dpi=150)
plt.show()
print("Plots saved.")