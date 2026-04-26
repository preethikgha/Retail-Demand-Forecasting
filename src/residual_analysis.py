import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf
import os

DATA_DIR = "data"
OUT_DIR  = "outputs/plots"
os.makedirs(OUT_DIR, exist_ok=True)

pred = pd.read_csv(f"{DATA_DIR}/xgboost_predictions.csv", index_col=0, parse_dates=True)
residuals = pred["actual"] - pred["predicted"]

print("Residual Summary")
print(f"  Mean   : {round(residuals.mean(), 2)}")
print(f"  Std    : {round(residuals.std(), 2)}")
print(f"  Min    : {round(residuals.min(), 2)}")
print(f"  Max    : {round(residuals.max(), 2)}")

# Ljung-Box test
lb = acorr_ljungbox(residuals, lags=[10, 20], return_df=True)
print("\nLjung-Box Test (are residuals autocorrelated?)")
print(lb)
if (lb["lb_pvalue"] > 0.05).all():
    print("Result : Residuals are WHITE NOISE — model assumptions satisfied")
else:
    print("Result : Residuals have autocorrelation — model missed some pattern")

# Plots
fig, axes = plt.subplots(2, 2, figsize=(14, 8))

# Residuals over time
axes[0, 0].plot(pred.index, residuals, color="crimson", linewidth=0.7)
axes[0, 0].axhline(0, color="black", linestyle="--", linewidth=0.8)
axes[0, 0].set_title("Residuals Over Time")
axes[0, 0].set_xlabel("Date")
axes[0, 0].set_ylabel("Error")

# Histogram
axes[0, 1].hist(residuals, bins=30, color="steelblue", edgecolor="white")
axes[0, 1].set_title("Residual Distribution")
axes[0, 1].set_xlabel("Error")
axes[0, 1].set_ylabel("Frequency")

# ACF of residuals
plot_acf(residuals, lags=30, ax=axes[1, 0])
axes[1, 0].set_title("ACF of Residuals")

# Actual vs Predicted scatter
axes[1, 1].scatter(pred["actual"], pred["predicted"], alpha=0.5, color="darkorange", s=20)
axes[1, 1].plot([pred["actual"].min(), pred["actual"].max()],
                [pred["actual"].min(), pred["actual"].max()],
                color="black", linestyle="--", linewidth=1)
axes[1, 1].set_title("Actual vs Predicted")
axes[1, 1].set_xlabel("Actual")
axes[1, 1].set_ylabel("Predicted")

fig.suptitle("Residual Analysis — XGBoost Model", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/residual_analysis.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nPlot saved.")