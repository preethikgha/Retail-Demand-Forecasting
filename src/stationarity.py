import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import os

DATA_DIR = "data"
OUT_DIR  = "outputs/plots"
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(f"{DATA_DIR}/ts_ca1_foods.csv", index_col="date", parse_dates=True)
df = df.asfreq("D").fillna(method="ffill")

def run_adf(series, label):
    result = adfuller(series.dropna())
    print(f"\nADF Test — {label}")
    print(f"  Test Statistic : {round(result[0], 4)}")
    print(f"  p-value        : {round(result[1], 6)}")
    print(f"  Critical (5%)  : {round(result[4]['5%'], 4)}")
    if result[1] < 0.05:
        print("  Result         : STATIONARY (p < 0.05)")
    else:
        print("  Result         : NON-STATIONARY (p >= 0.05)")
    return result[1]

p_raw = run_adf(df["sales"], "Raw Sales")

df["sales_diff"] = df["sales"].diff()
p_diff = run_adf(df["sales_diff"], "First Differenced Sales")

# ACF and PACF plots
fig, axes = plt.subplots(2, 2, figsize=(14, 8))

axes[0, 0].plot(df.index, df["sales"], color="steelblue", linewidth=0.7)
axes[0, 0].set_title("Raw Sales")

axes[0, 1].plot(df.index, df["sales_diff"], color="darkorange", linewidth=0.7)
axes[0, 1].set_title("First Differenced Sales")

plot_acf(df["sales_diff"].dropna(), lags=40, ax=axes[1, 0])
axes[1, 0].set_title("ACF - Differenced Series")

plot_pacf(df["sales_diff"].dropna(), lags=40, ax=axes[1, 1])
axes[1, 1].set_title("PACF - Differenced Series")

plt.suptitle("Stationarity Analysis", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/stationarity.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nPlot saved.")
