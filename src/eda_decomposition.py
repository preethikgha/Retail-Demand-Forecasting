import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from statsmodels.tsa.seasonal import seasonal_decompose
import os

OUT_DIR = "outputs/plots"
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv("data/ts_ca1_foods.csv", index_col="date", parse_dates=True)
df = df.asfreq("D").fillna(method="ffill")

# STL Decomposition — period=7 for weekly seasonality
result = seasonal_decompose(df["sales"], model="additive", period=7)


fig = plt.figure(figsize=(14, 10))
gs  = gridspec.GridSpec(4, 1, hspace=0.5)

ax1 = fig.add_subplot(gs[0])
ax1.plot(df.index, result.observed, linewidth=0.7, color="steelblue")
ax1.set_title("Observed")
ax1.set_ylabel("Sales")

ax2 = fig.add_subplot(gs[1])
ax2.plot(df.index, result.trend, linewidth=1, color="darkorange")
ax2.set_title("Trend")
ax2.set_ylabel("Sales")

ax3 = fig.add_subplot(gs[2])
ax3.plot(df.index, result.seasonal, linewidth=0.7, color="green")
ax3.set_title("Seasonality (Weekly)")
ax3.set_ylabel("Effect")

ax4 = fig.add_subplot(gs[3])
ax4.plot(df.index, result.resid, linewidth=0.7, color="red", alpha=0.6)
ax4.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax4.set_title("Residual")
ax4.set_ylabel("Error")
ax4.set_xlabel("Date")

plt.suptitle("STL Decomposition — CA_1 FOODS Daily Sales", fontsize=13, y=1.01)
plt.savefig(f"{OUT_DIR}/decomposition.png", dpi=150, bbox_inches="tight")
plt.show()

print("Trend range   :", round(result.trend.min()), "to", round(result.trend.max()))
print("Seasonal range:", round(result.seasonal.min()), "to", round(result.seasonal.max()))
print("Residual std  :", round(result.resid.std(), 2))
