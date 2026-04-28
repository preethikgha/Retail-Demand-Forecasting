import pandas as pd
import matplotlib.pyplot as plt
import os

#  Paths
OUT_DIR  = "outputs/plots"
os.makedirs(OUT_DIR, exist_ok=True)

#  Load files
print("Loading data...")
sales    = pd.read_csv(f"{DATA_DIR}/sales_train_evaluation.csv")
calendar = pd.read_csv(f"{DATA_DIR}/calendar.csv")

print(f"Sales shape    : {sales.shape}")
print(f"Calendar shape : {calendar.shape}")

#Filter: 1 store + 1 category (CA_1, FOODS) 
mask = (
    sales["store_id"] == "CA_1"
) & (
    sales["cat_id"] == "FOODS"
)
sales_filtered = sales[mask].copy()
print(f"\nFiltered rows (CA_1 + FOODS): {len(sales_filtered)}")

#  Aggregate daily sales 
day_cols = [c for c in sales_filtered.columns if c.startswith("d_")]
daily    = sales_filtered[day_cols].sum(axis=0)

# Map d_1, d_2 
cal = calendar[["d", "date"]].copy()
cal = cal[cal["d"].isin(daily.index)]
cal = cal.set_index("d")

df = pd.DataFrame({"sales": daily})
df.index = pd.to_datetime(cal.loc[df.index, "date"])
df.index.name = "date"
df = df.sort_index()

print(f"\nTime series range : {df.index.min()} → {df.index.max()}")
print(f"Total days        : {len(df)}")
print(df.head(10))

df.to_csv(f"{DATA_DIR}/ts_ca1_foods.csv")
print("\n Saved to data/ts_ca1_foods.csv")


plt.figure(figsize=(14, 4))
plt.plot(df.index, df["sales"], linewidth=0.8, color="steelblue")
plt.title("Daily Sales — CA_1 Store, FOODS Category")
plt.xlabel("Date")
plt.ylabel("Units Sold")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/raw_sales.png", dpi=150)
plt.show()
print(" Plot saved to outputs/plots/01_raw_sales.png")
