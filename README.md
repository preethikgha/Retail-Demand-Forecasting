# Retail Demand Forecasting

End-to-end demand forecasting pipeline that validates statistical model assumptions before shipping predictions — comparing ARIMA, Prophet, and XGBoost on 1,941 days of real Walmart retail sales data.

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Statsmodels](https://img.shields.io/badge/Statsmodels-4051B5?style=for-the-badge&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-FF6600?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)

---

## Overview

Most forecasting pipelines jump straight to model fitting without checking whether the data actually satisfies model assumptions. This project treats statistical validation as a first-class step — not an afterthought.

Applied to the Walmart M5 dataset (CA_1 store, FOODS category), the pipeline tests stationarity before fitting ARIMA, decomposes trend and seasonality before choosing a model family, and validates residuals after fitting to confirm the model captured all learnable structure.

**Dataset:** Walmart M5 Forecasting — Kaggle  
**Series:** Daily aggregated food sales, CA_1 store  
**Period:** January 2011 – May 2016  
**Days:** 1,941  
**Test window:** Last 90 days held out across all models

---

## Statistical Framework

- **ADF Test** — Augmented Dickey-Fuller confirms raw series is non-stationary (p=0.185). First differencing achieves stationarity (p=0.0), setting d=1 for ARIMA
- **STL Decomposition** — Trend, weekly seasonality (period=7), and residual components isolated before model selection
- **ACF / PACF Analysis** — Autocorrelation and partial autocorrelation plots used to determine ARIMA (p, q) parameters without relying on auto_arima blindly
- **Ljung-Box Test** — Post-fit residual test checks whether errors are white noise or carry unexplained autocorrelation
- **Three model families compared** — Classical stats (SARIMA), decomposition-based (Prophet), and ML with engineered features (XGBoost)

---

## Results

| Model | MAE | RMSE | MAPE |
|-------|-----|------|------|
| ARIMA | 304.65 | 394.39 | 9.26% |
| Prophet | 340.00 | 408.93 | 10.58% |
| **XGBoost** | **163.87** | **210.01** | **5.33%** |

XGBoost with lag and calendar features outperformed ARIMA by 43% and Prophet by 50% on MAPE.

**Residual Analysis — XGBoost**  
Ljung-Box at lag 10: p = 0.045 — mild short-term autocorrelation detected. Residual mean = 22.01, std = 210.03. Distribution is approximately normal with slight right skew from weekend peak misses. Addressable by adding lag-2 and lag-3 features.

---

## Key Findings

- Weekly seasonality is the dominant signal — weekend sales spike ~600 units above weekday baseline
- lag_28 and dayofweek are the two most predictive XGBoost features, explaining most variance
- Prophet consistently undershoots weekend peaks due to its smooth seasonality assumption
- ARIMA captures the weekly cycle but cannot leverage the non-linear interactions that XGBoost exploits through lag features
- One round of differencing is sufficient for stationarity — no seasonal differencing required beyond the SARIMA seasonal component

---


## Setup

```bash
git clone https://github.com/yourusername/retail-demand-forecasting.git
cd retail-demand-forecasting
pip install -r requirements.txt
```

Download the M5 dataset from [Kaggle](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data) and place these files in `data/`:
- `sales_train_evaluation.csv`
- `calendar.csv`
- `sell_prices.csv`

---

## Run

**Build pipeline and train all models:**
```bash
python src/load_data.py
python src/eda_decomposition.py
python src/stationarity.py
python src/arima_model.py
python src/prophet_model.py
python src/xgboost_model.py
python src/model_comparison.py
python src/residual_analysis.py
```

**Launch dashboard:**
```bash
streamlit run src/app.py
```

---

## Key Concepts

**Stationarity** — ARIMA requires a constant mean and variance over time. Fitting on a non-stationary series produces unreliable coefficients. ADF test makes this assumption explicit and measurable rather than assumed.

**STL Decomposition** — Separating trend, seasonality, and residual before modeling reveals the structure of the series and informs which model family is appropriate. A series with strong weekly seasonality needs a model that can represent it — Prophet and SARIMA handle it natively, XGBoost learns it through dayofweek features.

**Residual Autocorrelation** — If a model's errors are correlated with each other, the model has not extracted all learnable signal. Ljung-Box test formalizes this check. A p-value below 0.05 means the residuals are not white noise and the model can still be improved.

**Lag Features** — XGBoost cannot natively understand time ordering. Lag features (lag_1, lag_7, lag_28) encode historical values as explicit inputs, allowing the model to learn temporal dependencies through standard supervised learning.

---

Built by Preethikgha M
