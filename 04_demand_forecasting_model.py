"""
Project 4: Demand Forecasting Model
Predict future product demand using previous demand values and calendar features.

Install:
    pip install pandas scikit-learn

Run:
    python 04_demand_forecasting_model.py

This is a demo. Replace the generated data with real sales history for a
meaningful forecasting model.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

np.random.seed(42)

# Create 120 days of demo demand.
dates = pd.date_range("2025-01-01", periods=120, freq="D")
trend = np.linspace(80, 140, len(dates))
weekly_pattern = 12 * np.sin(2 * np.pi * np.arange(len(dates)) / 7)
noise = np.random.normal(0, 5, len(dates))

df = pd.DataFrame({
    "date": dates,
    "demand": np.maximum(10, trend + weekly_pattern + noise)
})

# Lag features: yesterday, 7 days ago, and 14 days ago.
df["lag_1"] = df["demand"].shift(1)
df["lag_7"] = df["demand"].shift(7)
df["lag_14"] = df["demand"].shift(14)
df["day_of_week"] = df["date"].dt.dayofweek
df["month"] = df["date"].dt.month

df = df.dropna().reset_index(drop=True)

features = ["lag_1", "lag_7", "lag_14", "day_of_week", "month"]
X = df[features]
y = df["demand"]

# Time-order split: do not randomly shuffle forecasting data.
split = int(len(df) * 0.80)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Mean Absolute Error:",
      round(mean_absolute_error(y_test, predictions), 2))

# Forecast the next day using the latest known demand values.
last = df.iloc[-1]
next_date = last["date"] + pd.Timedelta(days=1)

next_features = pd.DataFrame([{
    "lag_1": last["demand"],
    "lag_7": df.iloc[-7]["demand"],
    "lag_14": df.iloc[-14]["demand"],
    "day_of_week": next_date.dayofweek,
    "month": next_date.month
}])

forecast = model.predict(next_features)[0]

print("\nNext date:", next_date.date())
print("Predicted demand:", round(forecast, 2), "units")
