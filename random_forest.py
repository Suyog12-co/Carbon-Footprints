from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np
import pandas as pd
import model3 as xgb

df = pd.read_csv("cleaned_data.csv")
# Separate features and target
X = df.drop(columns=['CarbonEmission', 'CarbonEmission_log'])
y_log = df['CarbonEmission_log']
y_actual = df['CarbonEmission']  # keep for final kg-scale evaluation

# Split (fixed random_state for reproducibility — mention this in your paper)
X_train, X_test, y_train_log, y_test_log, y_train_actual, y_test_actual = train_test_split(
    X, y_log, y_actual, test_size=0.2, random_state=42
)

# Random Forest
rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train_log)
rf_pred = np.expm1(rf.predict(X_test))

r2_rf = r2_score(y_test_actual, rf_pred)
rmse_rf = np.sqrt(mean_squared_error(y_test_actual, rf_pred))
mae_rf = mean_absolute_error(y_test_actual, rf_pred)
print(f"Random Forest — R²: {r2_rf:.4f}, RMSE: {rmse_rf:.2f} kg, MAE: {mae_rf:.2f} kg")

# XGBoost
xgb_model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42)
xgb_model.fit(X_train, y_train_log)
xgb_pred = np.expm1(xgb_model.predict(X_test))

r2_xgb = r2_score(y_test_actual, xgb_pred)
rmse_xgb = np.sqrt(mean_squared_error(y_test_actual, xgb_pred))
mae_xgb = mean_absolute_error(y_test_actual, xgb_pred)
print(f"XGBoost — R²: {r2_xgb:.4f}, RMSE: {rmse_xgb:.2f} kg, MAE: {mae_xgb:.2f} kg")