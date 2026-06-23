from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np
import pandas as pd
import xgboost as xgb

df = pd.read_csv("cleaned_data.csv")
# Separate features and target
X = df.drop(columns=['CarbonEmission', 'CarbonEmission_log'])
y_log = df['CarbonEmission_log']
y_actual = df['CarbonEmission']  # keep for final kg-scale evaluation

# Split (fixed random_state for reproducibility — mention this in your paper)
X_train, X_test, y_train_log, y_test_log, y_train_actual, y_test_actual = train_test_split(
    X, y_log, y_actual, test_size=0.2, random_state=42
)

param_grid = {
    'n_estimators': [100, 200, 300, 400],
    'max_depth': [3, 4, 5, 6, 8],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0]
}

xgb_base = xgb.XGBRegressor(random_state=42)

random_search = RandomizedSearchCV(
    xgb_base, param_distributions=param_grid,
    n_iter=30, scoring='r2', cv=5, random_state=42, n_jobs=-1, verbose=1
)

random_search.fit(X_train, y_train_log)

print("Best params:", random_search.best_params_)

# Evaluate tuned model
best_xgb = random_search.best_estimator_
tuned_pred = np.expm1(best_xgb.predict(X_test))

r2_tuned = r2_score(y_test_actual, tuned_pred)
rmse_tuned = np.sqrt(mean_squared_error(y_test_actual, tuned_pred))
mae_tuned = mean_absolute_error(y_test_actual, tuned_pred)
print(f"Tuned XGBoost — R²: {r2_tuned:.4f}, RMSE: {rmse_tuned:.2f} kg, MAE: {mae_tuned:.2f} kg")

import joblib

joblib.dump(best_xgb, 'final_xgb_model.pkl')