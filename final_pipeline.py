# =====================================================
# CARBON FOOTPRINT PREDICTION PROJECT
# FINAL PIPELINE
# =====================================================

# STEP 1: IMPORT LIBRARIES

import pandas as pd
import numpy as np
import joblib
import shap

from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV
)

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)

import xgboost as xgb


# =====================================================
# STEP 2: LOAD DATA
# =====================================================

print("Loading Dataset...")

df = pd.read_csv("cleaned_data.csv")

print("Dataset Shape:", df.shape)


# =====================================================
# STEP 3: FEATURE / TARGET SPLIT
# =====================================================

X = df.drop(
    columns=["CarbonEmission", "CarbonEmission_log"]
)

y_log = df["CarbonEmission_log"]
y_actual = df["CarbonEmission"]


# =====================================================
# STEP 4: TRAIN TEST SPLIT
# =====================================================

X_train, X_test, y_train_log, y_test_log, \
y_train_actual, y_test_actual = train_test_split(
    X,
    y_log,
    y_actual,
    test_size=0.2,
    random_state=42
)

print("Train/Test Split Complete")


# =====================================================
# STEP 5: LINEAR REGRESSION
# =====================================================

lr = LinearRegression()

lr.fit(X_train, y_train_log)

pred_lr = np.expm1(
    lr.predict(X_test)
)

r2_lr = r2_score(y_test_actual, pred_lr)
rmse_lr = np.sqrt(
    mean_squared_error(
        y_test_actual,
        pred_lr
    )
)
mae_lr = mean_absolute_error(
    y_test_actual,
    pred_lr
)

print("\nLinear Regression")
print("R²:", round(r2_lr, 4))
print("RMSE:", round(rmse_lr, 2))
print("MAE:", round(mae_lr, 2))


# =====================================================
# STEP 6: RANDOM FOREST
# =====================================================

rf = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

rf.fit(X_train, y_train_log)

pred_rf = np.expm1(
    rf.predict(X_test)
)

r2_rf = r2_score(y_test_actual, pred_rf)
rmse_rf = np.sqrt(
    mean_squared_error(
        y_test_actual,
        pred_rf
    )
)
mae_rf = mean_absolute_error(
    y_test_actual,
    pred_rf
)

print("\nRandom Forest")
print("R²:", round(r2_rf, 4))
print("RMSE:", round(rmse_rf, 2))
print("MAE:", round(mae_rf, 2))


# =====================================================
# STEP 7: XGBOOST + TUNING
# =====================================================

param_grid = {
    "n_estimators": [100, 200, 300, 400],
    "max_depth": [3, 4, 5, 6, 8],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0]
}

xgb_base = xgb.XGBRegressor(
    random_state=42
)

search = RandomizedSearchCV(
    xgb_base,
    param_distributions=param_grid,
    n_iter=30,
    cv=5,
    scoring="r2",
    random_state=42,
    n_jobs=-1
)

search.fit(
    X_train,
    y_train_log
)

best_xgb = search.best_estimator_

pred_xgb = np.expm1(
    best_xgb.predict(X_test)
)

r2_xgb = r2_score(
    y_test_actual,
    pred_xgb
)

rmse_xgb = np.sqrt(
    mean_squared_error(
        y_test_actual,
        pred_xgb
    )
)

mae_xgb = mean_absolute_error(
    y_test_actual,
    pred_xgb
)

print("\nTuned XGBoost")
print("R²:", round(r2_xgb, 4))
print("RMSE:", round(rmse_xgb, 2))
print("MAE:", round(mae_xgb, 2))


# =====================================================
# STEP 8: SAVE MODEL
# =====================================================

joblib.dump(
    best_xgb,
    "final_xgb_model.pkl"
)

print("\nModel Saved")


# =====================================================
# STEP 9: SHAP EXPLAINABILITY
# =====================================================

explainer = shap.TreeExplainer(
    best_xgb
)

shap_values = explainer.shap_values(
    X_test
)

print("\nSHAP Analysis Complete")


# =====================================================
# STEP 10: SAMPLE RECOMMENDATION
# =====================================================

sample = X_test.iloc[[0]]

prediction = np.expm1(
    best_xgb.predict(sample)
)[0]

print("\n================================")
print("CARBON FOOTPRINT REPORT")
print("================================")

print(
    f"\nPredicted Carbon Emission: "
    f"{prediction:.2f} kg"
)

print("\nProject Pipeline Completed")