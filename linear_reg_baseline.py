from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np
import pandas as pd
df = pd.read_csv("cleaned_data.csv")
# Separate features and target
X = df.drop(columns=['CarbonEmission', 'CarbonEmission_log'])
y_log = df['CarbonEmission_log']
y_actual = df['CarbonEmission']  # keep for final kg-scale evaluation

# Split (fixed random_state for reproducibility — mention this in your paper)
X_train, X_test, y_train_log, y_test_log, y_train_actual, y_test_actual = train_test_split(
    X, y_log, y_actual, test_size=0.2, random_state=42
)

# Train baseline
lr = LinearRegression()
lr.fit(X_train, y_train_log)

# Predict (in log space), then inverse-transform back to kg
pred_log = lr.predict(X_test)
pred_actual = np.expm1(pred_log)

# Evaluate in ACTUAL kg units (this is what goes in your paper)
r2 = r2_score(y_test_actual, pred_actual)
rmse = np.sqrt(mean_squared_error(y_test_actual, pred_actual))
mae = mean_absolute_error(y_test_actual, pred_actual)

print(f"Linear Regression — R²: {r2:.4f}, RMSE: {rmse:.2f} kg, MAE: {mae:.2f} kg")

