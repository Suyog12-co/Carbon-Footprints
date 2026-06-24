import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df = pd.read_csv("cleaned_data.csv")

X = df.drop(columns=['CarbonEmission','CarbonEmission_log'])
y_log = df['CarbonEmission_log']
y_actual = df['CarbonEmission']

X_train, X_test, y_train, y_test_log, y_train_actual, y_test_actual = train_test_split(
    X, y_log, y_actual,
    test_size=0.2,
    random_state=42
)

model = joblib.load("final_xgb_model.pkl")

pred = np.expm1(model.predict(X_test))

plt.figure(figsize=(7,6))
plt.scatter(y_test_actual, pred, alpha=0.5)
plt.xlabel("Actual Carbon Emission")
plt.ylabel("Predicted Carbon Emission")
plt.title("Actual vs Predicted Carbon Emission")
plt.tight_layout()
plt.savefig("actual_vs_predicted.png", dpi=300)
plt.show()
