import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df = pd.read_csv("cleaned_data.csv")

X = df.drop(columns=['CarbonEmission', 'CarbonEmission_log'])
y = df['CarbonEmission_log']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = joblib.load("final_xgb_model.pkl")

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X_test)

# Summary Plot
shap.summary_plot(shap_values, X_test, show=False)
plt.tight_layout()
plt.savefig("shap_summary.png", dpi=300)
plt.show()