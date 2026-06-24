import pandas as pd
import numpy as np
import shap
import joblib
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("cleaned_data.csv")

# Features and target
X = df.drop(columns=['CarbonEmission', 'CarbonEmission_log'])
y = df['CarbonEmission_log']

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Load trained model
model = joblib.load("final_xgb_model.pkl")

# Select one test sample
sample = X_test.iloc[[0]]

# Predict footprint
prediction = np.expm1(model.predict(sample))[0]

print("\n==============================")
print("CARBON FOOTPRINT REPORT")
print("==============================")
print(f"\nPredicted Carbon Emission: {prediction:.2f} kg")

# SHAP analysis
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(sample)

# Create feature importance table
feature_imp = pd.DataFrame({
    "Feature": sample.columns,
    "SHAP_Value": shap_values[0]
})

feature_imp["Abs_SHAP"] = feature_imp["SHAP_Value"].abs()

# Top 5 contributors
top_features = feature_imp.sort_values(
    by="Abs_SHAP",
    ascending=False
).head(5)

print("\nTop Contributors:")
for i, row in enumerate(top_features.itertuples(), start=1):
    print(f"{i}. {row.Feature} ({row.SHAP_Value:.4f})")

# Recommendations
print("\nRecommendations:")

for feature in top_features["Feature"]:

    if "Vehicle Monthly Distance" in feature:
        print("- Reduce monthly driving distance.")

    elif "Frequency of Traveling by Air" in feature:
        print("- Reduce air travel frequency where possible.")

    elif "Transport_private" in feature:
        print("- Shift from private transport to public transport.")

    elif "Vehicle Type_petrol" in feature:
        print("- Consider switching to an electric or hybrid vehicle.")

    elif "How Many New Clothes Monthly" in feature:
        print("- Reduce fast-fashion purchases and reuse clothing.")

    elif "Waste Bag Weekly Count" in feature:
        print("- Reduce household waste and improve recycling.")

    elif "Heating Energy Source_coal" in feature:
        print("- Shift to cleaner energy sources.")

    elif "How Long TV PC Daily Hour" in feature:
        print("- Reduce unnecessary electricity consumption.")

print("\n==============================")