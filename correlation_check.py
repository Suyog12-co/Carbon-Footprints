import pandas as pd

df = pd.read_csv("cleaned_data.csv")

print(df.corr(numeric_only=True)["CarbonEmission"].sort_values(ascending=False))