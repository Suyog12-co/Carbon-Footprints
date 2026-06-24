import pandas as pd

df = pd.read_csv("cleaned_data.csv")

print("\nColumns:")
print(df.columns.tolist())

print("\nShape:")
print(df.shape)