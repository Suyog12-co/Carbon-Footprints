import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
df = pd.read_csv("cleaned_data.csv")
'''print(df.head())
print(df.info())
print (df.describe())
print(df.shape)
print(df.isnull().sum())
print(df["Vehicle Type"].value_counts(dropna=False))
print(df["Transport"].value_counts())

df['How Often Shower'].unique()
df['Energy efficiency'].unique()
df['Frequency of Traveling by Air'].unique()
df['Recycling'].unique()
df['Cooking_With'].unique()
df['Transport'].unique()
df['Vehicle Type'].unique()
df['CarbonEmission'].skew()
df['CarbonEmission'].hist(bins=50)
plt.show()

# Create log-transformed target
df["CarbonEmission_log"] = np.log1p(df["CarbonEmission"])

print(df[["CarbonEmission", "CarbonEmission_log"]].head())
print(df["CarbonEmission"].describe())
print(df["CarbonEmission_log"].describe())
print("Original skew:", df["CarbonEmission"].skew())
print("Log skew:", df["CarbonEmission_log"].skew())

cols = [
    'How Often Shower',
    'Energy efficiency',
    'Frequency of Traveling by Air',
    'Recycling',
    'Cooking_With',
    'Transport',
    'Vehicle Type'
]

for col in cols:
    print(f"\n{col}")
    print(df[col].unique())'''

df.isnull().sum().sum()  # confirm 0
df.shape  # confirm column count after encoding (~45-50 columns expected)
print("Total missing values:", df.isnull().sum().sum())
print("Shape:", df.shape)