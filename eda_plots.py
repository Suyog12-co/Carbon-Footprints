import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("cleaned_data.csv")

# 1. Correlation Heatmap
plt.figure(figsize=(12,8))
corr = df.corr(numeric_only=True)
sns.heatmap(corr, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig("heatmap.png", dpi=300)
plt.show()

# 2. Target Distribution
plt.figure(figsize=(8,5))
df["CarbonEmission"].hist(bins=40)
plt.xlabel("Carbon Emission")
plt.ylabel("Frequency")
plt.title("Distribution of Carbon Emission")
plt.savefig("target_distribution.png", dpi=300)
plt.show()