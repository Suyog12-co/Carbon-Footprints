import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("cleaned_data.csv")

# Heatmap
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("heatmap.png", dpi=300)
plt.show()

# Histogram
plt.figure(figsize=(8,5))
sns.histplot(df["CarbonEmission"], bins=40, kde=True)
plt.title("Carbon Emission Distribution")
plt.savefig("histogram.png", dpi=300)
plt.show()

# Top Features Scatter Plots
features = [
    "Vehicle Monthly Distance Km",
    "Frequency of Traveling by Air",
    "Monthly Grocery Bill",
    "How Many New Clothes Monthly",
    "Waste Bag Weekly Count"
]

for f in features:
    plt.figure(figsize=(6,4))
    sns.scatterplot(x=df[f], y=df["CarbonEmission"])
    plt.title(f"{f} vs Carbon Emission")
    plt.savefig(f"{f}.png", dpi=300)
    plt.show()