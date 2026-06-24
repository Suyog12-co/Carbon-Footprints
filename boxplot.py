import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load original dataset (not cleaned_data.csv)
df = pd.read_csv("dataset.csv")

# Transport Boxplot
plt.figure(figsize=(8,5))
sns.boxplot(x="Transport", y="CarbonEmission", data=df)
plt.title("Carbon Emission by Transport Type")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("transport_boxplot.png", dpi=300)
plt.show()

# Diet Boxplot
plt.figure(figsize=(8,5))
sns.boxplot(x="Diet", y="CarbonEmission", data=df)
plt.title("Carbon Emission by Diet Type")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("diet_boxplot.png", dpi=300)
plt.show()