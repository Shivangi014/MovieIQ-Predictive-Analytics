import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Create assets folder if it doesn't exist
os.makedirs("assets", exist_ok=True)

# Load cleaned dataset
df = pd.read_csv("data/clean_movies.csv")

# -----------------------------
# Budget vs Revenue
# -----------------------------
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x="budget", y="revenue")
plt.title("Budget vs Revenue")
plt.tight_layout()
plt.savefig("assets/budget_vs_revenue.png")
plt.show()

# -----------------------------
# Correlation Heatmap
# -----------------------------
plt.figure(figsize=(10, 8))

numeric_df = df.select_dtypes(include=["number"])

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("assets/correlation_heatmap.png")
plt.show()

# -----------------------------
# Popularity Distribution
# -----------------------------
plt.figure(figsize=(8, 5))
sns.histplot(df["popularity"], bins=30, kde=True)
plt.title("Popularity Distribution")
plt.tight_layout()
plt.savefig("assets/popularity_distribution.png")
plt.show()

# -----------------------------
# Vote Average Distribution
# -----------------------------
plt.figure(figsize=(8, 5))
sns.histplot(df["vote_average"], bins=20, kde=True)
plt.title("Vote Average Distribution")
plt.tight_layout()
plt.savefig("assets/vote_average_distribution.png")
plt.show()

print("\nEDA completed successfully!")
print("Charts saved inside the assets folder.")