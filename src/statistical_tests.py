import pandas as pd
from scipy.stats import ttest_ind, chi2_contingency

# Load cleaned dataset
df = pd.read_csv("data/clean_movies.csv")

print("=" * 60)
print("STATISTICAL TESTS")
print("=" * 60)

# -----------------------------
# T-Test
# -----------------------------
success = df[df["success"] == 1]["popularity"]
failure = df[df["success"] == 0]["popularity"]

t_stat, p_value = ttest_ind(success, failure)

print("\nT-Test (Popularity vs Success)")
print(f"T-Statistic : {t_stat:.4f}")
print(f"P-Value     : {p_value:.4f}")

if p_value < 0.05:
    print("Conclusion  : Popularity differs significantly between successful and unsuccessful movies.")
else:
    print("Conclusion  : No significant difference found.")

# -----------------------------
# Chi-Square Test
# -----------------------------
print("\n" + "=" * 60)
print("Chi-Square Test")

# Handle multiple genres
genre_df = df.copy()
genre_df["genres"] = genre_df["genres"].fillna("")
genre_df["genres"] = genre_df["genres"].str.split("|")
genre_df = genre_df.explode("genres")

contingency_table = pd.crosstab(
    genre_df["genres"],
    genre_df["success"]
)

chi2, p, dof, expected = chi2_contingency(contingency_table)

print(f"Chi2 Statistic : {chi2:.4f}")
print(f"P-Value        : {p:.4f}")

if p < 0.05:
    print("Conclusion     : Genre is associated with movie success.")
else:
    print("Conclusion     : No association found.")

print("\nAll statistical tests completed successfully!")