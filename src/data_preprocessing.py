import pandas as pd

# Load dataset
df = pd.read_csv("data/movies.csv")

print("=" * 50)
print("MovieIQ Dataset")
print("=" * 50)

# Dataset shape
print("\nDataset Shape:")
print(df.shape)

# Dataset information
print("\nDataset Information:")
print(df.info())

# First five rows
print("\nFirst Five Rows:")
print(df.head())

# Summary statistics
print("\nSummary Statistics:")
print(df.describe())

# Missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Check zeros
print("\nMovies with Budget = 0 :", (df["budget"] == 0).sum())
print("Movies with Revenue = 0 :", (df["revenue"] == 0).sum())

# Remove invalid rows
df = df.dropna()

df = df[
    (df["budget"] > 0) &
    (df["revenue"] > 0)
]

# Create target column
df["success"] = (df["revenue"] > df["budget"]).astype(int)

print("\nSuccess Distribution:")
print(df["success"].value_counts())

# Save cleaned dataset
df.to_csv("data/clean_movies.csv", index=False)

print("\nCleaned dataset saved successfully!")