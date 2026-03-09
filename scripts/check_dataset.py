import pandas as pd

# Load CLEAN dataset
data = pd.read_csv("../datasets/processed/sonyc_clean.csv")

# Separate features and labels
label_cols = [c for c in data.columns if c.endswith("_presence")]

X = data.drop(columns=["audio_filename"] + label_cols)
y = data[label_cols]

print("\n--- CLEAN DATASET CHECK ---")
print("Total samples:", len(data))
print("Feature shape:", X.shape)
print("Label shape:", y.shape)

print("\n--- FEATURE QUALITY ---")
print("Any NaN:", X.isna().any().any())
print("Any infinite:", (X == float("inf")).any().any())
print("Min value:", X.min().min())
print("Max value:", X.max().max())

print("\n--- LABEL COUNTS ---")
print(y.sum().sort_values())

