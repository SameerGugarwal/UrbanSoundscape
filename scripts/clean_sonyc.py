import pandas as pd

print("Loading data...")

X = pd.read_csv("../datasets/processed/mfcc_features.csv")
X = X.drop(columns=["class"], errors="ignore")

y = pd.read_csv("../datasets/raw/SONYC-UST/annotations.csv")

label_cols = [c for c in y.columns if c.endswith("_presence")]

# Convert -1,0,1 → binary (1 = present)
y[label_cols] = (y[label_cols] > 0).astype(int)

# Group by audio and take majority vote
y_clean = y.groupby("audio_filename")[label_cols].mean()
y_clean = (y_clean > 0.5).astype(int).reset_index()

print("Cleaned labels:", y_clean.shape)

# Merge
data = X.merge(y_clean, on="audio_filename", how="inner")

print("Final dataset:", data.shape)

# Save clean dataset
data.to_csv("../datasets/processed/sonyc_clean.csv", index=False)

print("Saved clean dataset!")
