import pandas as pd
import matplotlib.pyplot as plt

# Paths
TOP_LABELS = "results/dashboard_top_labels.csv"
COUNT_DIST = "results/dashboard_predicted_count_distribution.csv"
TOP_PAIRS  = "results/dashboard_top_pairs.csv"

OUT1 = "results/chart_top_labels.png"
OUT2 = "results/chart_predicted_count_distribution.png"
OUT3 = "results/chart_top_pairs.png"

# ---------- Chart 1: Top Labels ----------
df_labels = pd.read_csv(TOP_LABELS).head(10)
plt.figure(figsize=(12,6))
plt.bar(df_labels["label"], df_labels["count"])
plt.xticks(rotation=45, ha="right")
plt.title("Top 10 Most Frequent Predicted Urban Sounds")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(OUT1, dpi=200)
plt.close()

# ---------- Chart 2: Predicted Count Distribution ----------
df_dist = pd.read_csv(COUNT_DIST).sort_values("predicted_count")
plt.figure(figsize=(10,6))
plt.bar(df_dist["predicted_count"].astype(str), df_dist["count"])
plt.title("Distribution of Number of Predicted Sound Events per Audio Clip")
plt.xlabel("Predicted Labels per Clip")
plt.ylabel("Number of Samples")
plt.tight_layout()
plt.savefig(OUT2, dpi=200)
plt.close()

# ---------- Chart 3: Top Pairs ----------
df_pairs = pd.read_csv(TOP_PAIRS).head(10)
plt.figure(figsize=(12,6))
plt.bar(df_pairs["pair"], df_pairs["count"])
plt.xticks(rotation=45, ha="right")
plt.title("Top 10 Most Common Co-occurring Urban Sound Pairs")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(OUT3, dpi=200)
plt.close()

print("✅ Charts saved:")
print(OUT1)
print(OUT2)
print(OUT3)
