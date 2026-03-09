from pyspark.sql import SparkSession
from pyspark.sql import functions as F

PRED_PATH = "results/spark_predictions"  # folder of part CSVs
OUT_DIR = "results/dashboard"

spark = SparkSession.builder.appName("UrbanSoundscape-Dashboard-Analytics").getOrCreate()

# Read predictions
df = spark.read.option("header", True).csv(PRED_PATH)

print("\n=== BASIC CHECK ===")
print("Rows:", df.count())
df.select("audio_filename", "predicted_count").show(5, truncate=False)

# Ensure predicted_count is numeric
df = df.withColumn("predicted_count", F.col("predicted_count").cast("int"))

# -----------------------------
# 1) Distribution of predicted labels per sample
# -----------------------------
print("\n=== Predicted label count distribution ===")
pred_dist = df.groupBy("predicted_count").count().orderBy("predicted_count")
pred_dist.show(50, truncate=False)

# -----------------------------
# 2) Most frequent predicted labels
# -----------------------------
print("\n=== Top predicted labels (frequency) ===")
labels_df = df.withColumn(
    "label",
    F.explode(F.split(F.col("predicted_labels"), ","))
).filter((F.col("label").isNotNull()) & (F.col("label") != ""))

top_labels = labels_df.groupBy("label").count().orderBy(F.desc("count"))
top_labels.show(30, truncate=False)

# -----------------------------
# 3) Top co-occurring label pairs (SAFE)
# -----------------------------
print("\n=== Top co-occurring label pairs ===")

# Build an array of labels and keep only rows with 2+ labels
df_pairs_base = df.withColumn(
    "labels_arr",
    F.expr("filter(split(predicted_labels, ','), x -> x != '')")
).filter(F.size("labels_arr") >= 2)

# Create all pair combinations safely using explode + array_sort
pairs = df_pairs_base.select(
    F.explode("labels_arr").alias("a"),
    F.explode("labels_arr").alias("b")
).filter(F.col("a") < F.col("b")) \
 .withColumn("pair", F.concat(F.col("a"), F.lit(" + "), F.col("b")))

top_pairs = pairs.groupBy("pair").count().orderBy(F.desc("count"))
top_pairs.show(30, truncate=False)

# -----------------------------
# Save dashboard-ready CSV outputs
# -----------------------------
print("\n=== Saving dashboard tables ===")

spark.sql("set spark.sql.shuffle.partitions=8")

pred_dist.coalesce(1).write.mode("overwrite").option("header", True).csv(f"{OUT_DIR}/predicted_count_distribution")
top_labels.coalesce(1).write.mode("overwrite").option("header", True).csv(f"{OUT_DIR}/top_labels")
top_pairs.coalesce(1).write.mode("overwrite").option("header", True).csv(f"{OUT_DIR}/top_pairs")

print(f"\n✅ Saved outputs under {OUT_DIR}/")
spark.stop()
