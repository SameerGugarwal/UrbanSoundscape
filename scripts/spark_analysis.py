from pyspark.sql import SparkSession
from pyspark.sql import functions as F

DATA_PATH = "datasets/processed/sonyc_clean.csv"  # change to mfcc_features.csv if needed

spark = SparkSession.builder.appName("SONYC-Clean-Spark-Analysis").getOrCreate()

# ✅ Read as strings to avoid inferSchema crashes
df_raw = spark.read.option("header", True).option("inferSchema", False).csv(DATA_PATH)

print("\n=== BASIC INFO ===")
print("Rows:", df_raw.count())
print("Columns:", len(df_raw.columns))
print("First 15 columns:", df_raw.columns[:15])

# ✅ Missing values (handles empty strings too)
print("\n=== MISSING VALUES (top 20 columns) ===")
missing_counts = df_raw.select([
    F.sum(F.when(F.col(c).isNull() | (F.col(c) == ""), 1).otherwise(0)).alias(c)
    for c in df_raw.columns
]).collect()[0].asDict()

for col, miss in sorted(missing_counts.items(), key=lambda x: x[1], reverse=True)[:20]:
    print(f"{col}: {miss}")

# Detect label columns
presence_cols = [c for c in df_raw.columns if c.endswith("_presence")]
print("\nDetected _presence columns:", len(presence_cols))

# If we have multilabel presence columns, compute prevalence robustly
if len(presence_cols) > 0:
    # ✅ try_cast to int; invalid becomes NULL; treat NULL as 0
    df = df_raw
    for c in presence_cols:
        df = df.withColumn(c, F.expr(f"coalesce(try_cast(`{c}` as int), 0)"))

    total = df.count()

    sums = df.select([F.sum(F.col(c)).alias(c) for c in presence_cols]).collect()[0].asDict()

    prev_df = spark.createDataFrame(
        [(k, int(v), float(v) / total) for k, v in sums.items()],
        ["label", "positive_count", "positive_rate"]
    ).orderBy(F.desc("positive_rate"))

    print("\n=== TOP 20 MOST COMMON LABELS ===")
    prev_df.show(20, truncate=False)

    print("\n=== BOTTOM 20 RAREST LABELS ===")
    prev_df.orderBy(F.asc("positive_rate")).show(20, truncate=False)

    # Average positives per sample (sparsity)
    pos_sum_expr = None
    for c in presence_cols:
        expr = F.col(c)
        pos_sum_expr = expr if pos_sum_expr is None else (pos_sum_expr + expr)

    avg_labels = df.select(F.avg(pos_sum_expr).alias("avg_positive_labels")).collect()[0]["avg_positive_labels"]
    print(f"\nAverage positive labels per sample: {avg_labels:.4f}")

# If single-label dataset
elif "class" in df_raw.columns:
    print("\n⚠️ No _presence columns found. Found 'class' column (single-label).")
    print("\n=== CLASS DISTRIBUTION (Top 30) ===")
    df_raw.groupBy("class").count().orderBy(F.desc("count")).show(30, truncate=False)
    print("Unique classes:", df_raw.select("class").distinct().count())

else:
    print("\n⚠️ No _presence columns and no 'class' column found.")
    print("Columns are:", df_raw.columns)

print("\n✅ Spark analysis complete.")
spark.stop()

