from pyspark.sql import SparkSession
import pandas as pd
import joblib

# Start Spark
spark = SparkSession.builder.appName("UrbanSoundInference").getOrCreate()
print("Spark session started")

# Load clean dataset using Spark
df = spark.read.csv("../datasets/processed/sonyc_clean.csv", header=True, inferSchema=True)

print("Total rows in Spark DataFrame:", df.count())

# Convert to Pandas for ML model
pdf = df.toPandas()

# Load trained model
model = joblib.load("../models/sound_rf_multilabel.pkl")

# Separate features
label_cols = [c for c in pdf.columns if c.endswith("_presence")]
X = pdf.drop(columns=["audio_filename"] + label_cols)

print("Running ML inference from Spark data...")

preds = model.predict(X)

print("Prediction shape:", preds.shape)

spark.stop()
print("Spark stopped")
