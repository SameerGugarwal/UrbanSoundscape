import os
import argparse
import numpy as np
import pandas as pd

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType

# ---- Lazy-loaded globals (loaded once per executor) ----
_MODEL = None
_LABELS = None
_FEATURE_COLS = None

def load_model_and_metadata(model_path: str, labels_path: str, feature_cols: list[str]):
    global _MODEL, _LABELS, _FEATURE_COLS
    if _MODEL is None:
        import joblib
        _MODEL = joblib.load(model_path)

    if _LABELS is None:
        with open(labels_path, "r") as f:
            _LABELS = [line.strip() for line in f if line.strip()]

    if _FEATURE_COLS is None:
        _FEATURE_COLS = feature_cols

def predict_partition(pdf_iter, model_path: str, labels_path: str, feature_cols: list[str], threshold: float, top_k: int):
    # Spark will feed iterator of pandas DataFrames (one per partition)
    load_model_and_metadata(model_path, labels_path, feature_cols)

    for pdf in pdf_iter:
        # Keep filename if present
        if "audio_filename" in pdf.columns:
            filenames = pdf["audio_filename"].astype(str).fillna("")
        else:
            filenames = pd.Series([""] * len(pdf))

        # Build X exactly like training: feature columns only
        X = pdf[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)

        # MultiOutputClassifier.predict_proba -> list of length n_labels
        probs_list = _MODEL.predict_proba(X)

        # Convert to (n_rows, n_labels) of P(1)
        # Each probs_list[i] is shape (n_rows, 2): [P(0), P(1)]
        p1 = np.column_stack([p[:, 1] for p in probs_list])

        # Top-K per row
        top_idx = np.argsort(-p1, axis=1)[:, :top_k]

        top_labels = []
        top_scores = []
        pred_labels = []
        pred_count = []

        for r in range(p1.shape[0]):
            idxs = top_idx[r].tolist()
            labels_r = [_LABELS[i] for i in idxs]
            scores_r = [float(p1[r, i]) for i in idxs]

            top_labels.append(",".join(labels_r))
            top_scores.append(",".join([f"{s:.4f}" for s in scores_r]))

            preds = [ _LABELS[i] for i in range(p1.shape[1]) if p1[r, i] >= threshold ]
            pred_labels.append(",".join(preds))
            pred_count.append(int(len(preds)))

        out = pd.DataFrame({
            "audio_filename": filenames,
            "top_labels": top_labels,
            "top_scores": top_scores,
            "predicted_labels": pred_labels,
            "predicted_count": pred_count
        })

        yield out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="datasets/processed/sonyc_clean.csv")
    parser.add_argument("--model", default="models/sound_rf_multilabel.pkl")
    parser.add_argument("--labels", default="models/label_names.txt")
    parser.add_argument("--output", default="results/spark_predictions")
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    spark = SparkSession.builder.appName("UrbanSoundscape-Spark-Batch-Inference").getOrCreate()

    # Read as strings (robust) then we’ll cast in pandas
    df_raw = spark.read.option("header", True).option("inferSchema", False).csv(args.input)

    # Detect multilabel columns and feature columns exactly like your training script
    label_cols = [c for c in df_raw.columns if c.endswith("_presence")]
    feature_cols = [c for c in df_raw.columns if c not in (["audio_filename"] + label_cols)]

    print("\n=== Spark Batch Inference Setup ===")
    print("Input:", args.input)
    print("Rows:", df_raw.count())
    print("Detected labels:", len(label_cols))
    print("Detected features:", len(feature_cols))
    print("Threshold:", args.threshold, "Top-K:", args.top_k)

    # We only need audio_filename + feature columns for inference
    select_cols = (["audio_filename"] if "audio_filename" in df_raw.columns else []) + feature_cols
    df = df_raw.select(*select_cols)

    # Output schema
    out_schema = StructType([
        StructField("audio_filename", StringType(), True),
        StructField("top_labels", StringType(), True),
        StructField("top_scores", StringType(), True),
        StructField("predicted_labels", StringType(), True),
        StructField("predicted_count", IntegerType(), True),
    ])

    # Run distributed inference
    pred_df = df.mapInPandas(
        lambda it: predict_partition(it, args.model, args.labels, feature_cols, args.threshold, args.top_k),
        schema=out_schema
    )

    # Save results
    os.makedirs("results", exist_ok=True)
    pred_df.write.mode("overwrite").option("header", True).csv(args.output)

    print("\n✅ Saved Spark predictions to:", args.output)
    spark.stop()

if __name__ == "__main__":
    main()
