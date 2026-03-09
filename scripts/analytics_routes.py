from fastapi import APIRouter, HTTPException, Query
import pandas as pd
import os

router = APIRouter(prefix="/analytics", tags=["Analytics"])

TOP_LABELS_PATH = "results/dashboard_top_labels.csv"
COUNT_DIST_PATH = "results/dashboard_predicted_count_distribution.csv"
TOP_PAIRS_PATH  = "results/dashboard_top_pairs.csv"


def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"File not found: {path}")
    try:
        return pd.read_csv(path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read {path}: {e}")


@router.get("/top-labels")
def top_labels(limit: int = Query(10, ge=1, le=100)):
    df = load_csv(TOP_LABELS_PATH)
    df = df.sort_values("count", ascending=False).head(limit)
    return {"limit": limit, "data": df.to_dict(orient="records")}


@router.get("/predicted-count-distribution")
def predicted_count_distribution(max_count: int = Query(30, ge=0, le=200)):
    df = load_csv(COUNT_DIST_PATH)
    # Keep only up to max_count for cleaner charts
    df = df[df["predicted_count"] <= max_count].sort_values("predicted_count")
    return {"max_count": max_count, "data": df.to_dict(orient="records")}


@router.get("/top-pairs")
def top_pairs(limit: int = Query(10, ge=1, le=100)):
    df = load_csv(TOP_PAIRS_PATH)
    df = df.sort_values("count", ascending=False).head(limit)
    return {"limit": limit, "data": df.to_dict(orient="records")}

