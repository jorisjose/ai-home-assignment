import os
import io
import pandas as pd
from google.cloud import storage
from .config import SETTINGS

def _read_from_gcs(path: str) -> bytes:
    assert path.startswith("gs://")
    _, bucket_name, *blob_parts = path.split("/")
    blob_name = "/".join(blob_parts)
    client = storage.Client(project=SETTINGS.project_id)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    return blob.download_as_bytes()

def load_dataset(path: str) -> pd.DataFrame:
    if path.startswith("gs://"):
        raw = _read_from_gcs(path)
        return pd.read_csv(io.BytesIO(raw))
    else:
        return pd.read_csv(path)

def basic_clean(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    df = df.dropna(subset=[text_col]).copy()
    df[text_col] = df[text_col].astype(str).str.strip()
    df = df[df[text_col].str.len() > 0]
    return df

def eda_summary(df: pd.DataFrame, text_col: str) -> str:
    lengths = df[text_col].str.len()
    lines = [
        f"Rows: {len(df)}",
        f"Avg length: {lengths.mean():.1f}",
        f"Median length: {lengths.median():.1f}",
        f"95th pct length: {lengths.quantile(0.95):.1f}",
        f"Nulls in text col: {df[text_col].isna().sum()}",
        "Top 5 samples:",
    ]
    for i, t in enumerate(df[text_col].head(5).tolist(), 1):
        t = t.replace('\n', ' ')[:160]
        lines.append(f"{i}. {t}{'...' if len(t) == 160 else ''}")
    return "\n".join(lines)
