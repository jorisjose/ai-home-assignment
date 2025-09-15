import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Settings:
    project_id: str = os.getenv("GOOGLE_CLOUD_PROJECT", "")
    region: str = os.getenv("GCP_REGION", "us-central1")
    gcs_bucket: str = os.getenv("GCS_BUCKET", "")
    dataset_path: str = os.getenv("DATASET_PATH", "data/sample_reviews.csv")
    use_vertex_summary: bool = os.getenv("USE_VERTEX_SUMMARY", "true").lower() == "true"
    gemini_model: str = os.getenv("MODEL_GEMINI", "gemini-1.5-flash")
    text_col: str = os.getenv("TEXT_COL", "original_text")

SETTINGS = Settings()
