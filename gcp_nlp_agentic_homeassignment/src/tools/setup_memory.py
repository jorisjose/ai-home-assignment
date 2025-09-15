import argparse
import os
from src.config import SETTINGS
from src.data_prep import load_dataset, basic_clean


def build_faiss_index(faiss_dir: str):
    try:
        from src.memory.persistence import FAISSMemory
    except Exception as e:
        print(f"[Info] FAISS unavailable: {e}")
        return
    df = load_dataset(SETTINGS.dataset_path)
    df = basic_clean(df, SETTINGS.text_col)
    mem = FAISSMemory(index_dir=faiss_dir, api_key=SETTINGS.google_api_key)
    texts = df[SETTINGS.text_col].astype(str).tolist()
    metas = [{"source": "dataset", "row_index": int(i)} for i in range(len(texts))]
    mem.upsert_texts(texts, metas)
    print(f"[OK] FAISS index built at {faiss_dir}")


def init_bigquery(dataset: str, table: str):
    try:
        from src.memory.persistence import BigQueryLogger
        logger = BigQueryLogger(dataset, table)
        logger.ensure_table()
        print(f"[OK] BigQuery table ensured: {dataset}.{table}")
    except Exception as e:
        print(f"[Info] BigQuery ensure_table failed or unavailable: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--faiss-dir", type=str, default=SETTINGS.faiss_dir)
    ap.add_argument("--bq-dataset", type=str, default=SETTINGS.bq_dataset)
    ap.add_argument("--bq-table", type=str, default=SETTINGS.bq_table)
    ap.add_argument("--faiss", action="store_true", help="build FAISS index from dataset")
    ap.add_argument("--bq", action="store_true", help="ensure BigQuery dataset/table exist")
    args = ap.parse_args()

    if args.faiss:
        build_faiss_index(args.faiss_dir)
    if args.bq and args.bq_dataset and args.bq_table:
        init_bigquery(args.bq_dataset, args.bq_table)
    if not (args.faiss or args.bq):
        # Run both if none specified
        build_faiss_index(args.faiss_dir)
        if args.bq_dataset and args.bq_table:
            init_bigquery(args.bq_dataset, args.bq_table)


if __name__ == "__main__":
    main()

