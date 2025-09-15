import argparse
import os
import sys
import pandas as pd
from tqdm import tqdm
from .config import SETTINGS
from .data_prep import load_dataset, basic_clean, eda_summary
from .gcp_nlp import gcp_entities, gcp_sentiment
from .vertex_summarize import summarize_text
from .agent.workflow import run_agent

def _log(path, msg):
    with open(path, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

def pipeline(limit: int = None, text_col: str = None):
    os.makedirs("outputs", exist_ok=True)
    log_path = os.path.join("outputs", "log.txt")
    _log(log_path, f"Starting run; dataset={SETTINGS.dataset_path}")

    df = load_dataset(SETTINGS.dataset_path)
    text_col = text_col or SETTINGS.text_col
    df = basic_clean(df, text_col)
    if limit:
        df = df.head(limit)

    # EDA
    with open(os.path.join("outputs", "eda.txt"), "w", encoding="utf-8") as f:
        f.write(eda_summary(df, text_col))

    rows = []
    for i, row in tqdm(df.iterrows(), total=len(df)):
        text = row[text_col]
        try:
            ents = gcp_entities(text)
        except Exception as e:
            ents = {"error": str(e)}
        try:
            sent = gcp_sentiment(text)
        except Exception as e:
            sent = {"error": str(e)}
        try:
            summ = summarize_text(text)
        except Exception as e:
            summ = f"[Summary error] {e}"
        rows.append({"row_index": i, "original_text": text, "entities": ents, "sentiment": sent, "summary": summ})

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join("outputs", "results.csv"), index=False)
    _log(log_path, f"Completed. Wrote {len(out)} rows.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="process only first N rows")
    ap.add_argument("--text-col", type=str, default=None, help="override text column name")
    ap.add_argument("--agent", type=str, default=None, help="ask the lightweight agent a question")
    args = ap.parse_args()

    if args.agent:
        df = load_dataset(SETTINGS.dataset_path)
        df = basic_clean(df, SETTINGS.text_col)
        ans = run_agent(df, args.agent, SETTINGS.text_col)
        print("\n=== Agent Answer ===\n")
        print(ans["answer"])
        print("\n--- Support (top docs) ---")
        for i, item in enumerate(ans["support"], 1):
            print(f"\n[{i}] {item['summary'][:280]}")
        return

    pipeline(limit=args.limit, text_col=args.text_col)

if __name__ == "__main__":
    main()
