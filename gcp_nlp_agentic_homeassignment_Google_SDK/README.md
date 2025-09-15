# GCP NLP Agentic HomeAssignment

End-to-end prototype for **entity extraction, sentiment, and summarization** on GCP, plus a simple **agentic workflow** that chains these tools. Designed for fast setup and review in VS Code.

## Quick start

1) **Python & venv**
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

2) **Auth & env**
```bash
gcloud auth application-default login
cp .env.sample .env  # then edit with your project/bucket
```

3) **Run on a CSV dataset** (local path or `gs://` in `.env`)
```bash
python -m src.main --limit 10
```

This writes outputs to `outputs/`:
- `results.csv`: entities, sentiment, and summaries per row
- `eda.txt`: basic EDA
- `log.txt`

## Dataset format

Expect a CSV with a text column called `original_text`. If your column differs, pass `--text-col`.

## Agentic demo

The agent (`src/agent/workflow.py`) exposes a `run_agent(query)` function that:
1. Finds relevant rows
2. Extracts entities & sentiment
3. Generates a concise answer with supporting snippets

Use:
```bash
python -m src.main --agent "What are customers most upset about?"
```

## Notebooks

Drop any exploration notebooks in `notebooks/`. The codebase is the source of truth for the deliverables.

## Diagrams & Report

- `diagrams/architecture.png` – high-level GCP architecture (generated programmatically).
- `docs/Architecture_Agent_Design.pdf` – 2-page report with architecture & productionization notes.

## Notes

- If Vertex AI access isn't provisioned, set `USE_VERTEX_SUMMARY=false` to use TextRank fallback.
- To pull a CSV from GCS, set `DATASET_PATH` in `.env` to `gs://bucket/file.csv`.
