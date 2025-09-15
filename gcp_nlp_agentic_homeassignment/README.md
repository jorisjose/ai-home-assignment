# GCP NLP Agentic HomeAssignment

End-to-end prototype for **entity extraction, sentiment, and summarization** on GCP, plus a simple **agentic workflow** that chains these tools. Designed for fast setup and review in VS Code.

## Quick start

1) **Python & venv**
```bash
python -m venv .venv
# macOS/Linux (bash/zsh):
source .venv/bin/activate
# Windows PowerShell:
./.venv/Scripts/Activate.ps1
# Windows cmd.exe:
.venv\Scripts\activate.bat

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

LangGraph agent (recommended):
```bash
python -m src.main --agent "What are customers most upset about?" --agent-mode langgraph
```

## Notebooks

Drop any exploration notebooks in `notebooks/`. The codebase is the source of truth for the deliverables.

## Diagrams & Report

- `diagrams/architecture.png` – high-level GCP architecture (generated programmatically).
- `docs/Architecture_Agent_Design.pdf` – 2-page report with architecture & productionization notes.

## Notes

- If Vertex AI access isn't provisioned, set `USE_VERTEX_SUMMARY=false`.
- To pull a CSV from GCS, set `DATASET_PATH` in `.env` to `gs://bucket/file.csv`.

## Memory & Persistence (Optional)

- FAISS Vector Memory (local):
  - Build index from your dataset:
    - `python src/tools/setup_memory.py --faiss-dir outputs/faiss_index`
  - Use with LangGraph agent:
    - `python -m src.main --agent "..." --agent-mode langgraph --use-faiss --faiss-dir outputs/faiss_index`
  - Env options: `USE_FAISS_MEMORY=true`, `FAISS_DIR=outputs/faiss_index`

- BigQuery Logging (run history):
  - Create dataset/table and verify access:
    - `python src/tools/setup_memory.py --bq-dataset YOUR_DATASET --bq-table runs`
  - Use with agent:
    - `python -m src.main --agent "..." --agent-mode langgraph --use-bq --bq-dataset YOUR_DATASET --bq-table runs`
  - Env options: `BQ_DATASET=...`, `BQ_TABLE=...`

Notes:
- Requires ADC for BigQuery: `gcloud auth application-default login`
- FAISS requires a Gemini API key for embeddings (`GOOGLE_API_KEY`).
