from __future__ import annotations
from typing import List, Dict, Any, Optional
import os
import json
from datetime import datetime


class BigQueryLogger:
    def __init__(self, dataset: str, table: str):
        self.dataset = dataset
        self.table = table

    def _client(self):
        try:
            from google.cloud import bigquery  # type: ignore
        except Exception as e:
            raise ImportError("google-cloud-bigquery not available") from e
        return bigquery.Client()

    def ensure_table(self):
        client = self._client()
        dataset_ref = client.dataset(self.dataset)
        try:
            client.get_dataset(dataset_ref)
        except Exception:
            client.create_dataset(dataset_ref)
        table_ref = dataset_ref.table(self.table)
        from google.cloud import bigquery  # type: ignore
        schema = [
            bigquery.SchemaField("ts", "TIMESTAMP"),
            bigquery.SchemaField("query", "STRING"),
            bigquery.SchemaField("answer", "STRING"),
            bigquery.SchemaField("support", "STRING"),  # JSON string
        ]
        try:
            client.get_table(table_ref)
        except Exception:
            table = bigquery.Table(table_ref, schema=schema)
            client.create_table(table)

    def log_run(self, query: str, answer: str, support: List[Dict[str, Any]]):
        try:
            client = self._client()
            self.ensure_table()
            table_id = f"{client.project}.{self.dataset}.{self.table}"
            rows = [{
                "ts": datetime.utcnow().isoformat(),
                "query": query,
                "answer": answer,
                "support": json.dumps(support, ensure_ascii=False),
            }]
            errors = client.insert_rows_json(table_id, rows)
            if errors:
                raise RuntimeError(str(errors))
        except Exception:
            # Best-effort: don't fail agent on logging issues
            pass


class FAISSMemory:
    def __init__(self, index_dir: str, embedding_model: str = "text-embedding-004", api_key: Optional[str] = None):
        self.index_dir = index_dir
        self.embedding_model = embedding_model
        self.api_key = api_key
        self._vs = None

    def _embeddings(self):
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings  # type: ignore
        except Exception as e:
            raise ImportError("langchain-google-genai not available") from e
        return GoogleGenerativeAIEmbeddings(model=self.embedding_model, google_api_key=self.api_key)

    def _load(self):
        if self._vs is not None:
            return self._vs
        try:
            from langchain_community.vectorstores import FAISS  # type: ignore
            from langchain.docstore.document import Document  # type: ignore
        except Exception as e:
            raise ImportError("faiss or langchain community components not available") from e
        os.makedirs(self.index_dir, exist_ok=True)
        path = os.path.join(self.index_dir, "index")
        if os.path.exists(path):
            self._vs = FAISS.load_local(path, self._embeddings(), allow_dangerous_deserialization=True)
        else:
            # empty store
            self._vs = FAISS.from_documents([], self._embeddings())
            self._vs.save_local(path)
        return self._vs

    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        try:
            vs = self._load()
            docs = vs.similarity_search(query, k=k)
            return [{"text": d.page_content, **(d.metadata or {})} for d in docs]
        except Exception:
            return []

    def upsert_texts(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None):
        try:
            vs = self._load()
            from langchain.docstore.document import Document  # type: ignore
            docs = [Document(page_content=t, metadata=(metadatas[i] if metadatas else {})) for i, t in enumerate(texts)]
            vs.add_documents(docs)
            path = os.path.join(self.index_dir, "index")
            vs.save_local(path)
        except Exception:
            pass

