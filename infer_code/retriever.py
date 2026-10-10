import os
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import requests
from langchain_community.vectorstores import Chroma

from config import (
    EMBEDDING_MAX_RETRIES,
    EMBEDDING_TIMEOUT,
    PERSIST_DIR_PRE,
    STRICT_CORPUS_PATH,
)


HNSW_METADATA = {"hnsw:search_ef": 256}
@dataclass
class RetrieverBundle:
    vectorstore: Chroma


TEXT_RETRIEVER_CACHE: Dict[str, RetrieverBundle] = {}
IMAGE_RETRIEVER_CACHE: Dict[str, RetrieverBundle] = {}
TABLE_RETRIEVER_CACHE: Dict[str, RetrieverBundle] = {}


def _request_embeddings(endpoint: str, texts: List[str]) -> List[List[float]]:
    if not texts:
        return []

    last_error: Exception | None = None
    for attempt in range(1, max(1, EMBEDDING_MAX_RETRIES) + 1):
        try:
            response = requests.post(
                endpoint,
                json={"texts": texts},
                timeout=EMBEDDING_TIMEOUT,
            )
            response.raise_for_status()
            payload = response.json()
            embeddings = payload.get("embedding")
            if not isinstance(embeddings, list) or len(embeddings) != len(texts):
                raise ValueError(
                    "Embedding endpoint returned an invalid number of vectors: "
                    f"expected {len(texts)}, got "
                    f"{len(embeddings) if isinstance(embeddings, list) else type(embeddings)}"
                )
            return embeddings
        except Exception as exc:
            last_error = exc
            print(
                f"[Embedding] Attempt {attempt}/{EMBEDDING_MAX_RETRIES} "
                f"failed for {endpoint}: {exc}"
            )
            if attempt < EMBEDDING_MAX_RETRIES:
                time.sleep(min(2.0 * attempt, 5.0))

    raise RuntimeError(
        f"Embedding request failed after {EMBEDDING_MAX_RETRIES} attempts: {last_error}"
    ) from last_error


class RemoteBGETextEmbedding:
    def __init__(self, endpoint_url: str = "http://localhost:8500/bge-m3-embed"):
        self.endpoint = endpoint_url

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return _request_embeddings(self.endpoint, texts)

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


class RemoteImagePathEmbedding:
    def __init__(self, endpoint_url: str = "http://localhost:8005/query_embed"):
        self.endpoint = endpoint_url

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return _request_embeddings(self.endpoint, texts)

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


def _ensure_persist_dir(persist_dir: str, modality: str) -> None:
    if os.path.isdir(persist_dir):
        return
    message = (
        f"Persist directory does not exist for {modality}: {persist_dir}. "
        "Opening Chroma here would silently create an empty evaluation corpus."
    )
    if STRICT_CORPUS_PATH:
        raise FileNotFoundError(message)
    print(f"[WARNING] {message}")


def load_text_retriever(dataset: str) -> RetrieverBundle:
    if dataset in TEXT_RETRIEVER_CACHE:
        return TEXT_RETRIEVER_CACHE[dataset]

    persist_dir = f"{PERSIST_DIR_PRE}/{dataset}/text"
    _ensure_persist_dir(persist_dir, "text")
    vectorstore = Chroma(
        collection_name=f"{dataset}_text_data",
        embedding_function=RemoteBGETextEmbedding(),
        persist_directory=persist_dir,
        collection_metadata=HNSW_METADATA,
    )
    retriever = RetrieverBundle(vectorstore=vectorstore)
    TEXT_RETRIEVER_CACHE[dataset] = retriever
    return retriever


def load_table_retriever() -> RetrieverBundle:
    dataset = "mutihop"
    if dataset in TABLE_RETRIEVER_CACHE:
        return TABLE_RETRIEVER_CACHE[dataset]

    persist_dir = f"{PERSIST_DIR_PRE}/{dataset}/table"
    _ensure_persist_dir(persist_dir, "table")
    vectorstore = Chroma(
        collection_name="mutihop_table_data",
        embedding_function=RemoteBGETextEmbedding(),
        persist_directory=persist_dir,
        collection_metadata=HNSW_METADATA,
    )
    retriever = RetrieverBundle(vectorstore=vectorstore)
    TABLE_RETRIEVER_CACHE[dataset] = retriever
    return retriever


def load_image_retriever(dataset: str) -> RetrieverBundle:
    if dataset in IMAGE_RETRIEVER_CACHE:
        return IMAGE_RETRIEVER_CACHE[dataset]

    persist_dir = f"{PERSIST_DIR_PRE}/{dataset}/image"
    _ensure_persist_dir(persist_dir, "image")
    vectorstore = Chroma(
        collection_name=f"{dataset}_image_data",
        embedding_function=RemoteImagePathEmbedding(),
        persist_directory=persist_dir,
        collection_metadata=HNSW_METADATA,
    )
    retriever = RetrieverBundle(vectorstore=vectorstore)
    IMAGE_RETRIEVER_CACHE[dataset] = retriever
    return retriever


def _normalize_vector(raw_vector: Any) -> np.ndarray:
    vector = np.asarray(raw_vector, dtype=np.float32)
    if vector.ndim == 2:
        vector = vector.mean(axis=0)
    elif vector.ndim == 3:
        vector = vector.mean(axis=1).squeeze()
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError(f"Expected one non-empty embedding vector, got {vector.shape}")
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("Embedding vector has zero or non-finite norm")
    return vector / norm


def _retrieve_generic(
    retriever: RetrieverBundle,
    query_list: List[str],
    top_n: int,
    id_field: str,
) -> List[Dict[str, Any]]:
    """Retrieve directly from Chroma and raise infrastructure errors to the agent."""
    clean_queries = [" ".join(str(query).split()) for query in query_list if str(query).strip()]
    if not clean_queries or top_n <= 0:
        return []

    embedder = retriever.vectorstore._embedding_function
    raw_query_vectors = embedder.embed_documents(clean_queries)
    query_vectors = [_normalize_vector(vector) for vector in raw_query_vectors]

    stats: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"hit": 0, "max_sim": -float("inf"), "content": ""}
    )
    collection = retriever.vectorstore._collection

    for vector in query_vectors:
        results = collection.query(
            query_embeddings=[vector.tolist()],
            n_results=max(1, int(top_n)),
            include=["embeddings", "metadatas", "documents"],
        )
        raw_ids = results.get("ids")
        result_ids = raw_ids[0] if raw_ids is not None and len(raw_ids) else []
        if not result_ids:
            continue

        raw_embeddings = results.get("embeddings")
        raw_metadatas = results.get("metadatas")
        raw_documents = results.get("documents")
        embeddings = (
            raw_embeddings[0]
            if raw_embeddings is not None and len(raw_embeddings)
            else []
        )
        metadatas = (
            raw_metadatas[0]
            if raw_metadatas is not None and len(raw_metadatas)
            else [{} for _ in result_ids]
        )
        documents = (
            raw_documents[0]
            if raw_documents is not None and len(raw_documents)
            else ["" for _ in result_ids]
        )

        for record_id, document_vector, metadata, document_content in zip(
            result_ids,
            embeddings,
            metadatas,
            documents,
        ):
            metadata = metadata or {}
            document_id = str(metadata.get("doc_id") or record_id or "").strip()
            if not document_id:
                continue

            normalized_document_vector = _normalize_vector(document_vector)
            similarity = float(np.dot(vector, normalized_document_vector))
            stats[document_id]["hit"] += 1

            if similarity <= stats[document_id]["max_sim"]:
                continue

            stats[document_id]["max_sim"] = similarity
            # This pipeline queries Chroma directly; the stored document is the
            # authoritative raw text or image path used downstream.
            stats[document_id]["content"] = str(document_content or "")

    content_key = "image_path" if "image" in id_field else "text"
    ranked = sorted(
        [
            {
                id_field: document_id,
                content_key: state["content"],
                "hit": int(state["hit"]),
                "score": round(float(state["max_sim"]), 6),
            }
            for document_id, state in stats.items()
        ],
        key=lambda item: (-item["hit"], -item["score"]),
    )
    return ranked[:top_n]


def top_n_documents_by_queries(
    query_list: List[str],
    top_n: int,
    dataset: str,
) -> Dict[str, List[Dict[str, Any]]]:
    if top_n <= 0:
        return {"docs": []}

    if dataset != "mutihop":
        ranked = _retrieve_generic(
            load_text_retriever(dataset),
            query_list,
            top_n,
            "text_id",
        )
    else:
        text_count = (top_n + 1) // 2
        table_count = top_n - text_count
        text_results = _retrieve_generic(
            load_text_retriever(dataset),
            query_list,
            text_count,
            "text_id",
        )
        table_results = (
            _retrieve_generic(
                load_table_retriever(),
                query_list,
                table_count,
                "text_id",
            )
            if table_count > 0
            else []
        )
        ranked = sorted(
            text_results + table_results,
            key=lambda item: (-item["hit"], -item["score"]),
        )[:top_n]

    return {"docs": ranked}


def top_n_images_by_text_query(
    query_texts: List[str],
    top_n: int,
    dataset: str,
) -> Dict[str, List[Dict[str, Any]]]:
    if top_n <= 0:
        return {"images": []}
    ranked = _retrieve_generic(
        load_image_retriever(dataset),
        query_texts,
        top_n,
        "image_id",
    )
    return {"images": ranked}
