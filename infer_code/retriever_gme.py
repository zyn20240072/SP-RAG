#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import requests
import chromadb
from langchain_community.vectorstores import Chroma

from config import (
    EMBEDDING_MAX_RETRIES,
    EMBEDDING_TIMEOUT,
    STRICT_CORPUS_PATH,
)

GME_CORPUS_ROOT = os.getenv(
    "GME_CORPUS_ROOT",
    "./VRAG/final_before20260718/corpus_gme",
).rstrip("/")

GME_BASE_URL = os.getenv(
    "GME_BASE_URL",
    "http://127.0.0.1:8505",
).rstrip("/")

GME_TEXT_ENDPOINT = os.getenv(
    "GME_TEXT_ENDPOINT",
    f"{GME_BASE_URL}/gme/text",
)

GME_HEALTH_ENDPOINT = os.getenv(
    "GME_HEALTH_ENDPOINT",
    f"{GME_BASE_URL}/health",
)

TEXT_QUERY_INSTRUCTION = os.getenv(
    "GME_TEXT_QUERY_INSTRUCTION",
    "Given a query, retrieve relevant text passages that answer the query.",
)

IMAGE_QUERY_INSTRUCTION = os.getenv(
    "GME_IMAGE_QUERY_INSTRUCTION",
    "Find an image that matches the given text.",
)

GME_QUERY_BATCH_SIZE = max(
    1,
    int(os.getenv("GME_QUERY_BATCH_SIZE", "8")),
)


@dataclass
class RetrieverBundle:
    vectorstore: Chroma
    modality: str
    dataset: str
    collection_name: str


TEXT_RETRIEVER_CACHE: Dict[str, RetrieverBundle] = {}
IMAGE_RETRIEVER_CACHE: Dict[str, RetrieverBundle] = {}
TABLE_RETRIEVER_CACHE: Dict[str, RetrieverBundle] = {}


class RemoteGMEQueryEmbedding:
    def __init__(
        self,
        endpoint_url: str,
        query_instruction: str,
        *,
        timeout: int = EMBEDDING_TIMEOUT,
        max_retries: int = EMBEDDING_MAX_RETRIES,
    ):
        self.endpoint = endpoint_url
        self.query_instruction = query_instruction
        self.timeout = max(1, int(timeout))
        self.max_retries = max(1, int(max_retries))
        self.session = requests.Session()
        self.embedding_dim: Optional[int] = None

    def _request(
        self,
        texts: List[str],
        *,
        is_query: bool,
        instruction: Optional[str],
    ) -> List[List[float]]:
        if not texts:
            return []

        cleaned = [
            " ".join(str(text).split()) or "empty_query_placeholder"
            for text in texts
        ]

        last_error: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.session.post(
                    self.endpoint,
                    json={
                        "texts": cleaned,
                        "batch_size": min(len(cleaned), GME_QUERY_BATCH_SIZE),
                        "is_query": bool(is_query),
                        "instruction": instruction,
                    },
                    timeout=self.timeout,
                )
                response.raise_for_status()
                payload = response.json()
                embeddings = payload.get("embedding")

                if not isinstance(embeddings, list):
                    raise TypeError(
                        "GME endpoint returned a non-list 'embedding' field."
                    )

                if len(embeddings) != len(cleaned):
                    raise ValueError(
                        "GME endpoint returned an invalid number of vectors: "
                        f"expected={len(cleaned)}, actual={len(embeddings)}"
                    )

                validated: List[List[float]] = []
                for index, raw_vector in enumerate(embeddings):
                    if not isinstance(raw_vector, list) or not raw_vector:
                        raise ValueError(
                            f"GME vector {index} is empty or invalid."
                        )
                    vector = [float(value) for value in raw_vector]

                    if self.embedding_dim is None:
                        self.embedding_dim = len(vector)
                    elif len(vector) != self.embedding_dim:
                        raise ValueError(
                            "GME embedding dimension changed during retrieval: "
                            f"expected={self.embedding_dim}, actual={len(vector)}"
                        )
                    validated.append(vector)

                return validated

            except Exception as exc:
                last_error = exc
                detail = str(exc)
                response = getattr(exc, "response", None)
                if response is not None:
                    try:
                        detail += f" | response={response.text[:1000]}"
                    except Exception:
                        pass

                print(
                    f"[GME Query] Attempt {attempt}/{self.max_retries} "
                    f"failed: {detail}"
                )

                if attempt < self.max_retries:
                    time.sleep(min(2.0 * attempt, 5.0))

        raise RuntimeError(
            f"GME query embedding failed after "
            f"{self.max_retries} attempts: {last_error}"
        ) from last_error

    def embed_query(self, text: str) -> List[float]:
        return self._request(
            [text],
            is_query=True,
            instruction=self.query_instruction,
        )[0]

    def embed_queries(self, texts: List[str]) -> List[List[float]]:
        return self._request(
            texts,
            is_query=True,
            instruction=self.query_instruction,
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._request(
            texts,
            is_query=False,
            instruction=None,
        )


class RemoteGMETextQueryEmbedding(RemoteGMEQueryEmbedding):
    def __init__(self):
        super().__init__(
            endpoint_url=GME_TEXT_ENDPOINT,
            query_instruction=TEXT_QUERY_INSTRUCTION,
        )


class RemoteGMEImageQueryEmbedding(RemoteGMEQueryEmbedding):
    def __init__(self):
        super().__init__(
            endpoint_url=GME_TEXT_ENDPOINT,
            query_instruction=IMAGE_QUERY_INSTRUCTION,
        )


def _ensure_persist_dir(persist_dir: str, modality: str) -> None:
    if os.path.isdir(persist_dir):
        return

    message = (
        f"Persist directory does not exist for {modality}: {persist_dir}. "
        "Opening Chroma here would create an empty evaluation corpus."
    )
    if STRICT_CORPUS_PATH:
        raise FileNotFoundError(message)
    print(f"[WARNING] {message}")


def _list_collection_names(persist_dir: str) -> List[str]:
    client = chromadb.PersistentClient(path=persist_dir)
    raw = client.list_collections()

    names: List[str] = []
    for item in raw:
        if isinstance(item, str):
            names.append(item)
        else:
            name = getattr(item, "name", None)
            if name:
                names.append(str(name))
    return names


def _ensure_collection_exists(
    persist_dir: str,
    collection_name: str,
    modality: str,
) -> None:
    names = _list_collection_names(persist_dir)

    if collection_name in names:
        return

    raise RuntimeError(
        f"GME {modality} collection not found.\n"
        f"persist_dir={persist_dir}\n"
        f"expected={collection_name}\n"
        f"available={names}\n"
        "Refusing to create a new empty collection during evaluation."
    )


def _open_gme_collection(
    *,
    dataset: str,
    data_type: str,
    embedding_function: RemoteGMEQueryEmbedding,
) -> RetrieverBundle:
    persist_dir = os.path.join(
        GME_CORPUS_ROOT,
        dataset,
        data_type,
    )
    collection_name = f"gme_{dataset}_{data_type}_data"

    _ensure_persist_dir(persist_dir, data_type)
    _ensure_collection_exists(
        persist_dir,
        collection_name,
        data_type,
    )

    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_function,
        persist_directory=persist_dir,
    )

    count = int(vectorstore._collection.count())
    if count <= 0:
        raise RuntimeError(
            f"GME collection is empty: {collection_name} at {persist_dir}"
        )

    print(
        f"[GME Retriever] loaded dataset={dataset} "
        f"type={data_type} collection={collection_name} n={count}"
    )

    return RetrieverBundle(
        vectorstore=vectorstore,
        modality=data_type,
        dataset=dataset,
        collection_name=collection_name,
    )


def load_text_retriever(dataset: str) -> RetrieverBundle:
    if dataset not in TEXT_RETRIEVER_CACHE:
        TEXT_RETRIEVER_CACHE[dataset] = _open_gme_collection(
            dataset=dataset,
            data_type="text",
            embedding_function=RemoteGMETextQueryEmbedding(),
        )
    return TEXT_RETRIEVER_CACHE[dataset]


def load_table_retriever() -> RetrieverBundle:
    dataset = "mutihop"
    if dataset not in TABLE_RETRIEVER_CACHE:
        TABLE_RETRIEVER_CACHE[dataset] = _open_gme_collection(
            dataset=dataset,
            data_type="table",
            embedding_function=RemoteGMETextQueryEmbedding(),
        )
    return TABLE_RETRIEVER_CACHE[dataset]


def load_image_retriever(dataset: str) -> RetrieverBundle:
    if dataset not in IMAGE_RETRIEVER_CACHE:
        IMAGE_RETRIEVER_CACHE[dataset] = _open_gme_collection(
            dataset=dataset,
            data_type="image",
            embedding_function=RemoteGMEImageQueryEmbedding(),
        )
    return IMAGE_RETRIEVER_CACHE[dataset]


def _normalize_vector(raw_vector: Any) -> np.ndarray:
    vector = np.asarray(raw_vector, dtype=np.float32)

    if vector.ndim == 2:
        vector = vector.mean(axis=0)
    elif vector.ndim == 3:
        vector = vector.mean(axis=1).squeeze()

    if vector.ndim != 1 or vector.size == 0:
        raise ValueError(
            f"Expected one non-empty embedding vector, got {vector.shape}"
        )

    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError(
            "Embedding vector has zero or non-finite norm"
        )

    return vector / norm


def _canonical_result_id(
    *,
    record_id: Any,
    metadata: Dict[str, Any],
    modality: str,
) -> str:
    if modality in {"text", "table"}:
        value = (
            metadata.get("parent_id")
            or metadata.get("doc_id")
            or record_id
        )
    else:
        value = (
            metadata.get("doc_id")
            or metadata.get("parent_id")
            or record_id
        )

    return str(value or "").strip()


def _canonical_content(
    *,
    document_content: Any,
    metadata: Dict[str, Any],
    modality: str,
) -> str:
    if modality == "image":
        return str(
            metadata.get("source_path")
            or document_content
            or ""
        )

    return str(document_content or "")


def _retrieve_generic(
    retriever: RetrieverBundle,
    query_list: List[str],
    top_n: int,
    id_field: str,
) -> List[Dict[str, Any]]:
    clean_queries = [
        " ".join(str(query).split())
        for query in query_list
        if str(query).strip()
    ]

    if not clean_queries or top_n <= 0:
        return []

    embedder = retriever.vectorstore._embedding_function

    if hasattr(embedder, "embed_queries"):
        raw_query_vectors = embedder.embed_queries(clean_queries)
    else:
        raw_query_vectors = [
            embedder.embed_query(query)
            for query in clean_queries
        ]

    query_vectors = [
        _normalize_vector(vector)
        for vector in raw_query_vectors
    ]

    stats: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "hit": 0,
            "max_sim": -float("inf"),
            "content": "",
        }
    )

    collection = retriever.vectorstore._collection
    collection_count = int(collection.count())

    if collection_count <= 0:
        return []

    n_results = min(
        collection_count,
        max(1, int(top_n)),
    )

    for vector in query_vectors:
        results = collection.query(
            query_embeddings=[vector.tolist()],
            n_results=n_results,
            include=[
                "embeddings",
                "metadatas",
                "documents",
            ],
        )

        raw_ids = results.get("ids")
        result_ids = (
            raw_ids[0]
            if raw_ids is not None and len(raw_ids)
            else []
        )

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

        for (
            record_id,
            document_vector,
            metadata,
            document_content,
        ) in zip(
            result_ids,
            embeddings,
            metadatas,
            documents,
        ):
            metadata = metadata or {}

            document_id = _canonical_result_id(
                record_id=record_id,
                metadata=metadata,
                modality=retriever.modality,
            )
            if not document_id:
                continue

            normalized_document_vector = _normalize_vector(
                document_vector
            )

            similarity = float(
                np.dot(
                    vector,
                    normalized_document_vector,
                )
            )

            state = stats[document_id]
            state["hit"] += 1

            if similarity <= state["max_sim"]:
                continue

            state["max_sim"] = similarity
            state["content"] = _canonical_content(
                document_content=document_content,
                metadata=metadata,
                modality=retriever.modality,
            )

    content_key = (
        "image_path"
        if id_field == "image_id"
        else "text"
    )

    ranked = sorted(
        [
            {
                id_field: document_id,
                content_key: state["content"],
                "hit": int(state["hit"]),
                "score": round(
                    float(state["max_sim"]),
                    6,
                ),
            }
            for document_id, state in stats.items()
        ],
        key=lambda item: (
            -item["hit"],
            -item["score"],
        ),
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

        merged: Dict[str, Dict[str, Any]] = {}

        for item in text_results + table_results:
            text_id = str(item["text_id"])

            if text_id not in merged:
                merged[text_id] = dict(item)
                continue

            old = merged[text_id]
            old["hit"] = max(
                int(old.get("hit", 0)),
                int(item.get("hit", 0)),
            )

            if float(item.get("score", -1e9)) > float(
                old.get("score", -1e9)
            ):
                old["score"] = item["score"]
                old["text"] = item["text"]

        ranked = sorted(
            merged.values(),
            key=lambda item: (
                -int(item["hit"]),
                -float(item["score"]),
            ),
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


def check_gme_service() -> Dict[str, Any]:
    response = requests.get(
        GME_HEALTH_ENDPOINT,
        timeout=min(60, EMBEDDING_TIMEOUT),
    )
    response.raise_for_status()
    return response.json()


def preflight_gme_corpus(
    datasets: Optional[List[str]] = None,
) -> None:
    datasets = datasets or [
        "slide",
        "videoseek",
        "mmlongbench",
        "mutihop",
    ]

    health = check_gme_service()
    print(f"[GME] service health: {health}")
    print(f"[GME] corpus root: {GME_CORPUS_ROOT}")

    for dataset in datasets:
        text_bundle = load_text_retriever(dataset)
        image_bundle = load_image_retriever(dataset)

        print(
            f"[GME] {dataset}: "
            f"text={text_bundle.vectorstore._collection.count()} "
            f"image={image_bundle.vectorstore._collection.count()}"
        )

        if dataset == "mutihop":
            table_bundle = load_table_retriever()
            print(
                f"[GME] {dataset}: "
                f"table={table_bundle.vectorstore._collection.count()}"
            )


if __name__ == "__main__":
    preflight_gme_corpus()