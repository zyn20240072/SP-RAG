import os
import numpy as np
import requests
import functools
from typing import List, Dict, Tuple, Any
from collections import defaultdict
from langchain_community.vectorstores import Chroma
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain_core.documents import Document
PERSIST_DIR_PRE = "./VRAG/final_before20260718/corpus_v2/chroma_data"

from swift.utils import get_logger
logger = get_logger()

HNSW_METADATA = {"hnsw:search_ef": 256}  


def process_mllm_vector(raw_vec):
    v = np.array(raw_vec)
    if v.ndim == 2:
        v = np.mean(v, axis=0)
    elif v.ndim == 3:
        v = np.mean(v, axis=1).squeeze()
    norm = np.linalg.norm(v)
    return v / (norm + 1e-9)

def _compute_cosine_sim(vec1, vec2):
    v1 = process_mllm_vector(vec1)
    v2 = process_mllm_vector(vec2)
    return float(np.dot(v1, v2))



class RemoteBGETextEmbedding:
    def __init__(self, endpoint_url: str = "http://localhost:8500/bge-m3-embed"):
        self.endpoint = endpoint_url

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        try:
            response = requests.post(self.endpoint, json={"texts": texts}, timeout=10)
            response.raise_for_status()
            return response.json()["embedding"]
        except Exception as e:
            logger.error(f"[RemoteBGETextEmbedding] Error connecting to {self.endpoint}: {e}")
            raise

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


class RemoteImagePathEmbedding:
    def __init__(self, endpoint_url: str = "http://localhost:8005/query_embed"):
        self.endpoint = endpoint_url

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        try:
            response = requests.post(self.endpoint, json={"texts": texts}, timeout=10)
            response.raise_for_status()
            return response.json()["embedding"]
        except Exception as e:
            logger.error(f"[RemoteImagePathEmbedding] Error connecting to {self.endpoint}: {e}")
            raise

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


TEXT_RETRIEVER_CACHE = {}
IMAGE_RETRIEVER_CACHE = {}
TABLE_RETRIEVER_CACHE = {}

def load_text_retriever(dataset: str) -> MultiVectorRetriever:
    if dataset in TEXT_RETRIEVER_CACHE:
        return TEXT_RETRIEVER_CACHE[dataset]

    persist_dir = f"{PERSIST_DIR_PRE}/{dataset}/text"
    if not os.path.exists(persist_dir):
        logger.warning(f"[WARNING] Persist directory DOES NOT EXIST: {persist_dir}. Chroma will create an EMPTY database!")
        
    embedder = RemoteBGETextEmbedding()
    vectorstore = Chroma(
        collection_name=f"{dataset}_text_data",
        embedding_function=embedder,
        persist_directory=persist_dir,
        collection_metadata=HNSW_METADATA 
    )
    retriever = MultiVectorRetriever(vectorstore=vectorstore, docstore=InMemoryStore(), id_key="doc_id")
    TEXT_RETRIEVER_CACHE[dataset] = retriever
    return retriever

def load_table_retriever() -> MultiVectorRetriever:
    dataset = "mutihop"
    if dataset in TABLE_RETRIEVER_CACHE:
        return TABLE_RETRIEVER_CACHE[dataset]

    persist_dir = f"{PERSIST_DIR_PRE}/{dataset}/table"
    embedder = RemoteBGETextEmbedding()
    vectorstore = Chroma(
        collection_name="mutihop_table_data",
        embedding_function=embedder,
        persist_directory=persist_dir,
        collection_metadata=HNSW_METADATA
    )
    retriever = MultiVectorRetriever(vectorstore=vectorstore, docstore=InMemoryStore(), id_key="doc_id")
    TABLE_RETRIEVER_CACHE[dataset] = retriever
    return retriever

def load_image_retriever(dataset: str) -> MultiVectorRetriever:
    if dataset in IMAGE_RETRIEVER_CACHE:
        return IMAGE_RETRIEVER_CACHE[dataset]

    persist_dir = f"{PERSIST_DIR_PRE}/{dataset}/image"
    if not os.path.exists(persist_dir):
        logger.warning(f"[WARNING] Image persist directory DOES NOT EXIST: {persist_dir}.")
        
    embedder = RemoteImagePathEmbedding()
    vectorstore = Chroma(
        collection_name=f"{dataset}_image_data",
        embedding_function=embedder,
        persist_directory=persist_dir,
        collection_metadata=HNSW_METADATA
    )
    retriever = MultiVectorRetriever(vectorstore=vectorstore, docstore=InMemoryStore(), id_key="doc_id")
    IMAGE_RETRIEVER_CACHE[dataset] = retriever
    return retriever


def _retrieve_generic(retriever, query_list: List[str], top_n: int, id_field: str):
    try:
        embedder = retriever.vectorstore._embedding_function
        raw_query_vecs = embedder.embed_documents(query_list)
        query_vecs = [process_mllm_vector(q) for q in raw_query_vecs]

        stats = defaultdict(lambda: {"hit": 0, "max_sim": -float("inf"), "content": ""})
        collection = retriever.vectorstore._collection

        for vec in query_vecs:
            results = collection.query(
                query_embeddings=[vec.tolist()],
                n_results=top_n,
                include=["embeddings", "metadatas", "documents"]
            )
            
            if not results.get("ids") or not results["ids"][0]:
                continue
                
            doc_embeddings = results["embeddings"][0]
            metadatas = results["metadatas"][0]
            documents = results["documents"][0]
            
            for doc_emb, metadata, doc_content in zip(doc_embeddings, metadatas, documents):
                doc_id = metadata.get("doc_id") or metadata.get("id") or "unknown_id"
                stats[doc_id]["hit"] += 1
                
                db_v_norm = process_mllm_vector(doc_emb)
                sim_score = np.dot(vec, db_v_norm) 
                
                if sim_score > stats[doc_id]["max_sim"]:
                    stats[doc_id]["max_sim"] = sim_score
                    if hasattr(retriever, "docstore") and doc_id != "unknown_id":
                        full_docs = retriever.docstore.mget([doc_id])
                        stats[doc_id]["content"] = full_docs[0].page_content if full_docs and full_docs[0] else doc_content
                    else:
                        stats[doc_id]["content"] = doc_content

        ranked = sorted([
                {
                    id_field: doc_id,
                    "text" if "image" not in id_field else "image_path": stat["content"],
                    "hit": stat["hit"],
                    "score": round(float(stat["max_sim"]), 6)
                }
                for doc_id, stat in stats.items()
            ], key=lambda x: (-x["hit"], -x["score"])
        )
        return ranked[:top_n]
    except Exception as e:
        logger.error(f"[_retrieve_generic] FATAL ERROR during retrieval: {e}")
        return []

def top_n_documents_by_queries(query_list: List[str], top_n: int, dataset: str) -> Dict:
    if dataset != "mutihop":
        retriever = load_text_retriever(dataset)
        ranked = _retrieve_generic(retriever, query_list, top_n, "text_id")
    else:
        half = top_n // 2
        text_res = _retrieve_generic(load_text_retriever(dataset), query_list, half, "text_id")
        table_res = _retrieve_generic(load_table_retriever(), query_list, top_n - half, "text_id")
        combined = text_res + table_res
        ranked = sorted(combined, key=lambda x: (-x["hit"], -x["score"]))[:top_n]
    return {"docs": ranked}

def top_n_images_by_text_query(query_texts: List[str], top_n: int, dataset: str) -> Dict:
    retriever = load_image_retriever(dataset)
    ranked = _retrieve_generic(retriever, query_texts, top_n, "image_id")
    return {"images": ranked}