import os
import numpy as np
import requests
import functools
from typing import List, Dict, Tuple, Any
from langchain_community.vectorstores import Chroma

# ⚡️ 引入 logger，替代所有 print
from swift.utils import get_logger
from retriever import *

logger = get_logger()

# ===============================
# 公共向量处理函数 (确保打分和检索绝对对齐)
# ===============================
def process_mllm_vector(raw_vec):
    """通用的向量强归一化处理"""
    v = np.array(raw_vec)
    if v.ndim == 2:
        v = np.mean(v, axis=0)
    elif v.ndim == 3:
        v = np.mean(v, axis=1).squeeze()
    norm = np.linalg.norm(v)
    return v / (norm + 1e-9)

def _compute_cosine_sim(vec1, vec2):
    """计算两个原始向量归一化后的余弦相似度"""
    v1 = process_mllm_vector(vec1)
    v2 = process_mllm_vector(vec2)
    sim = np.dot(v1, v2)
    
    # 🛡️ 终极防御：强制转换为 Python 标量，杜绝任何矩阵类型的残留
    try:
        if hasattr(sim, 'item'):
            return float(sim.item())
        if isinstance(sim, (list, tuple, np.ndarray)):
            return float(np.array(sim).flatten()[0])
        return float(sim)
    except Exception as e:
        logger.error(f"[Toolbox] failed to convert sim to float: {e}")
        return 0.0

# -------------------------------
# Toolbox 类 (带有神级离线打分功能)
# -------------------------------
class Toolbox:
    def __init__(self, dataset: str, top_n: int = 5):
        if not dataset: raise ValueError("Dataset name cannot be empty.")
        self.dataset = dataset
        self.top_n = top_n
        
        self.text_embedder = RemoteBGETextEmbedding()
        self.image_embedder = RemoteImagePathEmbedding()
        logger.info(f"[Toolbox] Initialized for dataset '{self.dataset}' with top_n={self.top_n}.")

    # 🚀 并发缓存优化：拦截重复 Query，绝不请求两次！
    @functools.lru_cache(maxsize=2048)
    def _cached_text_embed(self, text: str) -> List[float]:
        return self.text_embedder.embed_query(text)

    @functools.lru_cache(maxsize=2048)
    def _cached_image_embed(self, query: str) -> List[float]:
        return self.image_embedder.embed_query(query)

    # 🌟 新增：通用文本对齐打分 (替代 BGE-Small)
    def calculate_text_sim_pair(self, text1: str, text2: str) -> float:
        if not text1 or not text2: return 0.0
        vec1 = self._cached_text_embed(text1)
        vec2 = self._cached_text_embed(text2)
        return _compute_cosine_sim(vec1, vec2)

    # 🌟 新增：Planner 离线文本库打分
    def calculate_text_sim(self, query: str, target_text_ids: List[str]) -> float:
        if not query or not target_text_ids: return 0.0

        q_vec = self._cached_text_embed(query)
        retriever = load_text_retriever(self.dataset)
        collection = retriever.vectorstore._collection

        try:
            # 🚀 极致性能：直接通过 Chroma 底层 API，提取已经算好的目标特征向量！
            results = collection.get(
                where={"doc_id": {"$in": target_text_ids}},
                include=["embeddings"]
            )
            target_embeddings = results.get("embeddings")

            # ✅ 修复点 1：安全判空，完美兼容 NumPy 数组和 Python 列表
            if target_embeddings is None or len(target_embeddings) == 0:
                docs = retriever.docstore.mget(target_text_ids)
                texts = [d.page_content for d in docs if d]
                if not texts: return 0.0
                target_embeddings = self.text_embedder.embed_documents(texts)

            max_sim = 0.0
            for t_vec in target_embeddings:
                sim = _compute_cosine_sim(q_vec, t_vec)
                if sim > max_sim: max_sim = sim
                
            return max_sim
        except Exception as e:
            logger.error(f"[Toolbox] Offline Text sim failed: {e}")
            return 0.0

    # 🌟 新增：Planner 离线跨模态打分 (BGE-VL)
    def calculate_image_sim(self, query: str, target_image_ids: List[str]) -> float:
        if not query or not target_image_ids: return 0.0

        q_vec = self._cached_image_embed(query)
        retriever = load_image_retriever(self.dataset)
        collection = retriever.vectorstore._collection

        try:
            # 🚀 极致跨模态：直接提取存好的原图特征！Query 文本直接与图像向量对撞！
            results = collection.get(
                where={"doc_id": {"$in": target_image_ids}},
                include=["embeddings"]
            )
            target_embeddings = results.get("embeddings")

            # ✅ 修复点 2：安全判空，完美兼容 NumPy 数组和 Python 列表
            if target_embeddings is None or len(target_embeddings) == 0:
                docs = retriever.docstore.mget(target_image_ids)
                paths = [d.page_content for d in docs if d]
                if not paths: return 0.0
                target_embeddings = self.image_embedder.embed_documents(paths)

            max_sim = 0.0
            for t_vec in target_embeddings:
                sim = _compute_cosine_sim(q_vec, t_vec)
                if sim > max_sim: max_sim = sim
                
            return max_sim
        except Exception as e:
            logger.error(f"[Toolbox] Offline Image sim failed: {e}")
            return 0.0

    # ------ 原始搜索方法保持不变 ------
    def text_search(self, query: str, top_n: int = None) -> List[Dict[str, Any]]:
        num_results = top_n if top_n is not None else self.top_n
        logger.info(f"[Toolbox] text_search executing: query='{query}'")
        retrieved_data = top_n_documents_by_queries(query_list=[query], top_n=num_results, dataset=self.dataset)
        assets = []
        for doc in retrieved_data.get("docs", []):
            assets.append({
                "asset_id": f"text_{doc.get('text_id', 'unknown')}",
                "type": "text",
                "content": doc.get('text', ''),
                "metadata": {"hit": doc.get('hit', 0), "score": doc.get('score', 0.0)}
            })
        logger.info(f"[Toolbox] Found {len(assets)} text assets")
        return assets

    def image_search(self, query: str, top_n: int = None) -> List[Dict[str, Any]]:
        num_results = top_n if top_n is not None else self.top_n
        logger.info(f"[Toolbox] image_search executing: query='{query}'")
        retrieved_data = top_n_images_by_text_query(query_texts=[query], top_n=num_results, dataset=self.dataset)
        assets = []
        for img in retrieved_data.get("images", []):
            assets.append({
                "asset_id": f"img_{img.get('image_id', 'unknown')}",
                "type": "image",
                "content": img.get('image_path', ''),
                "metadata": {"hit": img.get('hit', 0), "score": img.get('score', 0.0)}
            })
        logger.info(f"[Toolbox] Found {len(assets)} image assets")
        return assets