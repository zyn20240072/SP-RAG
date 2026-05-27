import os
import time
import json
import shutil
import requests
import glob
from tqdm import tqdm
from typing import List, Dict, Union
from langchain_community.vectorstores import Chroma
from langchain.storage import InMemoryStore
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# -------------------------------
# 嵌入器类定义 (强化日志版 Fallback)
# -------------------------------
class RemoteBGETextEmbedding:
    def __init__(self, endpoint_url: str = "http://localhost:8500/bge-m3-embed", batch_size: int = 4):
        self.endpoint = endpoint_url
        self.batch_size = batch_size

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for i in tqdm(range(0, len(texts), self.batch_size), desc="🔠 文本嵌入中"):
            batch = texts[i:i + self.batch_size]
            try:
                time.sleep(1)
                response = requests.post(self.endpoint, json={
                    "texts": batch,
                    "batch_size": len(batch), 
                    "max_length": 8192
                }, timeout=120)
                response.raise_for_status()
                
                batch_embeddings = response.json()["embedding"]
                
                if len(batch_embeddings) != len(batch):
                    print(f"\n⚠️ [警告] 批次对齐失败: 传给服务端 {len(batch)} 条文本，但只返回了 {len(batch_embeddings)} 个向量！")
                    print("⚠️ 启动单条隔离与修复机制...")
                    batch_embeddings = self._fallback_embed(batch)
                    
                embeddings.extend(batch_embeddings)
                
            except Exception as e:
                error_msg = str(e)
                if hasattr(e, 'response') and e.response is not None:
                    error_msg += f" | 响应内容: {e.response.text}"
                print(f"\n❌ [批次请求崩溃] 错误信息: {error_msg}")
                print("❌ 转入逐条单发模式排查具体是哪条文本导致的...")
                batch_embeddings = self._fallback_embed(batch)
                embeddings.extend(batch_embeddings)
                
        return embeddings

    def _fallback_embed(self, batch: List[str]) -> List[List[float]]:
        repaired_embeddings = []
        for idx, text in enumerate(batch):
            try:
                time.sleep(1)
                res = requests.post(self.endpoint, json={
                    "texts": [text],
                    "batch_size": 1,
                    "max_length": 8192
                }, timeout=30)
                res.raise_for_status()
                single_emb = res.json()["embedding"]
                
                if len(single_emb) == 1:
                    repaired_embeddings.append(single_emb[0])
                else:
                    raise ValueError(f"服务端异常：期望返回 1 个向量，实际返回 {len(single_emb)} 个。")
            except Exception as e:
                error_detail = str(e)
                if hasattr(e, 'response') and e.response is not None:
                    error_detail += f" | 服务端底层报错: {e.response.text}"
                
                print(f"\n  -> 🚨 【单条文本嵌入失败】 详细原因: {error_detail}")
                print(f"  -> 📝 【罪魁祸首文本】 (字符长度: {len(text)}):")
                print(f"     {repr(text)[:300]}...") 
                print(f"  -> 🛠️ 正在为其生成安全的占位符向量以保证数据不丢弃...\n")
                
                safe_res = requests.post(self.endpoint, json={
                    "texts": ["empty_document_placeholder"], 
                    "batch_size": 1
                })
                repaired_embeddings.append(safe_res.json()["embedding"][0])
                
        return repaired_embeddings

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


class RemoteImagePathEmbedding:
    def __init__(self, endpoint_url: str = "http://localhost:8005/embed"):
        self.endpoint = endpoint_url

    def embed_documents(self, image_paths: List[str]) -> List[List[float]]:
        embeddings = []
        for path in tqdm(image_paths, desc="🖼️ 图像嵌入中"):
            try:
                time.sleep(1)
                response = requests.post(self.endpoint, json={"image_path": path}, timeout=120)
                if response.status_code != 200:
                    print(f"❌ Image embedding failed for {path}: {response.text}")
                    embeddings.append(None)
                    continue
                embeddings.append(response.json()["embedding"])
            except Exception as e:
                print(f"❌ Failed to process {path}, error: {e}")
                embeddings.append(None)
        return embeddings

    def embed_query(self, image_path: str) -> List[float]:
        return self.embed_documents([image_path])[0]


EMBEDDERS = {
    "text": RemoteBGETextEmbedding(),
    "image": RemoteImagePathEmbedding(),
}


def add_embeddings_to_chroma(chroma_store, documents, embeddings, batch_size=500):
    assert len(documents) == len(embeddings), "❗文档数量与嵌入数量不一致！"
    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i:i + batch_size]
        batch_embeds = embeddings[i:i + batch_size]
        
        # ✨ 核心修复：动态获取主键，优先找 chunk_id，找不到再找 doc_id
        batch_ids = [
            doc.metadata.get("chunk_id") or doc.metadata.get("doc_id") 
            for doc in batch_docs
        ]
        
        # 兜底安全检查，防止有空的 ID 混入导致 Chroma 崩溃
        if not all(batch_ids):
            raise ValueError("🚨 发现缺少唯一 ID 的文档！请检查 metadata 构造逻辑。")

        chroma_store._collection.add(
            embeddings=batch_embeds,
            documents=[doc.page_content for doc in batch_docs],
            metadatas=[doc.metadata for doc in batch_docs],
            ids=batch_ids,
        )

# -------------------------------
# 数据规范化加载与格式化
# -------------------------------
def load_json(path: str) -> Union[List[Dict], Dict]:
    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line.strip()) for line in f if line.strip()]
    else:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

def normalize_dict_to_list(raw_data: Union[List[Dict], Dict], default_field: str) -> List[Dict]:
    if isinstance(raw_data, dict):
        data_list = []
        for k, v in raw_data.items():
            if isinstance(v, dict):
                v_copy = v.copy()
                v_copy["id"] = k
                data_list.append(v_copy)
            else:
                data_list.append({"id": k, default_field if default_field else "content": v})
        return data_list
    return raw_data

def format_table_to_markdown(doc: dict, field: str) -> str:
    content = doc.get(field) if field else doc
    if not isinstance(content, dict) or "table_rows" not in content:
        if isinstance(content, dict):
            return json.dumps(content, ensure_ascii=False)
        return str(content) if content is not None else ""

    global_title = doc.get("title", "Unknown Title")
    table_name = content.get("table_name", "Unknown Table")
    headers = []
    if "header" in content:
        headers = [h.get("column_name", "") for h in content["header"]]
        
    lines = []
    if global_title and global_title != "Unknown Title":
        lines.append(f"Document Title: {global_title}")
    if table_name and table_name != "Unknown Table":
        lines.append(f"Table Name: {table_name}")
    if lines: lines.append("") 
    
    if headers:
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("|" + "|".join(["---"] * len(headers)) + "|")
        
    for row in content.get("table_rows", []):
        row_texts = []
        for cell in row:
            cell_text = str(cell.get("text", "") if isinstance(cell, dict) else cell)
            cell_text = cell_text.replace("\n", " ").replace("|", "\|").strip()
            row_texts.append(cell_text)
        if headers: lines.append("| " + " | ".join(row_texts) + " |")
        else: lines.append(" | ".join(row_texts))
            
    return "\n".join(lines)


# -------------------------------
# Retriever 创建器 (注入 Cosine 灵魂)
# -------------------------------
persist_directory = "/root/local/corpus_v2/chroma_data/"

CHROMA_METADATA = {
    "hnsw:space": "cosine", 
    "hnsw:search_ef": 256
}

def get_vectorstore(dataset_name: str, data_type: str):
    """通用的 Chroma 获取器，不要 MultiVectorRetriever 了"""
    persist_path = os.path.join(persist_directory, dataset_name, data_type)
    return Chroma(
        collection_name=f"{dataset_name}_{data_type}_data",
        embedding_function=EMBEDDERS[data_type] if data_type == "image" else EMBEDDERS["text"],
        persist_directory=persist_path,
        collection_metadata=CHROMA_METADATA 
    )

# -------------------------------
# 路径一：普通文本处理
# -------------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=100,
    separators=["\n\n", "\n", "。", "！", "？", ".", " ", ""]
)

def process_text_dataset(dataset_name: str, config: dict):
    path, field, embedder = config["path"], config["field"], EMBEDDERS[config["embedder"]]
    print(f"\n📝 处理普通文本: {dataset_name} | 文件: {path}")

    vectorstore = get_vectorstore(dataset_name, "text")
    existing_ids = set(vectorstore.get(include=[])["ids"])

    try:
        raw_data = load_json(path)
        data = normalize_dict_to_list(raw_data, field)
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return

    sub_docs, sub_contents = [], []
    total_saved = 0
    
    for doc in tqdm(data, desc=f"{dataset_name}-text"):
        doc_id = str(doc.get("id") or doc.get("text_id"))
        content = str(doc.get(field) or "").strip()
        if not content: continue

        chunks = text_splitter.split_text(content)
        for i, chunk_text in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk_{i}"
            if chunk_id in existing_ids: continue 
            
            sub_docs.append(Document(
                page_content=chunk_text,
                metadata={
                    "chunk_id": chunk_id,
                    "parent_id": doc_id,         
                    "source_path": path,         
                    "dataset": dataset_name
                }
            ))
            sub_contents.append(chunk_text)

            # ✨ 防 OOM 优化：每攒够 1000 个切片，立即存盘释放内存
            if len(sub_docs) >= 1000:
                embeddings = embedder.embed_documents(sub_contents)
                add_embeddings_to_chroma(vectorstore, sub_docs, embeddings)
                vectorstore.persist()
                total_saved += len(sub_docs)
                sub_docs, sub_contents = [], [] # 清空池子

    # 处理最后剩下的一点尾巴
    if sub_docs: 
        embeddings = embedder.embed_documents(sub_contents)
        add_embeddings_to_chroma(vectorstore, sub_docs, embeddings)
        vectorstore.persist()
        total_saved += len(sub_docs)
        
    print(f"✅ 文本切块嵌入完成，共新增: {total_saved} 个切片")

# -------------------------------
# 路径二：表格处理
# -------------------------------
def split_table_to_contextual_chunks(doc: dict, field: str, rows_per_chunk: int = 15) -> List[str]:
    """智能表格切片：确保每个切片都包含完整的 Title、Table Name 和 Header"""
    content = doc.get(field) if field else doc
    if not isinstance(content, dict) or "table_rows" not in content:
        return [str(content)]

    global_title = doc.get("title", "Unknown Title")
    table_name = content.get("table_name", "Unknown Table")
    headers = [h.get("column_name", "") for h in content.get("header", [])]

    # 构建全局上下文前缀
    prefix_lines = []
    if global_title and global_title != "Unknown Title":
        prefix_lines.append(f"Document Title: {global_title}")
    if table_name and table_name != "Unknown Table":
        prefix_lines.append(f"Table Name: {table_name}")
    
    header_md = ""
    if headers:
        header_md = "| " + " | ".join(headers) + " |\n|" + "|".join(["---"] * len(headers)) + "|"

    rows = content.get("table_rows", [])
    chunks = []
    
    for i in range(0, len(rows), rows_per_chunk):
        chunk_rows = rows[i:i + rows_per_chunk]
        chunk_md_lines = []
        for row in chunk_rows:
            row_texts = []
            for cell in row:
                cell_text = str(cell.get("text", "") if isinstance(cell, dict) else cell)
                cell_text = cell_text.replace("\n", " ").replace("|", "\|").strip()
                row_texts.append(cell_text)
            
            chunk_md_lines.append("| " + " | ".join(row_texts) + " |" if headers else " | ".join(row_texts))
                
        chunk_content = "\n".join(prefix_lines) + "\n\n" + header_md + "\n" + "\n".join(chunk_md_lines)
        chunks.append(chunk_content.strip())

    return chunks

def process_table_dataset(dataset_name: str, config: dict):
    path, field, embedder = config["path"], config["field"], EMBEDDERS[config["embedder"]]
    print(f"\n📊 处理表格数据: {dataset_name} | 文件: {path}")

    vectorstore = get_vectorstore(dataset_name, "table")
    existing_ids = set(vectorstore.get(include=[])["ids"])

    try:
        raw_data = load_json(path)
        data = normalize_dict_to_list(raw_data, field)
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return

    sub_docs, sub_contents = [], []
    total_saved = 0

    for doc in tqdm(data, desc=f"{dataset_name}-table"):
        doc_id = str(doc.get("id") or doc.get("text_id"))
        
        # 智能表格切块 (调用之前写的带表头的切片函数)
        table_chunks = split_table_to_contextual_chunks(doc, field, rows_per_chunk=10)
        
        for i, chunk_text in enumerate(table_chunks):
            chunk_id = f"{doc_id}_chunk_{i}"
            if chunk_id in existing_ids: continue
            
            sub_docs.append(Document(
                page_content=chunk_text,
                metadata={
                    "chunk_id": chunk_id,
                    "parent_id": doc_id,         # 👈 外键
                    "source_path": path,         # 👈 指针
                    "dataset": dataset_name
                }
            ))
            sub_contents.append(chunk_text)

            # ✨ 防 OOM 优化：每攒够 1000 个切片，立即存盘释放内存
            if len(sub_docs) >= 1000:
                embeddings = embedder.embed_documents(sub_contents)
                add_embeddings_to_chroma(vectorstore, sub_docs, embeddings)
                vectorstore.persist()
                total_saved += len(sub_docs)
                sub_docs, sub_contents = [], [] # 清空池子

    # 处理最后剩下的一点尾巴
    if sub_docs: 
        embeddings = embedder.embed_documents(sub_contents)
        add_embeddings_to_chroma(vectorstore, sub_docs, embeddings)
        vectorstore.persist()
        total_saved += len(sub_docs)

    print(f"✅ 表格切块嵌入完成: {len(total_saved)} 个上下文切片")

# -------------------------------
# 路径三：图像专属处理
# -------------------------------
def process_image_folder(dataset_name: str, folder_path: str):
    print(f"\n🖼️ 处理图像文件夹: {dataset_name} | 路径: {folder_path}")

    # ✨ 核心改动：直接获取底层的 Chroma vectorstore
    vectorstore = get_vectorstore(dataset_name, "image")

    existing_ids = set(vectorstore.get(include=[])["ids"])
    print(f"🔍 向量库中已存在 {len(existing_ids)} 条数据，准备增量处理...")

    image_paths = [p for p in glob.glob(os.path.join(folder_path, "*"))]
    documents, paths_to_embed = [], []
    total_saved = 0 

    for image_path in tqdm(image_paths, desc=f"{dataset_name}-image"):
        try:
            image_filename = os.path.basename(image_path)
            image_id = os.path.splitext(image_filename)[0]

            if image_id in existing_ids:
                continue 

            # ✨ 核心改动：直接把绝对路径塞进 source_path，下游直接用
            documents.append(Document(
                page_content=image_filename,
                metadata={
                    "doc_id": image_id,           # Chroma的主键
                    "parent_id": image_id,        # 保持与文本一致的字段习惯
                    "source_path": image_path,    # 👈 绝对路径指针
                    "dataset": dataset_name,
                    "type": "image"
                }
            ))
            paths_to_embed.append(image_path)

            if len(documents) >= 500:
                _save_image_batch(vectorstore, documents, paths_to_embed)
                total_saved += len(documents)
                documents, paths_to_embed = [], []

        except Exception as e:
            print(f"❌ 处理图像失败：{image_path}，错误: {e}")
            continue 

    if documents:
        _save_image_batch(vectorstore, documents, paths_to_embed)
        total_saved += len(documents)

    if total_saved == 0:
        print("✅ 所有图片已存在或无新图片，跳过")
    else:
        print(f"✅ 图像嵌入完成，共新增: {total_saved} 张")


# ✨ 精简后的图片批处理保存函数
def _save_image_batch(vectorstore, documents, paths_to_embed):
    try:
        embeddings = EMBEDDERS["image"].embed_documents(paths_to_embed)
        
        filtered_documents, filtered_embeddings = [], []
        for doc, emb in zip(documents, embeddings):
            if emb is not None:
                filtered_documents.append(doc)
                filtered_embeddings.append(emb)

        if filtered_documents:
            add_embeddings_to_chroma(vectorstore, filtered_documents, filtered_embeddings)
            vectorstore.persist()        
    except Exception as e:
        print(f"❌ 向量数据库操作失败，错误: {e}")


# -------------------------------
# 数据路径配置 (✨ 新增 4 个数据集配置)
# -------------------------------
# 🚨 警告：请务必核对 slide 和 videoseek 的具体 JSON 文件名是否正确！
JSON_DATASETS = {
    "slide": {
        "text": {
            "path": "/root/nas/RAG/OURRAG/v9/texts/slide/ocr_slide_all.jsonl", # 👈 占位符：请核对真实文件名
            "field": "text",
            "embedder": "text"
        }
    },
    "videoseek": {
        "text": {
            "path": "/root/nas/RAG/OURRAG/v9/texts/vidoseek/ocr_vidoseek_all.jsonl", # 👈 占位符：请核对真实文件名
            "field": "text",
            "embedder": "text"
        }
    },
    "mmlongbench": {
        "text": {
            "path": "/root/nas/RAG/OURRAG/v9/texts/mmlongbench/mmlongbench_ocr_formatted.jsonl",
            "field": "text", 
            "embedder": "text"
        }
    },
    "mutihop": {
        "text": {
            "path": "/root/nas/RAG/OURRAG/v9/texts/multimodalqa/MMQA_texts.json",
            "field": "text",   
            "embedder": "text"
        },
        "table": {
            "path": "/root/nas/RAG/OURRAG/v9/texts/multimodalqa/MMQA_tables.json",
            "field": "table",       
            "embedder": "text"
        }
    }
}

IMAGE_FOLDER_DATASETS = {
    "slide": "/root/nas/RAG/OURRAG/v9/images/slide_vqa_img",             # 👈 占位符：请核对路径
    "videoseek": "/root/local/corpus_v2/chroma_data/videoseek/vidoseek_vqa_img",      # 👈 占位符：请核对路径
    "mmlongbench": "/root/local/corpus_v2/chroma_data/mmlongbench/mmlongbench_img",
    "mutihop": "/root/local/corpus_v2/chroma_data/mutihop/image/final_dataset_images",         # 👈 占位符：请核对路径
}


# -------------------------------
# 主执行流程
# -------------------------------
if __name__ == "__main__":
    
    # 1. 处理 JSON 文本与表格数据
    for dataset_name, type_map in JSON_DATASETS.items():
        for data_type, config in type_map.items():
            if data_type == "table":
                process_table_dataset(dataset_name, config)
            else:
                process_text_dataset(dataset_name, config)

    # # 2. 处理图像文件数据 
    # for dataset_name, folder_path in IMAGE_FOLDER_DATASETS.items():
    #     process_image_folder(dataset_name, folder_path)

    print(f"\n✅ 所有数据处理完成，保存至目录：{persist_directory}")