import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
from FlagEmbedding import BGEM3FlagModel
import torch
import uvicorn

# 模型加载
print("🔧 Loading BGE-M3 model...")
model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
print("✅ Model loaded.")

# FastAPI 初始化
app = FastAPI(title="BGE-M3 Embedding API")

# 请求体定义
class EmbeddingRequest(BaseModel):
    texts: List[str]
    max_length: Optional[int] = 2048
    batch_size: Optional[int] = 12

@app.post("/bge-m3-embed")
def embed_texts(req: EmbeddingRequest):
    if not req.texts:
        return JSONResponse(content={"error": "No texts provided."}, status_code=400)

    try:
        result = model.encode(
            req.texts,
            max_length=req.max_length,
            batch_size=req.batch_size
        )
        embeddings = [vec.tolist() for vec in result["dense_vecs"]]
        return JSONResponse(content={"embedding": embeddings})
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return JSONResponse(content={"error": "CUDA out of memory. Try reducing batch size or max_length."}, status_code=500)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# 启动服务（指定端口）
if __name__ == "__main__":
    port = 8500  # ← 你可以改这个
    print(f"🚀 Starting FastAPI embedding server at http://0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)