from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import AutoModel
import torch
import os
import uvicorn

# 使用第 1 张 GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

app = FastAPI()

# 加载模型
MODEL_NAME = "BAAI/BGE-VL-MLLM-S2"
model = AutoModel.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    torch_dtype=torch.float16, # 确保基础数据类型是半精度
    device_map="auto"  # 自动分配到多张 GPU
)
model.set_processor(MODEL_NAME)
model.eval()
# 请求结构
class ImagePathInput(BaseModel):
    image_path: str

@app.post("/embed")
async def get_image_embedding(input: ImagePathInput):
    image_path = input.image_path

    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail=f"Image not found: {image_path}")

    try:
        with torch.no_grad():
            # ✅ 图像预处理
            inputs = model.data_process(images=image_path, q_or_c="c")
            for k in inputs:
                if isinstance(inputs[k], torch.Tensor):
                    inputs[k] = inputs[k].cuda()

            # ✅ 计算嵌入
            output = model(**inputs, output_hidden_states=True)
            emb = output[:, -1, :]
            emb = torch.nn.functional.normalize(emb, dim=-1)

            # ✅ 返回前释放缓存
            return JSONResponse(content={"embedding": emb[0].tolist()})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # ✅ 清理 GPU 显存碎片（在所有情况下）
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    
class TextInput(BaseModel):
    texts: list[str]

@app.post("/query_embed")
async def get_text_embedding(input: TextInput):
    if not input.texts:
        raise HTTPException(status_code=400, detail="Input text list is empty.")

    with torch.no_grad():
        inputs = model.data_process(
            text=input.texts,
            q_or_c="q",
            task_instruction="Retrieve the target image that best meets the provided description:"
        )
        for k in inputs:
            if isinstance(inputs[k], torch.Tensor):
                inputs[k] = inputs[k].cuda()

        # 直接取输出结果
        emb = model(**inputs)  # 返回的是 [B, D] 的 tensor
        emb = torch.nn.functional.normalize(emb, dim=-1)

    return JSONResponse(content={"embedding": emb.cpu().tolist()})



##uvicorn main:app --host 0.0.0.0 --port 8000 --reload
if __name__ == "__main__":
    port = 8005  # ← 你可以改这个
    print(f"🚀 Starting FastAPI embedding server at http://0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)