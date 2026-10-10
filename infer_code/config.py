import os

DATA_TYPE_SELECT = "test"    # "train", "dev", "test"
TEMPERATURE = 0.0
RETRIEVAL_STRATEGY = os.getenv(
    "RAG_RETRIEVAL_STRATEGY",
    "intersect",
).strip().lower()
if RETRIEVAL_STRATEGY not in {"intersect", "independent", "union_rrf"}:
    raise ValueError(
        f"Unsupported RAG_RETRIEVAL_STRATEGY={RETRIEVAL_STRATEGY!r}; "
        "expected intersect / independent / union_rrf."
    )
     # intersect independent union_rrf

# Unified execution budgets for every open- and closed-source backbone.
MAX_RETRIEVAL_STEPS = 5
COGNITIVE_CONTEXT_TOKENS = 6000
COGNITIVE_MAX_ASSETS = 8
FINAL_CONTEXT_TOKENS = 8000
FINAL_MAX_ASSETS = 6
FINAL_MAX_BELIEFS = 12
COGNITIVE_MAX_BELIEFS = 12
VALIDATOR_MAX_WORKERS = 8
VALIDATOR_PASS_SCORE = 3

# Soft evidence admission: validator scores remain unchanged, while repeated
# retrieval under genuinely different queries can rescue a likely false negative.
REPEAT_HIT_BONUS = 0.5
MAX_REPEAT_HIT_BONUS = 1.0
REPEAT_RESCUE_MIN_VALIDATOR_SCORE = 2

# Preserve a very small part of the initial top-ranked retrieval so that the
# validator never becomes an irreversible information bottleneck.
INITIAL_ANCHORS_PER_MODALITY = 1

# API / embedding reliability. Environment variables can override deployment-specific values.
API_MAX_RETRIES = int(os.getenv("RAG_API_MAX_RETRIES", "3"))
API_RETRY_DELAY = float(os.getenv("RAG_API_RETRY_DELAY", "3"))
API_TIMEOUT = float(os.getenv("RAG_API_TIMEOUT", "300"))
API_PROXY = os.getenv("RAG_API_PROXY", "")
IMAGE_MAX_SIDE = int(os.getenv("RAG_IMAGE_MAX_SIDE", "1024"))
EMBEDDING_TIMEOUT = float(os.getenv("RAG_EMBEDDING_TIMEOUT", "20"))
EMBEDDING_MAX_RETRIES = int(os.getenv("RAG_EMBEDDING_MAX_RETRIES", "3"))
STRICT_CORPUS_PATH = os.getenv("RAG_STRICT_CORPUS_PATH", "1") != "0"
NUM_PROCESSES = int(os.getenv("RAG_NUM_PROCESSES", "6"))

# 针对多模型混合实验，定义一个专门的文件名后缀（替代原先的 MODEL_SELECT）
EXPERIMENT_NAME = os.getenv(
    "RAG_EXPERIMENT_NAME",
    "single_role_qwen3.5_vl_cog_plan_2400",
) 
ROLE_MODEL_NAME = os.getenv("RAG_MODEL_NAME", "cog_plan")  # openai/gpt-4.1-mini
ROLE_API_BASE = os.getenv("RAG_API_BASE", "http://localhost:8160/v1")  
ROLE_API_KEY = os.getenv("RAG_API_KEY", "")  

# ✨ 按角色分配模型配置
ROLE_MODEL_CONFIGS = {
    role: {
        "model_name": ROLE_MODEL_NAME,
        "api_key": ROLE_API_KEY,
        "api_base": ROLE_API_BASE,
        "max_new_tokens": 2048,
    }
    for role in [
        "validator",
        "cognitive",
        "planner",
        "responder",
        "fallback",
    ]
}

PERSIST_DIR_PRE = "./VRAG/final_before20260718/corpus_v2/chroma_data"

ALL_DATASET_CONFIGS = [
    {
        "name": "mmlongbench",
        "input_jsonl": f"./final_before20260718/test_data_ori/mmlongbench_TrainForTest_cleaned.jsonl",
        "output_jsonl": f"./results/mmlongbench_TrainForTest_cleaned_{RETRIEVAL_STRATEGY}_{EXPERIMENT_NAME}.jsonl",
        "prefix_image": "./VRAG/final_before20260718/images/mmlongbench_img"
    },
    {
        "name": "mutihop",
        "input_jsonl": f"./VRAG/final_before20260718/test_data_ori/multimodel_{DATA_TYPE_SELECT}.jsonl",
        "output_jsonl": f"./results/multimodel_{DATA_TYPE_SELECT}_{RETRIEVAL_STRATEGY}_{EXPERIMENT_NAME}.jsonl",
        "prefix_image": "./VRAG/final_before20260718/images/final_dataset_images"
    },
    {
        "name": "videoseek",
        "input_jsonl": f"./VRAG/final_before20260718/test_data_ori/vidoseek_{DATA_TYPE_SELECT}_labeled.jsonl",
        "output_jsonl": f"./results/vidoseek_{DATA_TYPE_SELECT}_{RETRIEVAL_STRATEGY}_{EXPERIMENT_NAME}.jsonl",
        "prefix_image": "./VRAG/final_before20260718/images/vidoseek_vqa_img"
    },
    {
        "name": "slide",
        "input_jsonl": f"./VRAG/final_before20260718/test_data_ori/slide_{DATA_TYPE_SELECT}_labeled.jsonl",
        "output_jsonl": f"./results/slide_{DATA_TYPE_SELECT}_{RETRIEVAL_STRATEGY}_{EXPERIMENT_NAME}.jsonl",
        "prefix_image": "./VRAG/final_before20260718/images/slide_vqa_img"
    }
] 

