import os

DATA_TYPE_SELECT = "test"
TEMPERATURE = 0.0
RETRIEVAL_STRATEGY = "independent" # intersect independent

EXPERIMENT_NAME = "multi_role_qwen-7b-grpo"

ROLE_MODEL_CONFIGS = {
    "validator": {
        "model_name": "sft",  
        "api_key": "YOUR_API_KEY",
        "api_base": "http://localhost:8040/v1",
        "max_new_tokens": 2048
    },
    "cognitive": {
        "model_name": "cog_plan_1500",   
        "api_key": "YOUR_API_KEY",
        "api_base": "http://localhost:8040/v1",
        "max_new_tokens": 2048
    },
    "planner": {
        "model_name": "cog_plan_1500",      
        "api_key": "YOUR_API_KEY",
        "api_base": "http://localhost:8040/v1",
        "max_new_tokens": 2048
    },
    "responder": {
        "model_name": "sft",   
        "api_key": "YOUR_API_KEY",
        "api_base": "http://localhost:8040/v1",
        "max_new_tokens": 2048
    },
    "fallback": {
        "model_name": "sft",    
        "api_key": "YOUR_API_KEY",
        "api_base": "http://localhost:8040/v1",
        "max_new_tokens": 2048
    }
}

PERSIST_DIR_PRE = "/root/local/corpus_v2/chroma_data"

ALL_DATASET_CONFIGS = [
    {
        "name": "mmlongbench",
        "input_jsonl": f"/root/nas/RAG/OURRAG/v9/test_data_ori/mmlongbench_TrainForTest_cleaned.jsonl",
        "output_jsonl": f"/root/nas/RAG/OURRAG/v9/code_v14.1/results/mmlongbench_TrainForTest_cleaned_{RETRIEVAL_STRATEGY}_{EXPERIMENT_NAME}.jsonl",
        "prefix_image": "/root/nas/RAG/OURRAG/v9/images/mmlongbench_img"
    },
    {
        "name": "mutihop",
        "input_jsonl": f"/root/nas/RAG/OURRAG/v9/test_data_ori/multimodel_{DATA_TYPE_SELECT}.jsonl",
        "output_jsonl": f"/root/nas/RAG/OURRAG/v9/code_v14.1/results/multimodel_{DATA_TYPE_SELECT}_{RETRIEVAL_STRATEGY}_{EXPERIMENT_NAME}.jsonl",
        "prefix_image": "/root/nas/RAG/OURRAG/v9/images/final_dataset_images"
    },
    {
        "name": "videoseek",
        "input_jsonl": f"/root/nas/RAG/OURRAG/v9/test_data_ori/vidoseek_{DATA_TYPE_SELECT}_labeled.jsonl",
        "output_jsonl": f"/root/nas/RAG/OURRAG/v9/code_v14.1/results/vidoseek_{DATA_TYPE_SELECT}_{RETRIEVAL_STRATEGY}_{EXPERIMENT_NAME}.jsonl",
        "prefix_image": "/root/nas/RAG/OURRAG/v9/images/vidoseek_vqa_img"
    },
    {
        "name": "slide",
        "input_jsonl": f"/root/nas/RAG/OURRAG/v9/test_data_ori/slide_{DATA_TYPE_SELECT}_labeled.jsonl",
        "output_jsonl": f"/root/nas/RAG/OURRAG/v9/code_v14.1/results/slide_{DATA_TYPE_SELECT}_{RETRIEVAL_STRATEGY}_{EXPERIMENT_NAME}.jsonl",
        "prefix_image": "/root/nas/RAG/OURRAG/v9/images/slide_vqa_img"
    }
] 

