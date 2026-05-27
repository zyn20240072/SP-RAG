export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER=GNU
export OMP_NUM_THREADS=1


CUDA_VISIBLE_DEVICES=$SELECTED_GPUS vllm serve /data/yuguangya/ALLYOUNEED/VL/Qwen2.5-VL-7B-Instruct \
    --served-model-name base \
    --max-model-len 40000 \
    -tp 1 \
    --port 8010 \
    --gpu-memory-utilization 0.9 \
    --max_lora_rank 64 \
    --enable-lora \
    --lora-modules sft=/root/nas/RAG/OURRAG/v9/code_v14_sft3_grpo/ckp/500
