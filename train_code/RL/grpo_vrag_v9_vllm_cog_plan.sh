#!/bin/bash

TARGET_FREE_MEM=32000
TARGET_GPU_COUNT=2
CHECK_INTERVAL=60 

SELECTED_GPUS=""

echo "=========================================================="
echo "⏳ Waiting for $TARGET_GPU_COUNT GPUs with at least ${TARGET_FREE_MEM}MB free memory..."
echo "=========================================================="

while true; do
    AVAILABLE_GPUS=()
    
    GPU_INFO=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits)
    
    while IFS=',' read -r idx mem; do
        idx=$(echo "$idx" | xargs)
        mem=$(echo "$mem" | xargs)
        
        if [ "$mem" -ge "$TARGET_FREE_MEM" ]; then
            AVAILABLE_GPUS+=("$idx") 
        fi
    done <<< "$GPU_INFO"

    if [ "${#AVAILABLE_GPUS[@]}" -ge "$TARGET_GPU_COUNT" ]; then
        SELECTED_GPUS=$(IFS=,; echo "${AVAILABLE_GPUS[*]:0:$TARGET_GPU_COUNT}")
        echo "✅ Success! Found enough GPUs: $SELECTED_GPUS"
        break
    else
        echo "[-] Found ${#AVAILABLE_GPUS[@]} valid GPUs, but need $TARGET_GPU_COUNT. Checking again in ${CHECK_INTERVAL}s..."
        sleep $CHECK_INTERVAL
    fi
done

echo "🚀 Starting GRPO training on GPUs: $SELECTED_GPUS"


export MAX_PIXELS=1000000
export WANDB_API_KEY="xxx"
export CUDA_VISIBLE_DEVICES=$SELECTED_GPUS
export MASTER_PORT=$((10000 + RANDOM % 50000))
export NPROC_PER_NODE=2
export HF_ENDPOINT="https://hf-mirror.com"
export ANONYMIZED_TELEMETRY=False 
export CHROMA_TELEMETRY_IMPL=none
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1


OUTPUT_DIR="./checkpoint/cog_plan-NewData-v18"

LATEST_CHECKPOINT=$(ls -td ${OUTPUT_DIR}/checkpoint-* 2>/dev/null | head -1)

if [ -n "$LATEST_CHECKPOINT" ]; then
    echo "♻️ Found checkpoint! Resuming from: $LATEST_CHECKPOINT"
    RESUME_ARG="--resume_from_checkpoint $LATEST_CHECKPOINT"
else
    echo "🌱 No checkpoint found. Starting fresh training..."
    RESUME_ARG=""
fi


swift rlhf \
  --rlhf_type grpo \
  --model /mnt/nvme/ALLYOUNEED/qwen_vl/qwen2.5_vl_7b \
  --template qwen2_5_vl \
  --tuner_type lora \
  --max_pixels 1000000 \
  --lora_rank 32 \
  --lora_alpha 64 \
  --target_modules all-linear \
  --freeze_vit true \
  --freeze_aligner true \
  --dataset ./VRAG/final/train_grpo_v3/grpo_data/manual_data/all.sample10000.balanced.jsonl \
  --save_only_model false \
  --output_dir $OUTPUT_DIR \
  $RESUME_ARG \
  --load_from_cache_file true \
  --external_plugins plugin_vrag_v18 \
  --reward_funcs agentic_rag_reward \
  --reward_weights 1 \
  --torch_dtype bfloat16 \
  --attn_impl flash_attn \
  --num_train_epochs 4 \
  --max_length 14000 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --save_steps 600 \
  --beta 0.005 \
  --learning_rate 2e-6 \
  --save_total_limit 10 \
  --logging_steps 1 \
  --warmup_ratio 0.05 \
  --dataloader_num_workers 4 \
  --max_completion_length 1200 \
  --num_generations 8 \
  --deepspeed zero2 \
  --temperature 0.9 \
  --top_p 1.0 \
  --top_k 80 \
  --log_completions true \
  --move_model_batches 40 \
  --sleep_level 0 \
  --use_vllm true \
  --vllm_mode colocate \
  --vllm_gpu_memory_utilization 0.35 \
  --vllm_max_model_len 14000 \
  --vllm_tensor_parallel_size 2