#!/bin/bash
export MAX_PIXELS=1048576
export WANDB_API_KEY="xxx"
export CUDA_VISIBLE_DEVICES=$SELECTED_GPUS
export MASTER_PORT=$((10000 + RANDOM % 50000))
export CUDA_VISIBLE_DEVICES="1"
export NPROC_PER_NODE=2
export HF_ENDPOINT="https://hf-mirror.com"
export ANONYMIZED_TELEMETRY=False 
export CHROMA_TELEMETRY_IMPL=none
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1


OUTPUT_DIR="./checkpoint/sft-1000-cog_plan/v0-20260515-132159"

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
  --model xxx \
  --template qwen2_5_vl \
  --tuner_type lora \
  --max_pixels 1048576 \
  --lora_rank 32 \
  --lora_alpha 64 \
  --target_modules all-linear \
  --freeze_vit true \
  --freeze_aligner true \
  --dataset xxx \
  --save_only_model false \
  --output_dir $OUTPUT_DIR \
  $RESUME_ARG \
  --load_from_cache_file true \
  --external_plugins plugin_vrag_v14 \
  --reward_funcs agentic_rag_reward \
  --reward_weights 1 \
  --torch_dtype bfloat16 \
  --attn_impl flash_attn \
  --num_train_epochs 4 \
  --max_length 12000 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --save_steps 150 \
  --beta 0.005 \
  --learning_rate 5e-6 \
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
  --vllm_gpu_memory_utilization 0.5 \
  --vllm_max_model_len 12000 \
  --vllm_tensor_parallel_size 2
