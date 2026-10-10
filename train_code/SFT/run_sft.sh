# #!/bin/bash

LOG_DIR="./sft_logs"
mkdir -p $LOG_DIR

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/qwen2_5_vl_7b_zero2_${TIMESTAMP}.log"

TARGET_FREE_MEM=38680

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

echo "ߚࠓtarting GRPO training on GPUs: $SELECTED_GPUS"

export CUDA_VISIBLE_DEVICES=$SELECTED_GPUS
export NPROC_PER_NODE=$TARGET_GPU_COUNT
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MASTER_PORT=$((10000 + RANDOM % 50000))
export MAX_PIXELS=1500000
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

echo "Starting Swift + DeepSpeed ZeRO-3 Training..."
echo "Log file: $LOG_FILE"

nohup swift sft \
    --model /mnt/nvme/ALLYOUNEED/qwen_vl/qwen2.5_vl_7b \
    --template qwen2_5_vl \
    --dataset ./final/train_data/sft/v9/sft_formatted_data/sft_mixed_dataset_vlm.jsonl \
    --tuner_type lora \
    --lora_rank 32 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-4 \
    --warmup_ratio 0.05 \
    --max_length 16000 \
    --gradient_checkpointing true \
    --deepspeed zero2 \
    --dataset_num_proc 4 \
    --dataloader_num_workers 4 \
    --save_steps 50 \
    --save_total_limit 50 \
    --logging_steps 1 \
    --output_dir ../swift_output/qwen2.5-vl-7b-r16 \
    > "$LOG_FILE" 2>&1


TRAIN_PID=$!
sleep 2

if kill -0 $TRAIN_PID 2>/dev/null; then
    echo "Training started with ZeRO-2! PID: $TRAIN_PID"
    echo "To view logs: tail -f $LOG_FILE"
else
    echo "Failed to start. Check $LOG_FILE"
fi