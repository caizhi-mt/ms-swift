#!/bin/bash
set -euo pipefail

export LD_LIBRARY_PATH=/usr/local/musa/lib:/usr/local/musa/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export PYTHONPATH=/home/Megatron-LM/:/home/megatron-lm-musa-patch/:${PYTHONPATH:-}
sed -i 's#if (implementation == "flash_attention_2" and is_fa2) or (implementation is None and is_fa2 and not is_fa3):#if True:#g' /usr/local/lib/python3.10/dist-packages/transformers/modeling_flash_attention_utils.py

bash install.sh

: "${NNODES:?NNODES is required}"
: "${NODE_RANK:?NODE_RANK is required}"
: "${MASTER_ADDR:?MASTER_ADDR is required}"
: "${MASTER_PORT:?MASTER_PORT is required}"

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

WORK_DIR="$(pwd)"
TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
LOG_DIR="${WORK_DIR}/logs/S5000_train_qwen3_5-27b_node${NODE_RANK}_${TIMESTAMP}"
mkdir -p "${LOG_DIR}"

OUTPUT_PATH="${WORK_DIR}/output"
DATA_PATH="${DATA_PATH:-/mnt/moer-train/public/liang.geng/alpaca-gpt4-data-zh}"
MODEL_PATH="${MODEL_PATH:-/mnt/moer-train/public/models/Qwen3.5-27B}"
#MODEL_PATH="${MODEL_PATH:-/mnt/moer-train/public/models/Qwen3.5-2B}"
#MODEL_PATH="${MODEL_PATH:-/mnt/moer-train/public/liang.geng/Qwen3.5-0.8B}"

SEQ_LENGTH="${SEQ_LENGTH:-8192}"
LR="${LR:-1e-5}"
MIN_LR="${MIN_LR:-1e-6}"

TP_SIZE="${TP_SIZE:-1}"
PP_SIZE="${PP_SIZE:-8}"
#EP_SIZE="${EP_SIZE:-4}"

MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-128}"

DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}"
DATASET_NUM_PROC="${DATASET_NUM_PROC:-8}"

VISIBLE_DEVICES="${VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

export PYTHONPATH=/home/Megatron-LM/:/home/megatron-lm-musa-patch/:${PYTHONPATH:-}
export MUSA_EXECUTION_TIMEOUT="${MUSA_EXECUTION_TIMEOUT:-3200000}"
export ACCELERATOR_BACKEND="${ACCELERATOR_BACKEND:-musa}"

export MCCL_PROTOS="${MCCL_PROTOS:-2}"
export MCCL_ALGOS="${MCCL_ALGOS:-1}"
export MCCL_BUFFSIZE="${MCCL_BUFFSIZE:-20971520}"
export MCCL_MAX_NCHANNELS="${MCCL_MAX_NCHANNELS:-14}"
export MCCL_CHECK_POINTERS="${MCCL_CHECK_POINTERS:-0}"
export MCCL_IB_GID_INDEX="${MCCL_IB_GID_INDEX:-3}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

# debug for musa
# export MUSA_LAUNCH_BLOCKING=1
# export MUDNN_LOG_LEVEL=INFO
# export MCCL_DEBUG=INFO
# export MCCL_DEBUG_SUBSYS=ALL
# export MUSA_FAST_DEBUG=1

# profiling for megatron musa patch
# export ENABLE_PROFILER=1
# export PROFILER_FREQ=6   # 1~3 for warmup, 4 for active
# export PROFILER_WARMUP_STEPS=3
# export PROFILER_ACTIVE_STEPS=3
# export PROFILER_SAVE_DIR="${LOG_DIR}/profiler"

# for musa env
export SWIFT_ENABLE_TORCHADA=1         # enable torchada
# export ENABLE_MEGATRON_MUSA_PATCH=1  # enbale megatron musa patch, not suggested for loading models wait too long, use with caution
export NO_LOSS_REDUCE=1

# swift
export SWIFT_USE_MCORE_GDN=1

echo "================ TRAIN ENV ================"
echo "HOSTNAME=$(hostname)"
echo "WORK_DIR=${WORK_DIR}"
echo "NNODES=${NNODES}"
echo "NODE_RANK=${NODE_RANK}"
echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "MUSA_VISIBLE_DEVICES=${VISIBLE_DEVICES}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "DATA_PATH=${DATA_PATH}"
echo "OUTPUT_PATH=${OUTPUT_PATH}"
echo "LOG_DIR=${LOG_DIR}"
echo "==========================================="

if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "[ERROR] MODEL_PATH not found: ${MODEL_PATH}"
    exit 1
fi

if [[ ! -e "${DATA_PATH}" ]]; then
    echo "[ERROR] DATA_PATH not found: ${DATA_PATH}"
    exit 1
fi

WORLD_SIZE_TOTAL=$((NNODES * NPROC_PER_NODE))
#PARALLEL_PRODUCT=$((TP_SIZE * PP_SIZE * EP_SIZE))

echo "[CHECK] WORLD_SIZE_TOTAL=${WORLD_SIZE_TOTAL}"
#echo "[CHECK] TP*PP*EP=${PARALLEL_PRODUCT}"

#if [[ "${WORLD_SIZE_TOTAL}" -ne "${PARALLEL_PRODUCT}" ]]; then
#    echo "[ERROR] total processes (${WORLD_SIZE_TOTAL}) != TP*PP*EP (${PARALLEL_PRODUCT})"
#    exit 1
#fi


NNODES="${NNODES}" \
NODE_RANK="${NODE_RANK}" \
NPROC_PER_NODE="${NPROC_PER_NODE}" \
MASTER_ADDR="${MASTER_ADDR}" \
MASTER_PORT="${MASTER_PORT}" \
MUSA_VISIBLE_DEVICES="${VISIBLE_DEVICES}" \
megatron pt \
    --model ${MODEL_PATH} \
    --save_safetensors true \
    --dataset ${DATA_PATH} \
    --load_from_cache_file true \
    --split_dataset_ratio 0.01 \
    --micro_batch_size 1 \
    --global_batch_size 128 \
    --num_train_epochs 1 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --tensor_model_parallel_size 1 \
    --pipeline_model_parallel_size 4 \
    --decoder-first-pipeline-num-layers 2 \
    --decoder-last-pipeline-num-layers 2 \
    --recompute-granularity selective \
    --recompute-modules moe_act layernorm mlp core_attn \
    --lr_warmup_fraction 0.02 \
    --lr ${MIN_LR} \
    --min_lr ${MIN_LR} \
    --freeze_llm false \
    --freeze_vit true \
    --freeze_aligner true \
    --output_dir ${OUTPUT_PATH} \
    --eval_steps 500 \
    --save_steps 500 \
    --max_length ${SEQ_LENGTH} \
    --dataloader_num_workers 8 \
    --dataset_num_proc 8 \
    --no_save_optim true \
    --no_save_rng true \
    --sequence_parallel false \
    --padding_free false \
    --model_author swift \
    --model_name swift-robot \
    --manual_gc true \
    --manual_gc_steps 100 \
    --apply_rope_fusion true \
    --attention_backend unfused \
    --packing \
    2>&1 | tee "${LOG_DIR}/train_${TIMESTAMP}.log"
    #--recompute_granularity full \
    #--recompute_method uniform \
    #--recompute_num_layers 1 \
    #--group_by_length true \
    #--recompute_granularity full \
    #--recompute_method uniform \
    #--recompute_num_layers 1 \
    # --attention_backend flash \ # unfuseds
    #--grad_reduce_in_bf16 \
    #--bf16 true \
