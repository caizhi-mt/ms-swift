#!/bin/bash

pkill -f /usr/bin/python

export PYTHONPATH=/home/Megatron-LM/:/home/megatron-lm-musa-patch/:$PYTHONPATH
export MUSA_EXECUTION_TIMEOUT=3200000
export ACCELERATOR_BACKEND="musa"
export MCCL_PROTOS=2
export MCCL_ALGOS=1
export MCCL_BUFFSIZE=20971520
export MCCL_MAX_NCHANNELS=14
export MCCL_CHECK_POINTERS=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
export MCCL_IB_GID_INDEX=3
export OMP_NUM_THREADS=4
export MCCL_CROSS_NIC=1
export MCCL_IB_TIMEOUT=20
export MCCL_IB_RETRY_CNT=7
export MCCL_BUFFSIZE=20971520
export MUSA_BLOCK_SCHEDULE_MODE=1
#export MUSA_LOG=0xffff
#export MUSA_LAUNCH_BLOCKING=1
export MUSA_FAST_DEBUG=1
#export MUDNN_LOG_LEVEL=INFO
#export SWIFT_USE_MCORE_GDN=1
export ENABLE_GDN_BF16=1


OUTPUT_DIR="/data/caizhi/zj_task1/ms-swift/output"
SEQ_LENGTH=8192
LR=1e-5
MIN_LR=1e-6

#NNODES=${WORLD SIZE} \
#NODE_RANK=${RANK} \
#MODEL_PATH="/data/caizhi/Qwen3.5-9B/"
#MODEL_PATH="/mnt/moer-train/public/models/Qwen3.5-2B"
#MODEL_PATH="/mnt/moer-train/public/models/Qwen3.5-9B"
MODEL_PATH="/mnt/moer-train/public/liang.geng/Qwen3.5-0.8B"
#PYTORCH_MUSA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=1 \
MUSA_VISIBLE_DEVICE5=0,1,2,3,4,5,6,7 \
megatron pt\
    --model ${MODEL_PATH} \
    --save_safetensors true \
    --dataset /mnt/moer-train/public/liang.geng/alpaca-gpt4-data-zh \
    --load_from_cache_file true \
    --split_dataset_ratio 0.01 \
    --micro_batch_size 1 \
    --global_batch_size 1 \
    --num_train_epochs 1 \
    --cross_entropy_loss_fusion true \
    --tensor_model_parallel_size 1 \
    --pipeline_model_parallel_size 1 \
    --lr_warmup_fraction 0.02 \
    --min_lr ${MIN_LR} \
    --output_dir ${OUTPUT_DIR} \
    --eval_steps 500 \
    --save_steps 500\
    --max_length ${SEQ_LENGTH} \
    --dataloader_num_workers 8 \
    --dataset_num_proc 8 \
    --no_save_optim true \
    --no_save_rng true \
    --sequence_parallel false \
    --padding_free false \
    --model_author swift \
    --model_name swift-robot \
    --attention_backend unfused \
    --manual_gc true \
    --manual_gc_steps 100 \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --packing \
    --bf16 true \
    --grad_reduce_in_bf16 \

    #--decoder_first_pipeline_num_layers 4 \
    #--decoder_last_pipeline_num_layers 4 \
    #--finetune false 
    #--group_by_length true \
    #--cross_entropy_fusion_impl te \
