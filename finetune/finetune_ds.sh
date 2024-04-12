#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
DIR=`pwd`

# 每个 GPU worker 上的 GPU 数量
GPUS_PER_NODE=$(python -c 'import torch; print(torch.cuda.device_count())')

# GPU worker 的数量，默认为 1
NNODES=${NNODES:-1}

# 当前 worker 的排名，应该在 {0, ..., WORKER_CNT-1} 范围内，默认为 0
NODE_RANK=${NODE_RANK:-0}

# 排名为 0 的 worker 的 IP 地址，单 worker 训练时设为 localhost
MASTER_ADDR=${MASTER_ADDR:-localhost}

# 通信端口
MASTER_PORT=${MASTER_PORT:-6001}

# 模型路径
MODEL="Qwen/Qwen-7B"
# 数据路径
DATA="path_to_data"

function usage() {
    echo '
Usage: bash finetune/finetune_ds.sh [-m MODEL_PATH] [-d DATA_PATH]
'
}

# 解析命令行参数
while [[ "$1" != "" ]]; do
    case $1 in
        -m | --model )
            shift
            MODEL=$1
            ;;
        -d | --data )
            shift
            DATA=$1
            ;;
        -h | --help )
            usage
            exit 0
            ;;
        * )
            echo "Unknown argument ${1}"
            exit 1
            ;;
    esac
    shift
done

# 分布式参数设置
DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

# 使用 torchrun 执行 finetune.py 脚本
torchrun $DISTRIBUTED_ARGS finetune.py \
    --model_name_or_path $MODEL \
    --data_path $DATA \
    --bf16 True \
    --output_dir output_qwen \
    --num_train_epochs 5 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 10 \
    --learning_rate 1e-5 \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --report_to "none" \
    --model_max_length 512 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --deepspeed finetune/ds_config_zero3.json
