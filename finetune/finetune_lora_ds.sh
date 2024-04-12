#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
DIR=`pwd`

# 设置每个 GPU 工作节点的 GPU 数量
GPUS_PER_NODE=$(python -c 'import torch; print(torch.cuda.device_count())')
# GPU 工作节点数量，对于单节点训练，请设置为 1
NNODES=${NNODES:-1}
# 此节点的排名，应在 {0, ..., WORKER_CNT-1} 中，对于单节点训练，请设置为 0
NODE_RANK=${NODE_RANK:-0}
# 排名-0 节点的 IP 地址，对于单节点训练，请设置为 localhost
MASTER_ADDR=${MASTER_ADDR:-localhost}
# 通信端口
MASTER_PORT=${MASTER_PORT:-6001}

# 模型路径
MODEL="Qwen/Qwen-7B"
# 数据路径，需指定到包含对话列表的 JSON 文件，请参阅 README 中的微调部分以获取更多信息
DATA="path_to_data"
# DeepSpeed 配置文件路径
DS_CONFIG_PATH="finetune/ds_config_zero2.json"

function usage() {
    echo '
Usage: bash finetune/finetune_lora_ds.sh [-m MODEL_PATH] [-d DATA_PATH] [--deepspeed DS_CONFIG_PATH]
'
}

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
        --deepspeed )
            shift
            DS_CONFIG_PATH=$1
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

# 调用 torchrun 启动分布式训练
torchrun $DISTRIBUTED_ARGS finetune.py \
    --model_name_or_path $MODEL \
    --data_path $DATA \
    --bf16 True \
    --output_dir output_qwen \
    --num_train_epochs 5 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 10 \
    --learning_rate 3e-4 \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --report_to "none" \
    --model_max_length 512 \
    --lazy_preprocess True \
    --use_lora \
    --gradient_checkpointing \
    --deepspeed ${DS_CONFIG_PATH}
