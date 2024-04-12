#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
DIR=`pwd`

# 该脚本支持在多GPU工作节点上进行分布式训练（以及单工作节点训练）。
# 请根据注释设置以下选项。
# 对于多GPU工作节点训练，应手动为每个工作节点设置这些选项。
# 设置选项后，请在每个工作节点上运行脚本。

# 每个GPU工作节点上的GPU数量
GPUS_PER_NODE=$(python -c 'import torch; print(torch.cuda.device_count())')

# GPU工作节点的数量，对于单工作节点训练，请设置为1
NNODES=${NNODES:-1}

# 此工作节点的排名，应在{0，...，WORKER_CNT-1}中，对于单工作节点训练，请设置为0
NODE_RANK=${NODE_RANK:-0}

# 排名0工作节点的IP地址，对于单工作节点训练，请设置为localhost
MASTER_ADDR=${MASTER_ADDR:-localhost}

# 通信端口
MASTER_PORT=${MASTER_PORT:-6001}

MODEL="Qwen/Qwen-7B-Chat-Int4"
# 如果不想直接从huggingface加载，请设置路径
# 注意：指定训练数据的路径，它应该是一个包含对话列表的json文件。
# 有关更多信息，请参阅README中的微调部分。
DATA="path_to_data"

function usage() {
    echo '
Usage: bash finetune/finetune_qlora_ds.sh [-m MODEL_PATH] [-d DATA_PATH]
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

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

# 由于autogptq，请记住使用--fp16而不是--bf16
torchrun $DISTRIBUTED_ARGS finetune.py \
    --model_name_or_path $MODEL \
    --data_path $DATA \
    --fp16 True \
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
    --q_lora \
    --gradient_checkpointing \
    --deepspeed finetune/ds_config_zero2.json
