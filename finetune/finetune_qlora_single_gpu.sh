#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
DIR=`pwd`

# 模型路径，如果不想直接从huggingface加载，请设置路径
MODEL="Qwen/Qwen-7B-Chat-Int4"

# 数据路径，请设置为包含对话列表的json文件的路径
DATA="path_to_data"

# 函数：用法说明
function usage() {
    echo '
Usage: bash finetune/finetune_qlora_single_gpu.sh [-m MODEL_PATH] [-d DATA_PATH]
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

# 设置可见的GPU设备
export CUDA_VISIBLE_DEVICES=0

# 执行finetune.py脚本
python finetune.py \
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
  --gradient_checkpointing \
  --use_lora \
  --q_lora \
  --deepspeed finetune/ds_config_zero2.json
