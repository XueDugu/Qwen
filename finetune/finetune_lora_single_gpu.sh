#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1

MODEL="Qwen/Qwen-7B" # 如果不想直接从HuggingFace加载，请设置模型路径
DATA="path_to_data" # 请指定训练数据的路径，应该是一个包含对话列表的json文件。更多信息请参阅README中的微调部分。

function usage() {
    echo '
Usage: bash finetune/finetune_lora_single_gpu.sh [-m MODEL_PATH] [-d DATA_PATH]
'
}

while [[ "$1" != "" ]]; do
    case $1 in
        -m | --model )
            shift
            MODEL=$1 # 指定模型路径
            ;;
        -d | --data )
            shift
            DATA=$1 # 指定数据路径
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

export CUDA_VISIBLE_DEVICES=0

python finetune.py \ # 运行Python脚本进行微调
  --model_name_or_path $MODEL \ # 模型名称或路径
  --data_path $DATA \ # 数据路径
  --bf16 True \ # 使用bf16精度
  --output_dir output_qwen \ # 输出目录
  --num_train_epochs 5 \ # 训练轮数
  --per_device_train_batch_size 2 \ # 每个设备的训练批次大小
  --per_device_eval_batch_size 1 \ # 每个设备的评估批次大小
  --gradient_accumulation_steps 8 \ # 梯度累积步骤
  --evaluation_strategy "no" \ # 评估策略
  --save_strategy "steps" \ # 保存策略
  --save_steps 1000 \ # 保存间隔步数
  --save_total_limit 10 \ # 总保存限制
  --learning_rate 3e-4 \ # 学习率
  --weight_decay 0.1 \ # 权重衰减
  --adam_beta2 0.95 \ # Adam优化器的beta2参数
  --warmup_ratio 0.01 \ # 热身比例
  --lr_scheduler_type "cosine" \ # 学习率调度程序类型
  --logging_steps 1 \ # 日志记录步骤
  --report_to "none" \ # 不报告到任何地方
  --model_max_length 512 \ # 模型最大长度
  --lazy_preprocess True \ # 惰性预处理
  --gradient_checkpointing \ # 梯度检查点
  --use_lora # 使用lora

# 如果使用fp16而不是bf16，请使用deepspeed
# --fp16 True --deepspeed finetune/ds_config_zero2.json
