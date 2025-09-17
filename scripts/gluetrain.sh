#!/bin/bash

# 定义要循环的数据集名称列表
datasets=("sst2" "cola" "qqp" "mnli_matched" "mnli_mismatched")
alg=fedls
cuda_device=3
# 循环遍历每个数据集
for dataset in "${datasets[@]}"
do
    echo "正在处理数据集: $dataset"
    
    # 执行您的命令，将dataset_name参数替换为当前数据集
    CUDA_VISIBLE_DEVICES=$cuda_device python main.py \
    --model_name FacebookAI/roberta-base \
    --task classification \
    --peft_lora_alpha 32 \
    --seq_len 128 \
    --peft_lora_r 16 \
    --epochs 2 \
    --batch_size 64 \
    --lr 0.0002 \
    --max_steps 30 \
    --split_strategy noniid \
    --alpha 0.5 \
    --fed_alg $alg \
    --dataset_name "$dataset"
done

echo "所有数据集处理完成！"

datasets=("mrpc" "rte")
# 循环遍历每个数据集
for dataset in "${datasets[@]}"
do
    echo "正在处理数据集: $dataset"
    
    # 执行您的命令，将dataset_name参数替换为当前数据集
    CUDA_VISIBLE_DEVICES=$cuda_device python main.py \
    --model_name FacebookAI/roberta-base \
    --task classification \
    --peft_lora_alpha 32 \
    --seq_len 128 \
    --peft_lora_r 16 \
    --epochs 2 \
    --batch_size 64 \
    --lr 0.0002 \
    --max_steps 10 \
    --split_strategy noniid \
    --alpha 0.5 \
    --fed_alg $alg \
    --dataset_name "$dataset"
done
