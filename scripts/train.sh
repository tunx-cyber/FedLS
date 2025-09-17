export HF_ENDPOINT="https://hf-mirror.com"

# qa
CUDA_VISIBLE_DEVICES=0 python main.py --fed_alg fedit --dataset_name zwhe99/commonsense_170k

CUDA_VISIBLE_DEVICES=1 python main.py --fed_alg ffalora --dataset_name zwhe99/commonsense_170k

CUDA_VISIBLE_DEVICES=2 python main.py --fed_alg flora --dataset_name zwhe99/commonsense_170k

CUDA_VISIBLE_DEVICES=3 python main.py --fed_alg fedex --dataset_name zwhe99/commonsense_170k

CUDA_VISIBLE_DEVICES=4 python main.py --fed_alg frlora --dataset_name zwhe99/commonsense_170k

CUDA_VISIBLE_DEVICES=5 python main.py --fed_alg fedls --dataset_name zwhe99/commonsense_170k


#mt mmlu
CUDA_VISIBLE_DEVICES=4 python main.py --fed_alg fedit 

CUDA_VISIBLE_DEVICES=5 python main.py --fed_alg ffalora 

CUDA_VISIBLE_DEVICES=6 python main.py --fed_alg flora 

CUDA_VISIBLE_DEVICES=7 python main.py --fed_alg fedex 

# glue
CUDA_VISIBLE_DEVICES=0 python main.py \
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
--fed_alg fedit \
--dataset_name sst2 
CUDA_VISIBLE_DEVICES=2 python main.py --model_name FacebookAI/roberta-base --task classification --dataset_name cola --peft_lora_alpha 32 --seq_len 128 --peft_lora_r 16 --epochs 2 --batch_size 64 --lr 0.0002 --max_steps 30 --num_rounds 200 --num_clients_per_round 2 --split_strategy noniid --alpha 0.5 --fed_alg fedls
CUDA_VISIBLE_DEVICES=7 python main.py --model_name FacebookAI/roberta-base --task classification --dataset_name qqp --peft_lora_alpha 32 --seq_len 128 --peft_lora_r 16 --epochs 2 --batch_size 64 --lr 0.0002 --max_steps 30 --num_rounds 200 --num_clients_per_round 2 --split_strategy noniid --alpha 0.5  --fed_alg fedex
CUDA_VISIBLE_DEVICES=0 python main.py --model_name FacebookAI/roberta-base --task classification --dataset_name mnli_matched --peft_lora_alpha 32 --seq_len 128 --peft_lora_r 16 --epochs 2 --batch_size 64 --lr 0.0002 --max_steps 30 --num_rounds 200 --num_clients_per_round 2 --split_strategy noniid --alpha 0.5
CUDA_VISIBLE_DEVICES=0 python main.py --model_name FacebookAI/roberta-base --task classification --dataset_name mnli_mismatched --peft_lora_alpha 32 --seq_len 128 --peft_lora_r 16 --epochs 2 --batch_size 64 --lr 0.0002 --max_steps 30 --num_rounds 200 --num_clients_per_round 2 --split_strategy noniid --alpha 0.5
CUDA_VISIBLE_DEVICES=3 python main.py --model_name FacebookAI/roberta-base --task classification --dataset_name mrpc --peft_lora_alpha 32 --seq_len 128 --peft_lora_r 16 --epochs 2 --batch_size 64 --lr 0.0002 --max_steps 10 --num_rounds 200 --num_clients_per_round 2 --split_strategy noniid --alpha 0.5 --fed_alg fedex
CUDA_VISIBLE_DEVICES=3 python main.py --model_name FacebookAI/roberta-base --task classification --dataset_name rte --peft_lora_alpha 32 --seq_len 128 --peft_lora_r 16 --epochs 2 --batch_size 64 --lr 0.0002 --max_steps 10 --num_rounds 200 --num_clients_per_round 2 --split_strategy noniid --alpha 0.5 --fed_alg fedex
# math
CUDA_VISIBLE_DEVICES=0 python main.py --fed_alg fedit --dataset_name meta-math/MetaMathQA
CUDA_VISIBLE_DEVICES=1 python main.py --fed_alg ffalora --dataset_name meta-math/MetaMathQA
CUDA_VISIBLE_DEVICES=2 python main.py --fed_alg fedex --dataset_name meta-math/MetaMathQA
CUDA_VISIBLE_DEVICES=3 python main.py --fed_alg flora --dataset_name meta-math/MetaMathQA

CUDA_VISIBLE_DEVICES=0 python main.py --fed_alg fedit --dataset_name meta-math/MetaMathQA
CUDA_VISIBLE_DEVICES=0 python main.py --fed_alg fedit --dataset_name meta-math/MetaMathQA