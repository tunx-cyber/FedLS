import torch
import numpy as np
import random
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from peft import prepare_model_for_kbit_training, LoraConfig, TaskType
import logging
import os
import matplotlib.pyplot as plt
import math
from datetime import datetime

def cosine_learning_rate(current_round, total_rounds, initial_lr=0.001, min_lr=0):
    """
    Compute the learning rate based on a cosine schedule.

    :param current_round: The current training round (0-indexed).
    :param total_rounds: The total number of training rounds.
    :param initial_lr: The initial learning rate.
    :param min_lr: The minimum learning rate.
    :return: The computed learning rate for the current round.
    """
    # Compute the cosine learning rate
    cosine_lr = min_lr + 0.5 * (initial_lr - min_lr) * (1 + math.cos(math.pi * current_round / total_rounds))
    return cosine_lr

def draw_loss_curve(x, y, args):
    plt.figure(figsize=(10, 5))
    plt.plot(x,y, label='Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.savefig(f"{args.fed_alg}_{str(args.dataset_name).replace('/','_')}_loss.png")

def setup_logger(name, log_file, level=logging.INFO):
    
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        for handler in logger.handlers:
            logger.removeHandler(handler)
    
    directory = os.path.dirname(log_file)
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(level)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    
    # console_handler = logging.StreamHandler()
    # console_handler.setLevel(level)
    # console_handler.setFormatter(formatter)
    # logger.addHandler(console_handler)
    
    return logger

def setup_seed(seed):
    torch.manual_seed(1+seed)
    torch.cuda.manual_seed_all(12+seed)
    np.random.seed(123+seed)
    random.seed(1234+seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def read_options():
    parser = argparse.ArgumentParser()

    parser.add_argument("--task",
                        type=str,
                        default="sft")
    
    parser.add_argument("--fed_alg",
                        help='name of fed method;',
                        type=str,
                        default='fedit')
    
    parser.add_argument("--dataset_name",
                        help='name of dataset;',
                        type=str,
                        default='vicgalle/alpaca-gpt4')
    
    parser.add_argument("--model_name",
                        help='training model;',
                        type=str,
                        default='huggyllama/llama-7b')# huggyllama/llama-7b # jingyaogong/MiniMind2-Small # meta-llama/Llama-2-7b-chat-hf # FacebookAI/roberta-base
    
    parser.add_argument('--split_strategy',
                        help='the split strategy. iid or non-iid, but in sft, we only support iid;',
                        type=str,
                        default="iid")
    
    parser.add_argument('--num_clients',
                        help='total clients',
                        type=int,
                        default=20)
    
    parser.add_argument('--num_clients_per_round',
                        help='number of clients trained per round;',
                        type=int,
                        default=2)
    
    parser.add_argument('--batch_size',
                        help='training data batch size',
                        type=int,
                        default=8)
    
    parser.add_argument('--max_steps',
                        help='max steps',
                        type=int,
                        default=20)
    
    parser.add_argument('--gradient_accumulation_steps',
                        help='gradient_accumulation_steps',
                        type=int,
                        default=1)
    
    parser.add_argument('--lr',
                        help='learning rate',
                        type=float,
                        default=5e-5)
    
    parser.add_argument('--epochs', 
                        help='number of epochs when clients train on data;',
                        type=int,
                        default=1)
    
    parser.add_argument('--num_rounds',
                        help='number of communication rounds when clients train on data;',
                        type=int,
                        default=200)
    
    parser.add_argument('--seed',
                        help='seed for randomness;',
                        type=int,
                        default=2025)
    
    parser.add_argument('--log_file',
                        help='logging output file',
                        type=str,
                        default="./log/log.txt")
    
    parser.add_argument('--load_in_8bit',
                        help='load_in_8bit',
                        type=bool,
                        default=False)
    
    parser.add_argument('--load_in_4bit',
                        help='load_in_4bit',
                        type=bool,
                        default=False)
    
    parser.add_argument('--use_peft',
                        help='use pert',
                        type=bool,
                        default=True)
    
    parser.add_argument('--dataset_sample',
                        help='dataset sample number which limits the dataset size',
                        type=int,
                        default=20000)
    
    parser.add_argument('--peft_lora_r',
                        help='lora rank',
                        type=int,
                        default=32)
    
    parser.add_argument('--peft_lora_alpha',
                        help='lora alpha',
                        type=int,
                        default=64)
    
    parser.add_argument('--seq_len',
                        help='max sequence length for training',
                        type=int,
                        default=512)
    
    parser.add_argument('--template',
                        help='chat template',
                        type=str,
                        default="alpaca")
    
    try: parsed = parser.parse_args()
    except IOError as msg: parser.error(str(msg))

    return parsed

def get_model_config(args):
    if args.load_in_8bit and args.load_in_4bit:
        raise ValueError("You can't load the model in 8 bits and 4 bits at the same time")
    elif args.load_in_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=args.load_in_8bit,
            llm_int8_has_fp16_weight=True
        )
        # # Copy the model to each device
        # device_map = {"": Accelerator().local_process_index}
        device_map = None
        torch_dtype = torch.bfloat16
    elif args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=args.load_in_4bit,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        # # Copy the model to each device
        # device_map = {"": Accelerator().local_process_index}
        device_map = None
        torch_dtype = torch.bfloat16
    else:
        device_map = None
        quantization_config = None
        torch_dtype = torch.bfloat16
    return device_map, quantization_config, torch_dtype

def get_model_and_tokenizer(args):
    device_map, quantization_config, torch_dtype = get_model_config(args)
    if args.task == "classification":
        label_map = {
            "sst2": 2,
            "cola": 2,
            "mrpc": 2,
            "qqp": 2,
            "stsb": 1,
            "qnli": 2,
            "rte": 2,
            "wnli": 2,
            "mnli_matched": 3,
            "mnli_mismatched": 3,
        }
        model = RobertaForSequenceClassification.from_pretrained(
            args.model_name, 
            num_labels=label_map[args.dataset_name],
            # quantization_config=quantization_config,
            # device_map=device_map,
            # trust_remote_code=True,
            # torch_dtype=torch_dtype,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            quantization_config=quantization_config,
            device_map=device_map,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
        )

    if args.load_in_4bit or args.load_in_8bit:
        model = prepare_model_for_kbit_training(model)

    if args.task == "classification":
        tokenizer = RobertaTokenizer.from_pretrained(args.model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, padding_side="right") # 在训练的时候在right 在推理的时候在left

    # if tokenizer.pad_token is None:
    if args.task == "sft":
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def get_peft_config(args):
    # ===== Define the LoraConfig =====
    if args.use_peft:
        peft_config = LoraConfig(
            r=args.peft_lora_r,
            lora_alpha=args.peft_lora_alpha,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM" if args.task == "sft" else TaskType.SEQ_CLS,
            target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"] if args.task == "sft" 
            else ["query", "key", "value","classifier.dense","classifier.out_proj"]
        )
    else:
        peft_config = None
    return peft_config


# def create_peft_model(num_labels, args):

#     model = RobertaForSequenceClassification.from_pretrained(
#         args.model, num_labels=num_labels
#     )

#     peft_config = LoraConfig(
#         task_type="SEQ_CLS",
#         r=args.lora_r,
#         lora_alpha=args.lora_alpha,
#         lora_dropout=args.lora_dropout,
#         use_rslora=args.rslora,
#         target_modules=["query", "value"],
#     )

#     model = get_peft_model(model, peft_config)

#     return model