# keep A fixed. train B.
from utils.utils import cosine_learning_rate,get_model_and_tokenizer,draw_loss_curve, setup_logger,get_peft_config
from utils.data import get_dataset, split_dataset, SFTDataset, get_dataset_this_round,get_classification_dataset,ClassificationDataset
from peft import get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
import torch
import torch.nn as nn
import math
from torch.utils.data import DataLoader
import numpy as np
import copy
from datetime import datetime
class FFALora:
    def __init__(self, args):
        self.args = args
    
    def run(self):
        args = self.args
        logger = setup_logger(args.fed_alg, f"./logs/{args.fed_alg}_{args.dataset_name}.txt")
        # init model and tokenizer
        model, tokenizer = get_model_and_tokenizer(args)
        peft_config = get_peft_config(args)
        model = get_peft_model(model, peft_config)
        # fix A and initialize B
        base_seed = 42
        for name, param in model.named_parameters():
            i = 0
            if "lora_A" in name:
                # Create a unique seed for each parameter (using hash of the parameter name)
                unique_seed = base_seed + i
                i += 1
                with torch.random.fork_rng(devices=[param.device]):
                    torch.random.manual_seed(unique_seed)
                    nn.init.kaiming_uniform_(param, a=math.sqrt(5))
                param.requires_grad = False
        
        model.print_trainable_parameters()
        model.config.use_cache = False
        model.cuda()

        # set up the global and local models
        global_dict = copy.deepcopy(get_peft_model_state_dict(model))
        # local_dict_list = [copy.deepcopy(global_dict) for i in range(args.num_clients)]

        # set up datasets
        if args.task == "classification":
            train_dataset, _, test_dataset = get_classification_dataset(args.dataset_name, args.dataset_sample)
            test_dataset = ClassificationDataset(
                args.dataset_name,
                test_dataset, 
                tokenizer, 
            )
            test_dataloder = DataLoader(test_dataset, batch_size=64)
            accuracy = self.eval_model(model, test_dataloder)
            print(f"Initial Test Accuracy: {accuracy}")
        elif args.task == "sft":
            train_dataset = get_dataset(args.dataset_name, args.dataset_sample)
        
        local_datasets = split_dataset(args, train_dataset)

        rounds_loss = []
        for r in range(args.num_rounds):
            print(f"Round {r + 1}/{args.num_rounds}")

            sample_num_list = []
            participants = np.random.choice(range(args.num_clients), args.num_clients_per_round, replace=False)
            if args.task == "sft":
                new_lr = cosine_learning_rate(r, args.num_rounds, args.lr, 1e-6)
            else:
                new_lr = args.lr  
            round_loss = []
            local_dict_list = []
            for client_id in participants:
                print(f">> ==================== Round {r+1} : {client_id} ====================")
                # send the global model to the client
                set_peft_model_state_dict(model, global_dict)
                # get dataloader this round
                if args.task == "sft":
                    dataset_this_round = get_dataset_this_round(local_datasets[client_id], r, args)
                    dataset_this_round = SFTDataset(dataset_this_round, 
                                                    tokenizer, 
                                                    template_name=args.template, 
                                                    max_len=args.seq_len, 
                                                    math_reason=True if "math" in args.dataset_name else False)
                elif args.task == "classification":
                    dataset_this_round = ClassificationDataset(
                        args.dataset_name,
                        local_datasets[client_id], 
                        tokenizer, 
                    )

                sample_num_list.append(len(dataset_this_round))
                local_dataloader = DataLoader(dataset_this_round, batch_size=args.batch_size, shuffle=True)
                # recieve the local model
                model, loss = self.local_train(model, local_dataloader, new_lr, args)
                # Save the local model state
                local_dict_list.append(copy.deepcopy(get_peft_model_state_dict(model)))
                round_loss.append(loss)
            # Aggregate the local models to update the global model
            self.aggerate_local_models(local_dict_list, global_dict, sample_num_list)

            rounds_loss.append(sum(round_loss)/len(round_loss))

            if args.task == "classification":
                set_peft_model_state_dict(model, global_dict)
                accuracy = self.eval_model(model, test_dataloder)
                print(f"Round {r+1} test Accuracy: {accuracy}")

        model.save_pretrained(f"./output/{args.fed_alg}/{args.dataset_name.replace('/','_')}")
        tokenizer.save_pretrained(f"./output/{args.fed_alg}/{args.dataset_name.replace('/','_')}")
        logger.info("loss data:")
        logger.info(rounds_loss)
        draw_loss_curve(range(args.num_rounds),rounds_loss,args)
    
    def local_train(self, model, local_dataloader, lr, args):
        # torch.compile(model)
        steps = 0
        training_loss = 0
        for epoch in range(args.epochs):
            print(f"Epoch {epoch + 1}/{args.epochs}")
            model.train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
            for idx, batch in enumerate(local_dataloader):
                batch = {k: v.to(model.device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                print(f"{datetime.now()} Loss: {loss.item()}")
                steps += 1
                training_loss += loss.item()
        
        return model, training_loss/steps
    
    def eval_model(self, model, eval_dataloader):
        model.eval()
        total = 0
        correct = 0
        with torch.no_grad():
            for idx, batch in enumerate(eval_dataloader):
                batch = {k: v.to(model.device) for k, v in batch.items()}
                outputs = model(**batch)
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)
                total += batch["labels"].size(0)
                correct += (predictions == batch["labels"]).sum().item()
        accuracy = correct / total
        return accuracy
    
    def aggerate_local_models(self, local_dict_list, global_dict, sample_num_list):
        total_samples = sum(sample_num_list)
        with torch.no_grad():
            for key in global_dict.keys():
                if "lora_B".lower() in key.lower():
                    global_dict[key] = sum([local_dict[key] * sample_num_list[idx] / total_samples 
                                            for idx, local_dict in enumerate(local_dict_list)])
        return global_dict