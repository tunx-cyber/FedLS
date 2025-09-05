# federated layerwise selection
from utils.utils import cosine_learning_rate,get_model_and_tokenizer,draw_loss_curve, setup_logger,get_peft_config
from utils.data import get_dataset, split_dataset, SFTDataset, get_dataset_this_round
from peft import get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
import torch
from torch.utils.data import DataLoader
import numpy as np
import copy
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.stats import entropy
import numpy as np
from collections import OrderedDict

def compute_weight_stats(param_dict):
    l2_norm = []
    entropy = []
    names = []
    for name, param in param_dict.items():
        l2_norm.append(torch.norm(param_dict[name], p = 2).cpu().item())
        P = torch.abs(param_dict[name])
        P = P / P.sum()
        P = P[P>0]
        H = - (P * torch.log(P)).sum()
        entropy.append(H.item())
        names.append(name)
    
    return l2_norm, entropy, names

def fixed_sample_layers(l2_norm, entropy, rho = 0.5, ritio = 0.3):
    l2_norm = np.array(l2_norm)
    entropy = np.array(entropy)
    if np.max(l2_norm) - np.min(l2_norm) < 1e-6:  # 所有分数相同
        l2_norm = np.ones(len(l2_norm)) * 0.5
    else:
        l2_norm = (l2_norm - np.min(l2_norm)) / (np.max(l2_norm) - np.min(l2_norm))
    
    if np.max(entropy) - np.min(entropy) < 1e-6:  # 所有分数相同
        entropy = np.ones(len(entropy)) * 0.5
    else:
        entropy = (entropy - np.min(entropy)) / (np.max(entropy) - np.min(entropy))
    
    scores = rho * l2_norm + (1-rho) * entropy

    idx = ritio * len(scores)
    sort_scores = sorted(scores)
    min_score = sort_scores[int(idx)]
    mask = [1] * len(scores)
    cnt = 0# avoid the shadow layer to be masked 
    for i in range(len(mask)):
        if scores[i] < min_score :
            mask[i] = 0
            cnt += 1
            if cnt > idx:
                break
    
    return mask

def sample_layers(l2_norm, entropy, rho = 0.5, min_prob=0.1, max_prob=0.9):

    l2_norm = np.array(l2_norm)
    entropy = np.array(entropy)
    if np.max(l2_norm) - np.min(l2_norm) < 1e-6:  # 所有分数相同
        l2_norm = np.ones(len(l2_norm)) * 0.5
    else:
        l2_norm = (l2_norm - np.min(l2_norm)) / (np.max(l2_norm) - np.min(l2_norm))
    
    if np.max(entropy) - np.min(entropy) < 1e-6:  # 所有分数相同
        entropy = np.ones(len(entropy)) * 0.5
    else:
        entropy = (entropy - np.min(entropy)) / (np.max(entropy) - np.min(entropy))

    # mid_score = np.median(l2_norm)
    scores = rho * l2_norm + (1-rho) * entropy
    
    # 线性映射到概率区间 [min_prob, max_prob]

    probs = min_prob + (max_prob - min_prob) * scores
    
    # 伯努利采样
    mask = np.random.binomial(1, probs)  #
    
    return mask

def plot_weight_stats(layer_norms, layer_entropies):
    """绘制权重矩阵统计量折线图"""
    layer_names = list(layer_norms.keys())
    x = np.arange(len(layer_names))
    
    # 创建图形和双纵轴
    fig, ax1 = plt.subplots(figsize=(14, 7))
    
    # 绘制平均二范数
    color = 'tab:blue'
    ax1.set_xlabel('Layer Weight Matrices')
    ax1.set_ylabel('Average L2 Norm', color=color)
    ax1.plot(x, list(layer_norms.values()), 'o-', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xticks(x)
    # ax1.set_xticklabels(layer_names, rotation=45, ha='right')
    
    # 创建第二个纵轴
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Normalized Entropy', color=color)
    ax2.plot(x, list(layer_entropies.values()), 's--', color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('Weight Matrix Statistics per Layer')
    fig.tight_layout()
    plt.grid(linestyle='--', alpha=0.7)
    plt.savefig(f"img/{datetime.now()}.png")
    plt.close()



class FedLS:
    def __init__(self, args):
        self.args = args
    
    def run(self):
        args = self.args
        logger = setup_logger(args.fed_alg, f"./logs/{args.fed_alg}_{args.dataset_name}.txt")
        # init model and tokenizer
        model, tokenizer = get_model_and_tokenizer(args)
        peft_config = get_peft_config(args)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        model.config.use_cache = False
        model.cuda()
        print(model.base_model_torch_dtype)
        # set up the global and local models
        global_dict = copy.deepcopy(get_peft_model_state_dict(model))
        # local_dict_list = [copy.deepcopy(global_dict) for i in range(args.num_clients)]

        # set up datasets
        dataset = get_dataset(args.dataset_name, args.dataset_sample)
        local_datasets = split_dataset(args, dataset)

        rounds_loss = []
        mask = None
        keys = None
        buffer = {}
        for r in range(args.num_rounds):
            print(f"Round {r + 1}/{args.num_rounds}")

            sample_num_list = []
            participants = np.random.choice(range(args.num_clients), args.num_clients_per_round, replace=False)
            new_lr = cosine_learning_rate(r, args.num_rounds, args.lr, 1e-6) 
            round_loss = []
            local_dict_list = []
            for client_id in participants:
                print(f">> ==================== Round {r+1} : {client_id} ====================")
                # send the global model to the client
                set_peft_model_state_dict(model, global_dict)
                # for name, param in model.named_parameters():
                #     if "lora" in name:
                #         param.requires_grad = True
                # if mask is not None:
                #     # print(mask)
                #     mask_dict = {}
                #     for i in range(len(mask)):
                #         mask_dict[keys[i].replace("weight","default.weight")] = mask[i]
                #     # print(keys)
                #     # base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight
                #     for name, param in model.named_parameters():
                #         # print(name)
                #         # base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight
                #         if name in mask_dict.keys() and mask_dict[name] == 0:
                #             param.requires_grad = False
                # model.print_trainable_parameters()
                # get dataloader this round
                dataset_this_round = get_dataset_this_round(local_datasets[client_id], r, args)
                dataset_this_round = SFTDataset(dataset_this_round, tokenizer, template_name=args.template, max_len=args.seq_len)
                sample_num_list.append(len(dataset_this_round))
                local_dataloader = DataLoader(dataset_this_round, batch_size=args.batch_size, shuffle=True)
                # recieve the local model
                model, loss = self.local_train(model, local_dataloader, new_lr, args)
                # Save the local model state
                local_dict_list.append(copy.deepcopy(get_peft_model_state_dict(model)))
                round_loss.append(loss)
            
            # Aggregate the local models to update the global model
            mask, keys = self.aggerate_local_models(local_dict_list, global_dict, sample_num_list, buffer)

            rounds_loss.append(sum(round_loss)/len(round_loss))

        model.save_pretrained(f"./output/{args.fed_alg}/{args.dataset_name.replace('/','_')}")
        tokenizer.save_pretrained(f"./output/{args.fed_alg}/{args.dataset_name.replace('/','_')}")
        logger.info("loss data:")
        logger.info(rounds_loss)
        draw_loss_curve(range(args.num_rounds),rounds_loss,args)
    
    def local_train(self, model, local_dataloader, lr, args):
        # torch.compile(model)
        steps = 0
        training_loss = 0
        model.train()
        for epoch in range(args.epochs):
            print(f"Epoch {epoch + 1}/{args.epochs}")
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
    # 分阶段 第一阶段全部参数参与，第二阶段开始逐渐剪枝，第三阶段cos概率激活一些被剪枝的
    def aggerate_local_models(self, local_dict_list, global_dict, sample_num_list, buffer):
        total_samples = sum(sample_num_list)
        avg_dict = {}
        for key in global_dict.keys():
            avg_dict[key] = sum([local_dict_list[idx][key] * sample_num_list[idx] / total_samples for idx, d in enumerate(local_dict_list)])
        
        delta_dict = {}
        for key in global_dict.keys():
            delta_dict[key] = avg_dict[key] - global_dict[key]
        
        # l2 = []
        # l2_keys = []
        # for key in delta_dict.keys():
        #     l2.append(torch.norm(delta_dict[key], p = 2).cpu().item())
        #     l2_keys.append(key)
        l2_norm, entropy, names = compute_weight_stats(delta_dict)

        l2_norm = []
        ent = []
        for i in range(len(local_dict_list)):
            local_l2_norm = []
            local_ent = []
            for key in global_dict.keys():
                delta = local_dict_list[i][key] - global_dict[key]
                local_l2_norm.append(torch.norm(delta, p = 2).cpu().item())

                P = torch.abs(delta)
                P = P / P.sum()
                P = P[P>0]
                H = - (P * torch.log(P)).sum().cpu().item()
                local_ent.append(H)
            l2_norm.append(local_l2_norm)
            ent.append(local_ent)

        # plt.figure(figsize=(22, 4))
        # fig, ax1 = plt.subplots()
        # ax1.plot(range(len(l2_norm)),l2_norm,label = "l2",color='b')
        # ax1.set_ylabel("l2", color='b')
        # ax2 = ax1.twinx()
        # ax2.plot(range(len(entropy)),entropy,label = "entropy",color='g')
        # # print(entropy)
        # ax2.set_ylabel("ent", color='g')
        # plt.savefig(f"img/{datetime.now()}.png")
        # plt.close()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        im1 = ax1.imshow(np.array(l2_norm), cmap='viridis', aspect='auto')
        ax1.set_title('L2 norm')
        ax1.set_xlabel('layer')
        ax1.set_ylabel('client number')
        fig.colorbar(im1, ax=ax1, shrink=0.8)  # 添加颜色条

        # 绘制第二个热力图
        im2 = ax2.imshow(np.array(ent), cmap='plasma', aspect='auto')
        ax2.set_title('Entropy')
        ax2.set_xlabel('layer')
        ax2.set_ylabel('client number')
        fig.colorbar(im2, ax=ax2, shrink=0.8)  # 添加颜色条

        # 调整布局
        plt.tight_layout()
        plt.savefig(f"img/{datetime.now()}.png")
        plt.close()
        
        # 分级采样
        # cos 概率
        
        # mask = sample_layers(l2_norm, entropy, rho=1, min_prob=0.4)

        # with torch.no_grad():
        #     if len(buffer.keys()) > 0:
        #         for key in global_dict.keys():
        #             global_dict[key] = avg_dict[key] + 0.05 * buffer[key]
        #             buffer[key] = 0.5 * buffer[key] + 0.5 * delta_dict[key]
        #     else:
        #         for key in global_dict.keys():
        #             global_dict[key] = avg_dict[key]
        #             buffer[key] = delta_dict[key]
        with torch.no_grad():
            for key in global_dict.keys():
                global_dict[key] = avg_dict[key]

        return None, None
