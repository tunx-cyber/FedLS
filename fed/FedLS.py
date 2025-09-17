# federated layerwise selection
from utils.utils import cosine_learning_rate,get_model_and_tokenizer,draw_loss_curve, setup_logger,get_peft_config
from utils.data import get_dataset, split_dataset, SFTDataset, get_dataset_this_round,get_classification_dataset,ClassificationDataset
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
from .FedBase import FedBase
import math
def cosine_probability(epoch, min_prob, max_prob, period_epochs):
    """
    基于训练轮数的余弦变化概率函数
    
    参数:
    epoch (int): 当前训练轮数
    min_prob (float): 最小概率值
    max_prob (float): 最大概率值
    period_epochs (int): 完整周期所需的轮数
    
    返回:
    float: 在当前轮数下的概率值
    """
    # 计算当前周期内的位置 (0到2π之间)
    cycle_position = 2 * math.pi * (epoch % period_epochs) / period_epochs
    
    # 使用余弦函数计算概率值 (从max_prob到min_prob再到max_prob)
    # 余弦函数在0时为1，π时为-1，2π时又回到1
    # 我们将其映射到[min_prob, max_prob]区间
    amplitude = (max_prob - min_prob) / 2
    midpoint = min_prob + amplitude
    probability = midpoint + amplitude * math.cos(cycle_position)
    
    return probability
def compute_weight_stats(param_dict, histroy_norm):
    l2_norm = []
    entropy = []
    names = []
    for name, param in param_dict.items():
        norm = torch.norm(param_dict[name]).cpu().item()
        # l2_norm.append(norm)
        l2_norm.append(norm)

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

    idx = (1-ritio) * len(scores)
    sort_scores = sorted(scores)
    min_score = sort_scores[int(idx)]
    mask = [1] * len(scores)
    cnt = 0# avoid the shadow layer to be masked 
    for i in range(len(mask)):
        if scores[i] <= min_score :
            mask[i] = 0
            # cnt += 1
            # if cnt > idx:
            #     break
    
    return mask
        
def sample_layers(l2_norm, entropy, rho = 0.5, min_prob=0.1, max_prob=0.9, half = False):

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
    
    # 线性映射到概率区间 [min_prob, max_prob]

    probs = min_prob + (max_prob - min_prob) * scores
    
    # 伯努利采样
    mask = np.random.binomial(1, probs)  #选择为1，否则为0，分数高为高概率被选择1
    
    return mask


class FedLS(FedBase):
    def __init__(self, args):
        super(FedLS,self).__init__(args)
        self.history_norm = {}
    
    def run(self):
        args = self.args
        logger = setup_logger(args.fed_alg, f"./logs/{args.fed_alg}/{args.dataset_name.replace('/','_')}.txt")
        # init model and tokenizer
        model, tokenizer = get_model_and_tokenizer(args)
        peft_config = get_peft_config(args)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        model.config.use_cache = False
        model.cuda()
        # print(model.base_model_torch_dtype)
        # set up the global and local models
        global_dict = copy.deepcopy(get_peft_model_state_dict(model))
        # local_dict_list = [copy.deepcopy(global_dict) for i in range(args.num_clients)]
        # init history
        for key in global_dict.keys():
            if key not in self.history_norm:
                self.history_norm[key] = []
        # set up datasets
        if args.task == "classification":
            train_dataset, test_dataset,_ = get_classification_dataset(args.dataset_name, args.dataset_sample)
            test_dataset = ClassificationDataset(
                args.dataset_name,
                test_dataset, 
                tokenizer, 
            )
            test_dataloder = DataLoader(test_dataset, batch_size=128)
            accuracy = self.eval_model(model, test_dataloder)
            print(f"Initial Test Accuracy: {accuracy}")
        elif args.task == "sft":
            train_dataset = get_dataset(args.dataset_name, args.dataset_sample)
        
        local_datasets = split_dataset(args, train_dataset)

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
                if args.task == "sft":
                    set_peft_model_state_dict(model, global_dict)
                else:
                    set_peft_model_state_dict(model, copy.deepcopy(global_dict))# fast fix bug here

                for name, param in model.named_parameters():
                    if "lora" in name:
                        param.requires_grad = True
                if mask is not None:
                    # print(mask)
                    mask_dict = {}
                    for i in range(len(mask)):
                        mask_dict[keys[i].replace("weight","default.weight")] = mask[i]
                    # print(keys)
                    # base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight
                    for name, param in model.named_parameters():
                        # print(name)
                        # base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight
                        if name in mask_dict.keys() and mask_dict[name] == 0:
                            param.requires_grad = False
                model.print_trainable_parameters()
                # get dataloader this round
                # get dataloader this round
                if args.task == "sft":
                    dataset_this_round = get_dataset_this_round(local_datasets[client_id], r, args)
                    dataset_this_round = SFTDataset(dataset_this_round, 
                                                    tokenizer, 
                                                    template_name=args.template, 
                                                    max_len=args.seq_len, 
                                                    math_reason=True if "math" in args.dataset_name else False)
                elif args.task == "classification":
                    dataset_this_round = get_dataset_this_round(local_datasets[client_id], r, args)
                    dataset_this_round = ClassificationDataset(
                        args.dataset_name,
                        dataset_this_round, 
                        tokenizer, 
                    )

                sample_num_list.append(len(dataset_this_round))
                local_dataloader = DataLoader(dataset_this_round, batch_size=args.batch_size, shuffle=True)
                # recieve the local model
                model, loss = self.local_train(model, local_dataloader, new_lr, args)
                # Save the local model state
                local_dict_list.append(copy.deepcopy(get_peft_model_state_dict(model)))
                round_loss.append(loss)
                torch.cuda.empty_cache()
            # Aggregate the local models to update the global model
            fixed = False
            if r > 100: 
                fixed = True
            if r+1 < args.num_rounds:
                mask, keys = self.aggerate_local_models(local_dict_list, global_dict, sample_num_list, buffer,r,mask, fixed)
            else:
                self.aggerate_local_models_svd(local_dict_list,global_dict, sample_num_list)
            rounds_loss.append(sum(round_loss)/len(round_loss))
            if args.task == "classification":
                set_peft_model_state_dict(model, copy.deepcopy(global_dict))
                accuracy = self.eval_model(model, test_dataloder)
                logger.info(f"Round {r+1} test Accuracy: {accuracy}")

        model.save_pretrained(f"./output/{args.fed_alg}/{args.dataset_name.replace('/','_')}")
        tokenizer.save_pretrained(f"./output/{args.fed_alg}/{args.dataset_name.replace('/','_')}")
        logger.info("loss data:")
        logger.info(rounds_loss)
        # draw_loss_curve(range(args.num_rounds),rounds_loss,args)
    
    def aggerate_local_models_svd(self, local_dict_list, global_dict, sample_num_list):# for last agg
        total_samples = sum(sample_num_list)
        # print(g_state_dict.keys())
        with torch.no_grad():
            for key in global_dict.keys():
                if "lora_A" in key:
                    lora_B_key = key.replace("lora_A", "lora_B")
                    # base_layer_key = key.replace("lora_A", "base_layer")
                    base_weight = sum(local_dict[lora_B_key] @ local_dict[key] * sample_num_list[idx] / total_samples
                                      for idx, local_dict in enumerate(local_dict_list))
                    # TODO 现在base的基础上加上这个然后再分解
                    # print(base_layer_key)
                    r = self.args.peft_lora_r
                    U, S, V = torch.svd_lowrank(base_weight,q=r,niter=2)

                    # 构造两个矩阵
                    M1 =  U @ torch.diag(S)   # m x r
                    M2 = V.T  # r x n
                    # print(global_dict[key].shape, global_dict[lora_B_key].shape, M1.shape, M2.shape)
                    # A_approx = M1 @ M2
                    # print("误差:", torch.norm(base_weight - A_approx).item())
                    global_dict[key] = M2
                    global_dict[lora_B_key] = M1
                    
        return global_dict
    
    def aggerate_local_models(self, local_dict_list, global_dict, sample_num_list, buffer, r, mask, fixed = False):
        total_samples = sum(sample_num_list)
        avg_dict = {}
        for key in global_dict.keys():
            avg_dict[key] = sum([local_dict_list[idx][key] * sample_num_list[idx] / total_samples for idx, d in enumerate(local_dict_list)])
        
        delta_dict = {}
        for key in global_dict.keys():
            delta_dict[key] = avg_dict[key] - global_dict[key]
            

        l2_norm, entropy, names = compute_weight_stats(delta_dict,self.history_norm)
        # l2_norm = []
        # ent = []
        # for i in range(len(local_dict_list)):
        #     local_l2_norm = []
        #     local_ent = []
        #     for key in global_dict.keys():
        #         delta = local_dict_list[i][key] - global_dict[key]
        #         local_l2_norm.append(torch.norm(delta, p = 2).cpu().item())

        #         P = torch.abs(delta)
        #         P = P / P.sum()
        #         P = P[P>0]
        #         H = - (P * torch.log(P)).sum().cpu().item()
        #         local_ent.append(H)
        #     l2_norm.append(local_l2_norm)
        #     ent.append(local_ent)
        # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 20))
        # im1 = ax1.imshow(np.array(l2_norm), cmap='viridis', aspect='auto')
        # ax1.set_title('L2 norm')
        # ax1.set_xlabel('layer')
        # ax1.set_ylabel('client number')
        # fig.colorbar(im1, ax=ax1, shrink=0.8)  # 添加颜色条

        # # 绘制第二个热力图
        # im2 = ax2.imshow(np.array(ent), cmap='plasma', aspect='auto')
        # ax2.set_title('Entropy')
        # ax2.set_xlabel('layer')
        # ax2.set_ylabel('client number')
        # fig.colorbar(im2, ax=ax2, shrink=0.8)  # 添加颜色条

        # # 调整布局
        # plt.tight_layout()
        # plt.savefig(f"img/{datetime.now()}.png")
        # plt.close()
        # if r == 0 or (r+1) % 5 == 0:
            # if fixed : 
            #     mask = fixed_sample_layers(l2_norm, entropy, rho=0.5, ritio=0.3)
            # else :
            # cosine decay

        # min_prob = cosine_probability(r,0.1,0.3,5)
        # max_prob = cosine_probability(r,0.7,0.9,5)
        if r == 0:
            mask = sample_layers(l2_norm, entropy, rho=0.5, min_prob=0.1, max_prob=0.9)

        
        with torch.no_grad():
            if r != 0:
                for key in global_dict.keys():
                    global_dict[key] = avg_dict[key]
            else:
                for key in global_dict.keys():
                    global_dict[key] = avg_dict[key]
                    buffer[key] = delta_dict[key]
        return mask, names
