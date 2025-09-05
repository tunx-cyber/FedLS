from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
def get_merged_model(model_name, lora_path, merge_path):
    base_model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # 加载训练好的 LoRA 适配器
    model = PeftModel.from_pretrained(base_model, lora_path)

    # 合并 LoRA 到基础模型并卸载 LoRA 结构
    merged_model = model.merge_and_unload()

    # 保存合并后的模型
    merged_model.save_pretrained(merge_path)
    tokenizer.save_pretrained(merge_path)

def sample_layers(scores, min_prob=0.1, max_prob=0.9):
    """
    根据层分数采样决定哪些层参与训练。
    
    参数:
        scores (list/np.array): 各层重要性分数
        min_prob (float): 最低采样概率（保证最弱层也有机会）
        max_prob (float): 最高采样概率（避免强制保留）
    
    返回:
        mask (np.array): 二进制掩码，1表示参与训练，0表示跳过
    """
    scores = np.array(scores)
    n = len(scores)
    
    # 归一化分数到 [0, 1]
    if np.max(scores) - np.min(scores) < 1e-6:  # 所有分数相同
        norm_scores = np.ones(n) * 0.5
    else:
        norm_scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
    
    # 线性映射到概率区间 [min_prob, max_prob]
    probs = min_prob + (max_prob - min_prob) * norm_scores
    print(probs)
    # 伯努利采样
    mask = np.random.binomial(1, probs)  # 1=保留, 0=跳过
    
    return mask
import numpy as np
if __name__ == "__main__":
    # model_name = ""
    # lora_path = "lora_model"
    # merge_path = "merged_model"
    # get_merged_model(model_name, lora_path, merge_path)
    # print(f"Model merged and saved to {merge_path}")
    # text = "the correct answer is a"
    # print(text.split("the correct answer is"))
    # name = "a/b"
    # print(name.replace("/", "_"))
    # 假设有5个层的分数（分数越高越重要）
    layer_scores = [0.2, 1.5, 0.8, 3.0, 0.5]

    # 采样（每次结果可能不同）
    mask = sample_layers(layer_scores)
    print("采样掩码 (参与=1, 跳过=0):", mask)