# already evaluated in trainning process
from utils.data import get_classification_dataset, ClassificationDataset
from peft import PeftModel
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.utils.data import DataLoader
import torch
import argparse
task = "sst2"
parser = argparse.ArgumentParser()
parser.add_argument('--lora', default='')
parser.add_argument('--task', default='')
try: args = parser.parse_args()
except IOError as msg: parser.error(str(msg))
task = args.task
lora_path = args.lora
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
base_model = RobertaForSequenceClassification.from_pretrained(
    "FacebookAI/roberta-base", 
    num_labels=label_map[task],
)
tokenizer = RobertaTokenizer.from_pretrained("FacebookAI/roberta-base")
# 加载训练好的 LoRA 适配器
model = PeftModel.from_pretrained(base_model, lora_path)

# 合并 LoRA 到基础模型并卸载 LoRA 结构
merged_model = model.merge_and_unload()

train_dataset, test_dataset,_ = get_classification_dataset(task, 20000)
test_dataset = ClassificationDataset(
                task,
                test_dataset, 
                tokenizer, 
            )
test_dataloder = DataLoader(test_dataset, batch_size=128)
merged_model.eval()
merged_model.cuda()
total = 0
correct = 0
with torch.no_grad():
    for idx, batch in enumerate(test_dataloder):
        batch = {k: v.to(merged_model.device) for k, v in batch.items()}
        outputs = merged_model(**batch)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        total += batch["labels"].size(0)
        correct += (predictions == batch["labels"]).sum().item()
accuracy = correct / total

print(f"{task}: accuracy:{accuracy}")