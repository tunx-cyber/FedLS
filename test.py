from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
model = AutoModelForCausalLM.from_pretrained("jingyaogong/MiniMind2-Small",device_map = "cuda:0")
tokenizer = AutoTokenizer.from_pretrained("jingyaogong/MiniMind2-Small")

alpaca_template = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{} 

### Response:"""

while True:
    prompt = input()
    prompt = alpaca_template.format(prompt)
    streamer = TextStreamer(tokenizer, skip_prompt=False, skip_special_tokens=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    _ = model.generate(**inputs, max_new_tokens=2048, streamer=streamer)

'''
C Answer me with only A or B or C or D
Which of the following is NOT a symptom of anaphylaxis? [ A. "Stridor.", B. "Bradycardia.", C. "Severe wheeze.", D. "Rash." ]
The following are multiple choice questions. Find the degree for the given field extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q. [A. 0, B. 4, C. 2, D. 6] 
In hypovolaemic shock, what percentage of blood can be lost before it is reflected in changes in heart rate and blood pressure? [ A. "5%", B. "10%", C. "20%", D. "30%" ]
There is a multiple choice question with optional answers. Choose the right answer. The question is "Let p = (1, 2, 5, 4)(2, 3) in S_5 . Find the index of <p> in S_5." and the optional answer is [ A. "8", B. "2", C. "24", D. "120" ]

There is a multiple choice question with optional answers. The question is "Find the degree for the given field extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q." and the optional answers are [A. "0", B. "4", C. "2", D. "6"]. Choose the right answer. Answer it with only "A" or "B" or "C" or "D"
'''
