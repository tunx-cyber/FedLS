# MT-Bench
import json
from openai import OpenAI
from vllm import LLM, SamplingParams
import argparse
alpaca_template = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{} 

### Response:
"""

api_key = "sk-ff8238fda8954c9baf9d8a24310310ff"
JUDGE_MODEL = "deepseek-chat"  # 可以换成 gpt-4o, gpt-4-turbo 等
client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
def generate_prompt(prompt, context=None):
    if context:
        # 如果有上下文，将其添加到 prompt 中
        prompt = f'The following is a conversation between a user and an assistant.\nPlease continue by answering the last question.\nConversation history:\nUser: {context[0]}\nAssistant:{context[1]}'
        prompt = prompt + "\nUser: " + prompt
        prompt = alpaca_template.format(prompt)
    else:
        prompt = alpaca_template.format(prompt)
    
    return prompt
    
def judge_answer(question, answer, turn):
    system_prompt = (
        "You are a professional dialogue evaluator. "
        "Score the given answer based on accuracy, completeness, reasoning quality, and clarity. "
        "Only return a number between 1 and 10, no explanations."
    )
    user_prompt = (
        f"Question (Turn {turn}): {question}\n"
        f"Model answer: {answer}\n"
        "Please provide a score (1-10):"
    )

    resp = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        stream=False
    )

    score_str = resp.choices[0].message.content.strip()
    try:
        score = float(score_str)
    except ValueError:
        score = 0.0
    return score

def test(model):
    # set up VLLM
    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=1024)
    llm = LLM(model=model, tensor_parallel_size=1)

    # load data
    mt_bench_questions = []
    with open("dataset/mt_bench.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:  # skip empty lines
                obj = json.loads(line)
                mt_bench_questions.append(obj)
        # mt_bench_questions = json.load(f)

    scores_round1 = []
    scores_round2 = []

    for i, q in enumerate(mt_bench_questions, start=1):
        turn1, turn2 = q["turns"]

        # first turn inference
        ans1 = llm.generate(generate_prompt(turn1),sampling_params=sampling_params)
        ans1 = ans1[0].outputs[0].text.strip()

        # second turn inference
        ans2 = llm.generate(generate_prompt(turn2, context=[turn1, ans1]),sampling_params=sampling_params)
        ans2 = ans2[0].outputs[0].text.strip()

        # score the two turns
        s1 = judge_answer(turn1, ans1, turn=1)
        s2 = judge_answer(turn2, ans2, turn=2)

        scores_round1.append(s1)
        scores_round2.append(s2)

        print(f"question {i}: turn1 {s1} score, turn2 {s2} score")


    avg1 = sum(scores_round1)/len(scores_round1)
    avg2 = sum(scores_round2)/len(scores_round2)
    avg_all = (sum(scores_round1) + sum(scores_round2)) / (len(scores_round1) + len(scores_round2))

    print("\n====== MT-Bench evaluation result ======")
    print(f"tunr 1 avg: {avg1:.2f}")
    print(f"turn 2 avg: {avg2:.2f}")
    print(f"avg: {avg_all:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",type=str,default='chat_model')
    try: parsed = parser.parse_args()
    except IOError as msg: parser.error(str(msg))

    model = parsed.model
    test(model)