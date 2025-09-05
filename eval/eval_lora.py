from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

# 1. 初始化 LLM 实例，启用 LoRA
llm = LLM(
    model="/home/models/llama-7b", # 基础模型路径
    enable_lora=True,              # 启用 LoRA
    max_lora_rank=64,              # 重要：设置最大 rank，匹配你的训练设置
    tensor_parallel_size=1,        # GPU 数量
)

# 2. 定义采样参数
sampling_params = SamplingParams(
    temperature=0.8,
    max_tokens=100,
)

# 3. 创建 LoRA 请求对象，指定要使用的适配器
lora_request = LoRARequest(
    lora_name="my-finance-lora",         # 名称，可自定义，但需唯一
    lora_int_id=1,                       # 一个唯一的整数 ID，用于内部标识
    lora_local_path="/home/loras/my-finance-lora" # LoRA 权重路径
)

# 4. 准备你的输入
prompts = ["什么是市盈率？"]

# 5. 生成文本，传入 lora_request 参数
outputs = llm.generate(
    prompts,
    sampling_params,
    lora_request=lora_request # 关键：传入 LoRA 请求
)

# 6. 输出结果
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")