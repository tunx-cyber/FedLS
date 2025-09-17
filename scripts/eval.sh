# commonsense
python merge.py --lora "/root/llm/output/fedls/zwhe99_commonsense_170k" --output "qa_model" && CUDA_VISIBLE_DEVICES=0,1,2,3 python -m eval.eval_qa --model qa_model

# mt-bench
python merge.py --lora "/root/llm/output/fedls/vicgalle_alpaca-gpt4" --output "chat_model" && CUDA_VISIBLE_DEVICES=1 python -m eval.eval_chat --model chat_model
CUDA_VISIBLE_DEVICES=7 python -m eval.eval_chat --model /root/llm/output/flora/vicgalle_alpaca-gpt4
# mmlu
python merge.py --lora "/root/llm/output/fedls/vicgalle_alpaca-gpt4" --output "chat_model" && CUDA_VISIBLE_DEVICES=4,5,6,7 python -m eval.eval_mmlu --model chat_model

# math
python merge.py --lora /root/llm/output/fedls/meta-math_MetaMathQA --output "math_model" && CUDA_VISIBLE_DEVICES=4,5,6,7 python -m eval.eval_math --model math_model
python merge.py --lora /root/llm/output/fedls/meta-math_MetaMathQA --output "math_model" && CUDA_VISIBLE_DEVICES=4,5,6,7 python -m eval.eval_gsm8k --model math_model


#glue
python -m eval.eval_glue --lora /root/llm/output/fedex/cola --task cola
