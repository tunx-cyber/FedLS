datasets=("sst2" "cola" "qqp" "mnli_matched" "mnli_mismatched" "mrpc" "rte")
alg=fedls
for dataset in "${datasets[@]}"
do
    CUDA_VISIBLE_DEVICES=1 python -m eval.eval_glue --lora /root/llm/output/$alg/$dataset --task $dataset
done
