sd_path=""
batch_size=4

CUDA_VISIBLE_DEVICES=0 python -m gen_data \
    --work-dir data/gen_voc \
    --json-path data/prompts/voc_prompts.json \
    --sd-path $sd_path \
    --batch-size $batch_size


# If you have 2 gpus, you can generate the dataset in parallel with
# ```
# CUDA_VISIBLE_DEVICES=0 python -m gen_data 
#     --work-dir data/gen_voc \
#     --json-path data/prompts/voc_prompts.json \
#     --sd-path $sd_path \
#     --batch-size $batch_size \
#     --start 0 --end 20000

# CUDA_VISIBLE_DEVICES=1 python -m gen_data 
#     --work-dir data/gen_voc \
#     --json-path data/prompts/voc_prompts.json \
#     --sd-path $sd_path \
#     --batch-size $batch_size \
#     --start 20000 --end 40000
# ```