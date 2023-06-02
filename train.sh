export MODEL_PATH="/apdcephfs_cq3/share_2973545/data/models/shibing624/chinese-alpaca-plus-7b-hf"
export SAVE_PATH="output_lora"
export DATA_PATH="data/alpaca_responses_hh_100.json"
export MASTER_ADDR="localhost"
export MASTER_PORT="22"
export WANDB_DISABLED=true
wandb offline



# python3 -m torch.distributed.launch --nproc_per_node=4 --use_env train.py \
#     --peft_type "no_lora" \
#     --model_name_or_path $MODEL_PATH \
#     --data_path $DATA_PATH \
#     --output_dir $SAVE_PATH \
#     --fp16 True \
#     --num_train_epochs 3 \
#     --per_device_train_batch_size 1 \
#     --per_device_eval_batch_size 1 \
#     --gradient_accumulation_steps 8 \
#     --evaluation_strategy "no" \
#     --save_strategy "steps" \
#     --save_steps 100 \
#     --save_total_limit 40 \
#     --learning_rate 2e-5 \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --fsdp "full_shard auto_wrap" \
#     --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
#     --model_max_length 256 \
#     --rrhf_weight 1 \


torchrun --nproc_per_node 4 train.py \
    --peft_type "LORA" \
    --model_name_or_path $MODEL_PATH \
    --data_path $DATA_PATH \
    --output_dir $SAVE_PATH \
    --fp16 True \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 40 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 512 \
    --rrhf_weight 1 \
