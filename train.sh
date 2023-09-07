# export MODEL_PATH="/apdcephfs/private_curvasong/ad_llm/llama_sogou_ad/output/merge_lora_llama"
export MODEL_PATH="/apdcephfs_cq3/share_2973545/mingyyi/chatglm_6b_lora_sogou_qa2kt_6epoch/merge_lora_chatglm"  #加载的底座模型路径
export MODEL_TYPE="chatglm"
export SAVE_PATH="/apdcephfs/private_curvasong/output/ad_glm_rrhf_ddp_datav3"
export TRAIN_DATA_PATH="data/v3/rrhf_sogou_llm_v3_train.json"
export EVAL_DATA_PATH="data/v3/rrhf_sogou_llm_v3_eval.json"
export MASTER_ADDR="localhost"
export MASTER_PORT="22"
export WANDB_DISABLED=true
wandb offline


#deepspeed方式训练
# torchrun --nproc_per_node 8 train.py \
#     --peft_type "LORA" \
#     --model_name_or_path $MODEL_PATH \
#     --model_type $MODEL_TYPE \
#     --data_path $DATA_PATH \
#     --output_dir $SAVE_PATH \
#     --fp16 True \
#     --num_train_epochs 1 \
#     --per_device_train_batch_size 2 \
#     --per_device_eval_batch_size 1 \
#     --gradient_accumulation_steps 8 \
#     --evaluation_strategy "no" \
#     --save_steps 1 \
#     --save_strategy "steps" \
#     --save_total_limit 1 \
#     --learning_rate 2e-5 \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --model_max_length 256 \
#     --rrhf_weight 1 \
#     --deepspeed ds_config_zero3.json \


# ddp方式训练
torchrun --nproc_per_node 8 train.py \
    --peft_type "LORA" \
    --model_name_or_path $MODEL_PATH \
    --model_type $MODEL_TYPE \
    --train_data_path $TRAIN_DATA_PATH \
    --eval_data_path $EVAL_DATA_PATH \
    --fp16 True \
    --output_dir $SAVE_PATH \
    --num_train_epochs 5 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 10 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 256 \
    --rrhf_weight 1 \
