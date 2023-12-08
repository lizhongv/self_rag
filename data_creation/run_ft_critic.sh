#cd data_creation
PATH_TO_TRAIN_DATA_FILE=/critic/gpt4_reward_all_0813_train.json
PATH_TO_CRITIC_MODEL=/critic/model

mkdir -p ${PATH_TO_CRITIC_MODEL}

torchrun --nproc_per_node=2 \
  --master_port=2568 train_special_tokens.py \
  --model_name_or_path meta-llama/Llama-2-7b-hf \
  --data_path ${PATH_TO_TRAIN_DATA_FILE} \
  --bf16  True \
  --output_dir ${PATH_TO_CRITIC_MODEL} \
  --num_train_epochs 3  \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 300 \
  --save_total_limit 1 \
  --learning_rate 2e-5 \
  --weight_decay 0. \
  --warmup_ratio 0.01 \
  --lr_scheduler_type "cosine" \
  --logging_steps 10 \
  --fsdp "full_shard auto_wrap"