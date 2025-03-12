accelerate launch --num_processes 1  --main_process_port 29557   train_t2v_lora_ddpo.py \
gradient_accumulation_steps=6 learning_rate=1e-4 \
ddpo_params.sample_batch_size=4 ddpo_params.sample_num_batches_per_epoch=4 ddpo_params.train_batch_size=4 \
reward_fn=aesthetic \
scheduler_type=ddim use_wandb=True \
validation_steps=1

