accelerate launch --num_processes 1 --main_process_port 29574 train_t2v_lora_dpo.py \
gradient_accumulation_steps=8 learning_rate=2e-4 \
dpo_params.sample_batch_size=4 dpo_params.sample_num_batches_per_epoch=4 \
dpo_params.num_train_inner_epochs=32 \
dpo_params.beta=5000 \
reward_fn=aesthetic \
scheduler_type=ddim use_wandb=True \
validation_steps=1