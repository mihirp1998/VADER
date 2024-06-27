accelerate launch --num_processes 4 train_t2v_lora.py \
gradient_accumulation_steps=4 \
prompt_fn=hps_custom \
reward_fn=aesthetic