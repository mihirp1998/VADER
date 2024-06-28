accelerate launch train_t2v_lora.py \
gradient_accumulation_steps=16 \
prompt_fn=hps_custom \
reward_fn=aesthetic