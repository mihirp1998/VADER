ckpt='checkpoints/base_512_v2/model.ckpt'
config='configs/inference_t2v_512_v2.0.yaml'
PORT=$((20000 + RANDOM % 10000))

accelerate launch --multi_gpu --main_process_port $PORT scripts/main/train_t2v_lora.py \
--seed 300 \
--mode 'base' \
--ckpt_path $ckpt \
--config $config \
--height 320 --width 512 \
--unconditional_guidance_scale 12.0 \
--ddim_steps 25 \
--ddim_eta 1.0 \
--frames 12 \
--prompt_fn 'chatgpt_custom_instruments' \
--gradient_accumulation_steps 8 \
--num_train_epochs 200 \
--train_batch_size 1 \
--val_batch_size 1 \
--num_val_runs 1 \
--reward_fn 'aesthetic_hps' \
--decode_frame '-1' \
--hps_version 'v2.1' \
--lr 0.0002 \
--validation_steps 10 \
--lora_rank 16 \
--max_train_steps 450 \
--is_sample_preview True

