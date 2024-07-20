ckpt='checkpoints/base_512_v2/model.ckpt'
config='configs/inference_t2v_512_v2.0.yaml'
PORT=$((20000 + RANDOM % 10000))

accelerate launch --multi_gpu --main_process_port $PORT scripts/main/train_t2v_lora.py \
--seed 200 \
--mode 'base' \
--ckpt_path $ckpt \
--config $config \
--height 320 --width 512 \
--unconditional_guidance_scale 12.0 \
--ddim_steps 25 \
--ddim_eta 1.0 \
--frames 24 \
--prompt_fn 'chatgpt_inference' \
--val_batch_size 1 \
--num_val_runs 1 \
--lora_rank 16 \
--inference_only True \
--project_dir ./project_dir/inference \
--lora_ckpt_path huggingface-pickscore \
--is_sample_preview True
