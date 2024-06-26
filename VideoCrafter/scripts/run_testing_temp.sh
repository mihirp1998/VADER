name="base_512_v2"

ckpt='checkpoints/base_512_v2/model.ckpt'
config='configs/inference_t2v_512_v2.0.yaml'

prompt_file="prompts/test_prompts.txt"
res_dir="results"

accelerate launch --num_processes 1 --main_process_port 29519 scripts/evaluation/train_t2v_lora.py \
--seed 300 \
--mode 'base' \
--ckpt_path $ckpt \
--config $config \
--n_samples 1 \
--bs 1 --height 320 --width 320 \
--unconditional_guidance_scale 12.0 \
--ddim_steps 5 \
--ddim_eta 1.0 \
--prompt_file $prompt_file \
--frames 4 \
--fps 24 \
--prompt_fn 'chatgpt_custom_actpred' \
--gradient_accumulation_steps 4 \
--num_train_epochs 200 \
--train_batch_size 1 \
--val_batch_size 1 \
--num_val_runs 1 \
--reward_fn 'hps' \
--decode_frame '-1' \
--hps_version 'v2.0' \
--lr 0.0002 \
--validation_steps 10 \
--lora_rank 16 \
--is_sample_preview True
