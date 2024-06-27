name="base_512_v2"

ckpt='checkpoints/base_512_v2/model.ckpt'
config='configs/inference_t2v_512_v2.0.yaml'

prompt_file="prompts/test_prompts.txt"
res_dir="results"

accelerate launch --num_processes 4 --main_process_port 29523 scripts/main/train_t2v_lora.py \
--seed 200 \
--mode 'base' \
--ckpt_path $ckpt \
--config $config \
--height 320 --width 512 \
--unconditional_guidance_scale 12.0 \
--ddim_steps 25 \
--ddim_eta 1.0 \
--frames 24 \
--prompt_fn 'chatgpt_custom_cute' \
--val_batch_size 2 \
--num_val_runs 8 \
--lora_rank 16 \
--inference_only True \
--project_dir ./project_dir/inference \
--is_sample_preview True
