name="base_512_v2"

ckpt='checkpoints/base_512_v2/model.ckpt'
config='configs/inference_t2v_512_v2.0.yaml'

prompt_file="prompts/test_prompts.txt"
res_dir="results"

nohup accelerate launch --main_process_port 29523 scripts/evaluation/train_t2v_lora.py \
--seed 200 \
--mode 'base' \
--ckpt_path $ckpt \
--config $config \
--n_samples 1 \
--bs 1 --height 384 --width 512 \
--unconditional_guidance_scale 12.0 \
--ddim_steps 25 \
--ddim_eta 1.0 \
--prompt_file $prompt_file \
--fps 24 \
--frames 24 \
--prompt_fn 'chatgpt_custom_animal_cute_select' \
--gradient_accumulation_steps 4 \
--num_train_epochs 200 \
--train_batch_size 1 \
--val_batch_size 4 \
--num_val_runs 8 \
--reward_fn 'aesthetic_hps' \
--hps_version 'v2.1' \
--lr 0.0001 \
--validation_steps 10 \
--lora_rank 16 \
--inference_only True \
--lora_ckpt_path project_dir/summer-wildflower-101/peft_model_532.pt \
--project_dir ./project_dir/inference/summer-wildflower-101/seed200/peft_model_532_fairytale \
--is_sample_preview True >>summer-wildflower-101_owenL1_inference.log &
