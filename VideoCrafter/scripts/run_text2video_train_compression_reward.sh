name="base_512_v2"

ckpt='checkpoints/base_512_v2/model.ckpt'
config='configs/inference_t2v_512_v2.0.yaml'

prompt_file="prompts/test_prompts.txt"
res_dir="results"

nohup accelerate launch --main_process_port 29520 scripts/evaluation/train_compression_reward.py \
--seed 400 \
--mode 'base' \
--ckpt_path $ckpt \
--config $config \
--n_samples 1 \
--bs 1 --height 320 --width 512 \
--unconditional_guidance_scale 12.0 \
--ddim_steps 25 \
--ddim_eta 1.0 \
--prompt_file $prompt_file \
--fps 24 \
--frames 4 \
--prompt_fn 'chatgpt_custom_compression_animals' \
--gradient_accumulation_steps 8 \
--num_train_epochs 200 \
--train_batch_size 2 \
--checkpointing_steps 1 \
--lr 0.001 >> chatgpt_custom_compression_animals_owenL1.log &

