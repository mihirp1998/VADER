accelerate launch --num_processes 1 --main_process_port 29517 scripts/train_t2v_lora.py configs/opensora-v1-2/vader/vader.py \
--num-frames 4 \
--resolution 144p \
--aspect-ratio 3:4 \
--prompt-path "../assets/chatgpt_custom_human_fashion.txt"