PORT=$((20000 + RANDOM % 10000))
accelerate launch --multi_gpu --main_process_port $PORT scripts/train_t2v_lora.py configs/opensora-v1-2/vader/vader_train.py \
--num-frames 2s \
--resolution 240p \
--aspect-ratio 3:4 \
--prompt-path "../assets/chatgpt_custom_human_fashion.txt"