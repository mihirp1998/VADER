nohup accelerate launch --main_process_port 29517 scripts/train_t2v_lora.py configs/opensora-v1-2/vader/vader.py \
--num-frames 2s \
--resolution 360p \
--aspect-ratio 3:4 \
--prompt-path "assets/chatgpt_custom_human_fashion.txt" \
>> chatgpt_custom_animal_fashion_pick_score_owenL3.log &