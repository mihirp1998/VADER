# 🎬 VADER-Open-Sora Documentation
This document provides a detailed guide on how to use Open-Sora based VADER for fine-tuning and inference.

- `VADER/VADER-Open-Sora/configs/opensora-v1-2/vader/vader_inference.py` is the configuration file for inference. You can modify the configuration file to change the inference settings.
    - `is_vader_training` is set to `False` if you want to use VADER for inference.
    - The rest of the parameters are the same as the training configuration file.


- `VADER/VADER-Open-Sora/configs/opensora-v1-2/vader/vader_train.py` is the configuration file for training. You can modify the configuration file to change the training settings.
    - `is_vader_training` is set to `True` if you want to use VADER for training.
    - `train_batch_size` is the batch size for training.
    - `val_batch_size` is the batch size for validation.
    - `num_val_runs` is the number of validation runs. The total number of validation videos generated will be `num_val_runs * val_batch_size * num_processes`.
    - `seed` is the random seed.
    - `lora_ckpt_path` is the path of the pretrained LoRA model. If it is not provided, the model will be initialized from scratch.
    - `project_dir` is the directory to save the checkpoints and sampled videos.
    - `lr` is to control the learning rate.
    - `reward_fn` is the reward function, which can be selected from `'aesthetic'`, `'hps'`, `'aesthetic_hps'`, and `'pick_score'`.`
    - `gradient_accumulation_steps` can be increased while reducing the `--num_processes` to alleviate bottleneck caused by the number of GPUs.
    - `--lora_rank` is the rank of LoRA. The larger the value, the more memory is used.
    - `dtype` is the data type of the model. It could be `'fp16'`, `'bf16'`, `'fp8'`, and `'fp32'`. For instance, it is recommended to use `'bf16'` for `'240p'` and `'360p'`.
    - `mixed_precision` is set to `'bf16'` as default. You can also set it to `'no'`, `'fp16'` or `'fp8'`.
    - `'logger_type'` is `'wandb'` as default. You can also set it to `'tensorboard'`.
    - `--use_wandb` is set to `True` if you want to use wandb to log the training process.
    - `wandb_entity` is the entity of wandb, whose default value is `''`.
    - `--validation_steps` is to control the frequency of validation, e.g., `1` means that we will generate validation videos every `1*num_processes` steps.
    - `--checkpointing_steps` is to control the frequency of saving checkpoints, e.g., `1` means that we will save checkpoints of LoRA model every `1*num_processes` steps.
    - `debug` is set to `False` as default.
    - `use_AdamW8bit` is set to `True` if you want to use AdamW8bit optimizer.
    - `hps_version` is the version of HPS, which can be `'v2.1'` or `'v2.0'`.
    - `num_train_epochs` is the number of training epochs.
    - `max_train_steps` is the maximum number of training steps.
    - `backprop_mode` is to control when we gather the gradient during backpropagation in LoRA. It could be `'last'` (gather the gradient only at the last DDIM step), `'rand'` (gather the gradient at a random step of DDIM), and `'specific'` (gather the gradient at the 15th DDIM step).
    - `decode_frame` is to control which frame of video to decode in the training process. It could be `'-1'` (a random frame), `'fml'` (first, middle, and last frames), `'all'` (all frames), and `'alt'` (alternate frames). It could also be any number in string type (not int type) like `'3'`, `'10'`, etc. Multiple frames mode can only be used when Actpred reward function is enabled.
    - `is_sample_preview` is set to `True` if you want to generate and save preview videos.
    - `grad_checkpoint` is set to `True` if you want to enable gradient checkpointing to save memory.
