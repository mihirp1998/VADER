<div align="center">

<!-- TITLE -->
# **Video Diffusion Alignment via Reward Gradient**
![VADER](../assets/vader_method.png)

[![arXiv](https://img.shields.io/badge/cs.LG-)]()
[![Website](https://img.shields.io/badge/🌎-Website-blue.svg)](http://vader-vid.github.io)
</div>

This is the official implementation of our paper [Video Diffusion Alignment via Reward Gradient](https://vader-vid.github.io/) by 

Mihir Prabhudesai*, Russell Mendonca*, Zheyang Qin*, Katerina Fragkiadaki, Deepak Pathak .


<!-- DESCRIPTION -->
## Abstract
We have made significant progress towards building foundational video diffusion models. As these models are trained using large-scale unsupervised data, it has become crucial to adapt these models to specific downstream tasks, such as video-text alignment or ethical video generation. Adapting these models via supervised fine-tuning requires collecting target datasets of videos, which is challenging and tedious. In this work, we instead utilize pre-trained reward models that are learned via preferences on top of powerful discriminative models. These models contain dense gradient information with respect to generated RGB pixels, which is critical to be able to learn efficiently in complex search spaces, such as videos. We show that our approach can enable alignment of video diffusion for aesthetic generations, similarity between text context and video, as well long horizon video generations that are 3X longer than the training sequence length. We show our approach can learn much more efficiently in terms of reward queries and compute than previous gradient-free approaches for video generation.


## Usage
### 🎬 Open-Sora
#### 🔧 Training
For our experiments, we used 4 A6000s- 48GB RAM to run our code.

```bash
cd Open-Sora
sh script/run_text2video_train.sh
```
- `Open-Sora/scripts/train_t2v_lora.py` is a script for fine-tuning the Open-Sora 1.2 using VADER via LoRA.
    - `--num-frames`, `'--resolution'`, `'fps'` and `'aspect-ratio'` are inherited from the original Open-Sora model. In short, you can set `'--num-frames'` as `'2s'`, `'4s'`, `'8s'`, and `'16s'`. Available resolutions are `'240p'`, `'360p'`, `'480p'`, and `'720p'`. The default value of `'fps'` is `24` and `'aspect-ratio'` is `3:4`. Please refer to the original [Open-Sora](https://github.com/hpcaitech/Open-Sora) repository for more details. One thing to keep in mind, for instance, is that if you set `--num-frames` to `2s` and `--resolution` to `'240p'`, it is better to use `bf16` mixed precision instead of `fp16`. Otherwise, the model may generate noise videos.
    - `--prompt-path` is the path of the prompt file. Unlike VideoCrafter, we do not provide prompt function for Open-Sora. Instead, you can provide a prompt file, which contains a list of prompts.
    - `--num-processes` is the number of processes for Accelerator. It is recommended to set it to the number of GPUs.
- `Open-Sora/configs/opensora-v1-2/vader/vader_train.py` is the configuration file for training. You can modify the configuration file to change the training settings.
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
    - `backprop_mode` is to control when we gather the gradient during backpropagation in LoRA. It could be `'last'` (gather the gradient only at the last DDIM step), `'rand'` (gather the gradient at random step of DDIM), and `'specific'` (do not gather the gradient at the 15th DDIM step).
    - `decode_frame` is to control which frame of video to decode in the training process. It could be `'-1'` (a random frame), `'fml'` (first, middle, and last frames), `'all'` (all frames), and `'alt'` (alternate frames). It could also be any number in string type (not int type) like `'3'`, `'10'`, etc. Multiple frames mode can only be used when Actpred reward function is enabled.
    - `is_sample_preview` is set to `True` if you want to generate and save preview videos.
    - `grad_checkpoint` is set to `True` if you want to enable gradient checkpointing to save memory.

#### 📺 Inference
```bash
cd Open-Sora
sh script/run_text2video_inference.sh
```
- `Open-Sora/scripts/train_t2v_lora.py` is also a script for do inference via the Open-Sora 1.2 using VADER.
    - Most of the arguments are the same as the training process. The main difference is that `is_vader_training` should be set to `False`. The `--lora_ckpt_path` should be set to the path of the pretrained LoRA model. Otherwise, the original Open-Sora model will be used for inference.

## 💡 Tutorial
This section is to provide a tutorial on how to implement the VADER method on Open-Sora by yourself. We will provide a step-by-step guide to help you understand the implementation details. Thus, you can easily adapt the VADER method to later versions of Open-Sora or other video generation models. This tutorial is based on the Open-Sora v1.2.0 version.

### Step 1: Install the dependencies
First, you need to install the dependencies according to the [Open-Sora repository](https://github.com/hpcaitech/Open-Sora). You can follow the instructions in the repository to install the dependencies.
```bash
conda create -n opensora python=3.9
conda activate opensora

# download the repo
git clone https://github.com/hpcaitech/Open-Sora
cd Open-Sora

# install torch, torchvision and xformers
pip install -r requirements/requirements-cu121.txt

# install the Open-Sora package
pip install -v -e .
```
There are a list of extra dependencies that you need to install for VADER. You can install them by running the following command.
```bash
pip install albumentations \
hpsv2 \
peft \
bitsandbytes \
accelerate \
inflect \
wandb \
ipdb \
pytorch_lightning
```

### Step 2: Write VADER training script
You can copy our `VADER/Open-Sora/scripts/train_t2v_lora.py` to the `scripts` directory of Open-Sora, namely `Open-Sora/scripts/`. It is better to copy `run_text2video_train.sh` and `run_text2video_inference.sh` to that directionary as well. Then, you need to copy All the files in `VADER/Core/` and `VADER/assets/` to the parent directory of Open-Sora, which means `Core/`, `assets` and `Open-Sora/` should be in the same directory. You have to also copy the `VADER/Open-Sora/configs/opensora-v1-2/vader/vader_train.py` and `VADER/Open-Sora/configs/opensora-v1-2/vader/vader_inference.py` to one directory of Open-Sora, namely `Open-Sora/configs/opensora-v1-2/vader/`. Now, you may have a directory structure like:
```bash
.
├── Core
│   ├── ...
├── Open-Sora
│   ├── scripts
│   │   ├── train_t2v_lora.py
│   ├── configs
│   │   ├── opensora-v1-2
│   │   │   ├── vader
│   │   │   │   ├── vader_train.py
│   │   │   │   ├── vader_inference.py
├── assets
│   ├── ...
```

### Step 3: Modify the Open-Sora Source Code
We will let you know which files you need to modify in the Open-Sora source code.

- Modify the function `sample()` in `Open-Sora/opensora/schedulers/rf/__init__.py` as in our implementation in `VADER/Core/schedulers/rf/__init__.py`.

Now you have all the files in the right place and modified the Open-Sora source code. You can run the training script by running the following command.
```bash
cd Open-Sora

# training
sh script/run_text2video_train.sh

# or inference
sh script/run_text2video_inference.sh
```


## Acknowledgement

Our codebase is directly built on top of [VideoCrafter](https://github.com/AILab-CVC/VideoCrafter), [Open-Sora](https://github.com/hpcaitech/Open-Sora), and [Animate Anything](https://github.com/alibaba/animate-anything/). We would like to thank the authors for open-sourcing their code.

## Citation

If you find this work useful in your research, please cite:

```bibtex

```
