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

## Demo
|         |          |       |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| <img src="../assets/videos/8.gif" width=""> | <img src="../assets/videos/5.gif" width=""> | <img src="../assets/videos/7.gif" width=""> |
| <img src="../assets/videos/10.gif" width=""> | <img src="../assets/videos/3.gif" width=""> | <img src="../assets/videos/4.gif" width=""> |
| <img src="../assets/videos/9.gif" width=""> | <img src="../assets/videos/1.gif" width=""> | <img src="../assets/videos/11.gif" width=""> |

## Usage
### 📀 VideoCrafter
- Please, download pretrained Text-to-Video [VideoCrafter2](https://huggingface.co/VideoCrafter/VideoCrafter2/blob/main/model.ckpt) model via Hugging Face, and put the `model.ckpt` in `VADER/VideoCrafter/checkpoints/base_512_v2/model.ckpt`.

#### 🔧 Training
For our experiments, we used 4 A6000s- 48GB RAM to run our code.

```bash
cd VideoCrafter
sh script/run_text2video_train.sh
```
- `VideoCrafter/scripts/main/train_t2v_lora.py` is a script for fine-tuning the VideoCrafter2 using VADER via LoRA.
    - `--height` and `--width` are the height and width of the video frames respectively.
    - `--n_samples` is the number of samples per prompt. It must be `1` during training process.
    - `--frames` is the number of frames to inference.
    - `--prompt_fn` is the prompt function, which can be the name of any functions in Core/prompts.py, like `'chatgpt_custom_instruments'`, `'chatgpt_custom_animal_technology'`, `'chatgpt_custom_ice'`, `'nouns_activities'`, etc. Note: If you set `--prompt_fn 'nouns_activities'`, you have to provide`--nouns_file` and `--nouns_file`, which will randomly select a noun and an activity from the files and form them into a single sentence as a prompt.
    - `--gradient_accumulation_steps` can be increased while reducing the `--num_processes` to alleviate bottleneck caused by the number of GPUs.
    - `--num_train_epochs` is the number of training epochs.
    - `--train_batch_size` is the batch size for training.
    - `--val_batch_size` is the batch size for validation.
    - `--num_val_runs` is the number of validation runs. The total number of validation videos generated will be `num_val_runs * val_batch_size * num_processes`.
    - `--reward_fn` is the reward function, which can be selected from `'aesthetic'`, `'hps'`, `'aesthetic_hps'`, `'pick_score'`, `'objectDetection'`, and `'actpred'`.
    - `--hps_version` is the version of HPS, which can be `'v2.1'` or `'v2.0'`.
    - `--decode_frame` is to control which frame of video to decode in the training process. It could be `'-1'` (a random frame), `'fml'` (first, middle, and last frames), `'all'` (all frames), and `'alt'` (alternate frames). It could also be any number in string type (not int type) like `'3'`, `'10'`, etc. Multiple frames mode can only be used when Actpred reward function is enabled.
    - `--lr` is to control the learning rate.
    - `--validation_steps` is to control the frequency of validation, e.g., `1` means that we will generate validation videos every `1*num_processes` steps.
    - `--checkpointing_steps` is to control the frequency of saving checkpoints, e.g., `1` means that we will save checkpoints of LoRA model every `1*num_processes` steps.
    - `--lora_rank` is the rank of LoRA. The larger the value, the more memory is used.
    - `--lora_ckpt_path` is the path of the pretrained LoRA model. If it is not provided, the model will be initialized from scratch.
    - `--is_sample_preview` is set to `True` if you want to generate and save preview videos.
    - `--detector_model` is used to switched the detection model among `'yolos-base'`, `'yolos-tiny'`, `'grounding-dino-base'`, and `'grounding-dino-tiny'`.
    - `--target_object` is used only when the reward function is `'objectDetection'`. It is the target object for object detection. The default value is `'book'`, which is used for YOLO models. Please do not add "." at the end of the object name for any YOLO models. However, if you are using grounding-dino model, you should instead set the object name to `'book.'` for example.
    - `--mixed_precision` is set to `'fp16'` as default. You can also set it to `'no'`, `'bf16'` or `'fp8'`.
    - `--project_dir` is the directory to save the checkpoints and sampled videos.
    - `--use_wandb` is set to `True` if you want to use wandb to log the training process.
    - `--wandb_entity` is the entity of wandb, whose default value is `''`.
    - `--use_AdamW8bit` is set to `True` if you want to use AdamW8bit optimizer.
    - `--inference_only` is set to `False` if you only want to do training.
    - `--backprop_mode` is to control when we gather the gradient during backpropagation in LoRA. It could be `'last'` (gather the gradient only at the last DDIM step), `'rand'` (gather the gradient at a random step of DDIM), and `'specific'` (gather the gradient at the 15th DDIM step).


#### 📺 Inference
```bash
cd VideoCrafter
sh script/run_text2video_inference.sh
```
- `VideoCrafter/scripts/main/train_t2v_lora.py` is also a script for inference of the VideoCrafter2 using VADER via LoRA.
    - Most of the arguments are the same as the training process. The main difference is that `--inference_only` should be set to `True`.
    - `--lora_ckpt_path` is required to set to the path of the pretrained LoRA model. Otherwise, the original VideoCrafter model will be used for inference.


## 💡 Tutorial
This section is to provide a tutorial on how to implement the VADER method on VideoCrafter by yourself. We will provide a step-by-step guide to help you understand the implementation details. Thus, you can easily adapt the VADER method to later versions of VideCrafter. This tutorial is based on the VideoCrafter2.

### Step 1: Install the dependencies
First, you need to install the dependencies according to the [VideoCrafter](https://github.com/AILab-CVC/VideoCrafter) repository. You can also follow the instructions in the repository to install the dependencies.
```bash
conda create -n videocrafter python=3.8.5
conda activate videocrafter
pip install -r requirements.txt
```

You have to download pretrained Text-to-Video [VideoCrafter2](https://huggingface.co/VideoCrafter/VideoCrafter2/blob/main/model.ckpt) model via Hugging Face, and put the `model.ckpt` in `VideoCrafter/checkpoints/base_512_v2/model.ckpt`.

There are a list of extra dependencies that you need to install for VADER. You can install them by running the following command.
```bash
# Install the HPS
git clone https://github.com/tgxs002/HPSv2.git
cd HPSv2/
pip install .
cd ..
rm -r HPSv2

# Install the dependencies
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

### Step 2: Transfer VADER scripts
You can copy our `VADER/VideoCrafter/scripts/main/train_t2v_lora.py` to the `VideoCrafter/scripts/evaluation/` directory of VideoCrafter. It is better to copy our `run_text2video_train.sh` and `run_text2video_inference.sh` to the directionary `VideoCrafter/scripts/` as well. Then, you need to copy All the files in `VADER/Core/` and `VADER/assets/` to the parent directory of VideoCrafter, which means `Core/`, `assets` and `VideoCrafter/` should be in the same directory. Now, you may have a directory structure like:
```bash
.
├── Core
│   ├── ...
├── VideoCrafter
│   ├── scripts
│   │   ├── evaluation
│   │   │   ├── train_t2v_lora.py
│   │   ├── run_text2video_train.sh
│   │   ├── run_text2video_inference.sh
│   ├── checkpoints
│   │   ├── base_512_v2
│   │   │   ├── model.ckpt
├── assets
│   ├── ...
```

### Step 3: Modify the VideoCrafter code
You need to modify the VideoCrafter code to adapt the VADER method. You can follow the instructions below to modify the code.

- Modify the `batch_ddim_sampling()` function in `VideoCrafter/scripts/evaluation/funcs.py` as our implementation in `VADER/VideoCrafter/scripts/main/funcs.py`.
- Modify the `DDIMSampler.__init__()`, `DDIMSampler.sample()` and `DDIMSampler.ddim_sampling` functions in  `VideoCrafter\lvdm\models\samplers\ddim.py` as our implementation in `VADER/VideoCrafter\lvdm\models\samplers\ddim.py`.
- Comment out the `@torch.no_grad()` before `DDIMSampler.sample()`, `DDIMSampler.ddim_sampling`, and `DDIMSampler.p_sample_ddim()` in `VideoCrafter\lvdm\models\samplers\ddim.py`. Also, comment out the `@torch.no_grad()` before `LatentDiffusion.decode_first_stage_2DAE()` in `VideoCrafter\lvdm\models\ddpm3d.py`.
- Because we have commented out the `@torch.no_grad()`, you can add `with torch.no_grad():` at some places in `VideoCrater/scripts/evaluation/inference.py` to avoid the gradient calculation.

### Step 4: Ready to Train
Now you have all the files in the right place and modified the VideoCrafter source code. You can run the training script by running the following command.
```bash
cd VideoCrafter

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