<div align="center">

<!-- TITLE -->
# **Video Diffusion Alignment via Reward Gradient**
![VADER](assets/vader_method.png)

[![arXiv](https://img.shields.io/badge/cs.LG-)]()
[![Website](https://img.shields.io/badge/🌎-Website-blue.svg)](http://vader-vid.github.io)
</div>

This is the official implementation of our paper [Video Diffusion Alignment via Reward Gradient](https://vader-vid.github.io/) by 

Mihir Prabhudesai*, Russell Mendonca*, Zheyang Qin*, Katerina Fragkiadaki, Deepak Pathak .


<!-- DESCRIPTION -->
## Abstract
We have made significant progress towards building foundational video diffusion models. As these models are trained using large-scale unsupervised data, it has become crucial to adapt these models to specific downstream tasks, such as video-text alignment or ethical video generation. Adapting these models via supervised fine-tuning requires collecting target datasets of videos, which is challenging and tedious. In this work, we instead utilize pre-trained reward models that are learned via preferences on top of powerful discriminative models. These models contain dense gradient information with respect to generated RGB pixels, which is critical to be able to learn efficiently in complex search spaces, such as videos. We show that our approach can enable alignment of video diffusion for aesthetic generations, similarity between text context and video, as well long horizon video generations that are 3X longer than the training sequence length. We show our approach can learn much more efficiently in terms of reward queries and compute than previous gradient-free approaches for video generation.


## Features
- [x] Adaptation of ModelScope Text2Video Model
- [ ] Adaptation of Stable Video Diffusion Image2Video Model

## Demo
|         |          |       |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| <img src="assets/videos/8.gif" width=""> | <img src="assets/videos/5.gif" width=""> | <img src="assets/videos/7.gif" width=""> |
| <img src="assets/videos/10.gif" width=""> | <img src="assets/videos/3.gif" width=""> | <img src="assets/videos/4.gif" width=""> |
| <img src="assets/videos/9.gif" width=""> | <img src="assets/videos/1.gif" width=""> | <img src="assets/videos/11.gif" width=""> |


## Code

### Installation 
Create Conda environments for VideoCrafter, Open-Sora, and ModelScope using the following commands:
#### 🎬 Videocrafter
```bash
cd VideoCrafter
conda create -n vader_videocrafter python=3.8.5
conda activate vader_videocrafter
pip install -r requirements.txt
git clone https://github.com/tgxs002/HPSv2.git
cd HPSv2/
pip install .
cd ..
rm -r HPSv2
```

#### 🎬 Open-Sora
```bash
cd Open-Sora
conda create -n vader_opensora python=3.9
conda activate vader_opensora
pip install -r requirements/requirements-cu121.txt
pip install -v -e .
git clone https://github.com/tgxs002/HPSv2.git
cd HPSv2/
pip install .
cd ..
rm -r HPSv2
```

#### 🎬 ModelScope
```bash
cd ModelScope
conda create -n vader_modelscope python=3.10
conda activate vader_modelscope
pip install -r requirements.txt
git clone https://github.com/tgxs002/HPSv2.git
cd HPSv2/
pip install .
cd ..
rm -r HPSv2
```

### Training Code
#### 🎬 Videocrafter
For our experiments, we used 4 A100s- 40GB RAM to run our code.

```bash
cd VideoCrafter
sh script/run_text2video_train.sh
```
- `VideoCrafter/scripts/main/train_t2v_lora.py` is a script for fine-tuning the VideoCrafter using VADER via LoRA.
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
    - `--inference_only` is set to `True` if you only want to generate videos without training. 
    - `--backprop_mode` is to control when we gather the gradient during backpropagation in LoRA. It could be `'last'` (gather the gradient only at the last DDIM step), `'rand'` (gather the gradient at random step of DDIM), and `'specific'` (do not gather the gradient at the 15th DDIM step).


#### 🎬 ModelScope
The current code can work on a single GPU with VRAM > 14GBs. The code can be further optimized to work with even lesser VRAM with deepspeed and CPU offloading.
For our experiments, we used 4 A100s- 40GB RAM to run our code.

#### Aesthetic Reward model.
Currently we early stop the code to prevent overfitting, however feel free to play with the `num_epochs` variable as per your needs.

```bash
accelerate launch --num_processes 4 train_t2v_lora.py gradient_accumulation_steps=4 prompt_fn=hps_custom reward_fn=aesthetic
```

If you are number of GPUs bottlenecked, increase the  `gradient_accumulation_steps`, while reducing the `num_processes` by an equivalent factor:


```bash
accelerate launch --num_processes 1 train_t2v_lora.py gradient_accumulation_steps=16 prompt_fn=hps_custom reward_fn=aesthetic
```

#### HPSv2 Reward model.

```bash
accelerate launch --num_processes 4 train_t2v_lora.py gradient_accumulation_steps=4 prompt_fn=hps_custom reward_fn=hps
```

If you are number of GPUs bottlenecked, increase the  `gradient_accumulation_steps`, while reducing the `num_process` by an equivalent factor:


```bash
accelerate launch --num_processes 1 train_t2v_lora.py gradient_accumulation_steps=16 prompt_fn=hps_custom reward_fn=hps
```

#### Video Action Classification Reward model.


```bash
accelerate launch --num_processes 4 train_t2v_lora.py gradient_accumulation_steps=4 prompt_fn=hps_custom reward_fn=hps
```



### Evaluation & Checkpoints
Please find the checkpoints for Aesthetic reward function [here](https://drive.google.com/file/d/1r7291awe3z37drfKyxLyqcNq6dHl6Egf/view?usp=sharing) and Hps-v2 reward function [here](https://drive.google.com/file/d/1nvSxwxf-OnDrKq4ob-j5islfUSif8lQb/view?usp=sharing)

Evaluates the model checkpoint, as per the `resume_from` variable in the config file.  Evaluation includes calculating the reward and storing/uploading the images to local/wandb.

#### normal evaluation.

```bash
accelerate launch --num_processes 1 train_t2v_lora.py only_val=True num_only_val_itrs=1000 val_batch_size=4 lora_path=media_vis/good-voice-252/checkpoint-592/lora 
```


### Acknowledgement

Our codebase is directly built on top of [Animate Anything](https://github.com/alibaba/animate-anything/). We would like to thank the authors for open-sourcing their code.

## Citation

If you find this work useful in your research, please cite:

```bibtex

```
