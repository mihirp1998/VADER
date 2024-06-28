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
### 🎥 ModelScope
#### 🔧 Training
The current code can work on a single GPU with VRAM > 14GBs. The code can be further optimized to work with even lesser VRAM with deepspeed and CPU offloading. For our experiments, we used 4 A100s- 40GB RAM to run our code.
```bash
cd ModelScope
sh run_text2video_train.sh
```
- `ModelScope/train_t2v_lora.py` is a script for fine-tuning ModelScope using VADER via LoRA.
    - `--num_processes` is the number of processes for Accelerator. It is recommended to set it to the number of GPUs.
    - `gradient_accumulation_steps` can be increased while reducing the `--num_processes` to alleviate bottleneck caused by the number of GPUs.
    - `prompt_fn` is the prompt function, which can be the name of any functions in Core/prompts.py, like `'chatgpt_custom_instruments'`, `'chatgpt_custom_animal_technology'`, `'chatgpt_custom_ice'`, `'nouns_activities'`, etc. Note: If you set `--prompt_fn 'nouns_activities'`, you have to provide`--nouns_file` and `--nouns_file`, which will randomly select a noun and an activity from the files and form them into a single sentence as a prompt.
    - `reward_fn` is the reward function, which can be selected from `'aesthetic'`, `'hps'`, and `'actpred'`.
- `ModelScope/config_t2v/config.yaml` is the configuration file for training. You can modify the configuration file to change the training settings following the comments in that file.



#### 📐 Evaluation & Checkpoints
Please find the checkpoints for Aesthetic reward function [here](https://drive.google.com/file/d/1r7291awe3z37drfKyxLyqcNq6dHl6Egf/view?usp=sharing) and Hps-v2 reward function [here](https://drive.google.com/file/d/1nvSxwxf-OnDrKq4ob-j5islfUSif8lQb/view?usp=sharing)

Evaluates the model checkpoint, as per the `resume_from` variable in the config file.  Evaluation includes calculating the reward and storing/uploading the images to local/wandb.

##### normal evaluation.

```bash
accelerate launch --num_processes 1 train_t2v_lora.py \
only_val=True \
num_only_val_itrs=1000 \
val_batch_size=4 \
lora_path=media_vis/good-voice-252/checkpoint-592/lora
```


## Acknowledgement

Our codebase is directly built on top of [VideoCrafter](https://github.com/AILab-CVC/VideoCrafter), [Open-Sora](https://github.com/hpcaitech/Open-Sora), and [Animate Anything](https://github.com/alibaba/animate-anything/). We would like to thank the authors for open-sourcing their code.

## Citation

If you find this work useful in your research, please cite:

```bibtex

```
