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
Create a conda environment for VideoCrafter, Open-Sora, and ModelScope respectively with the following commands:
#### 🎬 Videocrafter
```bash
cd VideoCrafter
conda create -n videocrafter python=3.8.5
conda activate videocrafter
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
conda create -n opensora python=3.9
conda activate opensora
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
conda create -n vader python=3.10
conda activate vader
pip install -r requirements.txt
git clone https://github.com/tgxs002/HPSv2.git
cd HPSv2/
pip install .
cd ..
rm -r HPSv2
```

### Training Code
The current code can work on a single GPU with VRAM > 14GBs. The code can be further optimized to work with even lesser VRAM with deepspeed and CPU offloading.
For our experiments, we used 4 A100s- 40GB RAM to run our code.

#### Aesthetic Reward model.
Currently we early stop the code to prevent overfitting, however feel free to play with the `num_epochs` variable as per your needs.

```bash
accelerate launch --num_processes 4 train_t2v_lora.py gradient_accumulation_steps=4 prompt_fn=hps_custom reward_fn=aesthetic
```

If you are number of GPUs bottlenecked, increase the  `gradient_accumulation_steps`, while reducing the `num_process` by an equivalent factor:


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
