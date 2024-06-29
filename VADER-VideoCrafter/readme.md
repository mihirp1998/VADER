<div align="center">

<!-- TITLE -->
# 🌟**VADER-VideoCrafter**
</div>



We **highly recommend** proceeding with the VADER-VideoCrafter model first, which performs better than the other two.

## ⚙️ Installation
Assuming you are in the `VADER/` directory, you are able to create a Conda environments for VADER-VideoCrafter using the following commands:
```bash
cd VADER-VideoCrafter
conda create -n vader_videocrafter python=3.10
conda activate vader_videocrafter
conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=12.1 -c pytorch -c nvidia
conda install xformers -c xformers
pip install -r requirements.txt
git clone https://github.com/tgxs002/HPSv2.git
cd HPSv2/
pip install -e .
cd ..
```


- We are using the pretrained Text-to-Video [VideoCrafter2](https://huggingface.co/VideoCrafter/VideoCrafter2/blob/main/model.ckpt) model via Hugging Face. If you unfortunately find the model is not automatically downloaded when you running inference or training script, you can manually download it and put the `model.ckpt` in `VADER/VADER-VideoCrafter/checkpoints/base_512_v2/model.ckpt`.


## 📺 Inference
Please run `accelerate config` as the first step to configure accelerator settings. If you are not familiar with the accelerator configuration, you can refer to VADER-VideoCrafter [documentation](../documentation/VADER-VideoCrafter.md).

Assuming you are in the `VADER/` directory, you are able to do inference using the following commands:
```bash
cd VADER-VideoCrafter
sh scripts/run_text2video_inference.sh
```
- We have tested on PyTorch 2.3.0 and CUDA 12.1. The inferece script works on a single GPU with 16GBs VRAM, when we set `val_batch_size=1` and use `fp16` mixed precision. It should also work with recent PyTorch and CUDA versions.
- `VADER/VADER-VideoCrafter/scripts/main/train_t2v_lora.py` is a script for inference of the VideoCrafter2 using VADER via LoRA.
    - Most of the arguments are the same as the training process. The main difference is that `--inference_only` should be set to `True`.
    - `--lora_ckpt_path` is required to set to the path of the pretrained LoRA model. Otherwise, the original VideoCrafter model will be used for inference.

## 🔧 Training
Please run `accelerate config` as the first step to configure accelerator settings. If you are not familiar with the accelerator configuration, you can refer to VADER-VideoCrafter [documentation](../documentation/VADER-VideoCrafter.md).

Assuming you are in the `VADER/` directory, you are able to train the model using the following commands:

```bash
cd VADER-VideoCrafter
sh scripts/run_text2video_train.sh
```
- Our experiments are conducted on PyTorch 2.3.0 and CUDA 12.1 while using 4 A6000s (48GB RAM). It should also work with recent PyTorch and CUDA versions. The training script have been tested on a single GPU with 16GBs VRAM, when we set `train_batch_size=1 val_batch_size=1` and use `fp16` mixed precision.
- `VADER/VADER-VideoCrafter/scripts/main/train_t2v_lora.py` is also a script for fine-tuning the VideoCrafter2 using VADER via LoRA.
    - You can read the VADER-VideoCrafter [documentation](../documentation/VADER-VideoCrafter.md) to understand the usage of arguments.

## 💡 Tutorial
This section is to provide a tutorial on how to implement the VADER method on VideoCrafter by yourself. We will provide a step-by-step guide to help you understand the implementation details. Thus, you can easily adapt the VADER method to later versions of VideCrafter. This tutorial is based on the VideoCrafter2.

### Step 1: Install the dependencies
First, you need to install the dependencies according to the [VideoCrafter](https://github.com/AILab-CVC/VideoCrafter) repository. You can also follow the instructions in the repository to install the dependencies.
```bash
conda create -n vader_videocrafter python=3.8.5
conda activate vader_videocrafter
pip install -r requirements.txt
```

You have to download pretrained Text-to-Video [VideoCrafter2](https://huggingface.co/VideoCrafter/VideoCrafter2/blob/main/model.ckpt) model via Hugging Face, and put the `model.ckpt` in the downloaded VideoCrafter directionary as `VideoCrafter/checkpoints/base_512_v2/model.ckpt`.

There are a list of extra dependencies that you need to install for VADER. You can install them by running the following command.
```bash
# Install the HPS
git clone https://github.com/tgxs002/HPSv2.git
cd HPSv2/
pip install -e .
cd ..

# Install the dependencies
pip install albumentations \
peft \
bitsandbytes \
accelerate \
inflect \
wandb \
ipdb \
pytorch_lightning
```

### Step 2: Transfer VADER scripts
You can copy our `VADER/VADER-VideoCrafter/scripts/main/train_t2v_lora.py` to the `VideoCrafter/scripts/evaluation/` directory of VideoCrafter. It is better to copy our `run_text2video_train.sh` and `run_text2video_inference.sh` to the directionary `VideoCrafter/scripts/` as well. Then, you need to copy All the files in `VADER/Core/` and `VADER/assets/` to the parent directory of VideoCrafter, which means `Core/`, `assets` and `VideoCrafter/` should be in the same directory. Now, you may have a directory structure like:
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

- Modify the `batch_ddim_sampling()` function in `VideoCrafter/scripts/evaluation/funcs.py` as our implementation in `VADER/VADER-VideoCrafter/scripts/main/funcs.py`.
- Modify the `DDIMSampler.__init__()`, `DDIMSampler.sample()` and `DDIMSampler.ddim_sampling` functions in  `VideoCrafter\lvdm\models\samplers\ddim.py` as our implementation in `VADER/VADER-VideoCrafter\lvdm\models\samplers\ddim.py`.
- Comment out the `@torch.no_grad()` before `DDIMSampler.sample()`, `DDIMSampler.ddim_sampling`, and `DDIMSampler.p_sample_ddim()` in `VideoCrafter\lvdm\models\samplers\ddim.py`. Also, comment out the `@torch.no_grad()` before `LatentDiffusion.decode_first_stage_2DAE()` in `VideoCrafter\lvdm\models\ddpm3d.py`.
- Because we have commented out the `@torch.no_grad()`, you can add `with torch.no_grad():` at some places in `VideoCrater/scripts/evaluation/inference.py` to avoid the gradient calculation.

### Step 4: Ready to Train
Now you have all the files in the right place and modified the VideoCrafter source code. You can run the training script by running the following command.
```bash
cd VideoCrafter

# training
sh scripts/run_text2video_train.sh

# or inference
sh scripts/run_text2video_inference.sh
```


## Acknowledgement

Our codebase is directly built on top of [VideoCrafter](https://github.com/AILab-CVC/VideoCrafter), [Open-Sora](https://github.com/hpcaitech/Open-Sora), and [Animate Anything](https://github.com/alibaba/animate-anything/). We would like to thank the authors for open-sourcing their code.

## Citation

If you find this work useful in your research, please cite:

```bibtex

```
