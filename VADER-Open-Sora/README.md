<div align="center">

<!-- TITLE -->
# 🎬 **VADER-Open-Sora**

</div>


## ⚙️ Installation
Assuming you are in the `VADER/` directory, you are able to create a Conda environments for VADER-Open-Sora using the following commands:
```bash
cd VADER-Open-Sora
conda create -n vader_opensora python=3.10
conda activate vader_opensora
conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=12.1 -c pytorch -c nvidia
conda install xformers -c xformers
pip install -v -e .
git clone https://github.com/tgxs002/HPSv2.git
cd HPSv2/
pip install -e .
cd ..
```

## 📺 Inference
Please run `accelerate config` as the first step to configure accelerator settings. If you are not familiar with the accelerator configuration, you can refer to VADER-Open-Sora [documentation](../documentation/VADER-Open-Sora.md).

Assuming you are in the `VADER/` directory, you are able to do inference using the following commands:
```bash
cd VADER-Open-Sora
sh scripts/run_text2video_inference.sh
```
- We have tested on PyTorch 2.3.0 and CUDA 12.1. If the `resolution` is set as `360p`, a GPU with 40GBs of VRAM is required when we set `val_batch_size=1` and use `bf16` mixed precision . It should also work with recent PyTorch and CUDA versions. Please refer to the original [Open-Sora](https://github.com/hpcaitech/Open-Sora) repository for more details about the GPU requirements and the model settings.
- `VADER/VADER-Open-Sora/scripts/train_t2v_lora.py` is a script for do inference via the Open-Sora 1.2 using VADER.
    - `--num-frames`, `'--resolution'`, `'fps'` and `'aspect-ratio'` are inherited from the original Open-Sora model. In short, you can set `'--num-frames'` as `'2s'`, `'4s'`, `'8s'`, and `'16s'`. Available values for `--resolution` are `'240p'`, `'360p'`, `'480p'`, and `'720p'`. The default value of `'fps'` is `24` and `'aspect-ratio'` is `3:4`. Please refer to the original [Open-Sora](https://github.com/hpcaitech/Open-Sora) repository for more details. One thing to keep in mind, for instance, is that if you set `--num-frames` to `2s` and `--resolution` to `'240p'`, it is better to use `bf16` mixed precision instead of `fp16`. Otherwise, the model may generate noise videos.
    - `--prompt-path` is the path of the prompt file. Unlike VideoCrafter, we do not provide prompt function for Open-Sora. Instead, you can provide a prompt file, which contains a list of prompts.
    - `--num-processes` is the number of processes for Accelerator. It is recommended to set it to the number of GPUs.
- `VADER/VADER-Open-Sora/configs/opensora-v1-2/vader/vader_inferece.py` is the configuration file for inference. You can modify the configuration file to change the inference settings following the guidance in the [documentation](../documentation/VADER-Open-Sora.md).
    - The main difference is that `is_vader_training` should be set to `False`. The `--lora_ckpt_path` should be set to the path of the pretrained LoRA model. Otherwise, the original Open-Sora model will be used for inference.


## 🔧 Training
Please run `accelerate config` as the first step to configure accelerator settings. If you are not familiar with the accelerator configuration, you can refer to VADER-Open-Sora [documentation](../documentation/VADER-Open-Sora.md).

Assuming you are in the `VADER/` directory, you are able to train the model using the following commands:

```bash
cd VADER-Open-Sora
sh scripts/run_text2video_train.sh
```
- Our experiments are conducted on PyTorch 2.3.0 and CUDA 12.1 while using 4 A6000s (48GB RAM). It should also work with recent PyTorch and CUDA versions. A GPU with 48GBs of VRAM is required for fine-tuning model when use `bf16` mixed precision as `resolution` is set as `360p` and `num_frames` is set as `2s`.
- `VADER/VADER-Open-Sora/scripts/train_t2v_lora.py` is a script for fine-tuning the Open-Sora 1.2 using VADER via LoRA.
    - The arguments are the same as the inference process above.
- `VADER/VADER-Open-Sora/configs/opensora-v1-2/vader/vader_train.py` is the configuration file for training. You can modify the configuration file to change the training settings.
    - You can read the VADER-Open-Sora [documentation](../documentation/VADER-Open-Sora.md) to understand the usage of arguments.




## 💡 Tutorial
This section is to provide a tutorial on how to implement the VADER method on Open-Sora by yourself. We will provide a step-by-step guide to help you understand the implementation details. Thus, you can easily adapt the VADER method to later versions of Open-Sora or other video generation models. This tutorial is based on the Open-Sora v1.2.0 version.

### Step 1: Install the dependencies
First, you need to install the dependencies according to the [Open-Sora](https://github.com/hpcaitech/Open-Sora) repository. You can also follow the instructions in the repository to install the dependencies.
```bash
conda create -n vader_opensora python=3.9
conda activate vader_opensora

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
You can copy our `VADER/VADER-Open-Sora/scripts/train_t2v_lora.py` to the `scripts` directory of Open-Sora, namely `Open-Sora/scripts/`. It is better to copy `run_text2video_train.sh` and `run_text2video_inference.sh` to that directionary as well. Then, you need to copy All the files in `VADER/Core/` and `VADER/assets/` to the parent directory of Open-Sora, which means `Core/`, `assets` and `Open-Sora/` should be in the same directory. You have to also copy the `VADER/VADER-Open-Sora/configs/opensora-v1-2/vader/vader_train.py` and `VADER/VADER-Open-Sora/configs/opensora-v1-2/vader/vader_inference.py` to one directory of Open-Sora, namely `Open-Sora/configs/opensora-v1-2/vader/`. Now, you may have a directory structure like:
```bash
.
├── Core
│   ├── ...
├── Open-Sora
│   ├── scripts
│   │   ├── train_t2v_lora.py
│   │   ├── run_text2video_train.sh
│   │   ├── run_text2video_inference.sh
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

- Modify the function `sample()` in `Open-Sora/opensora/schedulers/rf/__init__.py` as in our implementation of `VADER/Core/schedulers/rf/__init__.py`.

### Step 4: Ready to Train
Now you have all the files in the right place and modified the Open-Sora source code. You can run the training script by running the following command.
```bash
cd Open-Sora

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
