<div align="center">

<!-- TITLE -->
# 🎥 **ModelScope**
</div>


## ⚙️ Installation
Assuming you are in the `VADER/` directory, you are able to create a Conda environments for VADER-ModelScope using the following commands:
```bash
cd VADER-ModelScope
conda create -n vader_modelscope python=3.10
conda activate vader_modelscope
conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=12.1 -c pytorch -c nvidia
conda install xformers -c xformers
pip install -r requirements.txt
git clone https://github.com/tgxs002/HPSv2.git
cd HPSv2/
pip install -e .
cd ..
```

## 📺 Inference
Please run `accelerate config` as the first step to configure accelerator settings. Assuming you are in the `VADER/` directory, you are able to do inference using the following commands:
```bash
cd VADER-ModelScope
sh run_text2video_inference.sh
```

- Note: we do note set `lora_path` in the original inference script. You can set `lora_path` to the path of the pretrained LoRA model if you have one.

## 🔧 Training
The current code can work on a single GPU with VRAM > 14GBs. The code can be further optimized to work with even lesser VRAM with deepspeed and CPU offloading. For our experiments, we used 4 A100s- 40GB RAM to run our code.

Please run `accelerate config` as the first step to configure accelerator settings. Assuming you are in the `VADER/` directory, you are able to train the model using the following commands:
```bash
cd VADER-ModelScope
sh run_text2video_train.sh
```
- `VADER/VADER-ModelScope/train_t2v_lora.py` is a script for fine-tuning ModelScope using VADER via LoRA.
    - `gradient_accumulation_steps` can be increased while reducing the `--num_processes` of the accelerator to alleviate bottleneck caused by the number of GPUs. We tested with `gradient_accumulation_steps=4` and `--num_processes=4` on 4 A100s- 40GB RAM.
    - `prompt_fn` is the prompt function, which can be the name of any functions in Core/prompts.py, like `'chatgpt_custom_instruments'`, `'chatgpt_custom_animal_technology'`, `'chatgpt_custom_ice'`, `'nouns_activities'`, etc. Note: If you set `--prompt_fn 'nouns_activities'`, you have to provide`--nouns_file` and `--nouns_file`, which will randomly select a noun and an activity from the files and form them into a single sentence as a prompt.
    - `reward_fn` is the reward function, which can be selected from `'aesthetic'`, `'hps'`, and `'actpred'`.
- `VADER/VADER-ModelScope/config_t2v/config.yaml` is the configuration file for training. You can modify the configuration file to change the training settings following the comments in that file.


## Acknowledgement

Our codebase is directly built on top of [VideoCrafter](https://github.com/AILab-CVC/VideoCrafter), [Open-Sora](https://github.com/hpcaitech/Open-Sora), and [Animate Anything](https://github.com/alibaba/animate-anything/). We would like to thank the authors for open-sourcing their code.

## Citation

If you find this work useful in your research, please cite:

```bibtex

```
