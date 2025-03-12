import sys
sys.path.append('../')   # setting path to get Core and assets
import argparse
import datetime
import time
import hydra
from omegaconf import DictConfig 
from PIL import Image
import logging
from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
import hpsv2
import numpy as np
from accelerate.state import AcceleratorState
import inspect
import torchvision
from transformers.utils import ContextManagers
import math
import os
import random
import gc
import copy
import ipdb
import imageio
from Core.aesthetic_scorer import AestheticScorerDiff
from Core.actpred_scorer import ActPredScorer
st = ipdb.set_trace
import accelerate
from typing import Dict, Optional, Tuple
from omegaconf import OmegaConf
import Core.prompts as prompts_file
from accelerate import Accelerator
import cv2
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torchvision.transforms as T
import diffusers
import transformers
from collections import defaultdict

from torchvision import transforms
from tqdm.auto import tqdm

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

from models.unet_3d_condition import UNet3DConditionModel
from diffusers.models import AutoencoderKL
from diffusers import DPMSolverMultistepScheduler, DDPMScheduler, TextToVideoSDPipeline
from ms_custom_ddpo import CustomT2VDDPO
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, export_to_video
from diffusers.utils.import_utils import is_xformers_available
from diffusers.models.attention_processor import AttnProcessor2_0, Attention
from diffusers.models.attention import BasicTransformerBlock
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler

from transformers import CLIPTextModel, CLIPTokenizer
from transformers.models.clip.modeling_clip import CLIPEncoder
from utils.dataset import VideoJsonDataset, SingleVideoDataset, \
    ImageDataset, VideoFolderDataset, CachedDataset
from einops import rearrange, repeat
from utils.lora_handler import LoraHandler, LORA_VERSIONS

already_printed_trainables = False

logger = get_logger(__name__, log_level="INFO")

import numpy as np
from collections import deque
from ddim_with_logprob import ddim_step_with_logprob

class PerPromptStatTracker:
    def __init__(self, buffer_size, min_count):
        self.buffer_size = buffer_size
        self.min_count = min_count
        self.stats = {}

    def update(self, prompts, rewards):
        prompts = np.array(prompts)
        rewards = np.array(rewards)
        unique = np.unique(prompts)
        advantages = np.empty_like(rewards)
        for prompt in unique:
            prompt_rewards = rewards[prompts == prompt]
            if prompt not in self.stats:
                self.stats[prompt] = deque(maxlen=self.buffer_size)
            self.stats[prompt].extend(prompt_rewards)

            if len(self.stats[prompt]) < self.min_count:
                mean = np.mean(rewards)
                std = np.std(rewards) + 1e-6
            else:
                mean = np.mean(self.stats[prompt])
                std = np.std(self.stats[prompt]) + 1e-6
            advantages[prompts == prompt] = (prompt_rewards - mean) / std

        return advantages

    def get_stats(self):
        return {
            k: {"mean": np.mean(v), "std": np.std(v), "count": len(v)}
            for k, v in self.stats.items()
        }


def create_logging(logging, logger, accelerator):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

def accelerate_set_verbose(accelerator):
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

def get_train_dataset(dataset_types, train_data, tokenizer):
    train_datasets = []

    # Loop through all available datasets, get the name, then add to list of data to process.
    for DataSet in [VideoJsonDataset, SingleVideoDataset, ImageDataset, VideoFolderDataset]:
        for dataset in dataset_types:
            if dataset == DataSet.__getname__():
                train_datasets.append(DataSet(**train_data, tokenizer=tokenizer))

    if len(train_datasets) > 0:
        return train_datasets
    else:
        raise ValueError("Dataset type not found: 'json', 'single_video', 'folder', 'image'")

def extend_datasets(datasets, dataset_items, extend=False):
    biggest_data_len = max(x.__len__() for x in datasets)
    extended = []
    for dataset in datasets:
        if dataset.__len__() == 0:
            del dataset
            continue
        if dataset.__len__() < biggest_data_len:
            for item in dataset_items:
                if extend and item not in extended and hasattr(dataset, item):
                    print(f"Extending {item}")

                    value = getattr(dataset, item)
                    value *= biggest_data_len
                    value = value[:biggest_data_len]

                    setattr(dataset, item, value)

                    print(f"New {item} dataset length: {dataset.__len__()}")
                    extended.append(item)

def export_to_video(video_frames, output_video_path, fps):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    h, w, _ = video_frames[0].shape
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps=fps, frameSize=(w, h))
    for i in range(len(video_frames)):
        img = cv2.cvtColor(video_frames[i], cv2.COLOR_RGB2BGR)
        video_writer.write(img)

def deepspeed_zero_init_disabled_context_manager():
    """
    returns either a context list that includes one that will disable zero.Init or an empty context list
    """
    deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
    if deepspeed_plugin is None:
        return []

    return [deepspeed_plugin.zero3_init_context_manager(enable=False)]



def create_output_folders(output_dir, run_name, config):
    out_dir = os.path.join(output_dir, run_name)    
    
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(f"{out_dir}/samples", exist_ok=True)

    return out_dir

def load_primary_models(pretrained_model_path, scheduler_type):
    unet = UNet3DConditionModel.from_pretrained(pretrained_model_path, subfolder="unet")
    pipeline = CustomT2VDDPO.from_pretrained(pretrained_model_path, unet=unet)
    if scheduler_type == "dpm_multistep":
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    else:
        assert scheduler_type == "ddim", "Default scheudler is ddim"
    noise_scheduler = pipeline.scheduler
    tokenizer = pipeline.tokenizer
    text_encoder = pipeline.text_encoder
    vae = pipeline.vae
    unet = pipeline.unet
    
    return pipeline, noise_scheduler, tokenizer, text_encoder, vae, unet

def unet_and_text_g_c(unet, text_encoder, unet_enable, text_enable):
    if  isinstance(unet, UNet3DConditionModel):
        unet._set_gradient_checkpointing(value=unet_enable)
    else:
        unet.module._set_gradient_checkpointing(value=unet_enable)

    if isinstance(text_encoder, CLIPTextModel):
        text_encoder._set_gradient_checkpointing(CLIPEncoder, value=text_enable)
    else:
        text_encoder.module._set_gradient_checkpointing(CLIPEncoder, value=text_enable)

def freeze_models(models_to_freeze):
    for model in models_to_freeze:
        if model is not None: model.requires_grad_(False) 
            
def is_attn(name):
   return ('attn1' or 'attn2' == name.split('.')[-1])

def set_processors(attentions):
    for attn in attentions: attn.set_processor(AttnProcessor2_0()) 

def set_torch_2_attn(unet):
    optim_count = 0
    
    for name, module in unet.named_modules():
        if is_attn(name):
            if isinstance(module, torch.nn.ModuleList):
                for m in module:
                    if isinstance(m, BasicTransformerBlock):
                        set_processors([m.attn1, m.attn2])
                        optim_count += 1
    if optim_count > 0: 
        print(f"{optim_count} Attention layers using Scaled Dot Product Attention.")

def handle_memory_attention(enable_xformers_memory_efficient_attention, enable_torch_2_attn, unet): 
    try:
        is_torch_2 = hasattr(F, 'scaled_dot_product_attention')
        enable_torch_2 = is_torch_2 and enable_torch_2_attn
        
        if enable_xformers_memory_efficient_attention and not enable_torch_2:
            if is_xformers_available():
                from xformers.ops import MemoryEfficientAttentionFlashAttentionOp
                unet.enable_xformers_memory_efficient_attention(attention_op=MemoryEfficientAttentionFlashAttentionOp)
            else:
                raise ValueError("xformers is not available. Make sure it is installed correctly")
        
        if enable_torch_2:
            set_torch_2_attn(unet)
            
    except:
        print("Could not enable memory efficient attention for xformers or Torch 2.0.")

def param_optim(model, condition, extra_params=None, is_lora=False, negation=None):
    extra_params = extra_params if len(extra_params.keys()) > 0 else None
    return {
        "model": model, 
        "condition": condition, 
        'extra_params': extra_params,
        'is_lora': is_lora,
        "negation": negation
    }

def actpred_loss_fn(inference_dtype=None, device=None, num_frames = 14, target_size=224):
    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        scorer = ActPredScorer(device=device, num_frames = num_frames)
    scorer.requires_grad_(False)

    def preprocess_img(img):

        img = ((img/2) + 0.5).clamp(0,1)
        img = T.Resize((target_size, target_size), antialias = True)(img)
        img = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
        return img

    def loss_fn(vid_list, label_list):
        losses, scores = [], []
        for i in range(len(vid_list)):
            vid = vid_list[i]
            label = label_list[i]
            vid = torch.cat([ preprocess_img(img).unsqueeze(0) for img in vid])[None]
            loss, score, _ = scorer.get_loss_and_score(vid, label)
            losses.append(loss)
            scores.append(score)
        return torch.stack(losses), torch.stack(scores)
    
    return loss_fn
    
def aesthetic_loss_fn(aesthetic_target=None,
                     grad_scale=0,
                     device=None,
                     accelerator=None,
                     torch_dtype=None):
    
    target_size = 224
    normalize = torchvision.transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                                std=[0.26862954, 0.26130258, 0.27577711])
    
    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        scorer = AestheticScorerDiff(dtype=torch_dtype).to(device, dtype=torch_dtype)
    scorer.requires_grad_(False)
    target_size = 224
    def loss_fn(im_pix_un):
        im_pix = ((im_pix_un / 2) + 0.5).clamp(0, 1) 
        im_pix = torchvision.transforms.Resize(target_size)(im_pix)
        im_pix = normalize(im_pix).to(im_pix_un.dtype)
        rewards = scorer(im_pix)
        if aesthetic_target is None: # default maximization
            loss = -1 * rewards
        else:
            # using L1 to keep on same scale
            loss = abs(rewards - aesthetic_target)
        return loss * grad_scale, rewards
    return loss_fn


def create_optim_params(name='param', params=None, lr=5e-6, extra_params=None):
    params = {
        "name": name, 
        "params": params, 
        "lr": lr
    }
    if extra_params is not None:
        for k, v in extra_params.items():
            params[k] = v
    
    return params

def negate_params(name, negation):
    # We have to do this if we are co-training with LoRA.
    # This ensures that parameter groups aren't duplicated.
    if negation is None: return False
    for n in negation:
        if n in name and 'temp' not in name:
            return True
    return False


def create_optimizer_params(model_list, lr):
    import itertools
    optimizer_params = []

    for optim in model_list:
        model, condition, extra_params, is_lora, negation = optim.values()
        # Check if we are doing LoRA training.
        if is_lora and condition and isinstance(model, list): 
            params = create_optim_params(
                params=itertools.chain(*model), 
                extra_params=extra_params
            )
            optimizer_params.append(params)
            continue
            
        if is_lora and  condition and not isinstance(model, list):
            for n, p in model.named_parameters():
                if 'lora' in n:
                    params = create_optim_params(n, p, lr, extra_params)
                    optimizer_params.append(params)
            continue

        # If this is true, we can train it.
        if condition:
            for n, p in model.named_parameters():
                should_negate = 'lora' in n and not is_lora
                if should_negate: continue

                params = create_optim_params(n, p, lr, extra_params)
                optimizer_params.append(params)
    
    return optimizer_params

def get_optimizer(use_8bit_adam):
    if use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        return bnb.optim.AdamW8bit
    else:
        return torch.optim.AdamW

def is_mixed_precision(accelerator):
    weight_dtype = torch.float32

    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16

    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    return weight_dtype

def cast_to_gpu_and_type(model_list, accelerator, weight_dtype):
    for model in model_list:
        if model is not None: model.to(accelerator.device, dtype=weight_dtype)

def handle_cache_latents(
        should_cache, 
        output_dir, 
        train_dataloader, 
        train_batch_size, 
        vae, 
        cached_latent_dir=None,
        shuffle=False
    ):

    # Cache latents by storing them in VRAM. 
    # Speeds up training and saves memory by not encoding during the train loop.
    if not should_cache: return None
    vae.to('cuda', dtype=torch.float16)
    vae.enable_slicing()
    
    cached_latent_dir = (
        os.path.abspath(cached_latent_dir) if cached_latent_dir is not None else None 
        )

    if cached_latent_dir is None:
        cache_save_dir = f"{output_dir}/cached_latents"
        os.makedirs(cache_save_dir, exist_ok=True)

        for i, batch in enumerate(tqdm(train_dataloader, desc="Caching Latents.")):

            save_name = f"cached_{i}"
            full_out_path =  f"{cache_save_dir}/{save_name}.pt"

            pixel_values = batch['pixel_values'].to('cuda', dtype=torch.float16)
            batch['pixel_values'] = tensor_to_vae_latent(pixel_values, vae)
            for k, v in batch.items(): batch[k] = v[0]
        
            torch.save(batch, full_out_path)
            del pixel_values
            del batch

            # We do this to avoid fragmentation from casting latents between devices.
            torch.cuda.empty_cache()
    else:
        cache_save_dir = cached_latent_dir
        

    return torch.utils.data.DataLoader(
        CachedDataset(cache_dir=cache_save_dir), 
        batch_size=train_batch_size, 
        shuffle=shuffle,
        num_workers=0
    ) 

def handle_trainable_modules(model, trainable_modules=None, is_enabled=True, negation=None):
    global already_printed_trainables
    acc = []
    unfrozen_params = 0
    
    if trainable_modules is not None:
        unlock_all = any([name == 'all' for name in trainable_modules])
        if unlock_all:
            model.requires_grad_(True)
            unfrozen_params = len(list(model.parameters()))
        else:
            model.requires_grad_(False)
            for name, param in model.named_parameters():
                for tm in trainable_modules:
                    if all([tm in name, name not in acc, 'lora' not in name]):
                        param.requires_grad_(is_enabled)
                        acc.append(name)
                        unfrozen_params += 1
                        
    if unfrozen_params > 0 and not already_printed_trainables:
        already_printed_trainables = True 
        print(f"{unfrozen_params} params have been processed.")

def tensor_to_vae_latent(t, vae):
    video_length = t.shape[1]

    t = rearrange(t, "b f c h w -> (b f) c h w")
    latents = vae.encode(t).latent_dist.sample()
    latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
    latents = latents * 0.18215

    return latents

def sample_noise(latents, noise_strength, use_offset_noise=False):
    b ,c, f, *_ = latents.shape
    noise_latents = torch.randn_like(latents, device=latents.device)
    offset_noise = None

    if use_offset_noise:
        offset_noise = torch.randn(b, c, f, 1, 1, device=latents.device)
        noise_latents = noise_latents + noise_strength * offset_noise

    return noise_latents

def enforce_zero_terminal_snr(betas):
    """
    Corrects noise in diffusion schedulers.
    From: Common Diffusion Noise Schedules and Sample Steps are Flawed
    https://arxiv.org/pdf/2305.08891.pdf
    """
    # Convert betas to alphas_bar_sqrt
    alphas = 1 - betas
    alphas_bar = alphas.cumprod(0)
    alphas_bar_sqrt = alphas_bar.sqrt()

    # Store old values.
    alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
    alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()

    # Shift so the last timestep is zero.
    alphas_bar_sqrt -= alphas_bar_sqrt_T

    # Scale so the first timestep is back to the old value.
    alphas_bar_sqrt *= alphas_bar_sqrt_0 / (
        alphas_bar_sqrt_0 - alphas_bar_sqrt_T
    )

    # Convert alphas_bar_sqrt to betas
    alphas_bar = alphas_bar_sqrt ** 2
    alphas = alphas_bar[1:] / alphas_bar[:-1]
    alphas = torch.cat([alphas_bar[0:1], alphas])
    betas = 1 - alphas

    return betas

def should_sample(global_step, validation_steps, validation_data):
    return (global_step % validation_steps == 0 or global_step ==1)  \
    and validation_data.sample_preview

def save_pipe(
        path, 
        global_step,
        accelerator, 
        unet, 
        text_encoder, 
        vae, 
        output_dir,
        lora_manager: LoraHandler,
        unet_target_replace_module=None,
        text_target_replace_module=None,
        is_checkpoint=False,
        save_pretrained_model=True
    ):
    start_time = time.time()
    if is_checkpoint:
        save_path = os.path.join(output_dir, f"checkpoint-{global_step}")
        os.makedirs(save_path, exist_ok=True)
    else:
        save_path = output_dir

    # Save the dtypes so we can continue training at the same precision.
    u_dtype, t_dtype, v_dtype = unet.module.dtype, text_encoder.module.dtype, vae.dtype 

    unet_save = copy.deepcopy(unet.cpu())
    text_encoder_save = copy.deepcopy(text_encoder.cpu())

    unet_out = copy.deepcopy(accelerator.unwrap_model(unet_save, keep_fp32_wrapper=False))
    text_encoder_out = copy.deepcopy(accelerator.unwrap_model(text_encoder_save, keep_fp32_wrapper=False))

    pipeline = TextToVideoSDPipeline.from_pretrained(
        path,
        unet=unet_out,
        text_encoder=text_encoder_out,
        vae=vae,
    ).to(torch_dtype=torch.float32)
    
    lora_manager.save_lora_weights(model=pipeline, save_path=save_path, step=global_step)

    if save_pretrained_model:
        pipeline.save_pretrained(save_path)
    if is_checkpoint:
        unet, text_encoder = accelerator.prepare(unet, text_encoder)
        models_to_cast_back = [(unet, u_dtype), (text_encoder, t_dtype), (vae, v_dtype)]
        [x[0].to(accelerator.device, dtype=x[1]) for x in models_to_cast_back]

    logger.info(f"Saved model at {save_path} on step {global_step}")
    del pipeline
    del unet_out
    del text_encoder_out
    torch.cuda.empty_cache()
    gc.collect()


def hps_loss_fn(inference_dtype=None, device=None):
    model_name = "ViT-H-14"
    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        model, preprocess_train, preprocess_val = create_model_and_transforms(
            model_name,
            'laion2B-s32B-b79K',
            precision=inference_dtype,
            device=device,
            jit=False,
            force_quick_gelu=False,
            force_custom_text=False,
            force_patch_dropout=False,
            force_image_size=None,
            pretrained_image=False,
            image_mean=None,
            image_std=None,
            light_augmentation=True,
            aug_cfg={},
            output_dict=True,
            with_score_predictor=False,
            with_region_predictor=False
        )    
        tokenizer = get_tokenizer(model_name)
    
    checkpoint_path = f"{os.path.expanduser('~')}/.cache/hpsv2/HPS_v2_compressed.pt"
    # force download of model via score
    hpsv2.score([], "")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    tokenizer = get_tokenizer(model_name)
    model = model.to(device, dtype=inference_dtype)
    model.eval()

    target_size =  224
    normalize = torchvision.transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                                std=[0.26862954, 0.26130258, 0.27577711])
        
    def loss_fn(im_pix, prompts):    
        im_pix = ((im_pix / 2) + 0.5).clamp(0, 1) 
        x_var = torchvision.transforms.Resize(target_size)(im_pix)
        x_var = normalize(x_var).to(im_pix.dtype)        
        caption = tokenizer(prompts)
        caption = caption.to(device)
        outputs = model(x_var, caption)
        image_features, text_features = outputs["image_features"], outputs["text_features"]
        logits = image_features @ text_features.T
        scores = torch.diagonal(logits)
        loss = 1.0 - scores
        return  loss, scores
    
    return loss_fn
 

def replace_prompt(prompt, token, wlist):
    for w in wlist:
        if w in prompt: return prompt.replace(w, token)
    return prompt 


def sample_from_model(ddpo_params, log_step, epoch, prompt_fn, prompt_fn_kwargs, 
                    accelerator, pipeline, weight_dtype,
                    validation_params, 
                    output_dir, logger, stat_tracker,
                    reward_fn_type, loss_fn, decode_frame,
                    negative_prompt_embeds,
                    eta = 1.0,  num_rollouts = 2):

    samples = []
    for i in range(ddpo_params.sample_num_batches_per_epoch):

        prompts, prompt_metadata = zip(
                    *[prompt_fn(**prompt_fn_kwargs) for _ in range(ddpo_params.sample_batch_size)]
            )
        
        prompt_ids = pipeline.tokenizer(
                prompts,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=pipeline.tokenizer.model_max_length,
            ).input_ids.to(accelerator.device)
                
        prompts = list(prompts)

        with accelerator.autocast():
            with torch.no_grad():
                frames, latents, logprob, timesteps, prompt_embeds = pipeline.forward_with_logprob(
                    prompts,
                    width=validation_params.width,
                    height=validation_params.height,
                    num_frames=validation_params.num_frames,
                    num_inference_steps=validation_params.num_inference_steps,
                    guidance_scale=validation_params.guidance_scale,
                    generator=  torch.cuda.manual_seed( random.randint(0,100000)),
                    eta = eta,
                    negative_prompt_embeds = negative_prompt_embeds,
                    decode_frame = decode_frame,
                )
                
                if reward_fn_type == "actpred":
                    frames_ = frames.permute(1,0,2,3,4)
                    bs, nf, c, h, w = frames_.shape
                    frames_ = frames_.to(weight_dtype) 
                    loss, rewards = loss_fn(frames_, prompts) 
                 
                else:

                    frames_ = frames.permute(0,2,1,3,4)
                    bs, nf, c, h, w = frames_.shape
                    assert nf == 1 # reward should only be on single frame
                    frames_ = frames_.squeeze(1)
                    frames_ = frames_.to(weight_dtype) 

                    if reward_fn_type == "aesthetic":
                        loss, rewards = loss_fn(frames_)
                    else:
                        loss, rewards = loss_fn(frames_,prompts)                    
                    
                reward = rewards.squeeze()
                samples.append(
                    {
                        "prompt_embeds": prompt_embeds,
                        "prompt_ids": prompt_ids,
                        "timesteps": timesteps,
                        "latents": latents[:, :-1],
                        "next_latents": latents[:, 1:], 
                        "log_probs": logprob,
                        "rewards": reward,
                    }
                )
    samples = {k: torch.cat([s[k] for s in samples]) for k in samples[0].keys()}

    rewards = accelerator.gather(samples["rewards"]).cpu().numpy()

    accelerator.log(
            {
                "reward": rewards,
                "epoch": epoch,
                "reward_mean": rewards.mean(),
                "reward_std": rewards.std(),
            },
            step=log_step,
        )
 
    prompt_ids = accelerator.gather(samples["prompt_ids"]).cpu().numpy()
    prompts = pipeline.tokenizer.batch_decode(
                prompt_ids, skip_special_tokens=True
            )
    advantages = stat_tracker.update(prompts, rewards)

    samples["advantages"] = (
            torch.as_tensor(advantages)
            .reshape(accelerator.num_processes, -1)[accelerator.process_index]
            .to(accelerator.device)
        )
    del samples["rewards"]
    del samples["prompt_ids"]
    
    return samples, frames

def main(
    pretrained_model_path: str,
    output_dir: str,
    train_data: Dict,
    validation_data: Dict,
    extra_train_data: list = [],
    dataset_types: Tuple[str] = ('json'),
    shuffle: bool = True,
    validation_steps: int = 100,
    trainable_modules: Tuple[str] = None, # Eg: ("attn1", "attn2")
    trainable_text_modules: Tuple[str] = None, # Eg: ("all"), this also applies to trainable_modules
    extra_unet_params = None,
    extra_text_encoder_params = None,
    train_batch_size: int = 1,
    max_train_steps: int = 500,
    learning_rate: float = 5e-5,
    scale_lr: bool = False,
    load_videodataset = False,
    lr_scheduler: str = "constant",
    lr_warmup_steps: int = 0,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_weight_decay: float = 1e-2,
    adam_epsilon: float = 1e-08,
    max_grad_norm: float = 1.0,
    gradient_accumulation_steps: int = 1,
    gradient_checkpointing: bool = False,
    rwand = False,
    text_encoder_gradient_checkpointing: bool = False,
    checkpointing_steps: int = 500,
    resume_from_checkpoint: Optional[str] = None,
    resume_step: Optional[int] = None,
    mixed_precision: Optional[str] = "fp16",
    use_8bit_adam: bool = False,
    enable_xformers_memory_efficient_attention: bool = True,
    enable_torch_2_attn: bool = False,
    seed: Optional[int] = None,
    train_text_encoder: bool = False,
    use_offset_noise: bool = False,
    rescale_schedule: bool = False,
    offset_noise_strength: float = 0.1,
    extend_dataset: bool = False,
    cache_latents: bool = False,
    cached_latent_dir = None,
    prompt_fn = "",
    num_train_epochs = 200,
    lora_version: LORA_VERSIONS = LORA_VERSIONS[0],
    save_lora_for_webui: bool = False,
    only_lora_for_webui: bool = False,
    lora_bias: str = 'none',
    use_unet_lora: bool = False,
    use_text_lora: bool = False,
    unet_lora_modules: Tuple[str] = ["ResnetBlock2D"],
    text_encoder_lora_modules: Tuple[str] = ["CLIPEncoderLayer"],
    save_pretrained_model: bool = False,
    opt = None,
    lora_rank: int = 16,
    lora_path: str = '',
    lora_unet_dropout: float = 0.1,
    lora_text_dropout: float = 0.1,
    logger_type: str = 'wandb',
    scheduler_type = 'dpm_multistep',
    use_wandb = False,
    only_val = False,
    num_only_val_itrs = 5,
    ddpo_params=None,
    **kwargs
):

    *_, config = inspect.getargvalues(inspect.currentframe())

    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps * validation_data.num_inference_steps,
        mixed_precision=mixed_precision,
        log_with=logger_type,
        project_dir=output_dir
    )
    validation_steps = validation_steps*gradient_accumulation_steps
    checkpointing_steps = checkpointing_steps *gradient_accumulation_steps

    # Make one log on every process with the configuration for debugging.
    create_logging(logging, logger, accelerator)

    # Initialize accelerate, transformers, and diffusers warnings
    accelerate_set_verbose(accelerator)

    # If passed along, set the training seed now.
    if seed is not None:
        set_seed(seed)



    # Load scheduler, tokenizer and models.
    pipeline, noise_scheduler, tokenizer, text_encoder, vae, unet = load_primary_models(pretrained_model_path, scheduler_type)
    # Freeze any necessary models
    freeze_models([vae, text_encoder, unet])
    
    # Enable xformers if available
    handle_memory_attention(enable_xformers_memory_efficient_attention, enable_torch_2_attn, unet)

    if scale_lr:
        learning_rate = (
            learning_rate * gradient_accumulation_steps * train_batch_size * accelerator.num_processes
        )

    prompt_fn = getattr(prompts_file, prompt_fn)

    # Initialize the optimizer
    optimizer_cls = get_optimizer(use_8bit_adam)

    # Use LoRA if enabled.  
    lora_manager = LoraHandler(
        version=lora_version, 
        use_unet_lora=use_unet_lora,
        use_text_lora=use_text_lora,
        save_for_webui=save_lora_for_webui,
        only_for_webui=only_lora_for_webui,
        unet_replace_modules=unet_lora_modules,
        text_encoder_replace_modules=text_encoder_lora_modules,
        lora_bias=lora_bias
    )

    unet_lora_params, unet_negation = lora_manager.add_lora_to_model(
        use_unet_lora, unet, lora_manager.unet_replace_modules, lora_unet_dropout, lora_path, r=lora_rank) 

    text_encoder_lora_params, text_encoder_negation = lora_manager.add_lora_to_model(
        use_text_lora, text_encoder, lora_manager.text_encoder_replace_modules, lora_text_dropout, lora_path, r=lora_rank) 

    # Create parameters to optimize over with a condition (if "condition" is true, optimize it)
    extra_unet_params = extra_unet_params if extra_unet_params is not None else {}
    extra_text_encoder_params = extra_unet_params if extra_unet_params is not None else {}

    trainable_modules_available = trainable_modules is not None
    trainable_text_modules_available = (train_text_encoder and trainable_text_modules is not None)
    
    optim_params = [
        param_optim(unet, trainable_modules_available, extra_params=extra_unet_params, negation=unet_negation),
        param_optim(text_encoder, trainable_text_modules_available, 
                        extra_params=extra_text_encoder_params, 
                        negation=text_encoder_negation
                   ),
        param_optim(text_encoder_lora_params, use_text_lora, is_lora=True, 
                        extra_params={**{"lr": learning_rate}, **extra_text_encoder_params}
                    ),
        param_optim(unet_lora_params, use_unet_lora, is_lora=True, 
                        extra_params={**{"lr": learning_rate}, **extra_unet_params}
                    )
    ]

    params = create_optimizer_params(optim_params, learning_rate)
    
    # Create Optimizer
    optimizer = optimizer_cls(
        params,
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        weight_decay=adam_weight_decay,
        eps=adam_epsilon,
    )

    # Scheduler
    lr_scheduler = get_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
        num_training_steps=max_train_steps * gradient_accumulation_steps,
    )

    if load_videodataset:
        # Get the training dataset based on types (json, single_video, image)
        train_datasets = get_train_dataset(dataset_types, train_data, tokenizer)

        # If you have extra train data, you can add a list of however many you would like.
        # Eg: extra_train_data: [{: {dataset_types, train_data: {etc...}}}] 
        try:
            if extra_train_data is not None and len(extra_train_data) > 0:
                for dataset in extra_train_data:
                    d_t, t_d = dataset['dataset_types'], dataset['train_data']
                    train_datasets += get_train_dataset(d_t, t_d, tokenizer)

        except Exception as e:
            print(f"Could not process extra train datasets due to an error : {e}")

        # Extend datasets that are less than the greatest one. This allows for more balanced training.
        attrs = ['train_data', 'frames', 'image_dir', 'video_files']
        extend_datasets(train_datasets, attrs, extend=extend_dataset)

        # Process one dataset
        if len(train_datasets) == 1:
            train_dataset = train_datasets[0]
        # Process many datasets
        else:
            train_dataset = torch.utils.data.ConcatDataset(train_datasets) 

        # DataLoaders creation:
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=train_batch_size,
            shuffle=shuffle
        )

        # Latents caching
        cached_data_loader = handle_cache_latents(
            cache_latents, 
            output_dir,
            train_dataloader, 
            train_batch_size, 
            vae,
            cached_latent_dir
        ) 

        if cached_data_loader is not None: 
            train_dataloader = cached_data_loader
    else:
        train_dataloader = None

    # Prepare everything with our `accelerator`.
    unet, optimizer,train_dataloader, lr_scheduler, text_encoder = accelerator.prepare(
        unet, 
        optimizer, 
        train_dataloader, 
        lr_scheduler, 
        text_encoder
    )

    # Use Gradient Checkpointing if enabled.
    unet_and_text_g_c(
        unet, 
        text_encoder, 
        gradient_checkpointing, 
        text_encoder_gradient_checkpointing
    )
    
    # Enable VAE slicing to save memory.
    vae.enable_slicing()

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = is_mixed_precision(accelerator)

    # Move text encoders, and VAE to GPU
    models_to_cast = [text_encoder, vae]
    cast_to_gpu_and_type(models_to_cast, accelerator, weight_dtype)
   
    if kwargs['reward_fn'] == "aesthetic":
        loss_fn = aesthetic_loss_fn(grad_scale=0.1,
                                    aesthetic_target=10,
                                    accelerator = accelerator,
                                    torch_dtype = weight_dtype,
                                    device = accelerator.device)
    elif kwargs['reward_fn'] == "hps":
        loss_fn = hps_loss_fn(weight_dtype, accelerator.device)    
    elif kwargs['reward_fn'] == "actpred":
        assert opt.decode_frame in ['alt', 'all']
        num_score_frames = validation_data.num_frames//2 if opt.decode_frame == 'alt' else validation_data.num_frames
        loss_fn = actpred_loss_fn(weight_dtype, accelerator.device, num_frames = num_score_frames )
    else:
        raise NotImplementedError(f"Reward function {kwargs['reward_fn']} is not implemented.")

    # Fix noise schedules to predcit light and dark areas if available.
    if not use_offset_noise and rescale_schedule:
        noise_scheduler.betas = enforce_zero_terminal_snr(noise_scheduler.betas)
    
    if train_dataloader is not None:
        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)

        # Afterwards we recalculate our number of training epochs
        num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)
    else:
        pass

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process and use_wandb:
        import wandb
        wandb_args = {}
        
        if opt.wandb_entity != '':
            wandb_args['entity'] =  opt.wandb_entity    
        
        if opt.debug:
            wandb_args['mode'] = "disabled"
        
        opt_dict = dict(opt)
        accelerator.init_trackers("qvid", config=opt_dict, init_kwargs={"wandb": wandb_args})
        output_dir = create_output_folders(output_dir, wandb.run.name, config)

    # Train!
    samples_per_epoch = (
        ddpo_params.sample_batch_size
        * accelerator.num_processes
        * ddpo_params.sample_num_batches_per_epoch
    )
    total_train_batch_size = (
        ddpo_params.train_batch_size
        * accelerator.num_processes
        * gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {ddpo_params.num_epochs}")
    logger.info(f"  Sample batch size per device = {ddpo_params.sample_batch_size}")
    logger.info(f"  Train batch size per device = {ddpo_params.train_batch_size}")
    logger.info(
        f"  Gradient Accumulation steps = {gradient_accumulation_steps}"
    )
    logger.info("")
    logger.info(f"  Total number of samples per epoch = {samples_per_epoch}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}"
    )
    logger.info(
        f"  Number of gradient updates per inner epoch = {samples_per_epoch // total_train_batch_size}"
    )
    logger.info(f"  Number of inner epochs = {ddpo_params.num_train_inner_epochs}")


    global_step = 0
    first_epoch = 0
    
    if opt.prompt_fn_kwargs == 0:
        opt.prompt_fn_kwargs = {} 
      
    adv_stat_tracker = PerPromptStatTracker(
            ddpo_params.per_prompt_stat_tracking_buffer_size,
            ddpo_params.per_prompt_stat_tracking_min_count,
        )
    
    val_prompt, prompt_metadata = zip(
        *[prompt_fn(**opt.prompt_fn_kwargs) for _ in range(opt.val_batch_size)]
    )    
    val_prompt = list(val_prompt)    

    with torch.no_grad():
        neg_prompt_embed = pipeline.text_encoder(
            pipeline.tokenizer(
                [""],
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=pipeline.tokenizer.model_max_length,
            ).input_ids.to(accelerator.device)
        )[0]
        sample_neg_prompt_embeds = neg_prompt_embed.repeat(ddpo_params.sample_batch_size, 1, 1)
        train_neg_prompt_embeds = neg_prompt_embed.repeat(ddpo_params.train_batch_size, 1, 1)

    for epoch in range(first_epoch, ddpo_params.num_epochs):

        #################### SAMPLING ####################
        unet.eval()
        text_encoder.eval()
        with torch.no_grad():
            samples, frames = sample_from_model( ddpo_params, global_step, epoch,
                                        prompt_fn, opt.prompt_fn_kwargs,
                                        accelerator, pipeline, weight_dtype,
                                        validation_data, 
                                        output_dir, logger, adv_stat_tracker,
                                        kwargs['reward_fn'], loss_fn, kwargs['decode_frame'],
                                        sample_neg_prompt_embeds,
                                        eta=ddpo_params.sample_eta,
                                        num_rollouts = ddpo_params.sample_num_batches_per_epoch)

       

        ########################### TRAINING ##########################

        total_batch_size, num_timesteps = samples["timesteps"].shape
        for inner_epoch in range(ddpo_params.num_train_inner_epochs):
            # shuffle samples along batch dimension
            perm = torch.randperm(total_batch_size, device=accelerator.device)
            samples = {k: v[perm] for k, v in samples.items()}

            # shuffle along time dimension independently for each sample
            perms = torch.stack(
                [
                    torch.randperm(num_timesteps, device=accelerator.device)
                    for _ in range(total_batch_size)
                ]
            )
        
            for key in ["timesteps", "latents", "next_latents", "log_probs"]:
                samples[key] = samples[key][
                    torch.arange(total_batch_size, device=accelerator.device)[:, None],
                    perms,
                ]

            # rebatch for training
            samples_batched = {
                k: v.reshape(-1, ddpo_params.train_batch_size, *v.shape[1:])
                for k, v in samples.items()
            }
            # dict of lists -> list of dicts for easier iteration
            samples_batched = [
                dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())
            ]
            # train
            unet.train()
            info = defaultdict(list)

            for i, sample in tqdm(
                list(enumerate(samples_batched)),
                desc=f"Epoch {epoch}.{inner_epoch}: training",
                position=0,
                disable=not accelerator.is_local_main_process,
            ):
                embeds = torch.cat(
                        [train_neg_prompt_embeds, sample["prompt_embeds"]]
                    )

                for j in tqdm(
                    range(num_timesteps),
                    desc="Timestep",
                    position=1,
                    leave=False,
                    disable=not accelerator.is_local_main_process,
                ):
                    with accelerator.accumulate(unet) ,accelerator.accumulate(text_encoder):
                        with accelerator.autocast():
                            
                        
                            cur_timestep =  sample["timesteps"][:, j]
                            cur_latent = sample["latents"][:, j].permute(0,2,1,3,4)
                            latent_model_input = torch.cat([cur_latent] * 2)
                            latent_model_input = pipeline.scheduler.scale_model_input(latent_model_input, cur_timestep)

                            noise_pred = unet(
                                latent_model_input,
                                torch.cat([cur_timestep] * 2),  
                                encoder_hidden_states= embeds,
                            ).sample

                            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                            noise_pred = noise_pred_uncond + validation_data.guidance_scale * (noise_pred_text - noise_pred_uncond)
                            noise_pred = noise_pred.permute(0,2,1,3,4)
                            # compute the log prob of next_latents given latents under the current model
                            _, log_prob = ddim_step_with_logprob(
                                pipeline.scheduler,
                                noise_pred,
                                cur_timestep,
                                sample["latents"][:, j],
                                eta=ddpo_params.sample_eta,
                                prev_sample=sample["next_latents"][:, j],
                            )
                            log_prob = log_prob.reshape(train_batch_size, -1).mean(-1)

                        # ppo logic
                        advantages = torch.clamp(
                            sample["advantages"],
                            -ddpo_params.adv_clip_max,
                                ddpo_params.adv_clip_max,
                        )
                        ratio = torch.exp(log_prob - sample["log_probs"][:, j])
                        unclipped_loss = -advantages * ratio
                        clipped_loss = -advantages * torch.clamp(
                            ratio,
                            1.0 - ddpo_params.clip_range,
                            1.0 + ddpo_params.clip_range,
                        )
                        loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))
                        # John Schulman says that (ratio - 1) - log(ratio) is a better
                        # estimator, but most existing code uses this so...
                        # http://joschu.net/blog/kl-approx.html
                        info["approx_kl"].append(
                            0.5
                            * torch.mean((log_prob - sample["log_probs"][:, j]) ** 2)
                        )
                        info["clipfrac"].append(
                            torch.mean(
                                (
                                    torch.abs(ratio - 1.0) > ddpo_params.clip_range
                                ).float()
                            )
                        )
                        info["loss"].append(loss)

                        #backward pass
                        accelerator.backward(loss)
                        if accelerator.sync_gradients:
                            if any([train_text_encoder, use_text_lora]):
                                params_to_clip = list(unet.parameters()) + list(text_encoder.parameters())
                            else:
                                params_to_clip = list(unet.parameters())
                        
                            accelerator.clip_grad_norm_(params_to_clip, ddpo_params.max_grad_norm)
                            
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        
                    # Checks if the accelerator has performed an optimization step behind the scenes
                    if accelerator.sync_gradients:
                        assert (j == num_timesteps - 1)  
                        #and ( i + 1) % gradient_accumulation_steps == 0
                        #commented out the second line since using the text encoder too changes this logic
                        # log training-related stuff
                        info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                        info = accelerator.reduce(info, reduction="mean")
                        info.update({"epoch": epoch, "inner_epoch": inner_epoch})
                        accelerator.log(info, step=global_step)
                        global_step += 1
                        info = defaultdict(list)

                        if global_step % checkpointing_steps ==0:   
                            print("Saving checkpointing....")
                            save_pipe(
                                pretrained_model_path, 
                                global_step, 
                                accelerator, 
                                unet, 
                                text_encoder, 
                                vae, 
                                output_dir, 
                                lora_manager,
                                unet_lora_modules,
                                text_encoder_lora_modules,
                                is_checkpoint=True,
                                save_pretrained_model=False
                            )
                            print("Saving checkpoing end")
                
        # make sure we did an optimization step at the end of the inner epoch
        # assert accelerator.sync_gradients
        ########################## validation ##############################
        if should_sample(global_step, validation_steps, validation_data):
            print("Performing validation.")
            # print(accelerator.process_index)
            if accelerator.is_main_process:
                with accelerator.autocast():
                    import wandb
                    vis_dict = {}
                
                    unet.eval()
                    text_encoder.eval()
                    unet_and_text_g_c(unet, text_encoder, False, False)
                    lora_manager.deactivate_lora_train([unet, text_encoder], True)    

                    with torch.no_grad():
                        val_video_frames = pipeline(
                            val_prompt,
                            width=validation_data.width,
                            height=validation_data.height,
                            num_frames=validation_data.num_frames,
                            num_inference_steps=validation_data.num_inference_steps,
                            guidance_scale=validation_data.guidance_scale,
                            generator=torch.cuda.manual_seed(opt.seed),
                        ).frames
                    
                    for i in range(val_video_frames.shape[0]):
                        save_filename = f"{global_step}_dataset_{val_prompt[i]}_{i}"
                        out_file = f"{output_dir}/samples/{save_filename}.gif"
                       
                        video_frames_int = (val_video_frames[i] * 255).astype(np.uint8)
                        imageio.mimwrite(out_file, video_frames_int, duration=175, loop=0)
                        vis_dict[f"gen_video_{i}"] = wandb.Video(out_file, fps=2, format="gif")
                        logger.info(f"Saved a new sample to {out_file}")

                    accelerator.log(vis_dict, step=global_step) 
                    torch.cuda.empty_cache()
                    gc.collect()
                    
            unet_and_text_g_c(
                unet, 
                text_encoder, 
                gradient_checkpointing, 
                text_encoder_gradient_checkpointing
            )
            lora_manager.deactivate_lora_train([unet, text_encoder], False)    


        # if epoch != 0 and epoch % config.save_freq == 0 and accelerator.is_main_process:
        #     accelerator.save_state()
      
@hydra.main(config_path="config_t2v", config_name="config_dpo_ddpo")
def my_main(opt: DictConfig) -> None:
    main(**opt, opt=opt)


if __name__ == "__main__":
    my_main()
