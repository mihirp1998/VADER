# Build and modified from Open-Sora/scripts/inference.py
import os
import time
from pprint import pformat
import sys
sys.path.append('../')   # setting path to get Core and assets

import torch
import torch.distributed as dist
from pytorch_lightning import seed_everything
from tqdm import tqdm

from opensora.acceleration.parallel_states import set_sequence_parallel_group
from opensora.datasets import save_sample
from opensora.datasets.aspect import get_image_size, get_num_frames
from opensora.models.text_encoder.t5 import text_preprocessing
from opensora.registry import MODELS, SCHEDULERS, build_module
from opensora.utils.config_utils import parse_configs
from opensora.utils.inference_utils import (
    append_score_to_prompts,
    apply_mask_strategy,
    collect_references_batch,
    extract_json_from_prompts,
    extract_prompts_loop,
    load_prompts,
    merge_prompt,
    prepare_multi_resolution_info,
    split_prompt,
)
from opensora.utils.misc import create_logger, to_torch_dtype

import peft
import torchvision
from transformers import AutoProcessor, AutoModel
from Core.aesthetic_scorer import AestheticScorerDiff
from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
import hpsv2
import numpy as np
import bitsandbytes as bnb
from accelerate import Accelerator
from accelerate.utils import gather_object
import datetime
import torch.distributed as dist
import random
import logging
import gc
from PIL import Image
from opensora.acceleration.checkpoint import set_grad_checkpoint
# import ipdb
# st = ipdb.set_trace


def create_output_folders(output_dir, run_name):
    out_dir = os.path.join(output_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(f"{out_dir}/samples", exist_ok=True)
    return out_dir

def should_sample(global_step, validation_steps, is_sample_preview):
    return (global_step % validation_steps == 0 or global_step ==1)  \
    and is_sample_preview


def aesthetic_loss_fn(aesthetic_target=None,
                     grad_scale=0,
                     device=None,
                     torch_dtype=None):
    '''
    Args:
        aesthetic_target: float, the target value of the aesthetic score. it is 10 in this experiment
        grad_scale: float, the scale of the gradient. it is 0.1 in this experiment
        device: torch.device, the device to run the model. 
        torch_dtype: torch.dtype, the data type of the model.

    Returns:
        loss_fn: function, the loss function of the aesthetic reward function.
    '''
    target_size = (224, 224)
    normalize = torchvision.transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                                std=[0.26862954, 0.26130258, 0.27577711])
    
    scorer = AestheticScorerDiff(dtype=torch_dtype).to(device, dtype=torch_dtype)
    scorer.requires_grad_(False)

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
        return loss.mean() * grad_scale, rewards.mean()
    return loss_fn


def hps_loss_fn(inference_dtype=None, device=None, hps_version="v2.0"):
    '''
    Args:
        inference_dtype: torch.dtype, the data type of the model.
        device: torch.device, the device to run the model.
        hps_version: str, the version of the HPS model. It is "v2.0" or "v2.1" in this experiment.

    Returns:
        loss_fn: function, the loss function of the HPS reward function.
        '''
    model_name = "ViT-H-14"
    
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
    
    if hps_version == "v2.0":
        checkpoint_path = f"{os.path.expanduser('~')}/.cache/huggingface/hub/models--xswu--HPSv2/snapshots/697403c78157020a1ae59d23f111aa58ced35b0a/HPS_v2_compressed.pt"
    else:   # hps_version == "v2.1"
        checkpoint_path = f"{os.path.expanduser('~')}/.cache/huggingface/hub/models--xswu--HPSv2/snapshots/697403c78157020a1ae59d23f111aa58ced35b0a/HPS_v2.1_compressed.pt"
    
    # force download of model via score
    hpsv2.score([], "", hps_version=hps_version)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    tokenizer = get_tokenizer(model_name)
    model = model.to(device, dtype=inference_dtype)
    model.eval()

    target_size =  (224, 224)
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
        return  loss.mean(), scores.mean()
    
    return loss_fn

def aesthetic_hps_loss_fn(aesthetic_target=None,
                     grad_scale=0,
                     inference_dtype=None, 
                     device=None, 
                     hps_version="v2.1"):
    '''
    Args:
        aesthetic_target: float, the target value of the aesthetic score. it is 10 in this experiment
        grad_scale: float, the scale of the gradient. it is 0.1 in this experiment
        inference_dtype: torch.dtype, the data type of the model.
        device: torch.device, the device to run the model.
        hps_version: str, the version of the HPS model. It is "v2.0" or "v2.1" in this experiment.

    Returns:
        loss_fn: function, the loss function of a combination of aesthetic and HPS reward function.
    '''
    # HPS
    model_name = "ViT-H-14"
    
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
    
    if hps_version == "v2.0":
        checkpoint_path = f"{os.path.expanduser('~')}/.cache/huggingface/hub/models--xswu--HPSv2/snapshots/697403c78157020a1ae59d23f111aa58ced35b0a/HPS_v2_compressed.pt"
    else:
        print("=================== HPS v2.1 ===================")
        checkpoint_path = f"{os.path.expanduser('~')}/.cache/huggingface/hub/models--xswu--HPSv2/snapshots/697403c78157020a1ae59d23f111aa58ced35b0a/HPS_v2.1_compressed.pt"
    # force download of model via score
    hpsv2.score([], "", hps_version=hps_version)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    tokenizer = get_tokenizer(model_name)
    model = model.to(device, dtype=inference_dtype)
    model.eval()

    target_size =  (224, 224)
    normalize = torchvision.transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                                std=[0.26862954, 0.26130258, 0.27577711])
    # Aesthetic
    scorer = AestheticScorerDiff(dtype=inference_dtype).to(device, dtype=inference_dtype)
    scorer.requires_grad_(False)
    
    def loss_fn(im_pix_un, prompts):
        # Aesthetic
        im_pix = ((im_pix_un / 2) + 0.5).clamp(0, 1)
        im_pix = torchvision.transforms.Resize(target_size)(im_pix)
        im_pix = normalize(im_pix).to(im_pix_un.dtype)

        aesthetic_rewards = scorer(im_pix)
        if aesthetic_target is None: # default maximization
            aesthetic_loss = -1 * aesthetic_rewards
        else:
            # using L1 to keep on same scale
            aesthetic_loss = abs(aesthetic_rewards - aesthetic_target)
        aesthetic_loss = aesthetic_loss.mean() * grad_scale
        aesthetic_rewards = aesthetic_rewards.mean()

        # HPS
        caption = tokenizer(prompts)
        caption = caption.to(device)
        outputs = model(im_pix, caption)
        image_features, text_features = outputs["image_features"], outputs["text_features"]
        logits = image_features @ text_features.T
        scores = torch.diagonal(logits)
        hps_loss = abs(1.0 - scores)
        hps_loss = hps_loss.mean()
        hps_rewards = scores.mean()

        loss = (1.5 * aesthetic_loss + hps_loss) /2  # 1.5 is a hyperparameter. Set it to 1.5 because experimentally hps_loss is 1.5 times larger than aesthetic_loss
        rewards = (aesthetic_rewards + 15 * hps_rewards) / 2    # 15 is a hyperparameter. Set it to 15 because experimentally aesthetic_rewards is 15 times larger than hps_reward
        return loss, rewards
    
    return loss_fn

def pick_score_loss_fn(inference_dtype=None, device=None):
    '''
    Args:
        inference_dtype: torch.dtype, the data type of the model.
        device: torch.device, the device to run the model.

    Returns:
        loss_fn: function, the loss function of the PickScore reward function.
    '''
    processor_name_or_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    model_pretrained_name_or_path = "yuvalkirstain/PickScore_v1"
    processor = AutoProcessor.from_pretrained(processor_name_or_path, torch_dtype=inference_dtype)
    model = AutoModel.from_pretrained(model_pretrained_name_or_path, torch_dtype=inference_dtype).eval().to(device)
    model.requires_grad_(False)

    def loss_fn(im_pix_un, prompts):    # im_pix_un: b,c,h,w
        im_pix = ((im_pix_un / 2) + 0.5).clamp(0, 1)

        # reproduce the pick_score preprocessing
        im_pix = im_pix * 255   # b,c,h,w
        # if height is less than width
        if im_pix.shape[2] < im_pix.shape[3]:
            height = 224
            width = im_pix.shape[3] * height // im_pix.shape[2]    # keep the aspect ratio, so the width is w * 224/h
        else:
            width = 224
            height = im_pix.shape[2] * width // im_pix.shape[3]    # keep the aspect ratio, so the height is h * 224/w
        # interpolation and antialiasing should be the same as below
        im_pix = torchvision.transforms.Resize((height, width), 
                                               interpolation=torchvision.transforms.InterpolationMode.BICUBIC, 
                                               antialias=True)(im_pix)
        im_pix = im_pix.permute(0, 2, 3, 1)  # b,c,h,w -> (b,h,w,c)
        # crop the center 224x224
        startx = width//2 - (224//2)
        starty = height//2 - (224//2)
        im_pix = im_pix[:, starty:starty+224, startx:startx+224, :]
        # do rescale and normalize as CLIP
        im_pix = im_pix * 0.00392156862745098   # rescale factor
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(device)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(device)
        im_pix = (im_pix - mean) / std
        im_pix = im_pix.permute(0, 3, 1, 2)  # BHWC -> BCHW
        
        text_inputs = processor(
            text=prompts,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(device)

        
        # embed
        image_embs = model.get_image_features(pixel_values=im_pix)
        image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
    
        text_embs = model.get_text_features(**text_inputs)
        text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)
    
        # score
        scores = model.logit_scale.exp() * (text_embs @ image_embs.T)[0]
        loss = abs(1.0 - scores / 100.0)
        return loss.mean(), scores.mean()
    
    return loss_fn

def main():
    # ======================================================
    # configs & runtime variables
    # ======================================================
    # == parse configs ==
    cfg = parse_configs(training=False) # this setting is irrelevant to the vader training process

    # == device and dtype ==
    cfg_dtype = cfg.get("dtype", "bf16")
    assert cfg_dtype in ["fp16", "bf16", "fp32"], f"Unknown mixed precision {cfg_dtype}"
    dtype = to_torch_dtype(cfg.get("dtype", "bf16"))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_grad_enabled(cfg.get("is_vader_training", False))   # enable vader training!!!

    # == init distributed env ==
    enable_sequence_parallelism = False
    seed_everything(cfg.get("seed", 1024))

    accelerator = Accelerator(                                                  # Initialize Accelerator
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 1),
        mixed_precision=cfg.get("mixed_precision", "fp16"),
        log_with=cfg.get("logger_type", "wandb"),
        project_dir=cfg.get("project_dir", "./project_dir")
    )
    output_dir = cfg.get("project_dir", "./project_dir")
    device = accelerator.device

    validation_steps = cfg.get("validation_steps", 1) * cfg.get("gradient_accumulation_steps", 1)         # number of steps to run validation for
    checkpointing_steps = cfg.get("checkpointing_steps", 1) * cfg.get("gradient_accumulation_steps", 1)   # Saves a model every nth step.

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if cfg.get("use_wandb", False):
        if accelerator.is_main_process:
            import wandb
            wandb_args = {}
            
            if cfg.get("wandb_entity", '') != '':
                wandb_args['entity'] =  cfg.get("wandb_entity", '')   
            
            if cfg.get("debug", False):
                wandb_args['mode'] = "disabled"
            
            opt_dict = vars(cfg)   # convert args to dict
            accelerator.init_trackers("VADER-OpenSora", config=opt_dict, init_kwargs={"wandb": wandb_args})
            output_dir = create_output_folders(cfg.get("project_dir", "./project_dir"), wandb.run.name)    # all processes will create the same output folder
            # convert output_dir to broadcastable tensor, so that it can be broadcasted to all processes
            output_dir_broadcast = [output_dir]
            
        else:
            output_dir_broadcast = [None]
        
        # convert output_dir back to str
        dist.broadcast_object_list(output_dir_broadcast, src=0)    # broadcast the output_dir to all processes
        output_dir = output_dir_broadcast[0]
        print(f"+++++++++++++++++++output_dir: {output_dir}+++++++++++++++++++++++++++++++++")

    # == init logger ==
    logger = create_logger()
    logger.info("Vader configuration:\n %s", pformat(cfg.to_dict()))
    verbose = cfg.get("verbose", 1)
    progress_wrap = tqdm if verbose == 1 else (lambda x: x)

    # ======================================================
    # build model & load weights
    # ======================================================
    logger.info("Building models...")
    # == build text-encoder and vae ==
    text_encoder = build_module(cfg.text_encoder, MODELS, device=device)
    vae = build_module(cfg.vae, MODELS).to(device, dtype).eval()

    # == prepare video size ==
    image_size = cfg.get("image_size", None)
    if image_size is None:
        resolution = cfg.get("resolution", None)
        aspect_ratio = cfg.get("aspect_ratio", None)
        assert (
            resolution is not None and aspect_ratio is not None
        ), "resolution and aspect_ratio must be provided if image_size is not provided"
        image_size = get_image_size(resolution, aspect_ratio)
    num_frames = get_num_frames(cfg.num_frames)

    # == build diffusion model ==
    input_size = (num_frames, *image_size)
    latent_size = vae.get_latent_size(input_size)
    model = (
        build_module(
            cfg.model,
            MODELS,
            input_size=latent_size,
            in_channels=vae.out_channels,
            caption_channels=text_encoder.output_dim,
            model_max_length=text_encoder.model_max_length,
            enable_sequence_parallelism=enable_sequence_parallelism,
        )
        .to(device)
        .eval()
    )
    text_encoder.y_embedder = model.y_embedder  # HACK: for classifier-free guidance

    # == build scheduler ==
    scheduler = build_module(cfg.scheduler, SCHEDULERS)

    # == load weights ==
    # add LoRA using peft
    config = peft.LoraConfig(
            r=cfg.get("lora_rank", 16),
            target_modules=["qkv", "q_linear", "kv_linear"],        # only diffusion_model has these modules
            lora_dropout=0.01,
        )

    peft_model = peft.get_peft_model(model, config)
    peft_model.print_trainable_parameters()
    vae.requires_grad_(False)

    if cfg.get("grad_checkpoint", False):
        set_grad_checkpoint(model)

    # load the pretrained LoRA model
    if cfg.get("lora_ckpt_path", None) is not None:
        # load the pretrained LoRA model
        peft.set_peft_model_state_dict(peft_model, torch.load(cfg.get("lora_ckpt_path", None)))
    

    # ========= inference only mode starts. Skip this part if it is training =============
    if not cfg.get("is_vader_training", False): # if it is inference only mode
        
        peft_model = accelerator.prepare(peft_model)

        # == load prompts ==
        prompts = cfg.get("prompt", None)
        start_idx = cfg.get("start_index", 0)
        if prompts is None:
            if cfg.get("prompt_path", None) is not None:
                prompts = load_prompts(cfg.prompt_path, start_idx, cfg.get("end_index", None))
            else:
                prompts = [cfg.get("prompt_generator", "")] * 1_000_000  # endless loop

        # == prepare reference ==
        reference_path = cfg.get("reference_path", [""] * len(prompts))
        mask_strategy = cfg.get("mask_strategy", [""] * len(prompts))
        assert len(reference_path) == len(prompts), "Length of reference must be the same as prompts"
        assert len(mask_strategy) == len(prompts), "Length of mask_strategy must be the same as prompts"

        # == prepare arguments ==
        fps = cfg.fps
        save_fps = cfg.get("save_fps", fps // cfg.get("frame_interval", 1))
        multi_resolution = cfg.get("multi_resolution", None)
        loop = 1   # loop = 1 for vader inference
        align = cfg.get("align", None)



        # show the progress bar.
        global_step = 0
        progress_bar = tqdm(range(global_step, cfg.get('val_batch_size', 1) * accelerator.num_processes * cfg.get('num_val_runs', 1)))
        progress_bar.set_description("Inference Steps")
        
        ## Inference Step 5: generate new validation videos
        with torch.no_grad():
            vis_dict = {}

            # video validation loop
            for n in range(cfg.get('num_val_runs', 1)):
                # set random seed for each process
                random.seed(cfg.get("seed", 1024) + n)
                torch.manual_seed(cfg.get("seed", 1024) + n)

                # prepare batch prompts
                prompt_idx = random.sample(range(len(prompts)), cfg.get('val_batch_size', 1) * accelerator.num_processes)
                
                with accelerator.split_between_processes(prompt_idx) as val_idx:
                    val_prompt = [prompts[i] for i in val_idx]
                    val_refs = [mask_strategy[i] for i in val_idx]
                    val_ms = [reference_path[i] for i in val_idx]

                    assert len(val_prompt) == cfg.get("val_batch_size", 1)
                    # store output of generations in dict
                    results=dict(filenames=[],dir_name=[], prompt=[], gpu_no=[])

                    # Step 5.2.1: forward pass
                    # == get json from prompts ==
                    val_prompt, val_refs, val_ms = extract_json_from_prompts(val_prompt, val_refs, val_ms)

                    # == get reference for condition ==
                    val_refs = collect_references_batch(val_refs, vae, image_size)

                    # == multi-resolution info ==
                    model_args = prepare_multi_resolution_info(
                        multi_resolution, len(val_prompt), image_size, num_frames, fps, device, dtype
                    ) 

                    # == process prompts step by step ==
                    # 1. split prompt
                    # each element in the list is [prompt_segment_list, loop_idx_list]
                    batched_prompt_segment_list = []
                    batched_loop_idx_list = []
                    for prompt in val_prompt:
                        prompt_segment_list, loop_idx_list = split_prompt(prompt)
                        batched_prompt_segment_list.append(prompt_segment_list)
                        batched_loop_idx_list.append(loop_idx_list)

                    # 2. append score
                    for idx, prompt_segment_list in enumerate(batched_prompt_segment_list):
                        batched_prompt_segment_list[idx] = append_score_to_prompts(
                            prompt_segment_list,
                            aes=cfg.get("aes", None),
                            flow=cfg.get("flow", None),
                            camera_motion=cfg.get("camera_motion", None),
                        )

                    # 3. clean prompt with T5
                    for idx, prompt_segment_list in enumerate(batched_prompt_segment_list):
                        batched_prompt_segment_list[idx] = [text_preprocessing(prompt) for prompt in prompt_segment_list]

                    # 4. merge to obtain the final prompt
                    batch_prompts = []
                    for prompt_segment_list, loop_idx_list in zip(batched_prompt_segment_list, batched_loop_idx_list):
                        batch_prompts.append(merge_prompt(prompt_segment_list, loop_idx_list))

                    # == Iter over loop generation ==
                    assert loop == 1, "Only support loop=1 for training"
                    batch_prompts_loop = extract_prompts_loop(batch_prompts, loop)  # loop = 1 for vader training


                    with accelerator.autocast():            # mixed precision
                        # == sampling ==
                        z = torch.randn(len(batch_prompts), vae.out_channels, *latent_size, device=device, dtype=dtype)
                        masks = apply_mask_strategy(z, val_refs, val_ms, loop, align=align) # loop = 1 for vader training

                        samples = scheduler.sample(
                            peft_model.module,
                            text_encoder,
                            z=z,
                            prompts=batch_prompts_loop,
                            device=device,
                            additional_args=model_args,
                            progress=verbose >= 2,
                            mask=masks,
                            backprop_mode=cfg.get("backprop_mode", None),
                        )

                        samples = vae.decode(samples.to(dtype), num_frames=num_frames)         # samples: batch, c, t, h, w
                        
                    dir_name = os.path.join(output_dir, "samples", f"step_{global_step}")
                    # filenames should be related to the gpu index
                    filenames = [f"{n}_{accelerator.local_process_index}_{id+1:04d}" for id in range(len(batch_prompts))] # from 0 to batch size, n is the index of the batch
                    # if dir_name is not exists, create it
                    os.makedirs(dir_name, exist_ok=True)

                    for idx, batch_prompt in enumerate(batch_prompts):
                        save_path = os.path.join(dir_name, f"{filenames[idx]}")
                        video = samples[idx]
                        save_path = save_sample(
                            video,
                            fps=save_fps,
                            save_path=save_path,
                            verbose=verbose >= 2,
                        )

                    results["filenames"].extend(filenames)
                    results["dir_name"].extend([dir_name]*len(filenames))
                    results["prompt"].extend(batch_prompts)
                    results=[ results ] # transform to list, otherwise gather_object() will not collect correctly
                
                # collect inference results from all the GPUs
                results_gathered=gather_object(results)

                if accelerator.is_main_process:
                    filenames = []
                    dir_name = []
                    temp_prompts = []
                    for i in range(len(results_gathered)):
                        filenames.extend(results_gathered[i]["filenames"])
                        dir_name.extend(results_gathered[i]["dir_name"])
                        temp_prompts.extend(results_gathered[i]["prompt"])
                    # upload the video and their corresponding prompt to wandb
                    if cfg.get("use_wandb", False):
                        for i, filename in enumerate(filenames):
                            video_path = os.path.join(dir_name[i], f"{filename}.mp4")
                            vis_dict[f"{n}_sample_{i}"] = wandb.Video(video_path, fps=save_fps, caption=temp_prompts[i])

                    accelerator.log(vis_dict, step=global_step)
                    logger.info("Validation sample saved!")

                # release the memory of validation process
                del samples
                torch.cuda.empty_cache()
                gc.collect()

        return  # inference only mode does not need to train
    # ========================= inference only ends =============================

    
    # step 2.2: optimizer and loss function
    if cfg.get("use_AdamW8bit", False):
        optimizer = bnb.optim.AdamW8bit(peft_model.parameters(), lr=cfg.get("lr", 0.0001))
    else:
        optimizer = torch.optim.AdamW(peft_model.parameters(), lr=cfg.get("lr", 0.0001))

    if cfg.get("reward_fn", "aesthetic")== "aesthetic":
        loss_fn = aesthetic_loss_fn(grad_scale=0.1,
                                    aesthetic_target=10,
                                    torch_dtype = peft_model.dtype,
                                    device = accelerator.device)
    elif cfg.get("reward_fn", "aesthetic") == "hps":
        loss_fn = hps_loss_fn(dtype, accelerator.device, hps_version=cfg.get("hps_version", "v2.1"))
    elif cfg.get("reward_fn", "aesthetic") == "aesthetic_hps":
        # if use mixed precision, the dtype of peft_model is torch.float32
        # while the dtype of peft_model.model.first_stage_model is torch.float16
        loss_fn = aesthetic_hps_loss_fn(aesthetic_target=10,
                                        grad_scale=0.1,
                                        inference_dtype=dtype,
                                        device=accelerator.device,
                                        hps_version=cfg.get("hps_version", "v2.1"))
    elif cfg.get("reward_fn", "aesthetic") == "pick_score":
        loss_fn = pick_score_loss_fn(peft_model.dtype, accelerator.device)
    else:
        raise NotImplementedError(f"Reward function {cfg.get('reward_fn', 'aesthetic')} is not implemented for Open-Sora.")
        
    
    # accelerate the model
    peft_model, optimizer, loss_fn, text_encoder, vae = accelerator.prepare(peft_model, optimizer, loss_fn, text_encoder, vae)
        
    # ======================================================
    # inference
    # ======================================================
    # == load prompts ==
    prompts = cfg.get("prompt", None)
    start_idx = cfg.get("start_index", 0)
    if prompts is None:
        if cfg.get("prompt_path", None) is not None:
            prompts = load_prompts(cfg.prompt_path, start_idx, cfg.get("end_index", None))
        else:
            prompts = [cfg.get("prompt_generator", "")] * 1_000_000  # endless loop

    # == prepare reference ==
    reference_path = cfg.get("reference_path", [""] * len(prompts))
    mask_strategy = cfg.get("mask_strategy", [""] * len(prompts))
    assert len(reference_path) == len(prompts), "Length of reference must be the same as prompts"
    assert len(mask_strategy) == len(prompts), "Length of mask_strategy must be the same as prompts"

    # == prepare arguments ==
    fps = cfg.fps
    save_fps = cfg.get("save_fps", fps // cfg.get("frame_interval", 1))
    multi_resolution = cfg.get("multi_resolution", None)
    loop = 1   # loop = 1 for vader training
    condition_frame_length = cfg.get("condition_frame_length", 5)
    condition_frame_edit = cfg.get("condition_frame_edit", 0.0)
    align = cfg.get("align", None)

    sample_name = cfg.get("sample_name", None)
    prompt_as_path = cfg.get("prompt_as_path", False)


    # == training ==

    ## step 4: run training over samples
    ## -----------------------------------------------------------------
    # Train!
    total_batch_size = cfg.get("train_batch_size", 1) * accelerator.num_processes * cfg.get("gradient_accumulation_steps", 1)

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {cfg.get('num_train_epochs', 200)}")
    logger.info(f"  Instantaneous batch size per device = {cfg.get('train_batch_size', 1)}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {cfg.get('gradient_accumulation_steps', 1)}")
    logger.info(f"  Total optimization steps = {cfg.get('max_train_steps', 10000)}")
    
    first_epoch = 0
    global_step = 0

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, min(cfg.get("max_train_steps", 10000), cfg.get("num_train_epochs", 200) * 100 * cfg.get("train_batch_size", 1))))
    progress_bar.set_description("Training Steps")
    start = time.time()
    for epoch in range(first_epoch, cfg.get("num_train_epochs", 200)):
        default_iterations = 100
        for idx, step in enumerate(range(default_iterations)):
            # trainng step 1: randomize training process
            random.seed(datetime.datetime.now().microsecond)
            torch.manual_seed(datetime.datetime.now().microsecond)

            # trainng step 2: prepare the input
            if global_step % cfg.get("gradient_accumulation_steps", 1) == 0:
                train_loss_list = []    # this is for gradient accumulation
            
            # prepare batch prompts
            prompt_idx = random.sample(range(len(prompts)), cfg.get("train_batch_size", 1))
            batch_prompts = [prompts[i] for i in prompt_idx]    # prompt
            ms = [mask_strategy[i] for i in prompt_idx]         # mask_strategy
            refs = [reference_path[i] for i in prompt_idx]      # references

            batch_size = len(batch_prompts)
            
            # == get json from prompts ==
            batch_prompts, refs, ms = extract_json_from_prompts(batch_prompts, refs, ms)

            # == get reference for condition ==
            refs = collect_references_batch(refs, vae, image_size)

            # == multi-resolution info ==
            model_args = prepare_multi_resolution_info(
                multi_resolution, len(batch_prompts), image_size, num_frames, fps, device, dtype
            ) 

            # == process prompts step by step ==
            # 1. split prompt
            # each element in the list is [prompt_segment_list, loop_idx_list]
            batched_prompt_segment_list = []
            batched_loop_idx_list = []
            for prompt in batch_prompts:
                prompt_segment_list, loop_idx_list = split_prompt(prompt)
                batched_prompt_segment_list.append(prompt_segment_list)
                batched_loop_idx_list.append(loop_idx_list)

            # 2. append score
            for idx, prompt_segment_list in enumerate(batched_prompt_segment_list):
                batched_prompt_segment_list[idx] = append_score_to_prompts(
                    prompt_segment_list,
                    aes=cfg.get("aes", None),
                    flow=cfg.get("flow", None),
                    camera_motion=cfg.get("camera_motion", None),
                )

            # 3. clean prompt with T5
            for idx, prompt_segment_list in enumerate(batched_prompt_segment_list):
                batched_prompt_segment_list[idx] = [text_preprocessing(prompt) for prompt in prompt_segment_list]

            # 4. merge to obtain the final prompt
            batch_prompts = []
            for prompt_segment_list, loop_idx_list in zip(batched_prompt_segment_list, batched_loop_idx_list):
                batch_prompts.append(merge_prompt(prompt_segment_list, loop_idx_list))

            # == Iter over loop generation ==
            assert loop == 1, "Only support loop=1 for vader training"
            batch_prompts_loop = extract_prompts_loop(batch_prompts, loop)  # loop = 1 for vader training


            with accelerator.accumulate(peft_model):    # gradient accumulation
                with accelerator.autocast():            # mixed precision
                    # == sampling ==
                    z = torch.randn(len(batch_prompts), vae.out_channels, *latent_size, device=device, dtype=dtype)
                    masks = apply_mask_strategy(z, refs, ms, loop, align=align) # loop = 1 for vader training

                    
                    samples = scheduler.sample(
                        peft_model.module,
                        text_encoder,
                        z=z,
                        prompts=batch_prompts_loop,
                        device=device,
                        additional_args=model_args,
                        progress=verbose >= 2,
                        mask=masks,
                        backprop_mode=cfg.get("backprop_mode", None),
                        use_grad_checkpoint=cfg.get("grad_checkpoint", False),
                    )

                    if cfg.get("backprop_mode", None) is not None:   # it is for training now. Use one frame randomly to save memory
                        decode_frame = cfg.get("decode_frame", '-1')
                        try:
                            decode_frame=int(decode_frame)
                            #it's a int
                        except:
                            pass
                        if type(decode_frame) == int:
                            frame_index = random.randint(0,samples.shape[2]-1) if decode_frame == -1 else decode_frame        # samples: batch, c, t, h, w
                            samples = vae.decode(samples[:,:,frame_index:frame_index+1,:,:].to(dtype), num_frames=1)
                        elif decode_frame in ['alt', 'all']:
                            idxs = range(0, samples.shape[2], 2) if decode_frame == 'alt' else range(samples.shape[2])
                            samples = vae.decode(samples[:,:,idxs,:,:].to(dtype), num_frames=len(idxs))         # samples: batch, c, t, h, w

                    # == calculate the loss ==
                    video_frames_ = samples.permute(0,2,1,3,4)      # batch,channels,frames,height,width >> b,f,c,h,w
                    bs, nf, c_, h_, w_ = video_frames_.shape
                    assert nf == 1                                  # reward should only be on single frame for training, we only support decode_frame = -1
                    video_frames_ = video_frames_.squeeze(1)        # b,f,c,h,w >> b,c,h,w
                    video_frames_ = video_frames_.to(peft_model.module.dtype)
                    
                    # batch_prompts_loop ends with aesthetic score: rabbit in a black dress aesthetic score: 6.5. We need to remove the aesthetic score
                    pure_prompts = [prompt.split("aesthetic score:")[0].strip() for prompt in batch_prompts_loop]
                    
                    if cfg.get("reward_fn", "aesthetic") == "aesthetic" or cfg.get("reward_fn", "aesthetic") == "objectDetection" or cfg.get("reward_fn", "aesthetic") == "compression_score":
                        loss, rewards = loss_fn(video_frames_)      # video_frames_ in range [-1, 1]
                        
                    else:   # 'hps' or 'pick_score' or 'aesthetic_hps'
                        loss, rewards = loss_fn(video_frames_,pure_prompts) 
            
                # Gather the losses across all processes for logging (if we use distributed training).
                accum_loss = accelerator.gather(loss.repeat(cfg.get("train_batch_size", 1))) # loss.repeat() is to get the total loss for each sample in the batch
                avg_loss = accum_loss.mean()
                train_loss_list.append(avg_loss.item())

                # Step 4.2 backpropagation
                accelerator.backward(loss)

                if cfg.get("max_grad_norm", 1) > 0:  # gradient clipping is to prevent exploding gradients
                    if accelerator.sync_gradients:
                         # Unscale the gradients before clipping if using mixed precision
                        if accelerator.scaler is not None:
                            accelerator.scaler.unscale_(optimizer)
                        accelerator.clip_grad_norm_(peft_model.parameters(), cfg.get("max_grad_norm", 1))
                

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                # Step 4.3 logging and save checkpoint
                if accelerator.sync_gradients:
                    global_step += 1
                    log_dict_step = {"step_loss": loss.detach().item(), "step_reward": rewards.detach().item()}
                    accelerator.log(log_dict_step, step=global_step)
                    progress_bar.update(1)

                    if global_step % checkpointing_steps ==0:   
                        logger.info("Saving checkpointing....")
                        accelerator.wait_for_everyone()
                        if accelerator.is_local_main_process:
                            unwrapped_model = accelerator.unwrap_model(peft_model)
                            # save lora model only
                            peft_state_dict = peft.get_peft_model_state_dict(unwrapped_model)
                            peft_model_path = os.path.join(output_dir, f"peft_model_{global_step}.pt")
                            torch.save(peft_state_dict, peft_model_path)
                        logger.info("Saving checkpoing end")

                if global_step % cfg.get("gradient_accumulation_steps", 1) == 0:
                    avg_loss = sum(train_loss_list)/len(train_loss_list)
                    accelerator.log({"avg_loss": avg_loss}, step=global_step)
                
                if global_step >= cfg.get("max_train_steps", 10000):
                    break

            ## Step 5: Validation and save videos
            if should_sample(global_step, validation_steps, cfg.get("is_sample_preview", False)):
                ## 5.1 save the training sample
                if accelerator.is_local_main_process:
                    with torch.no_grad():
                        vis_dict = {}
                        if cfg.get("use_wandb", False):
                            # save as image
                            train_gen_frames = Image.fromarray((((video_frames_ + 1.0)/2.0)[0].permute(1,2,0).detach().cpu().clamp(0, 1).numpy() * 255).astype(np.uint8))
                            gen_image = wandb.Image(train_gen_frames, caption=pure_prompts[0])
                            vis_dict["gen image (train)"] = gen_image

                    accelerator.log(vis_dict, step=global_step)
                    logger.info("Training sample saved!")
                
                # release the memory
                del samples, video_frames_
                torch.cuda.empty_cache()
                gc.collect()

                ## Step 5.2: generate new validation videos
                with torch.no_grad():
                    vis_dict = {}
                    random.seed(cfg.get("seed", 1024))  # make sure the validation samples are the same for each epoch in order to compare the results
                    torch.manual_seed(cfg.get("seed", 1024))

                    # video validation loop
                    for n in range(cfg.get("num_val_runs", 1)):
                        prompt_idx = random.sample(range(len(prompts)), cfg.get("val_batch_size", 1) * accelerator.num_processes)

                        with accelerator.split_between_processes(prompt_idx) as val_idx:

                            val_prompt = [prompts[i] for i in val_idx]
                            val_refs = [mask_strategy[i] for i in val_idx]
                            val_ms = [reference_path[i] for i in val_idx]

                            assert len(val_prompt) == cfg.get("val_batch_size", 1)
                            # store output of generations in dict
                            results=dict(filenames=[],dir_name=[], prompt=[], gpu_no=[])

                            # Step 5.2.1: forward pass
                            # == get json from prompts ==
                            val_prompt, val_refs, val_ms = extract_json_from_prompts(val_prompt, val_refs, val_ms)

                            # == get reference for condition ==
                            val_refs = collect_references_batch(val_refs, vae, image_size)

                            # == multi-resolution info ==
                            model_args = prepare_multi_resolution_info(
                                multi_resolution, len(val_prompt), image_size, num_frames, fps, device, dtype
                            ) 

                            # == process prompts step by step ==
                            # 1. split prompt
                            # each element in the list is [prompt_segment_list, loop_idx_list]
                            batched_prompt_segment_list = []
                            batched_loop_idx_list = []
                            for prompt in val_prompt:
                                prompt_segment_list, loop_idx_list = split_prompt(prompt)
                                batched_prompt_segment_list.append(prompt_segment_list)
                                batched_loop_idx_list.append(loop_idx_list)

                            # 2. append score
                            for idx, prompt_segment_list in enumerate(batched_prompt_segment_list):
                                batched_prompt_segment_list[idx] = append_score_to_prompts(
                                    prompt_segment_list,
                                    aes=cfg.get("aes", None),
                                    flow=cfg.get("flow", None),
                                    camera_motion=cfg.get("camera_motion", None),
                                )

                            # 3. clean prompt with T5
                            for idx, prompt_segment_list in enumerate(batched_prompt_segment_list):
                                batched_prompt_segment_list[idx] = [text_preprocessing(prompt) for prompt in prompt_segment_list]

                            # 4. merge to obtain the final prompt
                            batch_prompts = []
                            for prompt_segment_list, loop_idx_list in zip(batched_prompt_segment_list, batched_loop_idx_list):
                                batch_prompts.append(merge_prompt(prompt_segment_list, loop_idx_list))

                            # == Iter over loop generation ==
                            assert loop == 1, "Only support loop=1 for training"
                            batch_prompts_loop = extract_prompts_loop(batch_prompts, loop)  # loop = 1 for vader training


                            with accelerator.autocast():            # mixed precision
                                # == sampling ==
                                z = torch.randn(len(batch_prompts), vae.out_channels, *latent_size, device=device, dtype=dtype)
                                masks = apply_mask_strategy(z, val_refs, val_ms, loop, align=align) # loop = 1 for vader training

                                
                                samples = scheduler.sample(
                                    peft_model.module,
                                    text_encoder,
                                    z=z,
                                    prompts=batch_prompts_loop,
                                    device=device,
                                    additional_args=model_args,
                                    progress=verbose >= 2,
                                    mask=masks,
                                    backprop_mode=cfg.get("backprop_mode", None),
                                )

                                samples = vae.decode(samples.to(dtype), num_frames=num_frames)         # samples: batch, c, t, h, w
                                
                            dir_name = os.path.join(output_dir, "samples", f"step_{global_step}")
                            # filenames should be related to the gpu index
                            filenames = [f"{n}_{accelerator.local_process_index}_{id+1:04d}" for id in range(len(batch_prompts))] # from 0 to batch size, n is the index of the batch
                            # if dir_name is not exists, create it
                            os.makedirs(dir_name, exist_ok=True)

                            for idx, batch_prompt in enumerate(batch_prompts):
                                save_path = os.path.join(dir_name, f"{filenames[idx]}")
                                video = samples[idx]
                                save_path = save_sample(
                                    video,
                                    fps=save_fps,
                                    save_path=save_path,
                                    verbose=verbose >= 2,
                                )

                            results["filenames"].extend(filenames)
                            results["dir_name"].extend([dir_name]*len(filenames))
                            results["prompt"].extend(batch_prompts)
                            results=[ results ] # transform to list, otherwise gather_object() will not collect correctly
                        
                        # collect inference results from all the GPUs
                        results_gathered=gather_object(results)
                        # accelerator.wait_for_everyone() # wait for all processes to finish saving the videos
                        if accelerator.is_main_process:
                            filenames = []
                            dir_name = []
                            temp_prompts = []
                            for i in range(len(results_gathered)):
                                filenames.extend(results_gathered[i]["filenames"])
                                dir_name.extend(results_gathered[i]["dir_name"])
                                temp_prompts.extend(results_gathered[i]["prompt"])
                            # upload the video and their corresponding prompt to wandb
                            if cfg.get("use_wandb", False):
                                for i, filename in enumerate(filenames):
                                    video_path = os.path.join(dir_name[i], f"{filename}.mp4")
                                    vis_dict[f"{n}_sample_{i}"] = wandb.Video(video_path, fps=save_fps, caption=temp_prompts[i])

                            accelerator.log(vis_dict, step=global_step)
                            logger.info("Validation sample saved!")

                        # release the memory of validation process
                        del samples
                        torch.cuda.empty_cache()
                        gc.collect()
                        
    logger.info("Vader finished.")


if __name__ == "__main__":
    main()
