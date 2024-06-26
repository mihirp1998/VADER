import argparse, os, sys, glob, yaml, math, random
import datetime, time
import numpy as np
from omegaconf import OmegaConf
from collections import OrderedDict
from tqdm import trange, tqdm
from einops import repeat
from einops import rearrange, repeat
from functools import partial
import torch
from pytorch_lightning import seed_everything

from funcs import load_model_checkpoint, load_prompts, load_image_batch, get_filelist, save_videos
from funcs import batch_ddim_sampling
from utils.utils import instantiate_from_config

import peft
import torchvision
from transformers.utils import ContextManagers
from transformers import AutoProcessor, AutoModel, AutoImageProcessor, AutoModelForObjectDetection
from aesthetic_scorer import AestheticScorerDiff
from weather_scorer import WeatherScorer
from compression_scorer import JpegCompressionScorer, jpeg_compressibility
from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
import hpsv2
import prompts as prompts_file
import bitsandbytes as bnb
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import gather_object
import torch.distributed as dist
import logging
import gc
from PIL import Image
import ipdb
st = ipdb.set_trace


logger = get_logger(__name__, log_level="INFO") # get logger for current module

def create_logging(logging, logger, accelerator):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

def create_output_folders(output_dir, run_name):
    out_dir = os.path.join(output_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(f"{out_dir}/samples", exist_ok=True)
    return out_dir


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20230211, help="seed for seed_everything")
    parser.add_argument("--mode", default="base", type=str, help="which kind of inference mode: {'base', 'i2v'}")
    parser.add_argument("--ckpt_path", type=str, default=None, help="checkpoint path")
    parser.add_argument("--config", type=str, help="config (yaml) path")
    parser.add_argument("--prompt_file", type=str, default=None, help="a text file containing many prompts")
    parser.add_argument("--savefps", type=str, default=10, help="video fps to generate")
    parser.add_argument("--n_samples", type=int, default=1, help="num of samples per prompt",)
    parser.add_argument("--ddim_steps", type=int, default=50, help="steps of ddim if positive, otherwise use DDPM",)
    parser.add_argument("--ddim_eta", type=float, default=1.0, help="eta for ddim sampling (0.0 yields deterministic sampling)",)
    parser.add_argument("--bs", type=int, default=1, help="batch size for inference")
    parser.add_argument("--height", type=int, default=512, help="image height, in pixel space")
    parser.add_argument("--width", type=int, default=512, help="image width, in pixel space")
    parser.add_argument("--frames", type=int, default=-1, help="frames num to inference")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--unconditional_guidance_scale", type=float, default=1.0, help="prompt classifier-free guidance")
    parser.add_argument("--unconditional_guidance_scale_temporal", type=float, default=None, help="temporal consistency guidance")
    ## for conditional i2v only
    parser.add_argument("--cond_input", type=str, default=None, help="data dir of conditional input")
    ## for training
    parser.add_argument("--lr", type=float, default=2e-4, help="learning rate")
    parser.add_argument("--val_batch_size", type=int, default=1, help="batch size for validation")
    parser.add_argument("--num_val_runs", type=int, default=1, help="total number of validation samples = num_val_runs * num_gpus * num_val_batch")
    parser.add_argument("--train_batch_size", type=int, default=1, help="batch size for training")
    parser.add_argument("--reward_fn", type=str, default="aesthetic", help="reward function: 'aesthetic', 'hps', 'aesthetic_hps', 'pick_score', 'rainy', 'snowy', 'objectDetection'")
    parser.add_argument("--compression_model_path", type=str, default=None, help="compression model path")
    parser.add_argument("--target_object", type=str, default="dog", help="target object for object detection reward function")
    parser.add_argument("--hps_version", type=str, default="v2.1", help="hps version: 'v2.0', 'v2.1'")
    parser.add_argument("--prompt_fn", type=str, default="hps_custom", help="prompt function")
    parser.add_argument("--nouns_file", type=str, default="simple_animals.txt", help="nouns file")
    parser.add_argument("--activities_file", type=str, default="activities.txt", help="activities file")
    parser.add_argument("--num_train_epochs", type=int, default=200, help="number of training epochs")
    parser.add_argument("--max_train_steps", type=int, default=10000, help="max training steps")
    parser.add_argument("--backprop_mode", type=str, default="last", help="backpropagation mode: 'last', 'rand', 'specific'")   # backprop_mode != None also means training mode
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="gradient accumulation steps")
    parser.add_argument("--mixed_precision", type=str, default='fp16', help="mixed precision training: 'no', 'fp8', 'fp16', 'bf16'")
    parser.add_argument("--logger_type", type=str, default="wandb", help="logger type: 'wandb', 'tensorboard'")
    parser.add_argument("--project_dir", type=str, default="./project_dir", help="project directory")
    parser.add_argument("--validation_steps", type=int, default=1, help="validation steps")
    parser.add_argument("--checkpointing_steps", type=int, default=1, help="checkpointing steps")
    parser.add_argument("--use_wandb", type=bool, default=True, help="use wandb for logging")
    parser.add_argument("--wandb_entity", type=str, default="", help="wandb entity")
    parser.add_argument("--debug", type=bool, default=False, help="debug mode")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="max gradient norm")
    parser.add_argument("--use_AdamW8bit", type=bool, default=False, help="use AdamW8bit optimizer")
    parser.add_argument("--is_sample_preview", type=bool, default=True, help="sample preview during training")
    parser.add_argument("--decode_frame", type=str, default="-1", help="decode frame: '-1', 'fml', 'all', 'alt'") # it could also be any number str like '3', '10'. alt: alternate frames, fml: first, middle, last frames, all: all frames. '-1': random frame
    parser.add_argument("--inference_only", type=bool, default=False, help="only do inference")
    parser.add_argument("--lora_ckpt_path", type=str, default=None, help="LoRA checkpoint path")
    parser.add_argument("--lora_rank", type=int, default=8, help="LoRA rank")

    return parser


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
    # with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
    #     scorer = AestheticScorerDiff(dtype=torch_dtype).to(device, dtype=torch_dtype)
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
        checkpoint_path = f"{os.path.expanduser('~')}/.cache/hpsv2/HPS_v2_compressed.pt"
    else:
        print("=================== HPS v2.1 ===================")
        checkpoint_path = f"{os.path.expanduser('~')}/.cache/hpsv2/HPS_v2.1_compressed.pt"
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
                     hps_version="v2.0"):
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
    
    # tokenizer = get_tokenizer(model_name)
    
    if hps_version == "v2.0":
        checkpoint_path = f"{os.path.expanduser('~')}/.cache/hpsv2/HPS_v2_compressed.pt"
    else:
        print("=================== HPS v2.1 ===================")
        checkpoint_path = f"{os.path.expanduser('~')}/.cache/hpsv2/HPS_v2.1_compressed.pt"
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

def pick_score_fn(inference_dtype=None, device=None):
    '''
    Args:
        inference_dtype: torch.dtype, the data type of the model.
        device: torch.device, the device to run the model.

    Returns:
        loss_fn: function, the loss function of the PickScore reward function.
    '''
    target_size =  (224, 224)
    normalize = torchvision.transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                                std=[0.26862954, 0.26130258, 0.27577711])
    processor_name_or_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    model_pretrained_name_or_path = "yuvalkirstain/PickScore_v1"
    processor = AutoProcessor.from_pretrained(processor_name_or_path, torch_dtype=inference_dtype)
    model = AutoModel.from_pretrained(model_pretrained_name_or_path, torch_dtype=inference_dtype).eval().to(device)
    model.requires_grad_(False)

    def loss_fn(im_pix_un, prompts):
        im_pix = ((im_pix_un / 2) + 0.5).clamp(0, 1)
        im_pix = torchvision.transforms.Resize(target_size)(im_pix)
        im_pix = normalize(im_pix).to(im_pix_un.dtype)

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
        loss = 1.0 - scores / 100.0
        return loss.mean(), scores.mean()
    
    return loss_fn

def weather_loss_fn(inference_dtype=None, device=None, weather="rainy", target=None, grad_scale=0):
    '''
    Args:
        inference_dtype: torch.dtype, the data type of the model.
        device: torch.device, the device to run the model.
        weather: str, the weather condition. It is "rainy" or "snowy" in this experiment.
        target: float, the target value of the weather score. It is 1.0 in this experiment.
        grad_scale: float, the scale of the gradient. It is 1 in this experiment.

    Returns:
        loss_fn: function, the loss function of the weather reward function.
    '''
    target_size = (224, 224)
    normalize = torchvision.transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                                std=[0.26862954, 0.26130258, 0.27577711])
    if weather == "rainy":
        reward_model_path = "./assets/rainy_reward.pt"
    elif weather == "snowy":
        reward_model_path = "./assets/snowy_reward.pt"
    else:
        raise NotImplementedError
    scorer = WeatherScorer(dtype=inference_dtype, model_path=reward_model_path).to(device, dtype=inference_dtype)
    scorer.requires_grad_(False)
    scorer.eval()
    def loss_fn(im_pix_un):
        im_pix = ((im_pix_un / 2) + 0.5).clamp(0, 1)
        im_pix = torchvision.transforms.Resize(target_size)(im_pix_un)
        im_pix = normalize(im_pix).to(im_pix_un.dtype)
        rewards = scorer(im_pix)
        
        if target is None:
            loss = rewards
        else:
            loss = abs(rewards - target)

        return loss.mean() * grad_scale, rewards.mean()
    return loss_fn

def objectDetection_loss_fn(inference_dtype=None, device=None, targetObject='dog'):
    '''
    This reward function is used to remove the target object from the generated video.
    We use yolo-s-tiny model to detect the target object in the generated video.

    Args:
        inference_dtype: torch.dtype, the data type of the model.
        device: torch.device, the device to run the model.
        targetObject: str, the object to detect. It is "dog" in this experiment.

    Returns:
        loss_fn: function, the loss function of the object detection reward function.
    '''
    image_processor = AutoImageProcessor.from_pretrained("hustvl/yolos-tiny", torch_dtype=inference_dtype)
    model = AutoModelForObjectDetection.from_pretrained("hustvl/yolos-tiny", torch_dtype=inference_dtype).to(device)
    model.requires_grad_(False)
    model.eval()

    def loss_fn(im_pix_un): # im_pix_un: b,c,h,w
        images = ((im_pix_un / 2) + 0.5).clamp(0.0, 1.0)

        # reproduce the yolo preprocessing
        height = 512
        width = 512 * images.shape[3] // images.shape[2]    # keep the aspect ratio, so the width is 512 * w/h
        images = torchvision.transforms.Resize((height, width))(images)
        images = images.permute(0, 2, 3, 1)  # b,c,h,w -> (b,h,w,c)

        image_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
        image_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

        images = (images - image_mean) / image_std
        normalized_image = images.permute(0,3,1,2)  # NHWC -> NCHW

        # Process images
        outputs = model(pixel_values=normalized_image)

        # Get target sizes for each image
        target_sizes = torch.tensor([normalized_image[0].shape[1:]]*normalized_image.shape[0]).to(device)

        # Post-process results for each image
        results = image_processor.post_process_object_detection(outputs, threshold=0.0, target_sizes=target_sizes)

        sum_avg_scores = 0
        for i, result in enumerate(results):
            id = model.config.label2id[targetObject]
            index = torch.where(result["labels"] == id)     # get index of targetObject's label
            if len(index[0]) == 0:  # index: ([],[]) so index[0] is the first list
                sum_avg_scores = torch.sum(result["scores"][0]) - torch.sum(result["scores"][0])    # set sum_avg_scores to 0
                continue
            scores = result["scores"][index]
            sum_avg_scores = sum_avg_scores +  (torch.sum(scores) / scores.shape[0])

        loss = sum_avg_scores / len(results)
        reward = 1 - loss

        return loss, reward
    return loss_fn

def should_sample(global_step, validation_steps, is_sample_preview):
    return (global_step % validation_steps == 0 or global_step ==1)  \
    and is_sample_preview


def run_training(args, gpu_num, gpu_no, **kwargs):
    ## step 1: accelerator setup
    ## -----------------------------------------------------------------
    accelerator = Accelerator(                                                  # Initialize Accelerator
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.logger_type,
        project_dir=args.project_dir
    )
    output_dir = args.project_dir

    validation_steps = args.validation_steps * args.gradient_accumulation_steps         # number of steps to run validation for
    checkpointing_steps = args.checkpointing_steps * args.gradient_accumulation_steps   # Saves a model every nth step.


    # Make one log on every process with the configuration for debugging.
    create_logging(logging, logger, accelerator)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.use_wandb:
        if accelerator.is_main_process:
            import wandb
            wandb_args = {}
            
            if args.wandb_entity != '':
                wandb_args['entity'] =  args.wandb_entity    
            
            if args.debug:
                wandb_args['mode'] = "disabled"
            
            opt_dict = vars(args)   # convert args to dict
            accelerator.init_trackers("VideoCrafter", config=opt_dict, init_kwargs={"wandb": wandb_args})
            output_dir = create_output_folders(args.project_dir, wandb.run.name)    # all processes will create the same output folder
            output_dir_broadcast = [output_dir]     # convert output_dir to broadcastable tensor, so that it can be broadcasted to all processes
            
        else:
            output_dir_broadcast = [None]
        
        # convert output_dir back to str
        dist.broadcast_object_list(output_dir_broadcast, src=0)    # broadcast the output_dir to all processes
        output_dir = output_dir_broadcast[0]
        print(f"+++++++++++++++++++output_dir: {output_dir}+++++++++++++++++++++++++++++++++")

    ## step 2: model config
    ## -----------------------------------------------------------------
    config = OmegaConf.load(args.config)
    model_config = config.pop("model", OmegaConf.create())
    model = instantiate_from_config(model_config)
    model = model.cuda(gpu_no)
    assert os.path.exists(args.ckpt_path), f"Error: checkpoint [{args.ckpt_path}] Not Found!"
    model = load_model_checkpoint(model, args.ckpt_path)

    # convert first_stage_model and cond_stage_model to torch.float16 if mixed_precision is True
    if args.mixed_precision != 'no':
        model.first_stage_model = model.first_stage_model.half()
        model.cond_stage_model = model.cond_stage_model.half()

    # step 2.1: add LoRA using peft
    config = peft.LoraConfig(
            r=args.lora_rank,
            target_modules=["to_k", "to_v", "to_q"],        # only diffusion_model contains these modules
            lora_dropout=0.01,
        )
    
    peft_model = peft.get_peft_model(model, config)

    peft_model.print_trainable_parameters()

    # load the pretrained LoRA model
    if args.lora_ckpt_path is not None:
        peft.set_peft_model_state_dict(peft_model, torch.load(args.lora_ckpt_path))

    peft_model.requires_grad_(False)    # turn it off becuase we train the reward model only

    # load pretrained weather reward model
    scorer = JpegCompressionScorer(dtype=peft_model.dtype, model_path=args.compression_model_path).to(accelerator.device)

    # set scorer to train mode
    scorer.clip.requires_grad_(False)
    scorer.score_generator.requires_grad_(True)
    
    # To generate label for generated frame. This label will be used to train compression reward model
    label_generator = jpeg_compressibility(accelerator.device)  # ground true label generator

    # step 2.2: optimizer and loss function
    if args.use_AdamW8bit:
        optimizer = bnb.optim.AdamW8bit(scorer.parameters(), lr=args.lr)
    else:
        optimizer = torch.optim.AdamW(scorer.parameters(), lr=args.lr)

    # loss function is L2 loss
    loss_fn = torch.nn.MSELoss()

    peft_model, optimizer, scorer, loss_fn = accelerator.prepare(peft_model, optimizer, scorer, loss_fn)

    ## sample shape
    assert (args.height % 16 == 0) and (args.width % 16 == 0), "Error: image size [h,w] should be multiples of 16!"
    
    ## latent noise shape
    h, w = args.height // 8, args.width // 8
    frames = peft_model.temporal_length if args.frames < 0 else args.frames
    channels = peft_model.channels


    ## step 3: load data
    ## -----------------------------------------------------------------
    prompt_fn = getattr(prompts_file, args.prompt_fn)    # Get the prompt function, default is nouns_activities()

    
    ## step 4: run training over samples
    ## -----------------------------------------------------------------
    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    
    first_epoch = 0
    global_step = 0

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, min(args.max_train_steps, args.num_train_epochs*100*args.bs)))
    progress_bar.set_description("Training Steps")
    start = time.time()


    for epoch in range(first_epoch, args.num_train_epochs):
        default_iterations = 100
        for idx, step in enumerate(range(default_iterations)): # default number of iterations is 100

            # randomize training process    
            random.seed(datetime.datetime.now().microsecond)
            torch.manual_seed(datetime.datetime.now().microsecond)

            # Step 4.1 forward pass
            if global_step % args.gradient_accumulation_steps == 0:
                train_loss_list = []    # this is for gradient accumulation

            train_prompt, promt_metadata = zip(
                *[prompt_fn(args.nouns_file, args.activities_file) for _ in range(args.train_batch_size)] # get train_batch_size prompts in a tuple
                )
            train_prompt = list(train_prompt)  # tuple to list ['', '', '', ...]

            batch_size = len(train_prompt)
            noise_shape = [batch_size, channels, frames, h, w]
            
            fps = torch.tensor([args.fps]*batch_size).to(accelerator.device).long()

            prompts = train_prompt
            if isinstance(prompts, str):
                prompts = [prompts]
            
            with accelerator.accumulate(scorer):    # gradient accumulation
                with accelerator.autocast():            # mixed precision
                    text_emb = peft_model.get_learned_conditioning(prompts)

                    if args.mode == 'base':
                        cond = {"c_crossattn": [text_emb], "fps": fps}
                    else:   # TODO: implement i2v mode training in the future
                        raise NotImplementedError

                    ## Step 4.1: inference, batch_samples shape: batch, <samples>, c, t, h, w
                    if isinstance(peft_model, torch.nn.parallel.DistributedDataParallel):
                        batch_samples = batch_ddim_sampling(peft_model.module, cond, noise_shape, args.n_samples, \
                                                            args.ddim_steps, args.ddim_eta, args.unconditional_guidance_scale, None, backprop_mode=args.backprop_mode, decode_frame=args.decode_frame, **kwargs)
                    else:
                        batch_samples = batch_ddim_sampling(peft_model, cond, noise_shape, args.n_samples, \
                                                                args.ddim_steps, args.ddim_eta, args.unconditional_guidance_scale, None, backprop_mode=args.backprop_mode, decode_frame=args.decode_frame, **kwargs)
                    
                    video_frames_ = batch_samples.permute(1,0,3,2,4,5)    # batch,samples,channels,frames,height,width >> s,b,f,c,h,w
                    s_, bs, nf, c_, h_, w_ = video_frames_.shape
                    assert s_ == 1 # samples should only be on single sample in training mode
                    video_frames_ = video_frames_.squeeze(0)    # s,b,f,c,h,w >> b,f,c,h,w
                    assert nf == 1 # reward should only be on single frame
                    video_frames_ = video_frames_.squeeze(1)    # b,f,c,h,w >> b,c,h,w 
                    video_frames_ = video_frames_.to(scorer.module.dtype)
                    
                    # start training the reward model
                    images = ((video_frames_ / 2) + 0.5).clamp(0, 1)    # images are in the range of [0, 1]
                    scores, images = label_generator(images)    # scores are the ground truth labels for the generated frame
                    

                    scores = torch.from_numpy(scores).to(accelerator.device).float()

                    # forward pass of the reward model
                    scores_pred = scorer(images)
                    
                    # calculate the loss
                    loss = loss_fn(scores_pred.to(images.dtype), scores.to(images.dtype))   # MSELoss
 
                # Gather the losses across all processes for logging (if we use distributed training).
                accum_loss = accelerator.gather(loss.repeat(args.train_batch_size)) # loss.repeat(args.train_batch_size) is to get the total loss for each sample in the batch
                avg_loss = accum_loss.mean()
                train_loss_list.append(avg_loss.item())


                # Step 4.2 backpropagation
                accelerator.backward(loss)

                if args.max_grad_norm > 0:  # gradient clipping is to prevent exploding gradients
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(scorer.parameters(), args.max_grad_norm)

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                # Step 4.3 logging and save checkpoint
                if accelerator.sync_gradients:
                    global_step += 1
                    # log_dict_step = {f"Step {global_step} | Step Loss: {loss.detach().item()} | Step Reward: {rewards.detach().item()}"}
                    log_dict_step = {"step_loss": loss.detach().item(), "predict_reward": scores_pred.mean().detach().item(), "ground_truth_reward": scores.mean().detach().item()}
                    accelerator.log(log_dict_step, step=global_step)
                    progress_bar.update(1)

                    if global_step % checkpointing_steps ==0:   
                        logger.info("Saving checkpointing....")
                        accelerator.wait_for_everyone()
                        if accelerator.is_local_main_process:
                            unwrapped_model = accelerator.unwrap_model(scorer)
                            # save score_generator model only
                            state_dict = unwrapped_model.score_generator.state_dict()
                            model_path = os.path.join(output_dir, f"compression_model_{global_step}.pt")
                            torch.save(state_dict, model_path)
                            # save the whole model
                            # state_dict = unwrapped_model.state_dict()
                            # model_path = os.path.join(output_dir, f"compression_whole_model_{global_step}.pt")
                            # torch.save(state_dict, model_path)
                            logger.info("Saving checkpoing end")

                if global_step % args.gradient_accumulation_steps == 0:
                    avg_loss = sum(train_loss_list)/len(train_loss_list)
                    accelerator.log({"avg_loss": avg_loss}, step=global_step)
                
                if global_step >= args.max_train_steps:
                    break
            # free memory
            del batch_samples, video_frames_
            torch.cuda.empty_cache()
            gc.collect()


if __name__ == '__main__':
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    print("@CoLVDM Inference: %s"%now)
    parser = get_parser()
    args = parser.parse_args()
    seed_everything(args.seed)
    rank, gpu_num = 0, 1
    run_training(args, gpu_num, rank)