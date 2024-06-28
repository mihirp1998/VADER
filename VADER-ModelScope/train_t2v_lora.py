import sys
sys.path.append('../')   # setting path to get Core and assets
import time
import hydra
__import__("builtins").st  = __import__("ipdb").set_trace
from omegaconf import DictConfig 
from PIL import Image
import logging
from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
import hpsv2
import numpy as np
import torchvision
from transformers.utils import ContextManagers
import os
import random
import gc
import copy
import imageio
from Core.aesthetic_scorer import AestheticScorerDiff
from Core.actpred_scorer import ActPredScorer
import accelerate
from accelerate import Accelerator
from accelerate.state import AcceleratorState
from typing import Dict, Optional, Tuple
import Core.prompts as prompts_file
import torch
import torchvision.transforms as T
import diffusers
import transformers
from tqdm.auto import tqdm
from accelerate.logging import get_logger
from accelerate.utils import set_seed

from models.unet_3d_condition import UNet3DConditionModel
from diffusers import DPMSolverMultistepScheduler,  TextToVideoSDPipeline
from ms_custom import CustomT2V
from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available
from diffusers.models.attention_processor import AttnProcessor2_0
from diffusers.models.attention import BasicTransformerBlock
from transformers import CLIPTextModel
from transformers.models.clip.modeling_clip import CLIPEncoder
from utils.lora_handler import LoraHandler, LORA_VERSIONS


logger = get_logger(__name__, log_level="INFO")

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


def deepspeed_zero_init_disabled_context_manager():
    """
    returns either a context list that includes one that will disable zero.Init or an empty context list
    """
    deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
    if deepspeed_plugin is None:
        return []

    return [deepspeed_plugin.zero3_init_context_manager(enable=False)]



def create_output_folders(output_dir, run_name):
    out_dir = os.path.join(output_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(f"{out_dir}/samples", exist_ok=True)
    return out_dir

def load_primary_models(pretrained_model_path, scheduler_type):
    unet = UNet3DConditionModel.from_pretrained(pretrained_model_path, subfolder="unet")
    pipeline = CustomT2V.from_pretrained(pretrained_model_path, unet=unet)
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
    def loss_fn(vid, target_action_label):
        vid = torch.cat([ preprocess_img(img).unsqueeze(0) for img in vid])[None]
        return scorer.get_loss_and_score(vid, target_action_label)
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
        return loss.mean() * grad_scale, rewards.mean()
    return loss_fn

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
    
    checkpoint_path = f"{os.path.expanduser('~')}/.cache/huggingface/hub/models--xswu--HPSv2/snapshots/697403c78157020a1ae59d23f111aa58ced35b0a/HPS_v2_compressed.pt"
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
        return  loss.mean(), scores.mean()
    
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
    if  isinstance(unet, UNet3DConditionModel):
        u_dtype, t_dtype, v_dtype = unet.dtype, text_encoder.dtype, vae.dtype 
    else:
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
    
    if accelerator.is_main_process:
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




def validation_run(log_step, accelerator, pipeline, 
                    prompt_fn, prompt_fn_kwargs, 
                    val_batch_size, validation_params, 
                    output_dir, logger, reward_fn_type, loss_fn, generator = None):
   
    vis_dict = {}
    random.seed(log_step)
    val_prompt, promt_metadata = zip(
        *[prompt_fn(**prompt_fn_kwargs) for _ in range(val_batch_size)]
    )    
    val_prompt = list(val_prompt)    

    with accelerator.autocast():
        import wandb
        with torch.no_grad():
            val_video_frames = pipeline(
                val_prompt,
                width=validation_params.width,
                height=validation_params.height,
                num_frames=validation_params.num_frames,
                num_inference_steps=validation_params.num_inference_steps,
                guidance_scale=validation_params.guidance_scale,
                generator= generator,
            ).frames
        for i in range(val_video_frames.shape[0]):
            save_filename = f"{log_step}_dataset_{val_prompt[i]}_{i}"
            out_file = f"{output_dir}/samples/{save_filename}.gif"

            if reward_fn_type == "actpred":
                _inp = torch.from_numpy(2*(val_video_frames[i] - 0.5))
                _inp = _inp.permute(0,3,1,2).to(accelerator.device)
                _inp = _inp[np.arange(0, len(_inp), 2)]
                with torch.no_grad():
                    loss, rewards, pred_cls = loss_fn(_inp, val_prompt[i])
                caption = f'{val_prompt[i]}, gt_prob: {np.round(rewards.detach().cpu().numpy(), 2)}, pred_cls: {pred_cls}'
            else:
                caption = f'{val_prompt[i]}'

            video_frames_int = (val_video_frames[i] * 255).astype(np.uint8)
            imageio.mimwrite(out_file, video_frames_int, duration=175, loop=0)
            vis_dict[f"gen_video_{i}"] = wandb.Video(out_file, fps=2, format="gif", caption=caption)
            logger.info(f"Saved a new sample to {out_file}")
        
        accelerator.log(vis_dict, step=log_step)
        torch.cuda.empty_cache()
        gc.collect()

def main(opt):
    output_dir = opt.output_dir
    accelerator = Accelerator(
        gradient_accumulation_steps=opt.gradient_accumulation_steps,
        mixed_precision=opt.mixed_precision,
        log_with=opt.logger_type,
        project_dir=output_dir
    )
    validation_steps = opt.validation_steps*opt.gradient_accumulation_steps
    checkpointing_steps = opt.checkpointing_steps *opt.gradient_accumulation_steps

    # Make one log on every process with the configuration for debugging.
    create_logging(logging, logger, accelerator)

    # Initialize accelerate, transformers, and diffusers warnings
    accelerate_set_verbose(accelerator)

    # If passed along, set the training seed now.
    if opt.seed is not None:
        set_seed(opt.seed)


    # Load scheduler, tokenizer and models.
    pipeline, noise_scheduler, tokenizer, text_encoder, vae, unet = load_primary_models(opt.pretrained_model_path, opt.scheduler_type)
    # Freeze any necessary models
    freeze_models([vae, text_encoder, unet])
    
    # Enable xformers if available
    handle_memory_attention(opt.enable_xformers_memory_efficient_attention, opt.enable_torch_2_attn, unet)

    if opt.scale_lr:
        learning_rate = (
            opt.learning_rate * opt.gradient_accumulation_steps * opt.train_batch_size * accelerator.num_processes
        )

    prompt_fn = getattr(prompts_file, opt.prompt_fn)

    # Initialize the optimizer
    optimizer_cls = get_optimizer(opt.use_8bit_adam)

    # Use LoRA if enabled.  
    lora_manager = LoraHandler(
        version=opt.lora_version, 
        use_unet_lora=opt.use_unet_lora,
        use_text_lora=opt.use_text_lora,
        save_for_webui=opt.save_lora_for_webui,
        only_for_webui=opt.only_lora_for_webui,
        unet_replace_modules=opt.unet_lora_modules,
        text_encoder_replace_modules=opt.text_encoder_lora_modules,
        lora_bias=opt.lora_bias
    )
    
    unet_lora_params, unet_negation = lora_manager.add_lora_to_model(
        opt.use_unet_lora, unet, lora_manager.unet_replace_modules, opt.lora_unet_dropout, opt.lora_path, r=opt.lora_rank) 

    text_encoder_lora_params, text_encoder_negation = lora_manager.add_lora_to_model(
        opt.use_text_lora, text_encoder, lora_manager.text_encoder_replace_modules, opt.lora_text_dropout, opt.lora_path, r=opt.lora_rank) 

    # Create parameters to optimize over with a condition (if "condition" is true, optimize it)
    extra_unet_params = opt.extra_unet_params if opt.extra_unet_params is not None else {}
    extra_text_encoder_params = extra_unet_params if extra_unet_params is not None else {}

    trainable_modules_available = opt.trainable_modules is not None
    trainable_text_modules_available = (opt.train_text_encoder and opt.trainable_text_modules is not None)

    optim_params = [
        param_optim(unet, trainable_modules_available, extra_params=extra_unet_params, negation=unet_negation),
        param_optim(text_encoder, trainable_text_modules_available, 
                        extra_params=extra_text_encoder_params, 
                        negation=text_encoder_negation
                   ),
        param_optim(text_encoder_lora_params, opt.use_text_lora, is_lora=True, 
                        extra_params={**{"lr": opt.learning_rate}, **extra_text_encoder_params}
                    ),
        param_optim(unet_lora_params, opt.use_unet_lora, is_lora=True, 
                        extra_params={**{"lr": opt.learning_rate}, **extra_unet_params}
                    )
    ]

    params = create_optimizer_params(optim_params, opt.learning_rate)
    
    # Create Optimizer
    optimizer = optimizer_cls(
        params,
        lr=opt.learning_rate,
        betas=(opt.adam_beta1, opt.adam_beta2),
        weight_decay=opt.adam_weight_decay,
        eps=opt.adam_epsilon,
    )

    # Scheduler
    lr_scheduler = get_scheduler(
        opt.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=opt.lr_warmup_steps * opt.gradient_accumulation_steps,
        num_training_steps=opt.max_train_steps * opt.gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    unet, optimizer,lr_scheduler, text_encoder = accelerator.prepare(
        unet, 
        optimizer, 
        lr_scheduler, 
        text_encoder
    )

    # Use Gradient Checkpointing if enabled.
    unet_and_text_g_c(
        unet, 
        text_encoder, 
        opt.gradient_checkpointing, 
        opt.text_encoder_gradient_checkpointing
    )
    
    # Enable VAE slicing to save memory.
    vae.enable_slicing()

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = is_mixed_precision(accelerator)

    # Move text encoders, and VAE to GPU
    models_to_cast = [text_encoder, vae]
    cast_to_gpu_and_type(models_to_cast, accelerator, weight_dtype)
    
    if opt.reward_fn == "aesthetic":
        loss_fn = aesthetic_loss_fn(grad_scale=0.1,
                                    aesthetic_target=10,
                                    accelerator = accelerator,
                                    torch_dtype = weight_dtype,
                                    device = accelerator.device)
    elif opt.reward_fn == "hps":
        loss_fn = hps_loss_fn(weight_dtype, accelerator.device)    
    elif opt.reward_fn == "actpred":
        assert opt.decode_frame in ['alt', 'last'], "decode_frame must be 'alt' or 'last' for actpred reward function."
        num_score_frames = opt.validation_data.num_frames//2 if opt.decode_frame == 'alt' else opt.validation_data.num_frames
        loss_fn = actpred_loss_fn(weight_dtype, accelerator.device, num_frames = num_score_frames )
    

    # Fix noise schedules to predcit light and dark areas if available.
    if not opt.use_offset_noise and opt.rescale_schedule:
        noise_scheduler.betas = enforce_zero_terminal_snr(noise_scheduler.betas)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process and opt.use_wandb:
        import wandb
        wandb_args = {}
        
        if opt.wandb_entity != '':
            wandb_args['entity'] =  opt.wandb_entity    
        
        if opt.debug:
            wandb_args['mode'] = "disabled"
        
        opt_dict = dict(opt)
        accelerator.init_trackers("vader", config=opt_dict, init_kwargs={"wandb": wandb_args})
        output_dir = create_output_folders(output_dir, wandb.run.name)
        

    # Train!
    total_batch_size = opt.train_batch_size * accelerator.num_processes * opt.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {opt.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {opt.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {opt.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {opt.max_train_steps}")
    global_step = 0
    first_epoch = 0
    
    if opt.prompt_fn_kwargs == 0:
        opt.prompt_fn_kwargs = {} 

   
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, opt.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Training Steps")
    
    if opt.only_val:
        generator = torch.cuda.manual_seed(opt.seed)
        for log_step in range(opt.num_only_val_itrs):
            validation_run(log_step, accelerator, pipeline, 
                        prompt_fn, opt.prompt_fn_kwargs, 
                        opt.val_batch_size, opt.validation_data, 
                        output_dir, logger, opt.reward_fn, loss_fn, generator = generator)
        return

    val_prompt, promt_metadata = zip(
    *[prompt_fn(**opt.prompt_fn_kwargs) for _ in range(opt.val_batch_size)]
    )    
    val_prompt = list(val_prompt)  
    
    for epoch in range(first_epoch, opt.num_train_epochs):
        train_loss = 0.0
        for idx, step in enumerate(range(100)):
            print("Training....")
            if global_step % opt.gradient_accumulation_steps == 0:
                train_loss_list = []                           

            with accelerator.accumulate(unet) ,accelerator.accumulate(text_encoder):
                with accelerator.autocast():
                    prompts, prompt_metadata = zip(
                        *[prompt_fn(**opt.prompt_fn_kwargs) for _ in range(opt.train_batch_size)]
                    )
                    
                    prompts = list(prompts)
                    
                    # randomize geneeration
                    random_seed = random.randint(0,100000)
                    gen = torch.cuda.manual_seed(random_seed)
                    
                    video_frames = pipeline.train_policy(
                            prompts,
                            width=opt.validation_data.width,
                            height=opt.validation_data.height,
                            num_frames=opt.validation_data.num_frames,
                            num_inference_steps=opt.validation_data.num_inference_steps,
                            guidance_scale=opt.validation_data.guidance_scale,
                            decode_frame=opt.decode_frame,
                            backprop_mode = opt.backprop_mode,
                            cpu_step_cutoff = opt.validation_data.cpu_step_cutoff,
                            decode_og = opt.decode_og,
                            generator=gen
                            
                        ) 
                   
                    if opt.reward_fn in ["actpred", "vidclip"]:
                        assert opt.decode_frame in  ['fml', 'all', 'alt']
                        assert len(prompts) ==1
                        if opt.reward_fn == "actpred":
                            loss, rewards, pred_cls = loss_fn(video_frames, prompts[0])
                        else:
                            loss, rewards = loss_fn(video_frames, prompts[0]) 
                    else:
                        video_frames_ = video_frames.permute(0,2,1,3,4)
                        bs, nf, c, h, w = video_frames_.shape
                        assert nf == 1 # reward should only be on single frame
                        video_frames_ = video_frames_.squeeze(1)
                        video_frames_ = video_frames_.to(weight_dtype) 
                        if opt.reward_fn == "aesthetic":
                            loss, rewards = loss_fn(video_frames_)
                        else:
                            loss, rewards = loss_fn(video_frames_,prompts)                    

                # Gather the losses across all processes for logging (if we use distributed training).
                
                accum_loss = accelerator.gather(loss.repeat(opt.train_batch_size))
                avg_loss = accum_loss.mean()
              
                train_loss_list.append(avg_loss.item())
                
                accelerator.backward(loss)
                
            
                if any([opt.train_text_encoder, opt.use_text_lora]):
                    params_to_clip = list(unet.parameters()) + list(text_encoder.parameters())
                else:
                    params_to_clip = unet.parameters()
                
                if opt.max_grad_norm > 0:
                    if accelerator.sync_gradients:
                        if any([opt.train_text_encoder, opt.use_text_lora]):
                            params_to_clip = list(unet.parameters()) + list(text_encoder.parameters())
                        else:
                            params_to_clip = list(unet.parameters())
                            
                        accelerator.clip_grad_norm_(params_to_clip, opt.max_grad_norm)
                        
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            
            # Checks if the accelerator has performed an optimization step behind the scenes
            global_step += 1
            log_dict_step = {"step_loss": loss.detach().item(), "step_reward": rewards.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            accelerator.log(log_dict_step, step=global_step)    
                    
            if accelerator.sync_gradients:
                progress_bar.update(1)
                if global_step % checkpointing_steps ==0:   
                    print("Saving checkpointing....")
                    # st()
                    save_pipe(
                        opt.pretrained_model_path, 
                        global_step, 
                        accelerator, 
                        unet, 
                        text_encoder, 
                        vae, 
                        output_dir, 
                        lora_manager,
                        opt.unet_lora_modules,
                        opt.text_encoder_lora_modules,
                        is_checkpoint=True,
                        save_pretrained_model=False
                    )
                    # st()
                    print("Saving checkpoing end")
                
                if global_step % opt.gradient_accumulation_steps == 0:
                    avg_loss = sum(train_loss_list)/len(train_loss_list)
                    accelerator.log({"avg_loss": avg_loss}, step=global_step)
               
                if should_sample(global_step, validation_steps, opt.validation_data):
                    print("Performing validation.")
                    if accelerator.is_main_process:
                        with accelerator.autocast():
                            import wandb
                            vis_dict = {}
                            if opt.decode_frame in ["fml", "all", "alt"]:
                                video_frames = torch.stack(video_frames).permute(0,2,3,1).detach().cpu()
                                video_frames_int = (((video_frames +1.0)/2.0).clamp(0,1).numpy() * 255).astype(np.uint8)
                               
                                out_file = f"{output_dir}/samples/train_{global_step}.gif"
                                imageio.mimwrite(out_file, video_frames_int, duration=175, loop=0)
                                if opt.reward_fn == "actpred":
                                    caption = f'{prompts[0]}, gt_prob: {np.round(rewards.detach().cpu().numpy(),2)}, pred_cls: {pred_cls}'
                                else:
                                    caption = f'{prompts[0]}, gt_prob: {np.round(rewards.detach().cpu().numpy(),2)}'
                                vis_dict["gen video (train)"] = wandb.Video(out_file, fps=2, format="gif", caption = caption)
                                logger.info(f"Saved a new sample to {out_file}")
                            else:

                                train_gen_frames = Image.fromarray((((video_frames_ + 1.0)/2.0)[0].permute(1,2,0).detach().cpu().clamp(0, 1).numpy() * 255).astype(np.uint8))
                                gen_image = wandb.Image(train_gen_frames, caption=prompts[0])                            
                                vis_dict["gen image (train)"] = gen_image

                            del video_frames    
                            gc.collect()

                            unet.eval()
                            text_encoder.eval()
                            unet_and_text_g_c(unet, text_encoder, False, False)
                            lora_manager.deactivate_lora_train([unet, text_encoder], True)    

                            generator = torch.cuda.manual_seed(opt.seed)
                            
                            with torch.no_grad():
                                val_video_frames = pipeline(
                                    val_prompt,
                                    width=opt.validation_data.width,
                                    height=opt.validation_data.height,
                                    num_frames=opt.validation_data.num_frames,
                                    num_inference_steps=opt.validation_data.num_inference_steps,
                                    guidance_scale=opt.validation_data.guidance_scale,
                                    generator= generator,
                                ).frames
                            
                            for i in range(val_video_frames.shape[0]):
                                save_filename = f"{global_step}_dataset_{val_prompt[i]}_{i}"
                                out_file = f"{output_dir}/samples/{save_filename}.gif"
                                if opt.reward_fn == "actpred":
                                    _inp = torch.from_numpy(2*(val_video_frames[i] - 0.5))
                                    _inp = _inp.permute(0,3,1,2).to(accelerator.device)
                                    _inp = _inp[np.arange(0, len(_inp), 2)]
                                    with torch.no_grad():
                                        loss, rewards, pred_cls = loss_fn(_inp, val_prompt[i])
                                    caption = f'{val_prompt[i]}, gt_prob: {np.round(rewards.detach().cpu().numpy(), 2)}, pred_cls: {pred_cls}'
                                else:
                                    caption = f'{val_prompt[i]}, gt_prob: {np.round(rewards.detach().cpu().numpy(), 2)}'

                                video_frames_int = (val_video_frames[i] * 255).astype(np.uint8)
                                imageio.mimwrite(out_file, video_frames_int, duration=175, loop=0)
                                vis_dict[f"gen_video_{i}"] = wandb.Video(out_file, fps=2, format="gif", caption=caption)
                                logger.info(f"Saved a new sample to {out_file}")

                            log_grad = False
                            
                            if log_grad:
                                for name, param in unet.named_parameters():
                                    if param.requires_grad:
                                        vis_dict[f"param/{name}"] =  param.norm().clone().cpu().data.numpy()
                            
                            accelerator.log(vis_dict, step=global_step)
                            torch.cuda.empty_cache()
                            gc.collect()

                    unet_and_text_g_c(
                        unet, 
                        text_encoder, 
                        opt.gradient_checkpointing, 
                        opt.text_encoder_gradient_checkpointing
                    )

                    lora_manager.deactivate_lora_train([unet, text_encoder], False)    


            if global_step >= opt.max_train_steps:
                break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_pipe(
                opt.pretrained_model_path, 
                global_step, 
                accelerator, 
                unet, 
                text_encoder, 
                vae, 
                output_dir, 
                lora_manager,
                opt.unet_lora_modules,
                opt.text_encoder_lora_modules,
                is_checkpoint=False,
                save_pretrained_model=opt.save_pretrained_model
        )     
    accelerator.end_training()


@hydra.main(config_path="config_t2v", config_name="config")
def my_main(opt: DictConfig) -> None:
    main(opt)


if __name__ == "__main__":
    my_main()
