resolution = "240p"
aspect_ratio = "9:16"
num_frames = 16
fps = 24
frame_interval = 1
save_fps = 24

multi_resolution = "STDiT2"
condition_frame_length = 5
align = 5

model = dict(
    type="STDiT3-XL/2",
    from_pretrained="hpcai-tech/OpenSora-STDiT-v3",
    qk_norm=True,
    enable_flash_attn=False,
    enable_layernorm_kernel=False,
)
vae = dict(
    type="OpenSoraVAE_V1_2",
    from_pretrained="hpcai-tech/OpenSora-VAE-v1.2",
    micro_frame_size=17,
    micro_batch_size=4,
)
text_encoder = dict(
    type="t5",
    from_pretrained="DeepFloyd/t5-v1_1-xxl",
    model_max_length=300,
)
scheduler = dict(
    type="rflow",
    use_timestep_transform=True,
    num_sampling_steps=30,
    cfg_scale=7.0,
)

aes = 6.5
flow = None

# training settings
is_vader_training = True            # if True, it will train the vader model, otherwise it will be inference only mode
train_batch_size = 1                # batch size for training
val_batch_size = 1                  # batch size for validation
num_val_runs = 2                    # number of validation runs. The total number of validation videos will be num_val_runs * val_batch_size * accelerator.num_processes
seed = 200                          # random seed
lora_ckpt_path = None               # path to the pretrained lora checkpoint
project_dir = "./project_dir"       # project directory, in which checkpoints, and other video files will be saved
lr = 0.0002                         # learning rate
reward_fn = "hps"                   # reward function in Open-Sora: ['aesthetic', 'hps', 'aesthetic_hps', 'pick_score']. It does not matter if is_vader_training is False
gradient_accumulation_steps = 8     # number of gradient accumulation steps
lora_rank = 16                      # rank of the LoRA model, the higher, the larger memory footprint

dtype = "bf16"                      # data type: it suggests to use bf16 for 240p resolution
mixed_precision = "bf16"            
logger_type = "wandb"
validation_steps = 10               # The frequency of validation, e.g., 1 means validate every 1*accelerator.num_processes steps
checkpointing_steps = 1             # The frequency of saving checkpoints
use_wandb = True                    # if True, it will log the training progress to wandb
wandb_entity = ""                   # wandb entity
debug = False                       
use_AdamW8bit = False               # if True, it will use AdamW8bit optimizer
hps_version = "v2.1"                # version of the HPS model
num_train_epochs = 200              # number of training epochs
max_train_steps = 800             # maximum number of training steps
backprop_mode = "last"              # backpropagation mode: ['last', 'rand', 'specific']
decode_frame = '-1'             # it could also be any number str like '3', '10'. 'alt': alternate frames, 'fml': first, middle, last frames, 'all': all frames. '-1': random frame
is_sample_preview = True        # if True, it will generate and save the validation video to wandb
grad_checkpoint = True          # if True, it will use gradient checkpointing
