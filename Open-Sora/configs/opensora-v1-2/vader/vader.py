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
is_vader_training = False            # if True, it will train the vader model, otherwise it will be inference only mode
train_batch_size = 1
val_batch_size = 2
num_val_runs = 8
seed = 200
lora_ckpt_path = "peft_model_1300.pt"
project_dir = "./project_dir/inference/cosmic-sun-65/seed200/origin"
lr = 0.0002
reward_fn = "pick_score"
gradient_accumulation_steps = 4

dtype = "bf16"
mixed_precision = "bf16"
logger_type = "wandb"
validation_steps = 10
checkpointing_steps = 1
use_wandb = True
wandb_entity = ""
debug = False
lora_rank = 16
use_AdamW8bit = False
hps_version = "v2.1"
num_train_epochs = 200
max_train_steps = 10000
backprop_mode = "last"
decode_frame = '-1'             # it could also be any number str like '3', '10'. alt: alternate frames, fml: first, middle, last frames, all: all frames. '-1': random frame
is_sample_preview = True        # if True, it will generate and save the validation video to wandb
grad_checkpoint = True          # if True, it will use gradient checkpointing
