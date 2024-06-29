# 🎥 ModelScope Documentation

## Accelerator Configuration
If you are not familiar with the accelerator configuration, you can refer to the following steps to set up the basic accelerator configuration.
```bash
accelerate config
```
Then, you can configure the accelerator following the prompts. If you have only one GPU on your machine, you can set as follows:
```bash
In which compute environment are you running?
Please select a choice using the arrow or number keys, and selecting with enter
➔  This machine
    AWS (Amazon SageMaker)


Which type of machine are you using?
Please select a choice using the arrow or number keys, and selecting with enter
➔  No distributed training
    multi-CPU
    multi-XPU
    multi-GPU
    multi-NPU
    TPU

Do you want to run your training on CPU only (even if a GPU / Apple Silicon / Ascend NPU device is available)? [yes/NO]:no

Do you wish to optimize your script with torch dynamo?[yes/NO]:no

Do you want to use DeepSpeed? [yes/NO]: no

What GPU(s) (by id) should be used for training on this machine as a comma-seperated list? [all]:all

Do you wish to use FP16 or BF16 (mixed precision)?                                                                                                                                                                   
Please select a choice using the arrow or number keys, and selecting with enter
    no                                     
➔  fp16
    bf16                                                   
    fp8
```

Or, if you have multiple GPUs (let's say 4) on your machine, you can set as follows:
```bash
In which compute environment are you running?
This machine

Which type of machine are you using?
multi-GPU

How many different machines will you use (use more than 1 for multi-node training)? [1]: 1

Should distributed operations be checked while running for errors? This can avoid timeout issues but will be slower. [yes/NO]: no

Do you wish to optimize your script with torch dynamo?[yes/NO]:no

Do you want to use DeepSpeed? [yes/NO]: no

Do you want to use FullyShardedDataParallel? [yes/NO]: no

Do you want to use Megatron-LM ? [yes/NO]: no

How many GPU(s) should be used for distributed training? [1]:4

What GPU(s) (by id) should be used for training on this machine as a comma-seperated list? [all]:all

Do you wish to use FP16 or BF16 (mixed precision)?
fp16
```