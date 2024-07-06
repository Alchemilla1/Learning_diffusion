from dataclasses import dataclass

@dataclass
class TrainingConfig:
    image_size = 64
    train_batch_size = 16
    eval_batch_size = 16
    num_epochs = 50
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    mixed_precision = "fp16"
    output_dir = "ddpm-animefaces-64"
    overwrite_output_dir = True

config = TrainingConfig()