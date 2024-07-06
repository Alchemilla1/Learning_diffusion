from accelerate import Accelerator
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.utils import make_image_grid, numpy_to_pil
import torch.nn.functional as F
import os
from config import config
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from DDPM.code.models import DDPM, model
import torch
from tqdm import tqdm

# data

dataset = load_dataset("huggan/anime-faces", split="train")
dataset = dataset.select(range(21551))

## preprocess


def get_transform():
    preprocess = transforms.Compose([
        transforms.Resize(config.image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    def transform(samples):
        images = [preprocess(img.convert("RGB")) for img in samples["image"]]
        return dict(images=images)

    return transform


dataset.set_transform(get_transform())

## dataloader

dataloader = DataLoader(dataset,
                        batch_size=config.train_batch_size,
                        shuffle=True)

if __name__ == '__main__':
    model = model.cuda()
    ddpm = DDPM()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(len(dataloader) * config.num_epochs),
    )
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(config.output_dir, "logs"),
    )
    model, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, dataloader, lr_scheduler)
    global_step = 0
    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(dataloader),
                            disable=not accelerator.is_local_main_process,
                            desc=f'Epoch {epoch}')

        for step, batch in enumerate(dataloader):
            clean_images = batch["images"]
            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape, device=clean_images.device)
            bs = clean_images.shape[0]
            # Sample a random timestep for each image
            timesteps = torch.randint(0,
                                      ddpm.num_train_timesteps, (bs, ),
                                      device=clean_images.device,
                                      dtype=torch.int64)
            # Add noise to the clean images according to the noise magnitude at each timestep
            noisy_images = ddpm.add_noise(clean_images, noise, timesteps)
            with accelerator.accumulate(model):
                # Predict the noise residual
                noise_pred = model(noisy_images, timesteps,
                                   return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            progress_bar.update(1)
            logs = {
                "loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "step": global_step
            }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        if accelerator.is_main_process:
            # evaluate
            images = ddpm.sample(model, config.eval_batch_size, 3,
                                 config.image_size)
            image_grid = make_image_grid(numpy_to_pil(images), rows=4, cols=4)
            samples_dir = os.path.join(config.output_dir, 'samples')
            os.makedirs(samples_dir, exist_ok=True)
            image_grid.save(os.path.join(samples_dir, f'{global_step}.png'))
            # save models
            model.save_pretrained(config.output_dir)