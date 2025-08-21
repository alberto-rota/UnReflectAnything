from PIL import Image
from matplotlib.pyplot import ginput
import numpy as np
import os
import sys

sys.path.append("/home/alberto/UnReflectAnything/")
from losses import SSIMLoss
from utilities.visualization import rgb, panelize
import torch
from dataset.scrream import SCRREAM
from rich import print
import yaml
from dotmap import DotMap
from pipelines.depth.depth import DepthPipeline
from projections import ReflectionWarp
import time
import wandb
### Load config
CONFIG_PATH = "config.yaml"

with open(CONFIG_PATH, "r") as f:
    config_yaml = yaml.safe_load(f)
    config_parameters = config_yaml["parameters"]
    config_training_dict = {
        k: v.get("value") for k, v in config_parameters.items() if v is not None
    }
    config = DotMap(config_training_dict)
    
### Load depth estimation
depthPipeline = DepthPipeline(config, model="", device="cuda")
reflection_warp = ReflectionWarp(config.IMAGE_HEIGHT, config.IMAGE_WIDTH)
reflection_warp = reflection_warp.cuda()  # Move to GPU

### Load dataset
print("Starting dataset creation")
dataset = SCRREAM(
        root_dir="/datasets/SCRREAM/",
        scene_names=["scene01_full_00"],
        rho_s=0.6,
        eps=1e-8
    )
    
# Create dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)

valdataset = SCRREAM(
        root_dir="/datasets/SCRREAM/",
        scene_names=["scene02_full_00"],
        rho_s=0.6,
        eps=1e-8
    )
    
# Create dataloader
valdataloader = torch.utils.data.DataLoader(valdataset, batch_size=config.BATCH_SIZE, shuffle=True)

### Load model
from models import DINOv3, DINOv3toDPTRGB

dinov3_config = {
    'model_name': "facebook/dinov3-vitb16-pretrain-lvd1689m",
    'image_size': config.IMAGE_HEIGHT,  # or any size divisible by 16
    'freeze_backbone': True,
    'return_selected_layers': [2, 5, 8, 11],  # DPT extraction points
    'return_as_feature_maps': True  # Need tokens for reassembly
}
dinov3_model = DINOv3(dinov3_config)

# Create the complete model with RGB decoder
model = DINOv3toDPTRGB(
    dinov3_model=dinov3_model,
    decoder_config={
        'use_bn': True,  # Use batch norm for training stability
        'readout_type': 'project'  # or 'project' for global context
    },
    selected_layers=[2, 5, 8, 11]  # Standard DPT layers
).cuda()

### Optimization
optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
loss = SSIMLoss()

if not config.NO_WANDB:
    wandb.init(project="UnReflectAnything")
    wandb.watch(model, log="all")

import torchvision
cropper = torchvision.transforms.CenterCrop(config.IMAGE_HEIGHT)
# Test loading a batch
for epoch in range(config.EPOCHS):
    # Training loop
    model.train()
    for batch_idx, batch in enumerate(dataloader):
        start_time = time.time()
        torch.cuda.empty_cache()
        ### Set these in the config
        light_position = torch.randn((1,3))*config.DEPTH_SCALE_FACTOR/2
        light_position[0,1:] = -torch.abs(light_position[0,1:])
        light_color = torch.tensor([1.0, 0.8, 0.8]).cuda()  # Warm light

        batch["rgb"] = cropper(batch["rgb"][0:1])
        torch.cuda.empty_cache()
        with torch.no_grad():
            depth_map = depthPipeline.depth(batch["rgb"].cuda())

        # Call with point light
            result = reflection_warp.forward_point_light(
                source_image=batch["rgb"][0:1].cuda(),
                depth_map=depth_map[0:1].cuda(),
                camera_intrinsics=batch["intrinsics"].cuda()[0:1],
                camera_pose=torch.eye(4)
                .unsqueeze(0)
                .repeat(batch["rgb"].shape[0], 1, 1)
                .cuda()[0:1],
                light_position=light_position.cuda(),
                light_intensity=100.0,
                light_color=light_color.cuda(),
                surface_roughness=0.1,  # Slightly rough surface
                reflection_strength=0.9,  # Strong reflections
                return_mask=True,
                return_artifacts=True,
            )
            batch["rgb_highlighted"] = result["warped"]
            batch["highlight_masks"] = result["mask"].float().mean(dim=1, keepdim=True).bool()
        # print(f"Time taken: {time.time() - start_time} seconds")
        # print("Batch shapes:")
        # for key, tensor in batch.items():
        #     print(f"  {key}: {tensor.shape}")  # All will be [B, C, H, W]
            
            
        # print(">> Inferencing")
        input_rgb = model.dinov3.preprocess_image(batch["rgb"].cuda())
        reconstructed = model(input_rgb)
        train_loss = loss(reconstructed, batch["rgb"].cuda())
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        print(f"Training loss: {train_loss.item()}")
        # Log to wandb
        if not config.NO_WANDB:
            log_dict = {
                "epoch": epoch,
                "batch": batch_idx,
                "loss/train": train_loss.item(),
            }
            
            # Log images every 10 batches to avoid overwhelming wandb
            if batch_idx % 4 == 0:
                # Convert tensors to numpy and ensure proper format for wandb
                original_imgs = batch["rgb"][:4].cpu().numpy()  # Log first 4 images [B, C, H, W]
                reconstructed_imgs = reconstructed[:4].detach().cpu().numpy()  # [B, C, H, W]
                
                # Convert from [B, C, H, W] to [B, H, W, C] for wandb
                original_imgs = original_imgs.transpose(0, 2, 3, 1)  # [B, H, W, C]
                reconstructed_imgs = reconstructed_imgs.transpose(0, 2, 3, 1)  # [B, H, W, C]
                
                # Ensure values are in [0, 1] range
                original_imgs = torch.clamp(torch.from_numpy(original_imgs), 0, 1)
                reconstructed_imgs = torch.clamp(torch.from_numpy(reconstructed_imgs), 0, 1)
                # print(original_imgs[0].shape, reconstructed_imgs[0].shape)
                log_dict.update({
                    "images/train_original": wandb.Image(original_imgs[0].numpy()),
                    "images/train_reconstructed": wandb.Image(reconstructed_imgs[0].numpy()),
                })
            
            wandb.log(log_dict)
    # Validation loop
    model.eval()
    val_losses = []
    with torch.no_grad():
        for val_batch_idx, val_batch in enumerate(valdataloader):
            torch.cuda.empty_cache()
            val_batch["rgb"] = cropper(val_batch["rgb"][0:1])
            # Generate validation depth maps
            val_depth_map = depthPipeline.depth(val_batch["rgb"].cuda())
            
            # Validation inference
            val_input_rgb = model.dinov3.preprocess_image(val_batch["rgb"].cuda())
            val_reconstructed = model(val_input_rgb)
            val_loss = loss(val_reconstructed, val_batch["rgb"].cuda())
            val_losses.append(val_loss.item())
            print(f"Validation loss: {val_loss.item()}")
            
            # Log validation images for first batch only
            if val_batch_idx == 0 and not config.NO_WANDB:
                val_original_imgs = val_batch["rgb"][:4].cpu().numpy()  # [B, C, H, W]
                val_reconstructed_imgs = val_reconstructed[:4].detach().cpu().numpy()  # [B, C, H, W]
                
                # Convert from [B, C, H, W] to [B, H, W, C] for wandb
                val_original_imgs = val_original_imgs.transpose(0, 2, 3, 1)  # [B, H, W, C]
                val_reconstructed_imgs = val_reconstructed_imgs.transpose(0, 2, 3, 1)  # [B, H, W, C]
                
                # Ensure values are in [0, 1] range
                val_original_imgs = torch.clamp(torch.from_numpy(val_original_imgs), 0, 1)
                val_reconstructed_imgs = torch.clamp(torch.from_numpy(val_reconstructed_imgs), 0, 1)
                
                val_log_dict = {
                    "epoch": epoch,
                    "images/val_original": wandb.Image(val_original_imgs[0].numpy()),
                    "images/val_reconstructed": wandb.Image(val_reconstructed_imgs[0].numpy()),
                }
                wandb.log(val_log_dict)
    
    # Log average validation loss
    avg_val_loss = sum(val_losses) / len(val_losses)
    print(f"Epoch {epoch}: Average validation loss: {avg_val_loss:.6f}")
    
    if not config.NO_WANDB:
        wandb.log({
            "epoch": epoch,
            "loss/val": avg_val_loss,
        })

