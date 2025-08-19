from PIL import Image
from matplotlib.pyplot import ginput
import numpy as np
import os
import sys

sys.path.append("/home/alberto/UnReflectAnything/")
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

### Load model
from models import DINOv3
from models import TransformerInpaintingDecoder

dinov3 = DINOv3({'return_as_feature_maps': True}).cuda()
decoder = TransformerInpaintingDecoder({'feature_dim': 768, 'hidden_dim': 512}).cuda()

### Optimization
optimizer = torch.optim.Adam(decoder.parameters(), lr=config.LEARNING_RATE)
loss = torch.nn.L1Loss()

wandb.init(project="UnReflectAnything")
wandb.watch(dinov3, log="all")
wandb.watch(decoder, log="all")
# Test loading a batch
for epoch in range(config.EPOCHS):
    for batch_idx, batch in enumerate(dataloader):
        start_time = time.time()
        torch.cuda.empty_cache()
        ### Set these in the config
        light_position = torch.randn((1,3))*config.DEPTH_SCALE_FACTOR/2
        light_position[0,1:] = -torch.abs(light_position[0,1:])
        light_color = torch.tensor([1.0, 0.8, 0.8]).cuda()  # Warm light

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
        print(f"Time taken: {time.time() - start_time} seconds")
        print("Batch shapes:")
        for key, tensor in batch.items():
            print(f"  {key}: {tensor.shape}")  # All will be [B, C, H, W]
            
            
        print(">> Inferencing")
        with torch.no_grad():
            features = dinov3(dinov3.preprocess_image(batch["rgb"]).cuda())['last_hidden_state']  # [B, 768, 56, 56]
        reconstructed = decoder(features, patch_h=56, patch_w=56)
        l1 = loss(reconstructed, dinov3.preprocess_image(batch["rgb"].cuda()))
        optimizer.zero_grad()
        l1.backward()
        optimizer.step()

        # Log to wandb
        if not config.NO_WANDB:
            log_dict = {
                "epoch": epoch,
                "batch": batch_idx,
                "loss/l1": l1.item(),
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
                print(original_imgs[0].shape, reconstructed_imgs[0].shape)
                log_dict.update({
                    "images/original": wandb.Image(original_imgs[0].numpy()),
                    "images/reconstructed": wandb.Image(reconstructed_imgs[0].numpy()),
                })
            
            wandb.log(log_dict)


