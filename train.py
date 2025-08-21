from PIL import Image
from matplotlib.pyplot import ginput
import numpy as np
import os
import sys

sys.path.append("/home/alberto/UnReflectAnything/")
from losses import SSIMLoss, specular_loss
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
dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.WORKERS)

valdataset = SCRREAM(
        root_dir="/datasets/SCRREAM/",
        scene_names=["scene02_full_00"],
        rho_s=0.6,
        eps=1e-8
    )
    
# Create dataloader
valdataloader = torch.utils.data.DataLoader(valdataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.WORKERS)

### Load model
from models import DINOv3, DPTRGBDecoder, RGBPOLDecomposer, POLViTEncoder

dinov3_cfg = {
    "model_name": "facebook/dinov3-vits16-pretrain-lvd1689m",
    "image_size": 448,
    "freeze_backbone": True,
    "return_last_hidden_state": True,
    "return_as_feature_maps": False,  # Need tokens for cross-attention
}

pol_enc_cfg = {
    "in_ch": 3,
    "embed_dim": 384,  # ViT-S dimension
    "depth": 4,
    "n_heads": 12,
    "patch_size": 16,
}
decS_cfg = {
    "use_bn": True,
    "readout_type": "ignore",
    "feature_dim": 384,
    "output_image_size": (448, 448),
}
decD_cfg = {
    "use_bn": True,
    "readout_type": "ignore",
    "feature_dim": 384,
    "output_image_size": (448, 448),
}
decH_cfg = {
    "use_bn": True,
    "readout_type": "ignore",
    "feature_dim": 384,
    "output_image_size": (448, 448),
}
pol_enc = POLViTEncoder(pol_enc_cfg)
dinov3 = DINOv3(dinov3_cfg)
decS = DPTRGBDecoder(decS_cfg)
decD = DPTRGBDecoder(decD_cfg)
decH = DPTRGBDecoder(decH_cfg)

model = RGBPOLDecomposer(
    dinov3=dinov3,
    pol_encoder=pol_enc,
    pol_preprocess=None,  # Use default
    pol_cross_attn=None,  # Use default
    spec_decoder=decS,
    diffuse_decoder=decD,
    highlight_decoder=decH,
).cuda()

### Optimization
optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
recon_loss = SSIMLoss()
spec_loss = SSIMLoss()

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

        cropped_fspec = model.pol_pre.prep_fn(images=batch["f_spec"], return_tensors="pt")["pixel_values"]
        cropped_rgb = model.pol_pre.prep_fn(images=batch["rgb"], return_tensors="pt")["pixel_values"]
        
        # batch["AoP"] = cropper(batch["AoP"])
        # batch["DoP"] = cropper(batch["rgb"])
        batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        torch.cuda.empty_cache()
        # with torch.no_grad():
            # depth_map = depthPipeline.depth(batch["rgb"].cuda())

        # Call with point light
            # result = reflection_warp.forward_point_light(
            #     source_image=cropped_rgb.cuda(),
            #     depth_map=depth_map[0:1].cuda(),
            #     camera_intrinsics=batch["intrinsics"].cuda()[0:1],
            #     camera_pose=torch.eye(4)
            #     .unsqueeze(0)
            #     .repeat(batch["rgb"].shape[0], 1, 1)
            #     .cuda()[0:1],
            #     light_position=light_position.cuda(),
            #     light_intensity=100.0,
            #     light_color=light_color.cuda(),
            #     surface_roughness=0.1,  # Slightly rough surface
            #     reflection_strength=0.9,  # Strong reflections
            #     return_mask=True,
            #     return_artifacts=True,
            # )
            # batch["rgb_highlighted"] = result["warped"]
            # batch["highlight_masks"] = result["mask"].float().mean(dim=1, keepdim=True).bool()
        # print(f"Time taken: {time.time() - start_time} seconds")
        # print("Batch shapes:")
        # for key, tensor in batch.items():
        #     print(f"  {key}: {tensor.shape}")  # All will be [B, C, H, W]
            
            
        # print(">> Inferencing")
        # input_rgb = model.dinov3.preprocess_image(batch["rgb"].cuda())
        decomposition = model(batch)
        recon = decomposition["specular"]+decomposition["diffuse"]#+decomposition["highlight"]
        recon = recon/recon.max()
        recon = torch.clamp(recon, 0, 1)
        decomposition["recon"] = recon
        # --- compute losses ---
        batch["f_spec"] = cropped_fspec
        batch["rgb"] = cropped_rgb
        losses = specular_loss(batch, decomposition, recon_loss=SSIMLoss())

        # total scalar to backprop
        lossval = losses["total"]

        # --- backward + update ---
        optimizer.zero_grad()
        lossval.backward()
        optimizer.step()
        print(f"Training loss: {lossval.item()}")
        # Log to wandb
        if not config.NO_WANDB:
            log_dict = {
                "epoch": epoch,
                "batch": batch_idx,
                "loss/train": lossval.item(),
            }
            
            # Log all individual losses from the losses dict
            for loss_name, loss_value in losses.items():
                if isinstance(loss_value, torch.Tensor):
                    log_dict[f"loss/{loss_name}"] = loss_value.item()
                else:
                    log_dict[f"loss/{loss_name}"] = loss_value
            
            # Log images every 10 batches to avoid overwhelming wandb
            if batch_idx % 4 == 0:
                # Convert tensors to numpy and ensure proper format for wandb
                original_imgs = batch["rgb"][:4].cpu().numpy()  # Log first 4 images [B, C, H, W]
                reconstructed_imgs = recon[:4].detach().cpu().numpy()  # [B, C, H, W]
                f_spec_imgs = batch["f_spec"][:4].cpu().numpy()  # [B, C, H, W]
                specular_imgs = decomposition["specular"][:4].detach().cpu().numpy()  # [B, C, H, W]
                
                # Convert from [B, C, H, W] to [B, H, W, C] for wandb
                original_imgs = original_imgs.transpose(0, 2, 3, 1)  # [B, H, W, C]
                reconstructed_imgs = reconstructed_imgs.transpose(0, 2, 3, 1)  # [B, H, W, C]
                f_spec_imgs = f_spec_imgs.transpose(0, 2, 3, 1)  # [B, H, W, C]
                specular_imgs = specular_imgs.transpose(0, 2, 3, 1)  # [B, H, W, C]
                
                # Ensure values are in [0, 1] range
                original_imgs = torch.clamp(torch.from_numpy(original_imgs), 0, 1)
                reconstructed_imgs = torch.clamp(torch.from_numpy(reconstructed_imgs), 0, 1)
                f_spec_imgs = torch.clamp(torch.from_numpy(f_spec_imgs), 0, 1)
                specular_imgs = torch.clamp(torch.from_numpy(specular_imgs), 0, 1)
                # print(original_imgs[0].shape, reconstructed_imgs[0].shape)
                log_dict.update({
                    "images/train_original": wandb.Image(original_imgs[0].numpy()),
                    "images/train_reconstructed": wandb.Image(reconstructed_imgs[0].numpy()),
                    "images/train_f_spec": wandb.Image(f_spec_imgs[0].numpy()),
                    "images/train_specular": wandb.Image(specular_imgs[0].numpy()),
                })
            
            wandb.log(log_dict)
            
    # Validation loop
    model.eval()
    val_losses = []
    with torch.no_grad():
        for val_batch_idx, val_batch in enumerate(valdataloader):
            torch.cuda.empty_cache()
            
            # Preprocess validation data the same way as training
            cropped_fspec_val = model.pol_pre.prep_fn(images=val_batch["f_spec"], return_tensors="pt")["pixel_values"]
            cropped_rgb_val = model.pol_pre.prep_fn(images=val_batch["rgb"], return_tensors="pt")["pixel_values"]
            
            # Move validation batch to CUDA (same as training)
            val_batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in val_batch.items()}
            
            # Update batch with cropped images (same as training)

            
            # Validation inference (same model call as training)
            val_decomposition = model(val_batch)
            val_recon = val_decomposition["specular"] + val_decomposition["diffuse"] #+ val_decomposition["highlight"]
            
            val_recon = val_recon/val_recon.max()
            val_recon = torch.clamp(val_recon, 0, 1)
            val_decomposition["recon"] = val_recon
            val_batch["f_spec"] = cropped_fspec_val.cuda()
            val_batch["rgb"] = cropped_rgb_val.cuda()
            # Compute validation losses (same as training)
            val_losses_dict = specular_loss(val_batch, val_decomposition, recon_loss=SSIMLoss())
            val_loss = val_losses_dict["total"]
            val_losses.append(val_loss.item())
            print(f"Validation loss: {val_loss.item()}")
            
            # Log validation images for first batch only
            if val_batch_idx == 0 and not config.NO_WANDB:
                val_original_imgs = val_batch["rgb"][:4].cpu().numpy()  # [B, C, H, W]
                val_reconstructed_imgs = val_recon[:4].detach().cpu().numpy()  # [B, C, H, W]
                val_f_spec_imgs = val_batch["f_spec"][:4].cpu().numpy()  # [B, C, H, W]
                val_specular_imgs = val_decomposition["specular"][:4].detach().cpu().numpy()  # [B, C, H, W]
                
                # Convert from [B, C, H, W] to [B, H, W, C] for wandb
                val_original_imgs = val_original_imgs.transpose(0, 2, 3, 1)  # [B, H, W, C]
                val_reconstructed_imgs = val_reconstructed_imgs.transpose(0, 2, 3, 1)  # [B, H, W, C]
                val_f_spec_imgs = val_f_spec_imgs.transpose(0, 2, 3, 1)  # [B, H, W, C]
                val_specular_imgs = val_specular_imgs.transpose(0, 2, 3, 1)  # [B, H, W, C]
                
                # Ensure values are in [0, 1] range
                val_original_imgs = torch.clamp(torch.from_numpy(val_original_imgs), 0, 1)
                val_reconstructed_imgs = torch.clamp(torch.from_numpy(val_reconstructed_imgs), 0, 1)
                val_f_spec_imgs = torch.clamp(torch.from_numpy(val_f_spec_imgs), 0, 1)
                val_specular_imgs = torch.clamp(torch.from_numpy(val_specular_imgs), 0, 1)
                
                val_log_dict = {
                    "epoch": epoch,
                    "images/val_original": wandb.Image(val_original_imgs[0].numpy()),
                    "images/val_reconstructed": wandb.Image(val_reconstructed_imgs[0].numpy()),
                    "images/val_f_spec": wandb.Image(val_f_spec_imgs[0].numpy()),
                    "images/val_specular": wandb.Image(val_specular_imgs[0].numpy()),
                }
                
                # Log all individual validation losses
                for loss_name, loss_value in val_losses_dict.items():
                    if isinstance(loss_value, torch.Tensor):
                        val_log_dict[f"loss/val_{loss_name}"] = loss_value.item()
                    else:
                        val_log_dict[f"loss/val_{loss_name}"] = loss_value
                
                wandb.log(val_log_dict)
    
    # Log average validation loss
    avg_val_loss = sum(val_losses) / len(val_losses)
    print(f"Epoch {epoch}: Average validation loss: {avg_val_loss:.6f}")
    
    if not config.NO_WANDB:
        wandb.log({
            "epoch": epoch,
            "loss/val": avg_val_loss,
        })

