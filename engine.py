import torch
import torch.nn as nn
import os
from rich import print
import pandas as pd
import numpy as np
import wandb
from contextlib import contextmanager, nullcontext
import gc
from typing import Union, Optional
from losses import SSIMLoss, specular_loss
from logger import get_logger
import shutil
import optimization
import utilities.engine_initializers as initialize

class Engine:
    def __init__(
        self,
        model: Union[nn.Module, str, None],
        dataset: dict,
        config: dict,
        notes: str = "",
        no_wandb: bool = False,
        **kwargs,
    ):
        """
        Initializes the Engine object for polarization-based reflection removal training.

        Args:
            model (nn.Module): The RGBPOLDecomposer model to be trained or model config.
            dataset (dict): Dictionary containing 'training', 'validation', and optionally 'test' datasets.
            config (dict): Dictionary containing config like BATCH_SIZE, LEARNING_RATE, etc.
            notes (str, optional): Additional notes for the training session. Defaults to "".
            no_wandb (bool): Whether to disable wandb logging.
            **kwargs: Additional keyword arguments.
        """
        # Set memory-efficient settings
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.autograd.set_detect_anomaly(False)  # Disable for performance
        
        # Store configuration
        self.config = config
        self.config["NOTES"] = notes
        self.no_wandb = no_wandb

        # Initialize device and directories
        device_dirs = initialize.device_and_directories()
        self.device = device_dirs["device"]
        self.RUNS_DIR = device_dirs["runs_dir"]

        # Function to set attributes from initialization functions
        def init(init_func, *args, **kwargs):
            result = init_func(*args, **kwargs)
            for key, value in result.items():
                setattr(self, key, value)
            return result

        # Initialize the model
        self.model = model
        
        # Initialize all components using engine_initializers
        init(initialize.dataloaders, dataset, config)
        init(initialize.dimensions, self.training_dl, config)
        init(initialize.hyperparameters, config)
        init(initialize.optimizers, self.model, config)
        init(initialize.loss_functions, config)
        init(initialize.schedulers, self.optimizer, config, self.training_dl)
        init(initialize.transforms, self.height, self.width)
        init(initialize.wandb, config, self.model, notes, no_wandb)
        init(initialize.tracking_metrics)
        init(initialize.setup_run_directories, self.RUNS_DIR, self.wandb, False)
        init(initialize.earlystopping, self.earlystopping_patience, self.MODELS_DIR, self.runname)
        self.config["name"] = self.runname

        
        # Save hyperparameters to json
        initialize.save_hyperparameters_json(self.RUN_DIR, self.config)
        
        # Once the run name is set, we move all the log files to the run directory
        TEMPORARY_LOG_DIR = os.path.join(self.RUNS_DIR, "temporary")
        for log_file in os.listdir(TEMPORARY_LOG_DIR):
            if log_file.endswith(".log"):
                shutil.move(os.path.join(TEMPORARY_LOG_DIR, log_file), os.path.join(self.RUN_DIR, log_file))

        # Initialize polarization-specific losses
        self.logger = get_logger(__name__, log_to_file=True, relative_log_dir=self.RUN_DIR)
        self.recon_loss = SSIMLoss()


    def trainloop(self):
        """
        The main training loop that runs through all epochs, trains the model,
        validates it, and handles early stopping and saving of the model.
        """
        for e in range(self.epochs):
            ### TRAINING + VALIDATION FOR EACH EPOCH
            self.train()  # Train the model for one epoch
            is_overfitting = self.validate()  # Train the model for one epoch

            self.csv_log_metrics() # Log the metrics to csv
            
            # Save checkpoint every few epochs
            if (e + 1) % self.config.get("SAVE_INTERVAL", 10) == 0:
                self._save_checkpoint(e)

            ### BREAK IF EARLYSTOP
            if is_overfitting == "EARLYSTOP":
                break  # Exit the training loop if early stopping condition is met

        # Log locations of important data at the end of training
        self.logger.info("TRAINING COMPLETE", context="SAVE")
        self.logger.info(
            f"Run directory: {os.path.abspath(self.RUN_DIR)}", context="SAVE"
        )
        self.logger.info(
            f"Checkpoints  : {os.path.abspath(self.MODELS_DIR)}", context="SAVE"
        )
        self.logger.info(f"Metrics      :", context="SAVE")
        self.logger.info(
            f"Training     : {os.path.abspath(os.path.join(self.RUN_DIR, 'training_metrics.csv'))}",
            context="SAVE",
        )
        self.logger.info(
            f"Validation   : {os.path.abspath(os.path.join(self.RUN_DIR, 'validation_metrics.csv'))}",
            context="SAVE",
        )

        # Remove unused IMAGES_DIR if it exists and is empty
        images_dir = os.path.join(self.RUN_DIR, "images")
        if os.path.exists(images_dir) and not os.listdir(images_dir):
            try:
                os.rmdir(images_dir)
                self.logger.info(
                    f"Removed unused directory: {images_dir}", context="SAVE"
                )
            except OSError:
                pass

    def train(self):
        """Training phase for one epoch"""
        return self.run_epoch(phase="Training")

    def validate(self):
        """Validation phase for one epoch"""
        result = self.run_epoch(phase="Validation")
        
        # Early stopping logic
        if result is not None:
            self.step["epoch"] += 1  # Increasing epoch counter
            self.LRschedulerPlateau.step(float(result))
            self.earlystopping(
                float(result),
                self.model,
                self.step["epoch"] - 1,
            )
            if self.earlystopping.early_stop:
                self.logger.info(">> [EARLYSTOPPING]: Patience Reached, Stopping Training", context="TRAINING")
                return "EARLYSTOP"
            return "IMPROVED"
        return "CONTINUE"

    def test(self):
        """Test phase"""
        result = self.run_epoch(phase="Test")
        if self.wandb is not None:
            self.log_tests()
        return result

    def log_tests(self):
        """
        Logs the test metrics to Weights and Biases.
        """
        self.logger.info(">> TEST REPORT", context="TEST")
        self.logger.info(self.metrics["Test"].describe(), context="TEST")
        self.metrics["Test"].to_csv(os.path.join(self.RUN_DIR, "test_metrics.csv"))
        if self.wandb:
            self.wandb.log({"Test/Summary": wandb.Table(dataframe=self.metrics["Test"])})

        # Log locations of important data
        self.logger.info(">> RUN DATA LOCATIONS", context="SAVE")
        self.logger.info(f"Run data directory: {os.path.abspath(self.RUN_DIR)}", context="SAVE")
        self.logger.info(f"Models saved at: {os.path.abspath(self.MODELS_DIR)}", context="SAVE")
        self.logger.info(f"Metrics CSV files:", context="SAVE")
        self.logger.info(
            f"  - Training: {os.path.abspath(os.path.join(self.RUN_DIR, 'training_metrics.csv'))}",
            context="SAVE",
        )
        self.logger.info(
            f"  - Validation: {os.path.abspath(os.path.join(self.RUN_DIR, 'validation_metrics.csv'))}",
            context="SAVE",
        )
        self.logger.info(
            f"  - Test: {os.path.abspath(os.path.join(self.RUN_DIR, 'test_metrics.csv'))}",
            context="SAVE",
        )

        # Log WandB URLs again for convenience
        if hasattr(self.wandb, "url") and self.wandb.url:
            self.logger.info(f"WandB run URL: {self.wandb.url}", context="WANDB")
            project_url = self.wandb.url.rsplit("/", 1)[0]
            self.logger.info(f"WandB project URL: {project_url}", context="WANDB")

    def csv_log_metrics(self):
        """Save metrics to CSV files"""
        if not self.metrics["Training"].empty:
            self.metrics["Training"].to_csv(
                os.path.join(self.RUN_DIR, "training_metrics.csv")
            )
        if not self.metrics["Validation"].empty:
            self.metrics["Validation"].to_csv(
                os.path.join(self.RUN_DIR, "validation_metrics.csv")
            )

    def console_log_metrics(
        self,
        stage,
        epoch=None,
        batch_idx=None,
        dataloader_len=None,
        extra_info=None,
    ):
        """
        Print metrics and status information for training, validation, or test.

        Parameters:
        -----------
        stage : str
            The current stage ('Training', 'Validation', or 'Test').
        epoch : int, optional
            Current epoch number.
        batch_idx : int, optional
            Current batch index.
        dataloader_len : int, optional
            Length of the dataloader being used.
        extra_info : str, optional
            Additional information to display in the phase indicator.
        """
        # Simple alignment function (since we don't have the original utilities)
        def align(text, width, direction="left"):
            if direction == "left":
                return f"{text:<{width}}"
            elif direction == "right":
                return f"{text:>{width}}"
            else:  # center
                return f"{text:^{width}}"
        
        epoch_batch_info = align(
            f"E {str(epoch+1)}/{self.epochs} ", 10, "right"
        ) + align(f"B {str(batch_idx+1)}/{dataloader_len} ", 10, "left")
        
        if extra_info is not None:
            phase_indicator = f"[purple]{extra_info}[/purple]"
        
        # Print header with run name and status information
        if "offline" in self.runname:
            printedrunname = "run"
        else:
            printedrunname = f'{self.runname.split("-")[0][0]}{self.runname.split("-")[1][0]}{self.runname.split("-")[2]}'
        
        metricstring = (
            align(f"{printedrunname}:", 6, "right")
            + epoch_batch_info
        )

        # Generate metrics string
        metrs = ""
        # Print metrics from the appropriate metrics dictionary
        if stage in self.metrics.keys():
            # First print the Loss column if it exists
            if (
                "Loss" in self.metrics[stage].columns
                and self.metrics[stage]["Loss"].iloc[-1] is not None
            ):
                metrs += (
                    f"[yellow]Loss[/yellow]"
                    + "="
                    + align(
                        f"{self.metrics[stage]['Loss'].iloc[-1]:.4f}",
                        6,
                        "left",
                    )
                    + " "
                )

            # Then print other columns that don't have a "/" in their name
            for m in self.metrics[stage].columns:
                if (
                    m != "Loss"
                    and "/" not in m
                    and self.metrics[stage][m].iloc[-1] is not None
                    and not m.startswith("Step/")  # Skip step metrics
                    and not m.startswith("HyperParameters/")  # Skip hyperparameter metrics
                ):
                    # Use full metric name for better readability
                    display_name = m if len(m) <= 6 else m[:6]
                    metrs += (
                        f"[yellow]{display_name}[/yellow]"
                        + "="
                        + align(
                            f"{self.metrics[stage][m].iloc[-1]:.4f}",
                            6,
                            "left",
                        )
                        + " "
                    )
        self.logger.info(metricstring + metrs, context=stage.upper())

    def log_loaded_paths(self, paths, phase):
        """Log loaded file paths for debugging"""
        if hasattr(self, 'paths_file'):
            with open(self.paths_file, mode="a") as file:
                file.write(f"{self.step[f'{phase}_batch']},{paths}\n")

    def backward_pass(self, loss_tensor, accumulate_gradients=False, phase="Training"):
        """
        Performs the backward pass, including gradient calculation, clipping, and optimization steps.

        Args:
            loss_tensor (torch.Tensor): The loss tensor to backpropagate
            accumulate_gradients (bool): If True, will update weights after backpropagation
                                        assuming gradient accumulation is complete
            phase (str): Current phase ("Training", "Validation", "Test")

        Returns:
            dict: A dictionary containing gradient norms and error status
        """
        if phase != "Training":
            loss_tensor.detach()
            torch.cuda.empty_cache()
            return {
                "grad_norm": np.nan,
                "weight_norm": np.nan,
            }

        ERROR_IN_BACKWARD_PASS = False
        try:
            loss_tensor.backward()
        except RuntimeError as e:
            self.logger.error(
                f">> [ERROR]: {e} - Skipping batch {self.step['Training_batch']} in epoch {self.step['epoch']}"
            )
            ERROR_IN_BACKWARD_PASS = True

        # Calculate gradient and weight norms using the matching pipeline model
        grad_norm, weight_norm = optimization.get_norms(
            self.model.parameters()
        )
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.GRADIENT_CLIPPING_MAX_NORM)

        # Step only if warmup phase is finished and we are backpropagating the accumulated gradients
        if accumulate_gradients:
            if not ERROR_IN_BACKWARD_PASS:
                self.optimizer.step()
                self.optimizer.zero_grad()
            self.LRscheduler.step()

        loss_tensor.detach()
        torch.cuda.empty_cache()

        return {
            "ERROR_IN_BACKWARD_PASS": ERROR_IN_BACKWARD_PASS,
            "grad_norm": grad_norm,
            "weight_norm": weight_norm,
        }

    def switch_optimizer(self, current_epoch):
        """
        Switches the optimizer if the current epoch matches the switch epoch and
        the bootstrap and refining optimizers are different.

        Args:
            current_epoch (int): The current epoch number

        Returns:
            bool: True if the optimizer was switched, False otherwise
        """
        if (
            current_epoch == self.switch_optimizer_epoch
            and self.optimizer_bootstrap_name != self.optimizer_refining_name
        ):
            self.logger.info(
                f">> [OPTIMIZER]: "
                f"Switching from [{self.optimizer_bootstrap_name}] to [{self.optimizer_refining_name}]"
            )
            self.in_optswitch_phase = True
            self.optimizer = getattr(optimization, self.optimizer_refining_name)(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
            return True
        return False



    def run_epoch(self, phase: str) -> Optional[float]:
        """
        Run one epoch of training, validation, or test.
        Adapted for polarization-based reflection removal with memory optimizations.
        
        Args:
            phase: "Training", "Validation", or "Test"

        Returns:
            Average loss for the epoch (if applicable)
        """
        # Phase setup
        is_training = phase == "Training"
        is_validation = phase == "Validation"
        is_test = phase == "Test"
        
        if is_training:
            self.model.train()
        else:
            self.model.eval()
            
        # Get dataset from the initialized dataset structure
        dataset = self.dataset[phase]
        
        if dataset is None:
            self.logger.warning(f"No dataset available for {phase}", context=phase.upper())
            return None
            
        # Create dataloader using the initialized parameters
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.config.WORKERS,
            drop_last=True,
            pin_memory=self.config.PIN_MEMORY,
            prefetch_factor=self.config.PREFETCH_FACTOR,
            shuffle=self.config.SHUFFLE,
        )
        
        if len(dataloader) == 0:
            self.logger.warning(f"Empty dataloader, skipping epoch.", context=phase.upper())
            return None
            
        epoch_losses = []
        images_logged = False
        
        # Switch optimizer if needed (for training)
        if is_training:
            self.switch_optimizer(self.step["epoch"])
        
        base_lr = self.optimizer.param_groups[0]["lr"]
        
        # Get image logging frequency from config
        image_log_interval = self.config.get("IMAGE_LOG_INTERVAL", 20)
        
        with self.choose_if_grad(phase):
            for batch_idx, sample in enumerate(dataloader):
                
                # Memory management - clear cache at start
                torch.cuda.empty_cache()
                
                # Calculate step for warmup logic
                step = self.step["epoch"] * len(dataloader) + batch_idx
                
                # Determine if we should log images on this batch
                log_images_this_batch = (
                    batch_idx > 0
                    and batch_idx % image_log_interval == 0
                    and image_log_interval > 1
                ) or (batch_idx == len(dataloader) - 1 and not images_logged)
                
                # Warmup logic
                if is_training and step < self.warmup_steps:
                    warmup_factor = step / self.warmup_steps
                    current_lr = base_lr * warmup_factor
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = current_lr
                
                # Create batch dictionary for the model
                batch = {
                    "rgb": sample["rgb"].to(self.device),
                    "AoP": sample["AoP"].to(self.device),
                    "DoP": sample["DoP"].to(self.device),
                    "f_spec": sample["f_spec"].to(self.device),
                }
                # Forward pass through the model
                decomposition = self.model(batch)
                specular = decomposition["specular"]  # [B, C, H, W]
                diffuse = decomposition["diffuse"]    # [B, C, H, W]
                
                ### Compositing Specular and Diffuse RGBA
                if specular.shape[1] == 4 and diffuse.shape[1] == 4:  # RGBA format
                    # For RGBA, use alpha compositing with diffuse as background, specular as foreground
                    spec_rgb = specular[:, :3]  # [B, 3, H, W] - foreground RGB
                    spec_alpha = specular[:, 3:4]  # [B, 1, H, W] - foreground alpha
                    diff_rgb = diffuse[:, :3]  # [B, 3, H, W] - background RGB
                    diff_alpha = diffuse[:, 3:4]  # [B, 1, H, W] - background alpha
                    
                    # Alpha compositing: C_out = C_fg * α_fg + C_bg * α_bg * (1 - α_fg)
                    # Final alpha: α_out = α_fg + α_bg * (1 - α_fg)
                    recon_rgb = spec_rgb * spec_alpha + diff_rgb * diff_alpha * (1 - spec_alpha)
                    recon_alpha = spec_alpha + diff_alpha * (1 - spec_alpha)
                    recon_alpha = torch.clamp(recon_alpha, 0, 1)
                    decomposition["recon"] = recon_rgb
                    # recon = torch.cat([recon_rgb, recon_alpha], dim=1)  # [B, 4, H, W]
                else:  # RGB format
                    # Simple addition for RGB
                    recon = specular + diffuse  # [B, 3, H, W]
                    recon = recon / recon.max()
                    recon = torch.clamp(recon, 0, 1)
                    decomposition["recon"] = recon
                
                # Compute losses using the specular_loss function
                losses = specular_loss(batch, decomposition, recon_loss=self.recon_loss)
                loss_value = losses["total"]
                
                # Backward pass for training
                if is_training:
                    try:
                        # Check if we should accumulate gradients
                        accumulate_gradients = (step >= self.warmup_steps and (batch_idx + 1) % self.gradient_accumulation_steps == 0)
                        
                        # Use the backward_pass method
                        backward_output = self.backward_pass(
                            loss_value,
                            accumulate_gradients=accumulate_gradients,
                            phase=phase
                        )
                        
                    except Exception as e:
                        self.logger.error(f"Error in backward pass: {e}", context=phase.upper())
                        continue
                
                # Track metrics
                epoch_losses.append(loss_value.item())
                self.step[f"{phase}_batch"] += 1
                
                # Update metrics dataframe 
                metrics = {
                    "Loss": loss_value.item(),
                    "HyperParameters/LR": self.optimizer.param_groups[0]["lr"],
                    f"Step/{'val' if phase == 'Validation' else ''}batch": self.step[f"{phase}_batch"],
                    f"Step/{'idx' if phase == 'Test' else 'epoch'}": self.step["epoch"],
                }
                
                # Add individual loss components if available
                if 'losses' in locals() and isinstance(losses, dict):
                    for loss_name, loss_val in losses.items():
                        if isinstance(loss_val, torch.Tensor) and loss_name != "total":
                            # Use the loss name directly (without "Loss_" prefix) for better display
                            metrics[loss_name] = loss_val.item()
                
                # Add gradient information if available
                if 'backward_output' in locals() and backward_output.get("grad_norm") is not None:
                    metrics["Gradients/GradNorm"] = backward_output["grad_norm"]
                    metrics["Gradients/WeightNorm"] = backward_output["weight_norm"]
                
                # Update the metrics dataframe
                self.metrics[phase] = pd.concat(
                    [self.metrics[phase], pd.DataFrame(metrics, index=[0])],
                    ignore_index=True,
                )
                
                # Image logging to wandb
                if log_images_this_batch and self.wandb:
                    try:
                        images = self.create_visualization_images(
                            batch, decomposition, sample
                        )
                        if images:
                            metrics.update(images)
                            images_logged = True
                    except Exception as e:
                        self.logger.warning(f"Failed to create visualization images: {e}", context=phase.upper())
                
                # Console logging
                if batch_idx % self.config.get("LOG_INTERVAL", 10) == 0:
                    extra_info = "W" if is_training and step < self.warmup_steps else None
                    self.console_log_metrics(
                        stage=phase,
                        epoch=self.step["epoch"],
                        batch_idx=batch_idx,
                        dataloader_len=len(dataloader),
                        extra_info=extra_info
                    )
                
                # WandB logging the batch metrics
                if self.wandb and batch_idx % self.logfreq_wandb == 0:
                    # Use the metrics_for_wandb function to format metrics properly
                    wandb_metrics = metrics_for_wandb(metrics, phase)
                    # Add the batch number
                    if phase == "Training":
                        batch_str = "batch"
                    elif phase == "Validation":
                        batch_str = "valbatch"
                    wandb_metrics[f"Step/{batch_str}"] = self.step[f"{phase}_batch"]
                    self.wandb.log(wandb_metrics)
                
                # Memory cleanup
                if 'batch' in locals():
                    del batch
                if 'decomposition' in locals():
                    del decomposition
                if 'recon' in locals():
                    del recon
                if 'losses' in locals():
                    del losses
                if 'backward_output' in locals():
                    del backward_output
                torch.cuda.empty_cache()
                gc.collect()
        
        # Compute average loss for epoch
        avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
        self.logger.info(f"Epoch {self.step['epoch']+1} - Average Loss: {avg_loss:.6f}", context=phase.upper())
        
        # Log epoch metrics to wandb
        if self.wandb:
            epochstr = "idx" if phase == "Test" else "epoch"
            epoch_metrics = metrics_for_wandb(
                self.metrics[phase][
                    self.metrics[phase][f"Step/{epochstr}"] == self.step["epoch"]
                ].mean(),
                phase,
            )
            # Format epoch metrics properly
            epoch_metrics = {
                key.replace(phase, f"{phase}/{epochstr}"): value
                for key, value in epoch_metrics.items()
                if phase in key
            }
            epoch_metrics[f"Step/{epochstr}"] = self.step["epoch"]
            self.wandb.log(epoch_metrics)
            
        return avg_loss
    
    def _save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            # 'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.MODELS_DIR, f'checkpoint_epoch_{epoch+1}.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.MODELS_DIR, 'best_model.pth')
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best model at epoch {epoch+1}", context="SAVE")

    @contextmanager
    def choose_if_grad(self, mode):
        """Conditionally use torch.no_grad based on the given mode."""
        with torch.no_grad() if mode in ["Validation", "Test"] else nullcontext():
            yield

    def create_visualization_images(self, batch, decomposition, sample, batch_idx=0):
        """
        Creates visualization images for polarization-based reflection removal training.
        
        Args:
            batch (dict): Input batch containing rgb, AoP, DoP, f_spec
            decomposition (dict): Model output containing specular, diffuse, recon
            sample (dict): Original sample from dataset
            batch_idx (int): Batch index to visualize
            
        Returns:
            dict: Dictionary of wandb.Image objects for visualization
        """
        try:
            import wandb
            from PIL import Image
            import torchvision.transforms as transforms
            
            # Convert tensors to CPU and detach for visualization
            def to_cpu_image(tensor):
                if tensor is None:
                    return None
                    
                if tensor.dim() == 4:  # [B, C, H, W]
                    tensor = tensor[batch_idx]  # Take first batch
                elif tensor.dim() == 3:  # [C, H, W]
                    pass
                elif tensor.dim() == 2:  # [H, W] - single channel
                    tensor = tensor.unsqueeze(0)  # Add channel dimension
                else:
                    return None
                
                # Convert to PIL Image
                tensor = tensor.cpu().detach().clamp(0, 1)
                to_pil = transforms.ToPILImage()
                return to_pil(tensor)
            
            # Create visualization dictionary
            visualization_dict = {}
            
            # Original RGB image
            # Specular component
            if 'specular' in decomposition:
                spec_img = to_cpu_image(decomposition['specular'])
                if spec_img:
                    visualization_dict[f"images/PRED_Specular"] = wandb.Image(spec_img, caption="Predicted Specular Component")
            
            # Diffuse component
            if 'diffuse' in decomposition:
                diff_img = to_cpu_image(decomposition['diffuse'])
                if diff_img:
                    visualization_dict[f"images/PRED_Diffuse"] = wandb.Image(diff_img, caption="Predicted Diffuse Component")
            
            # Reconstruction
            if 'recon' in decomposition:
                recon_img = to_cpu_image(decomposition['recon'])
                if recon_img:
                    visualization_dict[f"images/PRED_Reconstruction"] = wandb.Image(recon_img, caption="Reconstruction (Specular + Diffuse)")
            
            if 'rgb' in batch:
                rgb_img = to_cpu_image(batch['rgb'])
                if rgb_img:
                    visualization_dict[f"images/GT_RGB"] = wandb.Image(rgb_img, caption="Input RGB Image")
            
            # Ground truth specular/diffuse if available
            if 'f_spec' in batch:
                fspec_img = to_cpu_image(batch['f_spec'])
                if fspec_img:
                    visualization_dict[f"images/GT_FSpec"] = wandb.Image(fspec_img, caption="Specular Fraction")
            if 'specular' in sample:
                gt_spec_img = to_cpu_image(sample['specular'])
                if gt_spec_img:
                    visualization_dict[f"images/GT_Specular"] = wandb.Image(gt_spec_img, caption="Ground Truth Specular")
            
            if 'diffuse' in sample:
                gt_diff_img = to_cpu_image(sample['diffuse'])
                if gt_diff_img:
                    visualization_dict[f"images/GT_Diffuse"] = wandb.Image(gt_diff_img, caption="Ground Truth Diffuse")
            
            return visualization_dict
            
        except ImportError:
            self.logger.warning("wandb or PIL not available for image visualization", context="VISUALIZATION")
            return {}
        except Exception as e:
            self.logger.warning(f"Error creating visualization images: {e}", context="VISUALIZATION")
            return {}

    def reinstantiate_model_from_checkpoint(self, checkpoint_path=None):
        """
        Reinstantiate the model from checkpoint.
        """
        if checkpoint_path is None:
            # Try to load best model
            checkpoint_path = os.path.join(self.MODELS_DIR, 'best_model.pth')
            
        if not os.path.exists(checkpoint_path):
            self.logger.warning(f"Checkpoint not found at {checkpoint_path}", context="SAVE")
            return
            
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.logger.info(f"Model reinstantiated from checkpoint: {checkpoint_path}", context="SAVE")
        except Exception as e:
            self.logger.error(f"Error loading checkpoint: {e}", context="SAVE")


def metrics_for_wandb(metrics, phase):
    """Simple fallback for metrics formatting"""
    formatted_metrics = {}
    for k, v in metrics.items():
        if "Step" in k:
            # Keep Step metrics unchanged
            formatted_metrics[k] = v
        else:
            # Add phase prefix to non-Step metrics
            formatted_metrics[f"{phase}/{k}"] = v
    return formatted_metrics
