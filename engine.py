import gc
import os
import shutil
from contextlib import contextmanager, nullcontext
from typing import Optional, Union
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import math

import optimization
import utilities.engine_initializers as initialize
import utilities.system_ops as system_ops
import wandb
from logger import get_logger
from losses import UnReflectLoss
from polar_highlighter import PolarHighlighter, get_soft_highlight_map
import torchvision.transforms as transforms
from utilities.visualization import panelize, rgb
from utilities.ablation import Ablation

# Module-level ablation context so engine code can optionally use:
#   with ablation:
# or instance-specific via:
#   with self.ablation:
ablation = Ablation(False)


class Engine:
    def __init__(
        self,
        model: Union[nn.Module, str, None],
        dataset: dict,
        config: dict,
        notes: str = "",
        no_wandb: bool = False,
        resume_run_id: str = None,
        resume_info: Optional[dict] = None,
        will_resume: bool = False,
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
        # Bind ablation flag from config to both instance and module-level context
        try:
            enabled = bool(self.config.get("ABLATE", False))
        except Exception:
            enabled = False
        ablation.set(enabled)
        self.ablation = ablation
        # Mark if this Engine is expected to resume from an existing run
        # This must be set before any directory setup happens
        self._will_resume = bool(will_resume)

        # Initialize device and directories
        device_dirs = initialize.device_and_directories(config)
        self.device = device_dirs["device"]
        self.RUNS_DIR = device_dirs["runs_dir"]

        # Function to set attributes from initialization functions
        def init(init_func, *args, **kwargs):
            result = init_func(*args, **kwargs)
            for key, value in result.items():
                setattr(self, key, value)
            return result

        # Initialize the models
        self.model = model

        # Initialize all components using engine_initializers
        init(initialize.dataloaders, dataset, config)
        init(initialize.dimensions, self.training_dl, config)
        init(initialize.hyperparameters, config)
        init(initialize.optimizers, self.model, config)
        init(initialize.schedulers, self.optimizer, config, self.training_dl)
        init(initialize.transforms, self.height, self.width)
        # Check if we need to resume wandb run
        resume_wandb_run_id = resume_run_id
        init(initialize.wandb, config, self.model, notes, no_wandb, resume_wandb_run_id)
        init(initialize.tracking_metrics)
        
        # Skip directory setup during initialization if we're going to resume
        # The directories will be set up during resume_from_run()
        if not self._will_resume:
            init(initialize.setup_run_directories, self.RUNS_DIR, self.wandb, False)
            init(
                initialize.earlystopping,
                self.earlystopping_patience,
                self.MODELS_DIR,
                self.runname,
            )
            self.config["name"] = self.runname
        else:
            # For resume mode, if resume_info is provided we bind the existing directories immediately
            if resume_info is not None:
                self.runname = os.path.basename(resume_info.get("run_dir"))
                self.RUN_DIR = resume_info.get("run_dir")
                self.MODELS_DIR = resume_info.get("models_dir")
                self.TEST_DIR = os.path.join(self.RUN_DIR, "tests")
                self.paths_file = os.path.join(self.RUN_DIR, "loadeddata.csv")
                self.config["name"] = self.runname
            else:
                # Temporary placeholder; will be updated upon resume
                self.config["name"] = "resuming"
        self.add_polar_highlights = PolarHighlighter(
            height=self.height, width=self.width
        ).to(self.device)

        # Save hyperparameters to json only when not resuming
        if not self._will_resume:
            initialize.save_hyperparameters_json(self.RUN_DIR, self.config)

        # Once the run name is set, we move all the log files to the run directory
        if not self._will_resume:
            TEMPORARY_LOG_DIR = os.path.join(self.RUNS_DIR, "temporary")
            if os.path.exists(TEMPORARY_LOG_DIR):
                for log_file in os.listdir(TEMPORARY_LOG_DIR):
                    if log_file.endswith(".log"):
                        shutil.move(
                            os.path.join(TEMPORARY_LOG_DIR, log_file),
                            os.path.join(self.RUN_DIR, log_file),
                        )

        # Initialize polarization-specific losses
        self.logger = get_logger(
            __name__, log_to_file=True, relative_log_dir=self.RUN_DIR
        )
        self.loss = UnReflectLoss(
            weight_specular_loss=self.config.SPECULAR_LOSS_WEIGHT,
            weight_diffuse_loss=self.config.DIFFUSE_LOSS_WEIGHT,
            weight_highlight_loss=self.config.HIGHLIGHT_LOSS_WEIGHT,
            weight_component_matching=self.config.COMPONENT_MATCHING_LOSS_WEIGHT,
            weight_image_reconstruction=self.config.IMAGE_RECONSTRUCTION_LOSS_WEIGHT,
            weight_alpha_regularization=self.config.ALPHA_REGULARIZATION_LOSS_WEIGHT,
            weight_spatial_consistency=self.config.SPATIAL_CONSISTENCY_LOSS_WEIGHT,
            # New diffuse regularizers (optional; use defaults if missing)
            weight_diffuse_saturation=self.config.get("DIFFUSE_SATURATION_LOSS_WEIGHT", 0.0),
            diffuse_saturation_max=self.config.get("DIFFUSE_SATURATION_MAX", 0.35),
            weight_diffuse_ring=self.config.get("DIFFUSE_RING_LOSS_WEIGHT", 0.0),
            ring_kernel_size=int(self.config.get("RING_KERNEL_SIZE", 7)),
            ring_var_weight=float(self.config.get("RING_VAR_WEIGHT", 0.5)),
            ring_texture_weight=float(self.config.get("RING_TEXTURE_WEIGHT", 1.0)),
            # HL regression extra knobs (optional in config; keep defaults if missing)
            hlreg_balance_mode=self.config.get("HLREG_BALANCE_MODE", "none"),
            hlreg_pos_weight=self.config.get("HLREG_POS_WEIGHT", 1.0),
            hlreg_focal_gamma=self.config.get("HLREG_FOCAL_GAMMA", 0.0),
        ).to(self.device)

        # Memory management settings for optimal GPU memory usage
        self.memory_cleanup_frequency = config.get(
            "MEMORY_CLEANUP_FREQUENCY", 5
        )  # Clean every N batches
        self.aggressive_cleanup = config.get(
            "AGGRESSIVE_MEMORY_CLEANUP", True
        )  # Use gpuClean utility
        self.memory_monitoring = config.get(
            "MEMORY_MONITORING", False
        )  # Log memory usage

    def composite_specular_diffuse(
        self, specular: torch.Tensor, diffuse: torch.Tensor
    ) -> torch.Tensor:
        """
        Composite specular and diffuse components into a reconstructed image.
        Optimized for memory efficiency with strategic cleanup.

        Args:
            specular (torch.Tensor): Specular component [B, C, H, W]
            diffuse (torch.Tensor): Diffuse component [B, C, H, W]

        Returns:
            torch.Tensor: Reconstructed image [B, 3, H, W] or [B, 4, H, W]
        """
        # Log memory usage before compositing if monitoring is enabled
        if self.memory_monitoring:
            self._log_memory_usage("Before compositing")

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

            # Clean up intermediate tensors to free memory
            del spec_rgb, spec_alpha, diff_rgb, diff_alpha
            torch.cuda.empty_cache()

            return recon_rgb
        else:  # RGB format
            # Simple addition for RGB
            recon = specular + diffuse  # [B, 3, H, W]
            recon = recon / recon.max()
            recon = torch.clamp(recon, 0, 1)
            return recon

    def trainloop(self):
        """
        The main training loop that runs through all epochs, trains the model,
        validates it, and handles early stopping and saving of the model.
        """
        # Determine starting epoch (for resume functionality)
        start_epoch = getattr(self, 'start_epoch', 0)
        
        for e in range(start_epoch, self.epochs):
            ### TRAINING + VALIDATION FOR EACH EPOCH
            self.train()  # Train the model for one epoch
            is_overfitting = self.validate()  # Train the model for one epoch

            self.csv_log_metrics()  # Log the metrics to csv

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
        self.logger.info("Metrics      :", context="SAVE")
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
                self.optimizer,
                self.config,
                self.wandb,
            )
            if self.earlystopping.early_stop:
                self.logger.info(
                    ">> [EARLYSTOPPING]: Patience Reached, Stopping Training",
                    context="TRAINING",
                )
                return "EARLYSTOP"
            return "IMPROVED"
        return "CONTINUE"

    def test(self):
        """Test phase"""
        # Persistent test index: auto test after training is idx 0
        test_idx = self._load_and_increment_test_index()
        result = self.run_epoch(phase="Test", test_idx=test_idx)
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
            self.wandb.log(
                {"Test/Summary": wandb.Table(dataframe=self.metrics["Test"])}
            )

        # Log locations of important data
        self.logger.info(">> RUN DATA LOCATIONS", context="SAVE")
        self.logger.info(
            f"Run data directory: {os.path.abspath(self.RUN_DIR)}", context="SAVE"
        )
        self.logger.info(
            f"Models saved at: {os.path.abspath(self.MODELS_DIR)}", context="SAVE"
        )
        self.logger.info("Metrics CSV files:", context="SAVE")
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
            f"E {str(epoch + 1)}/{self.epochs} ", 10, "right"
        ) + align(f"B {str(batch_idx + 1)}/{dataloader_len} ", 10, "left")

        # Note: extra_info is used in console logging but phase_indicator was unused

        # Print header with run name and status information
        if "offline" in self.runname:
            printedrunname = "run"
        else:
            printedrunname = f"{self.runname.split('-')[0][0]}{self.runname.split('-')[1][0]}{self.runname.split('-')[2]}"

        metricstring = align(f"{printedrunname}:", 6, "right") + epoch_batch_info

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
                    "[yellow]Loss[/yellow]"
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
                    and not m.startswith(
                        "HyperParameters/"
                    )  # Skip hyperparameter metrics
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
        if hasattr(self, "paths_file"):
            with open(self.paths_file, mode="a") as file:
                file.write(f"{self.step[f'{phase}_batch']},{paths}\n")

    def backward_pass(
        self,
        loss_tensor,
        accumulate_gradients=False,
        phase="Training",
        submodules_to_monitor=None,
    ):
        """
        Performs the backward pass, including gradient calculation, clipping, and optimization steps.
        Optimized for memory efficiency with strategic cleanup.

        Args:
            loss_tensor (torch.Tensor): The loss tensor to backpropagate
            accumulate_gradients (bool): If True, will update weights after backpropagation
                                        assuming gradient accumulation is complete
            phase (str): Current phase ("Training", "Validation", "Test")
            submodules_to_monitor (dict, optional): Dictionary mapping submodule names to submodule objects
                                                   for separate gradient/weight norm monitoring

        Returns:
            dict: A dictionary containing gradient norms, weight norms, and error status
        """
        # Initialize return values
        grad_norm = np.nan
        weight_norm = np.nan
        submodule_norms = {}

        if phase != "Training":
            loss_tensor.detach()
            torch.cuda.empty_cache()
            return {
                "ERROR_IN_BACKWARD_PASS": False,
                "grad_norm": grad_norm,
                "weight_norm": weight_norm,
                "submodule_norms": submodule_norms,
            }

        # Log memory usage before backward pass if monitoring is enabled
        if self.memory_monitoring:
            self._log_memory_usage("Before backward pass")

        ERROR_IN_BACKWARD_PASS = False
        try:
            loss_tensor.backward()
        except RuntimeError as e:
            self.logger.error(
                f">> [ERROR]: {e} - Skipping batch {self.step['Training_batch']} in epoch {self.step['epoch']}"
            )
            ERROR_IN_BACKWARD_PASS = True

        # Gradient clipping with memory optimization
        if (
            self.config.GRADIENT_CLIPPING_MAX_NORM is not None
            and not ERROR_IN_BACKWARD_PASS
        ):
            try:
                # Calculate gradient and weight norms for the whole model
                grad_norm, weight_norm = optimization.get_norms(self.model.parameters())

                # Calculate norms for specific submodules if requested
                if submodules_to_monitor is not None:
                    for submodule_name, submodule in submodules_to_monitor.items():
                        if submodule is None:
                            continue
                        try:
                            sub_grad_norm, sub_weight_norm = optimization.get_norms(
                                submodule.parameters()
                            )
                            submodule_norms[submodule_name] = {
                                "grad_norm": sub_grad_norm,
                                "weight_norm": sub_weight_norm,
                            }
                        except Exception as e:
                            self.logger.warning(
                                f"Failed to compute norms for submodule {submodule_name}: {e}"
                            )
                            submodule_norms[submodule_name] = {
                                "grad_norm": np.nan,
                                "weight_norm": np.nan,
                            }

                if self.config.GRADIENT_CLIPPING_MAX_NORM > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=self.config.GRADIENT_CLIPPING_MAX_NORM,
                    )
            except Exception as e:
                self.logger.warning(f"Gradient clipping failed: {e}")

        # Step only if warmup phase is finished and we are backpropagating the accumulated gradients
        if accumulate_gradients:
            if not ERROR_IN_BACKWARD_PASS:
                self.optimizer.step()
                self.optimizer.zero_grad()
            self.LRscheduler.step()

        # Aggressive memory cleanup after backward pass
        loss_tensor.detach()
        del loss_tensor

        # Clean up gradients and intermediate computations
        if self.aggressive_cleanup:
            # Only clear gradients if we actually performed an optimizer step; otherwise
            # leave them to accumulate for gradient accumulation.
            if accumulate_gradients:
                for param in self.model.parameters():
                    if param.grad is not None:
                        param.grad.detach_()
                        param.grad = None

            # Force garbage collection and cache clearing
            torch.cuda.empty_cache()
            gc.collect()

        return {
            "ERROR_IN_BACKWARD_PASS": ERROR_IN_BACKWARD_PASS,
            "grad_norm": grad_norm,
            "weight_norm": weight_norm,
            "submodule_norms": submodule_norms,
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

    def run_epoch(self, phase: str, test_idx: Optional[int] = None) -> Optional[float]:
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

        if is_training:
            self.model.train()
        else:
            self.model.eval()

        # Get dataset from the initialized dataset structure
        dataset = self.dataset[phase]

        if dataset is None:
            self.logger.warning(
                f"No dataset available for {phase}", context=phase.upper()
            )
            return None

        cpu_affinity = os.sched_getaffinity(os.getpid())
        AUTO_NUM_WORKERS = int(math.floor(0.9 * len(list(cpu_affinity))))
        # Create dataloader using the initialized parameters
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=AUTO_NUM_WORKERS
            if self.config.NUM_WORKERS == "auto"
            else self.config.NUM_WORKERS,
            drop_last=True,
            pin_memory=self.config.PIN_MEMORY,
            prefetch_factor=self.config.PREFETCH_FACTOR,
            shuffle=self.config.SHUFFLE,
        )

        if len(dataloader) == 0:
            self.logger.warning(
                "Empty dataloader, skipping epoch.", context=phase.upper()
            )
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
                # Strategic memory cleanup at batch start
                if batch_idx == 0 or batch_idx % self.memory_cleanup_frequency == 0:
                    torch.cuda.empty_cache()
                    if self.aggressive_cleanup:
                        gc.collect()

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

                ### Add polarization highlights
                random_light_pos = self.add_polar_highlights.sample_light_source(
                    dist_to_camera=self.config.LIGHT_DISTANCE_RANGE,
                    left_right_angle=self.config.LIGHT_LEFT_RIGHT_ANGLE,
                    above_below_angle=self.config.LIGHT_ABOVE_BELOW_ANGLE,
                    batch_size=self.batch_size,
                    device=self.device,
                )
                highlight_result = self.add_polar_highlights(
                    rgb=sample["diffuse"].to(self.device, non_blocking=True),
                    light_pos=random_light_pos,
                    noise=random.uniform(0, float(self.config.NOISE)),
                    noise_type=self.config.NOISE_TYPE, 
                    noise_octaves=random.randint(1, int(self.config.NOISE_OCTAVES)) if int(self.config.NOISE_OCTAVES) > 1 else 1,
                    noise_persistence=random.uniform(0, float(self.config.NOISE_PERSISTENCE)),
                    surface_roughness=random.uniform(4, float(self.config.SURFACE_ROUGHNESS)),
                    intensity=random.uniform(0.5, float(self.config.INTENSITY)),
                )

                # Compute soft highlight map
                real_highlight_soft_mask = get_soft_highlight_map(
                    sample["diffuse"].to(self.device, non_blocking=True),
                    threshold=self.config.SOFT_HIGHLIGHT_THRESHOLD,
                )
                # Compute inverse binary mask to mask out real highlights from the loss computation
                real_highlight_inverse_binary_mask = torch.logical_not(
                    torch.nn.functional.max_pool2d(
                        real_highlight_soft_mask,
                        kernel_size=self.config.REAL_HIGHLIGHT_DILATION,
                        stride=1,
                        padding=self.config.REAL_HIGHLIGHT_DILATION // 2,
                    )
                    > 0
                ).int()

                # We use the inverse binary mask only at training time
                if phase == "Training":
                    lossmask = real_highlight_inverse_binary_mask
                else:
                    lossmask = None
                # Include BOTH virtual and real highlights in GT highlight map
                # Model learns to predict ALL highlights in the highlight channel
                # But diffuse supervision is masked in real highlight regions
                # Model must learn to inpaint/generalize diffuse behind real highlights
                real_and_virtual_highlights = (
                    highlight_result["highlight"] + real_highlight_soft_mask
                ).clamp(0, 1)
                

                ### Constructing ground truth dict
                rgb_highlighted = highlight_result["rgb_highlighted"]
                specular = sample["specular"].to(self.device, non_blocking=True)
                
                # Ground truth diffuse: original RGB (contains real highlights)
                diffuse = sample["diffuse"].to(self.device, non_blocking=True)
                
                gt_decomposition = {
                    "diffuse": diffuse,  # Contains real highlights, but masked during loss
                    "rgb_highlighted": rgb_highlighted,
                    "specular": specular,
                    "highlight": real_and_virtual_highlights,  # BOTH real and virtual
                }
                # Polarization data is added only if available.
                if "AoP" in sample:
                    gt_decomposition["AoP"] = sample["AoP"].to(
                        self.device, non_blocking=True
                    )
                if "DoP" in sample:
                    gt_decomposition["DoP"] = sample["DoP"].to(
                        self.device, non_blocking=True
                    )
                if "f_spec" in sample:
                    gt_decomposition["f_spec"] = sample["f_spec"].to(
                        self.device, non_blocking=True
                    )
                del sample  # Clean up. All necessary data is already on GPU.

                # Constructing model input dict
                model_input = {"rgb": gt_decomposition["rgb_highlighted"]}
                if self.config.MODEL.MODEL_CLASS == "RGBPOLDecomposer":
                    model_input["AoP"] = gt_decomposition["AoP"]
                    model_input["DoP"] = gt_decomposition["DoP"]

                # Log memory usage before forward pass if monitoring
                if self.memory_monitoring and batch_idx % 10 == 0:
                    self._log_memory_usage(f"Before forward pass - batch {batch_idx}")
                
                if self.ablation:
                    print("This is an ablation block")
                
                ### Forward pass
                pred_decomposition = self.model(model_input)

                ### COMPUTE LOSS FUNCTION
                losses = self.loss(
                    prediction=pred_decomposition,
                    ground_truth=gt_decomposition,
                    mask=lossmask,
                )
                loss_value = losses["total"]

                # Compositing the reconstructed image - for visualization purposes
                pred_decomposition["rgb_highlighted"] = self.loss.reconstruct_image(
                    pred_decomposition
                )
                # Adding the loss mask to the gt_decomposition to log it on wandb
                if phase == "Training":
                    gt_decomposition["lossmask"] = lossmask
                    gt_decomposition["masked_diffuse"] = diffuse * lossmask
                ### Backward pass
                backward_output = None
                if is_training:
                    try:
                        # Check if we should accumulate gradients
                        accumulate_gradients = (
                            step >= self.warmup_steps
                            and (batch_idx + 1) % self.gradient_accumulation_steps == 0
                        )

                        # Use the backward_pass method
                        backward_output = self.backward_pass(
                            loss_value,
                            accumulate_gradients=accumulate_gradients,
                            phase=phase,
                            submodules_to_monitor={
                                "highlight_decoder": self.model.decoders["highlight"]
                                if "highlight" in self.model.decoders
                                else None,
                                "diffuse_decoder": self.model.decoders["diffuse"]
                                if "diffuse" in self.model.decoders
                                else None,
                                "specular_decoder": self.model.decoders["specular"]
                                if "specular" in self.model.decoders
                                else None,
                                "dinov3": self.model.dinov3,
                            },
                        )

                    except Exception as e:
                        self.logger.error(
                            f"Error in backward pass: {e}", context=phase.upper()
                        )
                        continue

                # Track metrics
                epoch_losses.append(loss_value.item())
                self.step[f"{phase}_batch"] += 1

                # Update metrics dataframe
                metrics = {
                    "Loss": loss_value.item(),
                    "HyperParameters/LR": self.optimizer.param_groups[0]["lr"],
                    f"Step/{'val' if phase == 'Validation' else ''}batch": self.step[
                        f"{phase}_batch"
                    ],
                    f"Step/{'idx' if phase == 'Test' else 'epoch'}": self.step["epoch"],
                }
                if phase == "Test" and test_idx is not None:
                    metrics["Step/test_idx"] = test_idx
                    metrics["Index"] = float(test_idx)

                # Add individual loss components if available
                if isinstance(losses, dict):
                    for loss_name, loss_val in losses.items():
                        if isinstance(loss_val, torch.Tensor) and loss_name != "total":
                            # Use the loss name directly (without "Loss_" prefix) for better display
                            metrics[loss_name] = loss_val.item()

                # Add gradient information if available
                if (
                    backward_output is not None
                    and backward_output.get("grad_norm") is not None
                ):
                    metrics["Gradients/MODEL_GradNorm"] = backward_output["grad_norm"]
                    metrics["Gradients/MODEL_WeightNorm"] = backward_output[
                        "weight_norm"
                    ]

                    # Add submodule gradient norms if available
                    if "submodule_norms" in backward_output:
                        for submodule_name, norms in backward_output[
                            "submodule_norms"
                        ].items():
                            metrics[f"Gradients/{submodule_name}_GradNorm"] = norms[
                                "grad_norm"
                            ]
                            metrics[f"Gradients/{submodule_name}_WeightNorm"] = norms[
                                "weight_norm"
                            ]

                # Update the metrics dataframe
                self.metrics[phase] = pd.concat(
                    [self.metrics[phase], pd.DataFrame(metrics, index=[0])],
                    ignore_index=True,
                )

                # Image logging to wandb - with aggressive cleanup after
                if log_images_this_batch and self.wandb:
                    # try:
                        # Create a copy of sample for visualization (since we deleted it earlier)
                        gt_data = {
                            "rgb": gt_decomposition["rgb_highlighted"].cpu(),
                        }
                        # Add optional polarization data if available
                        if "AoP" in gt_decomposition:
                            gt_data["AoP"] = gt_decomposition["AoP"].cpu()
                        if "DoP" in gt_decomposition:
                            gt_data["DoP"] = gt_decomposition["DoP"].cpu()
                        if "f_spec" in gt_decomposition:
                            gt_data["f_spec"] = gt_decomposition["f_spec"].cpu()
                        if "highlight" in gt_decomposition:
                            gt_data["highlight"] = gt_decomposition["highlight"].cpu()
                        if "rgb_highlighted" in gt_decomposition:
                            gt_data["rgb_highlighted"] = gt_decomposition[
                                "rgb_highlighted"
                            ].cpu()

                        images = self.create_visualization_images(
                            gt_decomposition,
                            pred_decomposition,
                            gt_data,
                            as_single_panel=True,
                            batch_idx=batch_idx,
                            phase=phase,
                            test_idx=test_idx if phase == "Test" else None,
                            # Ensure images are logged under the correct Test prefix inside helper
                            # (helper will no-op prefixing for non-Test phases)
                        )
                        if images:
                            # Images are logged directly inside helper with proper prefixes.
                            # Avoid adding to metrics to prevent duplicate logging.
                            images_logged = True

                        # Clean up sample copy immediately
                        del gt_data

                    # except Exception as e:
                    #     self.logger.warning(
                    #         f"Failed to create visualization images: {e}",
                    #         context=phase.upper(),
                    #     )

                # Console logging
                if batch_idx % self.config.get("LOG_INTERVAL", 10) == 0:
                    extra_info = (
                        "W" if is_training and step < self.warmup_steps else None
                    )
                    self.console_log_metrics(
                        stage=phase,
                        epoch=self.step["epoch"],
                        batch_idx=batch_idx,
                        dataloader_len=len(dataloader),
                        extra_info=extra_info,
                    )

                # WandB logging the batch metrics
                if self.wandb and batch_idx % self.logfreq_wandb == 0:
                    # Use the self._prepare_metrics_for_wandb function to format metrics properly
                    wandb_metrics = self._prepare_metrics_for_wandb(metrics, phase)
                    # Add the batch number
                    if phase == "Training":
                        batch_str = "batch"
                    elif phase == "Validation":
                        batch_str = "valbatch"
                    elif phase == "Test":
                        batch_str = f"test_idx_{test_idx}"
                    wandb_metrics[f"Step/{batch_str}"] = self.step[f"{phase}_batch"]
                    if phase == "Test" and test_idx is not None:
                        wandb_metrics["Step/test_idx"] = test_idx
                        # Rewrite all Test metric keys to include test index as prefix
                        prefixed_metrics = {}
                        for key, value in wandb_metrics.items():
                            if isinstance(key, str) and key.startswith("Test/"):
                                rest = key[len("Test/"):]
                                new_key = f"Test/test_idx_{test_idx}/{rest}"
                                prefixed_metrics[new_key] = value
                            else:
                                prefixed_metrics[key] = value
                        wandb_metrics = prefixed_metrics
                    self.wandb.log(wandb_metrics)

                # Strategic memory cleanup - more aggressive for larger batches
                self._strategic_memory_cleanup(
                    phase, batch_idx, exclude_vars=["self", "dataloader"]
                )

                # Clean up variables that are no longer needed
                del gt_decomposition, pred_decomposition, losses, loss_value
                if backward_output is not None:
                    del backward_output

                # Force cleanup every few batches for maximum memory efficiency
                if batch_idx % self.memory_cleanup_frequency == 0:
                    torch.cuda.empty_cache()
                    gc.collect()

        # Compute average loss for epoch
        avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
        self.logger.info(
            f"Epoch {self.step['epoch'] + 1} - Average Loss: {avg_loss:.6f}",
            context=phase.upper(),
        )

        # Log epoch metrics to wandb
        if self.wandb:
            # For Test phase, filter using the stable column 'Step/test_idx'.
            # For Training/Validation, filter using 'Step/epoch' as before.
            if phase == "Test" and test_idx is not None:
                df = self.metrics[phase]
                if "Step/test_idx" in df.columns:
                    selector = df["Step/test_idx"] == test_idx
                    df_mean = df[selector].mean(numeric_only=True)
                else:
                    # Fallback: no explicit test_idx column found; average all rows
                    df_mean = df.mean(numeric_only=True)
                epoch_label = f"test_idx_{test_idx}"
            else:
                df_mean = self.metrics[phase][
                    self.metrics[phase]["Step/epoch"] == self.step["epoch"]
                ].mean()
                epoch_label = "epoch"

            epoch_metrics = self._prepare_metrics_for_wandb(df_mean, phase)
            # Build final logging dict with strict Test prefixing convention
            final_epoch_metrics = {}
            if phase == "Test" and test_idx is not None:
                for key, value in epoch_metrics.items():
                    if key.startswith("Test/"):
                        rest = key[len("Test/"):]
                        new_key = f"Test/test_idx_{test_idx}/{rest}"
                        final_epoch_metrics[new_key] = value
                # Record step keys
                final_epoch_metrics[f"Step/test_idx_{test_idx}"] = test_idx
                final_epoch_metrics["Step/test_idx"] = test_idx
            else:
                # Training/Validation: keep standard epoch prefixing
                for key, value in epoch_metrics.items():
                    if key.startswith(phase + "/"):
                        final_epoch_metrics[key.replace(phase, f"{phase}/{epoch_label}")] = value
                final_epoch_metrics[f"Step/{epoch_label}"] = self.step["epoch"]

            self.wandb.log(final_epoch_metrics)

        return avg_loss

    def _save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint with enhanced state information"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "model_class_name": self.model.__class__.__name__,
            "model_class_module": self.model.__class__.__module__,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
            "runname": getattr(self, 'runname', None),
            "wandb_run_id": getattr(self.wandb, 'id', None) if self.wandb else None,
        }
        
        # Add scheduler states if available (save what Engine actually uses)
        if hasattr(self, 'LRscheduler') and self.LRscheduler is not None:
            try:
                checkpoint["LRscheduler_state_dict"] = self.LRscheduler.state_dict()
            except Exception as e:
                self.logger.warning(f"Could not save LRscheduler state: {e}")
        if hasattr(self, 'LRschedulerPlateau') and self.LRschedulerPlateau is not None:
            try:
                checkpoint["LRschedulerPlateau_state_dict"] = self.LRschedulerPlateau.state_dict()
            except Exception as e:
                self.logger.warning(f"Could not save LRschedulerPlateau state: {e}")
        
        # Add early stopping state if available
        if hasattr(self, 'earlystopping') and self.earlystopping is not None:
            checkpoint["earlystopping_state"] = {
                "val_loss_min": getattr(self.earlystopping, 'val_loss_min', float('inf')),
                "counter": getattr(self.earlystopping, 'counter', 0),
                "patience": getattr(self.earlystopping, 'patience', 0),
            }
        
        # Add training metrics history if available
        if hasattr(self, 'training_metrics') and self.training_metrics:
            checkpoint["training_metrics_history"] = self.training_metrics
        if hasattr(self, 'validation_metrics') and self.validation_metrics:
            checkpoint["validation_metrics_history"] = self.validation_metrics

        # Save regular checkpoint
        checkpoint_path = os.path.join(
            self.MODELS_DIR, f"checkpoint_epoch_{epoch + 1}.pth"
        )
        torch.save(checkpoint, checkpoint_path)

        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.MODELS_DIR, "best_model.pth")
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best model at epoch {epoch + 1}", context="SAVE")

    @contextmanager
    def choose_if_grad(self, mode):
        """Conditionally use torch.no_grad based on the given mode."""
        with torch.no_grad() if mode in ["Validation", "Test"] else nullcontext():
            yield

    def create_visualization_images(
        self,
        gt_decomposition,
        pred_decomposition,
        sample,
        as_single_panel=True,
        batch_idx=0,
        phase: str = None,
        test_idx: Optional[int] = None,
    ):
        """
        Creates visualization images for polarization-based reflection removal training.
        Optimized for memory efficiency with aggressive cleanup.

        Args:
            gt_decomposition (dict): Input gt_decomposition containing rgb, AoP, DoP, f_spec
            pred_decomposition (dict): Model output containing specular, diffuse, recon
            sample (dict): Original sample from dataset
            as_single_panel (bool): Whether to create a single panel image
            batch_idx (int): Batch index to visualize

        Returns:
            dict: Dictionary of wandb.Image objects for visualization
        """
        if as_single_panel:
            visualization_dict = {}

            # Collect all unique keys from both dicts
            all_keys = list(sorted(set(pred_decomposition.keys()) | set(gt_decomposition.keys())))

            def make_black_image(size=(448, 448)):
                # Returns a black RGB image tensor [3, H, W]
                return torch.zeros(3, size[0], size[1])

            # Build prediction row, using black image if key missing
            prediction_row = panelize(
                *[
                    rgb(
                        pred_decomposition[comp_name][0][:3].detach() if comp_name in pred_decomposition else make_black_image(),
                        as_tensor=True,
                        resize=(448, 448),
                        colormap="gray",
                        # border={"color": "#ffffff   ", "thickness": 1 if comp_name not in pred_decomposition else 0} ,
                        label={
                            "position": "top-left",
                            "height": 40,
                            "margin": 1 if comp_name not in pred_decomposition else 0,
                            "text": f"PRED {comp_name.capitalize()}" if comp_name in pred_decomposition else "NA",
                        },
                    )
                    for comp_name in all_keys
                ],
                mode="horizontal",
            )

            # Build GT row, using black image if key missing
            gt_row = panelize(
                *[
                    rgb(
                        gt_decomposition[comp_name][0][:3].detach() if comp_name in gt_decomposition else make_black_image(),
                        as_tensor=True,
                        resize=(448, 448),
                        colormap="gray",
                        # border={"color": "#ffffff   ", "thickness": 1 if comp_name not in gt_decomposition else 0} ,
                        label={
                            "position": "top-left",
                            "height": 40,
                            "margin": 1 if comp_name not in gt_decomposition else 0,
                            "text": f"GT {comp_name.capitalize()}" if comp_name in gt_decomposition else "NA",
                        },
                    )
                    for comp_name in all_keys
                ],
                mode="horizontal",
            )

            prediction_panel_loggable = panelize(
                prediction_row, gt_row, mode="vertical", resize_to_match=False
            )
            self._add_image_safely(
                visualization_dict,
                "images/Comparison_panel",
                prediction_panel_loggable,
                caption="Comparison Panel",
                batch_idx=batch_idx,
                phase=phase,
                test_idx=test_idx,
            )
            return visualization_dict
        else:
        
            # Create visualization dictionary
            visualization_dict = {}

            # Predicted components - dynamically detect available components
            # First add known special components
            if "recon" in pred_decomposition:
                self._add_image_safely(
                    visualization_dict,
                    "images/PRED_Reconstruction",
                    pred_decomposition["recon"],
                    "Reconstruction",
                    batch_idx,
                    phase=phase,
                    test_idx=test_idx,
                )

            # Then add any other model output components
            for comp_name, comp_tensor in pred_decomposition.items():
                if (
                    comp_name != "recon"
                    and isinstance(comp_tensor, torch.Tensor)
                    and comp_tensor.dim() == 4
                ):
                    # Create nice display name
                    display_name = comp_name.replace("_", " ").title()
                    wandb_key = f"images/PRED_{comp_name.capitalize()}"
                    caption = f"Predicted {display_name} Component"
                    self._add_image_safely(
                        visualization_dict,
                        wandb_key,
                        comp_tensor,
                        caption,
                        batch_idx,
                        phase=phase,
                        test_idx=test_idx,
                    )

            # Input images
            input_images = [
                ("images/GT_Diffuse", "diffuse", "Input RGB Image"),
                (
                    "images/GT_RGB_Highlighted",
                    "rgb_highlighted",
                    "Input RGB Highlighted Image",
                ),
                ("images/GT_Highlight", "highlight", "Input RGB Highlighted Image"),
                ("images/GT_FSpec", "f_spec", "Specular Fraction"),
                (
                    "images/GT_MaskedDiffuse",
                    "masked_diffuse",
                    "Masked Diffuse Image",
                ),
            ]

            for key, tensor_key, caption in input_images:
                if (
                    tensor_key in gt_decomposition
                    and gt_decomposition[tensor_key] is not None
                ):
                    self._add_image_safely(
                        visualization_dict,
                        key,
                        gt_decomposition[tensor_key],
                        caption,
                        batch_idx,
                        phase=phase,
                        test_idx=test_idx,
                    )

            # Ground truth components - dynamically detect available components
            if sample is not None:
                # Add ground truth components dynamically
                for comp_name, comp_tensor in sample.items():
                    # Skip non-component keys
                    if comp_name in [
                        "rgb",
                        "AoP",
                        "DoP",
                        "f_spec",
                        "rgb_highlighted",
                        "intrinsics",
                    ] or not isinstance(comp_tensor, torch.Tensor):
                        continue

                    # Create nice display name
                    display_name = comp_name.replace("_", " ").title()
                    wandb_key = f"images/GT_{comp_name.capitalize()}"
                    caption = f"Ground Truth {display_name}"
                    self._add_image_safely(
                        visualization_dict,
                        wandb_key,
                        comp_tensor,
                        caption,
                        batch_idx,
                        phase=phase,
                        test_idx=test_idx,
                    )

            # Final cleanup
            torch.cuda.empty_cache()
            gc.collect()

            return visualization_dict

    def reinstantiate_model_from_checkpoint(self, checkpoint_path=None):
        """
        Reinstantiate the model from checkpoint.
        """
        if checkpoint_path is None:
            # Try to load best model
            checkpoint_path = os.path.join(self.MODELS_DIR, "weights_best.pt")

        if not os.path.exists(checkpoint_path):
            self.logger.warning(
                f"Checkpoint not found at {checkpoint_path}", context="SAVE"
            )
            return

        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            # Load schedulers if present (support both new and legacy keys)
            try:
                if "LRscheduler_state_dict" in checkpoint and hasattr(self, 'LRscheduler') and self.LRscheduler is not None:
                    self.LRscheduler.load_state_dict(checkpoint["LRscheduler_state_dict"])
                if "LRschedulerPlateau_state_dict" in checkpoint and hasattr(self, 'LRschedulerPlateau') and self.LRschedulerPlateau is not None:
                    self.LRschedulerPlateau.load_state_dict(checkpoint["LRschedulerPlateau_state_dict"])
                # Backward-compatibility: legacy single scheduler key
                if "scheduler_state_dict" in checkpoint:
                    loaded = False
                    if hasattr(self, 'LRscheduler') and self.LRscheduler is not None:
                        try:
                            self.LRscheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                            loaded = True
                        except Exception:
                            pass
                    if not loaded and hasattr(self, 'LRschedulerPlateau') and self.LRschedulerPlateau is not None:
                        try:
                            self.LRschedulerPlateau.load_state_dict(checkpoint["scheduler_state_dict"])
                            loaded = True
                        except Exception:
                            pass
                    if not loaded:
                        self.logger.warning("Found legacy scheduler_state_dict but no compatible scheduler attribute to load into", context="SAVE")
            except Exception as e:
                self.logger.warning(f"Could not load scheduler state(s): {e}")
            self.logger.info(
                f"Model reinstantiated from checkpoint: {checkpoint_path}",
                context="SAVE",
            )
        except Exception as e:
            self.logger.error(f"Error loading checkpoint: {e}", context="SAVE")

    @staticmethod
    def create_model_from_checkpoint(checkpoint_path, device=None):
        """
        Create a model instance from checkpoint information.

        Args:
            checkpoint_path (str): Path to the checkpoint file
            device (torch.device, optional): Device to load the model on

        Returns:
            tuple: (model, optimizer, scheduler, epoch, config) or None if loading fails
        """
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found at {checkpoint_path}")
            return None

        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

            # Extract model class information
            model_class_name = checkpoint.get("model_class_name")
            model_class_module = checkpoint.get("model_class_module")

            if model_class_name and model_class_module:
                # Dynamically import the model class
                import importlib

                module = importlib.import_module(model_class_module)
                model_class = getattr(module, model_class_name)

                # Create model instance (you may need to pass config or other parameters)
                # This is a basic implementation - you might need to adapt based on your model's __init__
                model = model_class()

                # Load state dict
                model.load_state_dict(checkpoint["model_state_dict"])

                if device:
                    model = model.to(device)

                return {
                    "model": model,
                    "optimizer_state_dict": checkpoint.get("optimizer_state_dict"),
                    # Expose both new and legacy scheduler keys for callers
                    "LRscheduler_state_dict": checkpoint.get("LRscheduler_state_dict"),
                    "LRschedulerPlateau_state_dict": checkpoint.get("LRschedulerPlateau_state_dict"),
                    "scheduler_state_dict": checkpoint.get("scheduler_state_dict"),
                    "epoch": checkpoint.get("epoch"),
                    "config": checkpoint.get("config"),
                    "model_class_name": model_class_name,
                    "model_class_module": model_class_module,
                }
            else:
                print("Warning: Model class information not found in checkpoint")
                return None

        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return None

    def resume_from_run(self, run_identifier: str) -> bool:
        """
        Resume training from an existing run.
        
        Args:
            run_identifier (str): Run name or run ID to resume from
            
        Returns:
            bool: True if resume was successful, False otherwise
        """
        from utilities.run_resume import get_resume_info, load_checkpoint_for_resume
        
        # Get resume information
        resume_info = get_resume_info(run_identifier, self.RUNS_DIR)
        if resume_info is None:
            self.logger.error(f"Failed to get resume info for run: {run_identifier}")
            return False
        
        # Set up directories using the existing run information
        self._setup_resume_directories(resume_info)
        
        # Load the latest checkpoint
        checkpoint_data = load_checkpoint_for_resume(
            resume_info['latest_checkpoint'], self.device
        )
        if checkpoint_data is None:
            self.logger.error("Failed to load checkpoint data")
            return False
        
        # Load model state
        try:
            self.model.load_state_dict(checkpoint_data['model_state_dict'])
            self.logger.info("Loaded model state from checkpoint", context="RESUME")
        except Exception as e:
            self.logger.error(f"Failed to load model state: {e}")
            return False
        
        # Load optimizer state
        try:
            self.optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
            self.logger.info("Loaded optimizer state from checkpoint", context="RESUME")
        except Exception as e:
            self.logger.error(f"Failed to load optimizer state: {e}")
            return False
        
        # Load scheduler states if available (support both new and legacy keys)
        try:
            if 'LRscheduler_state_dict' in checkpoint_data and hasattr(self, 'LRscheduler') and self.LRscheduler is not None:
                self.LRscheduler.load_state_dict(checkpoint_data['LRscheduler_state_dict'])
                self.logger.info("Loaded LRscheduler state from checkpoint", context="RESUME")
            if 'LRschedulerPlateau_state_dict' in checkpoint_data and hasattr(self, 'LRschedulerPlateau') and self.LRschedulerPlateau is not None:
                self.LRschedulerPlateau.load_state_dict(checkpoint_data['LRschedulerPlateau_state_dict'])
                self.logger.info("Loaded LRschedulerPlateau state from checkpoint", context="RESUME")
            # Backward-compatibility: legacy single scheduler key
            if 'scheduler_state_dict' in checkpoint_data:
                loaded = False
                if hasattr(self, 'LRscheduler') and self.LRscheduler is not None:
                    try:
                        self.LRscheduler.load_state_dict(checkpoint_data['scheduler_state_dict'])
                        loaded = True
                        self.logger.info("Loaded legacy scheduler state into LRscheduler", context="RESUME")
                    except Exception:
                        pass
                if not loaded and hasattr(self, 'LRschedulerPlateau') and self.LRschedulerPlateau is not None:
                    try:
                        self.LRschedulerPlateau.load_state_dict(checkpoint_data['scheduler_state_dict'])
                        loaded = True
                        self.logger.info("Loaded legacy scheduler state into LRschedulerPlateau", context="RESUME")
                    except Exception:
                        pass
                if not loaded:
                    self.logger.warning("Found legacy scheduler_state_dict but no compatible scheduler attribute to load into", context="RESUME")
        except Exception as e:
            self.logger.warning(f"Failed to load scheduler state(s): {e}")
        
        # Restore early stopping state if available
        if 'earlystopping_state' in checkpoint_data and hasattr(self, 'earlystopping') and self.earlystopping is not None:
            try:
                es_state = checkpoint_data['earlystopping_state']
                self.earlystopping.val_loss_min = es_state.get('val_loss_min', float('inf'))
                self.earlystopping.counter = es_state.get('counter', 0)
                self.earlystopping.patience = es_state.get('patience', 0)
                self.logger.info("Restored early stopping state", context="RESUME")
            except Exception as e:
                self.logger.warning(f"Failed to restore early stopping state: {e}")
        
        # Restore metrics history if available
        if 'training_metrics_history' in checkpoint_data:
            self.training_metrics = checkpoint_data['training_metrics_history']
            self.logger.info("Restored training metrics history", context="RESUME")
        
        if 'validation_metrics_history' in checkpoint_data:
            self.validation_metrics = checkpoint_data['validation_metrics_history']
            self.logger.info("Restored validation metrics history", context="RESUME")
        
        # Set the starting epoch
        self.start_epoch = checkpoint_data['epoch'] + 1
        self.logger.info(f"Resuming training from epoch {self.start_epoch}", context="RESUME")
        
        # Update wandb run ID if available and re-initialize wandb
        if 'wandb_run_id' in checkpoint_data and checkpoint_data['wandb_run_id']:
            self.resume_wandb_run_id = checkpoint_data['wandb_run_id']
            self.logger.info(f"Will resume wandb run: {self.resume_wandb_run_id}", context="RESUME")
            
            # Re-initialize wandb with the correct run ID
            self._reinitialize_wandb()
        
        return True

    def _setup_resume_directories(self, resume_info):
        """Set up directories using existing run information"""
        # Extract run name from the run directory path
        run_dir = resume_info['run_dir']
        runname = os.path.basename(run_dir)
        
        # Set up directory structure using existing run
        self.runname = runname
        self.RUN_DIR = run_dir
        self.MODELS_DIR = resume_info['models_dir']
        self.TEST_DIR = os.path.join(run_dir, "tests")
        self.paths_file = os.path.join(run_dir, "loadeddata.csv")
        
        # Create test directory if it doesn't exist
        os.makedirs(self.TEST_DIR, exist_ok=True)
        
        # Initialize early stopping with the existing run name
        from utilities import engine_initializers as initialize
        earlystopping_result = initialize.earlystopping(
            self.earlystopping_patience,
            self.MODELS_DIR,
            self.runname,
        )
        
        # Set early stopping attributes
        for key, value in earlystopping_result.items():
            setattr(self, key, value)
        
        # Update config with the run name
        self.config["name"] = self.runname
        
        self.logger.info(f"Set up resume directories for run: {self.runname}", context="RESUME")
        self.logger.info(f"Run directory: {self.RUN_DIR}", context="RESUME")

    def _reinitialize_wandb(self):
        """Re-initialize wandb with the correct run ID for resuming"""
        if self.wandb is not None:
            # Finish the current wandb run
            self.wandb.finish()
        
        # Re-initialize wandb with the resume run ID
        from utilities import engine_initializers as initialize
        wandb_result = initialize.wandb(
            self.config, 
            self.model, 
            self.config.get("NOTES", ""), 
            False,  # no_wandb
            self.resume_wandb_run_id
        )
        
        # Update the wandb reference
        self.wandb = wandb_result.get("wandb")
        self.logger.info(f"Re-initialized wandb with run ID: {self.resume_wandb_run_id}", context="RESUME")

    def _log_memory_usage(self, context: str = ""):
        """Log current GPU memory usage for monitoring"""
        if self.memory_monitoring and torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            reserved = torch.cuda.memory_reserved() / 1024**3  # GB
            self.logger.info(
                f"GPU Memory - {context}: Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB"
            )

    def _aggressive_memory_cleanup(self, exclude_vars: list = None):
        """
        Perform aggressive memory cleanup using the gpuClean utility.

        Args:
            exclude_vars (list): Variables to exclude from cleanup
        """
        if not self.aggressive_cleanup:
            return

        try:
            # Use the existing gpuClean utility
            freed_count, memory_freed = system_ops.gpuClean(
                frame_up=1, exclude_vars=exclude_vars or [], verbose=False
            )

            if self.memory_monitoring and freed_count > 0:
                self.logger.info(
                    f"Memory cleanup: Freed {freed_count} tensors, {memory_freed:.2f}MB"
                )

        except Exception as e:
            self.logger.warning(f"Memory cleanup failed: {e}")

    def _strategic_memory_cleanup(
        self, phase: str, batch_idx: int, exclude_vars: list = None
    ):
        """
        Perform strategic memory cleanup based on training phase and batch index.

        Args:
            phase (str): Current phase (Training, Validation, Test)
            batch_idx (int): Current batch index
            exclude_vars (list): Variables to exclude from cleanup
        """
        # Clean up every N batches or at specific intervals
        should_cleanup = batch_idx % self.memory_cleanup_frequency == 0 or (
            phase == "Training" and batch_idx % (self.memory_cleanup_frequency * 2) == 0
        )

        if should_cleanup:
            self._aggressive_memory_cleanup(exclude_vars)

    def _cleanup_tensor_dict(self, tensor_dict: dict, keys_to_keep: list = None):
        """
        Clean up a dictionary of tensors, optionally keeping specified keys.

        Args:
            tensor_dict (dict): Dictionary containing tensors
            keys_to_keep (list): Keys to preserve in the dictionary
        """
        if tensor_dict is None:
            return

        keys_to_keep = keys_to_keep or []
        keys_to_delete = [k for k in tensor_dict.keys() if k not in keys_to_keep]

        for key in keys_to_delete:
            if isinstance(tensor_dict[key], torch.Tensor):
                tensor_dict[key].detach_()
                if tensor_dict[key].is_cuda:
                    tensor_dict[key].cpu()
                del tensor_dict[key]
            elif isinstance(tensor_dict[key], dict):
                self._cleanup_tensor_dict(tensor_dict[key])
                del tensor_dict[key]

    def _prepare_metrics_for_wandb(self, metrics, phase):
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

    def _to_cpu_image(self, tensor, batch_idx=0):
        """Convert tensor to PIL Image with memory optimization"""
        if tensor is None:
            return None

        # Handle different tensor dimensions
        if tensor.dim() == 4:  # [B, C, H, W]
            tensor = tensor[batch_idx].clone()
        elif tensor.dim() == 3:  # [C, H, W]
            tensor = tensor.clone()
        elif tensor.dim() == 2:  # [H, W] - single channel
            tensor = tensor.unsqueeze(0).clone()

        # Convert to CPU and detach immediately
        tensor = tensor.cpu().detach().clamp(0, 1)

        # Convert to PIL Image
        to_pil = transforms.ToPILImage()
        pil_image = to_pil(tensor)

        # Clean up tensor immediately
        del tensor
        torch.cuda.empty_cache()

        return pil_image

    def _add_image_safely(self, viz_dict, key, tensor, caption, batch_idx=0, phase: str = None, test_idx: Optional[int] = None):
        """Safely add image to visualization dictionary"""

        image = wandb.Image(self._to_cpu_image(tensor))
        # Construct logging key with phase prefixing
        log_key = key
        if isinstance(phase, str):
            # Highest precedence: Test with explicit index
            if phase == "Test" and test_idx is not None:
                if key.startswith("Test/") and not key.startswith(f"Test/test_idx_{test_idx}/"):
                    # Upgrade existing Test/ prefix to include test_idx
                    log_key = key.replace("Test/", f"Test/test_idx_{test_idx}/", 1)
                elif not key.startswith(f"Test/test_idx_{test_idx}/"):
                    log_key = f"Test/test_idx_{test_idx}/{key}"
            else:
                # Generic phase prefix for Training/Validation/Test
                if not key.startswith(("Training/", "Validation/", "Test/")):
                    log_key = f"{phase}/{key}"
        viz_dict[log_key] = image
        payload = {log_key: image}
        # Attach step index for grouping if available
        if phase == "Test" and test_idx is not None:
            payload["Step/test_idx"] = int(test_idx)
        elif not self.metrics["Test"].empty and "Step/test_idx" in self.metrics["Test"].columns:
            # Fallback: infer last test_idx from metrics table
            try:
                payload["Step/test_idx"] = int(self.metrics["Test"]["Step/test_idx"].iloc[-1])
            except Exception:
                pass
        wandb.log(payload)

    def _load_and_increment_test_index(self) -> int:
        """Load current test index from RUN_DIR and increment it for next test.
        Returns the current index to be used for this test.
        """
        idx_path = os.path.join(self.RUN_DIR, "test_index.txt")
        current = 0
        try:
            if os.path.exists(idx_path):
                with open(idx_path, "r") as f:
                    content = f.read().strip()
                    if content != "":
                        current = int(content)
        except Exception:
            current = 0
        # Increment and persist for the next test
        try:
            with open(idx_path, "w") as f:
                f.write(str(current + 1))
        except Exception:
            pass
        return current

    # except Exception as e:
    #     self.logger.warning(f"Failed to create {key} visualization: {e}")
