import gc
import os
import shutil
from contextlib import contextmanager, nullcontext
from typing import Optional, Union

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
from losses import DecompositionLoss, alpha_composite
from polar_highlighter import PolarHighlighter, get_soft_highlight_map


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
        init(initialize.wandb, config, self.model, notes, no_wandb)
        init(initialize.tracking_metrics)
        init(initialize.setup_run_directories, self.RUNS_DIR, self.wandb, False)
        init(
            initialize.earlystopping,
            self.earlystopping_patience,
            self.MODELS_DIR,
            self.runname,
        )
        self.config["name"] = self.runname
        self.add_polar_highlights = PolarHighlighter(
            height=self.height, width=self.width
        ).to(self.device)

        # Save hyperparameters to json
        initialize.save_hyperparameters_json(self.RUN_DIR, self.config)

        # Once the run name is set, we move all the log files to the run directory
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
        self.loss = DecompositionLoss(
            weight_specular_loss=self.config.SPECULAR_LOSS_WEIGHT,
            weight_diffuse_loss=self.config.DIFFUSE_LOSS_WEIGHT,
            weight_highlight_loss=self.config.HIGHLIGHT_LOSS_WEIGHT,
            weight_component_matching=self.config.COMPONENT_MATCHING_LOSS_WEIGHT,
            weight_image_reconstruction=self.config.IMAGE_RECONSTRUCTION_LOSS_WEIGHT,
            weight_alpha_regularization=self.config.ALPHA_REGULARIZATION_LOSS_WEIGHT,
            weight_spatial_consistency=self.config.SPATIAL_CONSISTENCY_LOSS_WEIGHT,
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
        for e in range(self.epochs):
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

                highlight_result = self.add_polar_highlights(
                    rgb=sample["rgb"].to(self.device, non_blocking=True),
                    pol=sample["stokes"].to(self.device, non_blocking=True)
                    if "stokes" in sample
                    else None,
                    intrinsic=sample["intrinsics"].to(self.device, non_blocking=True),
                    shininess=self.config.SHININESS,
                    ks=self.config.KS,
                )
                soft_highlight_map = get_soft_highlight_map(
                    sample["rgb"].to(self.device, non_blocking=True),
                    threshold=self.config.SOFT_HIGHLIGHT_THRESHOLD,
                )
                highlight_mask = (
                    highlight_result["highlight"] + soft_highlight_map
                ).clamp(0, 1)
                rgb_highlighted = highlight_result["rgb_highlighted"]

                # Create input_data dictionary for the model. Only contains model inputs and GTs
                # Prefer cropped rect if present; keep its native size
                # rgb_img = sample["rect_crop"].to(self.device, non_blocking=True) if "rect_crop" in sample else sample["rgb"].to(self.device, non_blocking=True)
                input_data = {
                    "diffuse": sample["rgb"].to(self.device, non_blocking=True),
                    "rgb": rgb_highlighted,
                    "specular": sample["specular"].to(self.device, non_blocking=True),
                    "highlight": highlight_mask,  # .repeat(1, 4, 1, 1),
                }

                # Add optional polarization data if available
                if "AoP" in sample:
                    input_data["AoP"] = sample["AoP"].to(self.device, non_blocking=True)
                if "DoP" in sample:
                    input_data["DoP"] = sample["DoP"].to(self.device, non_blocking=True)
                if "f_spec" in sample:
                    input_data["f_spec"] = sample["f_spec"].to(
                        self.device, non_blocking=True
                    )

                # Clean up sample from CPU memory immediately
                del sample

                # Log memory usage before forward pass if monitoring
                if self.memory_monitoring and batch_idx % 10 == 0:
                    self._log_memory_usage(f"Before forward pass - batch {batch_idx}")

                ### Forward pass
                decomposition = self.model(input_data)

                ### Loss Computation
                losses = self.loss(decomposition, input_data)
                loss_value = losses["total"]

                ### Compositing available components with alpha channels
                # Find components with alpha channels (4 channels) for composition
                components_for_composition = []
                for comp_name, comp_tensor in decomposition.items():
                    if (
                        isinstance(comp_tensor, torch.Tensor)
                        and comp_tensor.dim() == 4
                        and comp_tensor.shape[1] == 4
                    ):
                        components_for_composition.append(comp_tensor)

                # Only do alpha composition if we have components with alpha channels
                if len(components_for_composition) > 0:
                    decomposition["recon"] = alpha_composite(components_for_composition)
                else:
                    # If no alpha channels, just sum RGB components (fallback)
                    rgb_components = []
                    for comp_name, comp_tensor in decomposition.items():
                        if (
                            isinstance(comp_tensor, torch.Tensor)
                            and comp_tensor.dim() == 4
                            and comp_tensor.shape[1] == 3
                        ):
                            rgb_components.append(comp_tensor)

                    if len(rgb_components) > 0:
                        decomposition["recon"] = sum(rgb_components)
                    else:
                        # No valid components found - this shouldn't happen but handle gracefully
                        decomposition["recon"] = input_data["rgb"]
                # Backward pass for training
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
                                "highlight_decoder": self.model.decoders["highlight"] if "highlight" in self.model.decoders else None,
                                "diffuse_decoder": self.model.decoders["diffuse"] if "diffuse" in self.model.decoders else None,
                                "specular_decoder": self.model.decoders["specular"] if "specular" in self.model.decoders else None,
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
                    metrics["Gradients/MODEL_WeightNorm"] = backward_output["weight_norm"]

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
                    try:
                        # Create a copy of sample for visualization (since we deleted it earlier)
                        gt_data = {
                            "rgb": input_data["rgb"].cpu(),
                        }
                        # Add optional polarization data if available
                        if "AoP" in input_data:
                            gt_data["AoP"] = input_data["AoP"].cpu()
                        if "DoP" in input_data:
                            gt_data["DoP"] = input_data["DoP"].cpu()
                        if "f_spec" in input_data:
                            gt_data["f_spec"] = input_data["f_spec"].cpu()
                        if "highlight" in input_data:
                            gt_data["highlight"] = input_data["highlight"].cpu()
                        if "rgb_highlighted" in input_data:
                            gt_data["rgb_highlighted"] = input_data[
                                "rgb_highlighted"
                            ].cpu()

                        images = self.create_visualization_images(
                            input_data, decomposition, gt_data
                        )
                        if images:
                            metrics.update(images)
                            images_logged = True

                        # Clean up sample copy immediately
                        del gt_data

                    except Exception as e:
                        self.logger.warning(
                            f"Failed to create visualization images: {e}",
                            context=phase.upper(),
                        )

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
                    # Use the metrics_for_wandb function to format metrics properly
                    wandb_metrics = metrics_for_wandb(metrics, phase)
                    # Add the batch number
                    if phase == "Training":
                        batch_str = "batch"
                    elif phase == "Validation":
                        batch_str = "valbatch"
                    elif phase == "Test":
                        batch_str = "idx"
                    wandb_metrics[f"Step/{batch_str}"] = self.step[f"{phase}_batch"]
                    self.wandb.log(wandb_metrics)

                # Strategic memory cleanup - more aggressive for larger batches
                self._strategic_memory_cleanup(
                    phase, batch_idx, exclude_vars=["self", "dataloader"]
                )

                # Clean up variables that are no longer needed
                del input_data, decomposition, losses, loss_value
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
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            # 'scheduler_state_dict': self.scheduler.state_dict(),
            "config": self.config,
        }

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

    def create_visualization_images(self, batch, decomposition, sample, batch_idx=0):
        """
        Creates visualization images for polarization-based reflection removal training.
        Optimized for memory efficiency with aggressive cleanup.

        Args:
            batch (dict): Input batch containing rgb, AoP, DoP, f_spec
            decomposition (dict): Model output containing specular, diffuse, recon
            sample (dict): Original sample from dataset
            batch_idx (int): Batch index to visualize

        Returns:
            dict: Dictionary of wandb.Image objects for visualization
        """
        try:
            import torchvision.transforms as transforms
            import wandb

            def to_cpu_image(tensor):
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
                else:
                    return None

                # Convert to CPU and detach immediately
                tensor = tensor.cpu().detach().clamp(0, 1)

                # Convert to PIL Image
                to_pil = transforms.ToPILImage()
                pil_image = to_pil(tensor)

                # Clean up tensor immediately
                del tensor
                torch.cuda.empty_cache()

                return pil_image

            def add_image_safely(viz_dict, key, tensor, caption):
                """Safely add image to visualization dictionary"""
                try:
                    img = to_cpu_image(tensor)
                    if img:
                        viz_dict[key] = wandb.Image(img, caption=caption)
                        del img
                except Exception as e:
                    self.logger.warning(f"Failed to create {key} visualization: {e}")

            # Create visualization dictionary
            visualization_dict = {}

            # Predicted components - dynamically detect available components
            # First add known special components
            if "recon" in decomposition:
                add_image_safely(
                    visualization_dict,
                    "images/PRED_Reconstruction",
                    decomposition["recon"],
                    "Reconstruction",
                )

            # Then add any other model output components
            for comp_name, comp_tensor in decomposition.items():
                if (
                    comp_name != "recon"
                    and isinstance(comp_tensor, torch.Tensor)
                    and comp_tensor.dim() == 4
                ):
                    # Create nice display name
                    display_name = comp_name.replace("_", " ").title()
                    wandb_key = f"images/PRED_{comp_name.capitalize()}"
                    caption = f"Predicted {display_name} Component"
                    add_image_safely(
                        visualization_dict, wandb_key, comp_tensor, caption
                    )

            # Input images
            input_images = [
                ("images/GT_RGB", "diffuse", "Input RGB Image"),
                ("images/GT_RGB_Highlighted", "rgb", "Input RGB Highlighted Image"),
                ("images/GT_Highlight", "highlight", "Input RGB Highlighted Image"),
                ("images/GT_FSpec", "f_spec", "Specular Fraction"),
            ]

            for key, tensor_key, caption in input_images:
                if tensor_key in batch:
                    add_image_safely(
                        visualization_dict, key, batch[tensor_key], caption
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
                    add_image_safely(
                        visualization_dict, wandb_key, comp_tensor, caption
                    )

            # Final cleanup
            torch.cuda.empty_cache()
            gc.collect()

            return visualization_dict

        except ImportError:
            self.logger.warning(
                "wandb or PIL not available for image visualization",
                context="VISUALIZATION",
            )
            return {}
        except Exception as e:
            self.logger.warning(
                f"Error creating visualization images: {e}", context="VISUALIZATION"
            )
            return {}

    def reinstantiate_model_from_checkpoint(self, checkpoint_path=None):
        """
        Reinstantiate the model from checkpoint.
        """
        if checkpoint_path is None:
            # Try to load best model
            checkpoint_path = os.path.join(self.MODELS_DIR, "best_model.pth")

        if not os.path.exists(checkpoint_path):
            self.logger.warning(
                f"Checkpoint not found at {checkpoint_path}", context="SAVE"
            )
            return

        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            self.logger.info(
                f"Model reinstantiated from checkpoint: {checkpoint_path}",
                context="SAVE",
            )
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
