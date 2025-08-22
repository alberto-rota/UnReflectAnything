

from readline import redisplay
import torch
import torch.nn as nn
import os
from rich import print
import pandas as pd

from utilities import *
import utilities.engine_initializers as initialize

import os
import wandb
import numpy as np
import augmentation as aug

# import matching as match
import utilities.visualization as viz

from contextlib import contextmanager, nullcontext
import gc
from typing import Union
from logger import get_logger, LogContext


class Engine:
    def __init__(
        self,
        model: Union[nn.Module, str, None],
        dataset: dict,
        config: dict,
        notes: str = "",
        **kwargs,
    ):
        """
        Initializes the Engine object with model, dataset adn config

        Args:
            model (nn.Module): The neural network model to be trained.
            dataset (dict): Dictionary containing 'training', 'validation', and 'test' datasets.
            config (dict): Dictionary containing config like BATCH_SIZE, LEARNING_RATE, etc.
            notes (str, optional): Additional notes for the training session. Defaults to "".
            **kwargs: Additional keyword arguments.
        """
        torch.autograd.set_detect_anomaly(True)
        # Store the model and config
        self.config = config
        self.config["NOTES"] = notes

        # Initialize device and directories
        device_dirs = initialize.device_and_directories()
        self.device = device_dirs["device"]
        self.RUNS_DIR = device_dirs["runs_dir"]

        # Initialize the model
        self.model = model

        # Function to set attributes from initialization functions
        def init(init_func, *args, **kwargs):
            result = init_func(*args, **kwargs)
            for key, value in result.items():
                setattr(self, key, value)
            return result

        # Initialize all components
        init(initialize.dataloaders, dataset, config)
        init(initialize.dimensions, self.training_dl, config)
        init(initialize.hyperparameters, config)
        init(
            initialize.projections,
            self.height,
            self.width,
            self.device,
            self.learning_rate,
        )
        init(initialize.matching_pipeline, config, model, self.device)
        init(initialize.optimizers, self.model, config)
        init(initialize.loss_functions, config)
        init(initialize.schedulers, self.optimizer, config, self.training_dl)
        init(initialize.transforms, self.height, self.width)
        init(
            initialize.wandb,
            config,
            self.model,
            notes,
            kwargs.get("no_wandb", False),
        )

        init(initialize.tracking_metrics)
        init(initialize.setup_run_directories, self.RUNS_DIR, self.wandb, False)
        self.config["name"] = self.runname

        # Initialize early stopping
        self.earlystopping = initialize.earlystopping(
            self.earlystopping_patience, self.MODELS_DIR, self.runname
        )

        # Save hyperparameters to json
        initialize.save_hyperparameters_json(self.RUN_DIR, self.config)
        self.logger = get_logger(__name__, log_to_file=True, log_dir=self.RUN_DIR)

        # Log run information at initialization time
        # self.logger.info(f"Run initialized: {self.runname}", context="INIT")
        # if hasattr(self.wandb, "url") and self.wandb.url:
        #     self.logger.info(f"WandB run URL: {self.wandb.url}", context="WANDB")
        #     project_url = self.wandb.url.rsplit("/", 1)[0]
        #     self.logger.info(f"WandB project URL: {project_url}", context="WANDB")

    def trainloop(self):
        """
        The main training loop that runs through all epochs, trains the model,
        validates it, and handles early stopping and saving of the model.
        """

        for e in range(self.epochs):

            ### TRAINING + VALIDATION FOR EACH EPOCH
            self.train()  # Train the model for one epoch
            training_status = self.validate()  # Train the model for one epoch

            ### RESET SAMPLERS FOR INCREASED ROBUSTNESS
            self.dataset["Training"].reset_sampler()
            self.dataset["Validation"].reset_sampler()

            # Step the curriculum learning
            if e % self.dataset["Training"].max_steps_frameskip == 0 and e > 0:
                self.dataset["Training"].step_frameskip_curriculum()

            self.csv_log_metrics()

            ### BREAK IF EARLYSTOP
            if training_status == "EARLYSTOP":
                break  # Exit the training loop if early stopping condition is met

        # Log locations of important data at the end of training
        self.logger.info("TRAINING COMPLETE", context="SAVE")
        self.logger.info(
            f" Run directory: {os.path.abspath(self.RUN_DIR)}", context="SAVE"
        )
        self.logger.info(
            f" Checkpoints  : {os.path.abspath(self.MODELS_DIR)}", context="SAVE"
        )
        self.logger.info(f" Metrics      :", context="SAVE")
        self.logger.info(
            f" Training     : {os.path.abspath(os.path.join(self.RUN_DIR, 'training_metrics.csv'))}",
            context="SAVE",
        )
        self.logger.info(
            f" Validation   : {os.path.abspath(os.path.join(self.RUN_DIR, 'validation_metrics.csv'))}",
            context="SAVE",
        )

        # Log WandB URLs again for convenience
        # if hasattr(self.wandb, "url") and self.wandb.url:
        #     self.logger.info(f"WandB run URL: {self.wandb.url}", context="WANDB")
        #     project_url = self.wandb.url.rsplit("/", 1)[0]
        #     self.logger.info(f"WandB project URL: {project_url}", context="WANDB")

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
        return self.run_epoch(phase="Training")

    def validate(self):
        return self.run_epoch(phase="Validation")

    def test(self):
        self.run_epoch(phase="Test")
        if self.wandb is not None:
            self.log_tests()

    def run_epoch(self, phase=str, epoch=None) -> None:
        """
        Trains the model for one epoch using 3D warping to create pixel correspondences,
        then applies a triplet loss on the extracted embeddings.

        Args:
            epoch (int): The current epoch index.
        """
        assert phase in [
            "Training",
            "Validation",
            "Test",
        ], "Invalid phase. Choose 'Training', 'Validation' or 'Test'."

        PHASE = phase
        TRAINING = PHASE == "Training"
        VALIDATION = PHASE == "Validation"
        TEST = PHASE == "Test"
        if epoch is None:
            epoch = self.step["idx" if TEST else "epoch"]
        if TRAINING:
            self.model.train()
        else:
            self.model.eval()
        images_logged = False
        dataset = self.dataset[PHASE]

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.config.WORKERS,
            drop_last=True,
            sampler=dataset.sampler,
            pin_memory=self.config.PIN_MEMORY,
            prefetch_factor=self.config.PREFETCH_FACTOR,
        )

        self.switch_optimizer(epoch)
        base_lr = self.optimizer.param_groups[0]["lr"]

        if len(dataloader) == 0:
            print("[DATASET]: Empty dataloader, skipping epoch.")
            return

        # Base learning rate for warmup
        with self.choose_if_grad(PHASE):
            for batch_idx, sample in enumerate(dataloader):
                ### INITIALIZATION
                step = epoch * len(dataloader) + batch_idx
                self.step[f"{PHASE}_batch"] += 1

                ### Batch Flags
                # True if we should log images on this batch
                log_images_this_batch = (
                    batch_idx > 0
                    and batch_idx % self.logfreq_wandb == 0
                    and self.logfreq_wandb > 1
                ) or (batch_idx == len(dataloader) - 1 and not images_logged)
                # True if we should scale the learning rate for the warmup phase
                warming_up = step < self.warmup_steps
                # True if we should accumulate gradients, false if grads should be backpropagated
                accumulate_gradients = (
                    step >= self.warmup_steps
                    and (batch_idx + 1) % self.gradient_accumulation_steps == 0
                )

                # Warmup logic
                if warming_up:
                    warmup_factor = step / self.warmup_steps
                    current_lr = base_lr * warmup_factor
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = current_lr

                ### ACCESSING SAMPLE ELEMENTS
                framestack = sample["framestack"].to(self.device)  # [B, T, C, H, W]
                camera_pose_gt = sample["Ts2t"].to(self.device)  # [B, 6] or [B, 4, 4]
                fundamental_gt = sample["fundamental"].to(self.device)
                paths = list(
                    zip(*sample["paths"])
                )  # list of tuples (source_path, target_path)
                K = sample["intrinsics"].to(self.device)  # [B, 3, 3]

                ### DEPTHMAP INFERENCE (or loading from sample)
                if "depthstack" in sample.keys():
                    depthstack = (
                        sample["depthstack"] + self.config.DEPTH_BIAS_FACTOR
                    ).to(self.device) * self.config.DEPTH_SCALE_FACTOR
                else:
                    with torch.no_grad():
                        depthstack = (
                            self.model.depth(framestack)
                            + self.config.DEPTH_BIAS_FACTOR
                        ) * self.config.DEPTH_SCALE_FACTOR
                ### GEOMETRIC AUGMENTATION
                framestack, camera_pose_gt, depthstack = aug.geometric_augmentation(
                    framestack,
                    camera_pose_gt,
                    (
                        depthstack
                        if "depthstack" in sample.keys()
                        else depthstack.unsqueeze(1).repeat(1, 2, 1, 1, 1)
                    ),
                    p=(
                        self.config.DATASETS[
                            list(self.config.DATASETS.keys())[0]
                        ].AUGMENTATION_PROBABILITY["GEOMETRIC"]
                        if TRAINING
                        else 0.0
                    ),
                    # target_only=True,
                )
                # tprint(K)
                # display(channels(depthstack))
                ### SYNTHETIZE GROUND TRUTH - TO BE USE FOR TRIPLET MINING
                (
                    warped,
                    source_matched_points,
                    target_matched_points_true,
                    embedding_mask,
                ) = self.matchingPipeline.synthethize_ground_truth(
                    framestack, K, camera_pose_gt, depthstack[:, 0]
                ).values()
                # Creating the synthetic framestack with the warped image at training time
                synthetic_framestack = framestack.clone()
                # display(rgb(synthetic_framestack))
                if (
                    not TEST
                ):  # Only warp the image at training and validation. At test time match the framestack as is
                    synthetic_framestack[:, 1] = warped.clone()
                # display(rgb(synthetic_framestack))
                ### AUGMENTATIONS
                synthetic_framestack, camera_pose_gt = aug.color_augmentation(
                    synthetic_framestack,
                    camera_pose_gt,
                    p=(
                        self.config.DATASETS[
                            list(self.config.DATASETS.keys())[0]
                        ].AUGMENTATION_PROBABILITY["COLOR"]
                        if TRAINING
                        else 0.0
                    ),
                    target_only=True,
                )
                ### BACKBONE FORWARD PASS
                descriptors = self.model(synthetic_framestack)

                ### MINE TRIPLETS
                if not TEST:
                    triplets = self.matchingPipeline.mine_triplets(
                        descriptors,
                        source_matched_points,
                        target_matched_points_true,
                        embedding_mask,
                    )
                    A, P, N = [
                        triplets.get(key) for key in ["anchor", "positive", "negative"]
                    ]

                    ### BACKBONE BACKWARD PASS
                    loss_tensor = (
                        self.loss_fn(A, P, N) / self.gradient_accumulation_steps
                    )
                    backward_output = self.backward_pass(
                        loss_tensor,
                        accumulate_gradients=accumulate_gradients,
                        phase=PHASE,
                    )
                    loss_tensor.detach()
                torch.cuda.empty_cache()
                ### FINDING PIXEL CORRESPONDENCES
                (
                    source_pixels_matched,
                    target_pixels_matched,
                    batch_idx_match,
                    descriptor_scores,
                    refinement_scores,
                    sim_matrix,
                ) = self.matchingPipeline.compute_correspondences(
                    descriptors,
                    synthetic_framestack,
                    embedding_mask if not TEST else None,
                ).values()

                ### FUNDAMENTAL MATRIX ESTIMATION
                fundamental_pred, inliers, epipolar_scores = (
                    self.matchingPipeline.RANSAC(
                        source_pixels_matched,
                        target_pixels_matched,
                        batch_idx_match,
                    ).values()
                )
                scores = self.matchingPipeline.combine_scores(
                    descriptor_scores,
                    refinement_scores,
                    epipolar_scores,
                    self.config.SCORE_WEIGHTS,
                )

                ### RETRIEVING PSUDO-GROUND TRUTH - TO EVALUATE PIXEL CORRESPONDENCES
                if not TEST:
                    (
                        warped,
                        source_pixels_matched,
                        true_pixels_matched,
                        embedding_mask,
                    ) = self.matchingPipeline.synthethize_ground_truth(
                        synthetic_framestack,
                        K,
                        camera_pose_gt,
                        depthstack[:, 0],
                        source_pixels_matched,
                        batch_idx_match,
                    ).values()
                # These variables are not calculated at test time
                if TEST:
                    true_pixels_matched = None
                    loss_tensor = None
                    backward_output = {}
                    triplets, A, P, N = None, None, None, None

                ### METRICS COMPUTATION
                metrics = self.matchingPipeline.compute_metrics(
                    self.matchingPipeline,
                    source_pixels_matched,
                    target_pixels_matched,
                    true_pixels_matched,
                    batch_idx_match,
                    scores,
                    fundamental_pred,
                    fundamental_gt,
                )
                metrics.update(
                    {
                        "Loss": (
                            loss_tensor.detach().item()
                            if loss_tensor is not None
                            else None
                        ),
                        "InlierCount": inliers.count_nonzero().item(),
                        "InlierPercentage": inliers.count_nonzero().item()
                        / inliers.numel(),
                        "NTripletsMined": (
                            (len(A) / self.batch_size) if not TEST else None
                        ),
                        "Gradients/GradNorm": backward_output.get("grad_norm"),
                        "Gradients/WeightNorm": backward_output.get("weight_norm"),
                        "HyperParameters/LR": self.optimizer.param_groups[0]["lr"],
                        f"Step/{'val' if VALIDATION else ''}batch": self.step[
                            f"{PHASE}_batch"
                        ],
                        f"Step/{'idx' if TEST else 'epoch'}": epoch,
                    }
                )

                # Updating the local metrics dataframe
                self.metrics[PHASE] = pd.concat(
                    [self.metrics[PHASE], pd.DataFrame(metrics, index=[0])],
                    ignore_index=True,
                )
                if log_images_this_batch:
                    # Adding logged images only if is time to do so
                    images = self.create_all_images(
                        synthetic_framestack,
                        warped,
                        sim_matrix,
                        source_pixels_matched,
                        target_pixels_matched,
                        true_pixels_matched,
                        scores,
                        batch_idx_match,
                        triplets,
                        fundamental_pred,
                    )
                    metrics.update(images)
                    images_logged = True

                ### LOGGING
                self.console_log_metrics(
                    stage=PHASE,
                    epoch=epoch,
                    batch_idx=batch_idx,
                    dataloader_len=len(dataloader),
                    extra_info="W" if step < self.warmup_steps else None,
                )
                if self.wandb is not None:
                    self.wandb.log(metrics_for_wandb(metrics, PHASE))
                self.log_loaded_paths(paths, PHASE)

                ### CLEANUP
                del (
                    framestack,
                    synthetic_framestack,
                    warped,
                    sim_matrix,
                    source_pixels_matched,
                    target_pixels_matched,
                    true_pixels_matched,
                    scores,
                    batch_idx_match,
                    triplets,
                    fundamental_pred,
                    loss_tensor,
                    inliers,
                    A,
                    metrics,
                    backward_output,
                    descriptors,
                    camera_pose_gt,
                    depthstack,
                    embedding_mask,
                    descriptor_scores,
                    refinement_scores,
                    epipolar_scores,
                    K,
                )
                if "backward_output" in locals():
                    del backward_output
                torch.cuda.empty_cache()
                gc.collect()

            self.step[f"{PHASE}_batch"] += 1

        ### LOGGING Epoch metrics
        epochstr = (
            "idx" if TEST else "epoch"
        )  # Tests need to have the test index in the key
        epoch_metrics = metrics_for_wandb(
            self.metrics[PHASE][
                self.metrics[PHASE][f"Step/{epochstr}"] == epoch
            ].mean(),
            PHASE,
        )
        # Assuming epoch_metrics is already defined
        epoch_metrics = {
            key.replace(PHASE, f"{PHASE}/{epochstr}"): value
            for key, value in epoch_metrics.items()
            if PHASE in key
        }
        if self.wandb is not None:
            self.wandb.log(epoch_metrics)

        torch.cuda.empty_cache()
        gc.collect()
        ### EARLY-STOPPING
        if VALIDATION:
            self.step[epochstr] += 1  # Increasing epoch/idx counter
            self.LRschedulerPlateau.step(float(epoch_metrics[f"{PHASE}/epoch/Loss"]))
            self.earlystopping(
                float(epoch_metrics[f"{PHASE}/epoch/Loss"]),
                self.model,
                epoch,
            )
            if self.earlystopping.early_stop:
                print(">> [EARLYSTOPPING]: Patience Reached, Stopping Training")
                return "EARLYSTOP"
            return "IMPROVED"

    @contextmanager
    def choose_if_grad(self, mode):
        """Conditionally use torch.no_grad based on the given mode."""
        with torch.no_grad() if mode in ["Validation", "Test"] else nullcontext():
            yield

    def backward_pass(self, loss_tensor, accumulate_gradients=False, phase="Training"):
        """
        Performs the backward pass, including gradient calculation, clipping, and optimization steps.

        Args:
            loss_tensor (torch.Tensor): The loss tensor to backpropagate
            accumulate_gradients (bool): If True, will update weights after backpropagation
                                        assuming gradient accumulation is complete

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
            print(
                f">> [ERROR]: {e} - Skipping batch {self.step['Training_batch']} in epoch {self.step['epoch']}"
            )
            ERROR_IN_BACKWARD_PASS = True

        grad_norm, weight_norm = optimization.get_norms(
            self.model.parameters()
        )
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=1.0
        )

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
            print(
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

    def log_tests(self):
        """
        Logs the test metrics to Weights and Biases.

        Args:
            test_idx (int): Index of the current test run.
            test_name (str): Name of the test (used for logging). Default is "TEST".

        Returns:
            None
        """

        self.logger.info(">> TEST REPORT", context="TEST")
        self.logger.info(self.metrics["Test"].describe(), context="TEST")
        self.metrics["Test"].to_csv(self.TEST_DIR + "/test_metrics.csv")
        self.wandb.log({"Test/Summary": wandb.Table(dataframe=self.metrics["Test"])})

        # Log locations of important data
        self.logger.info(">> RUN DATA LOCATIONS", context="SAVE")
        self.logger.info(
            f"Run data directory: {os.path.abspath(self.RUN_DIR)}", context="SAVE"
        )
        self.logger.info(
            f"Models saved at: {os.path.abspath(self.MODELS_DIR)}", context="SAVE"
        )
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
            f"  - Test: {os.path.abspath(os.path.join(self.TEST_DIR, 'test_metrics.csv'))}",
            context="SAVE",
        )

        # Log WandB URLs again for convenience
        if hasattr(self.wandb, "url") and self.wandb.url:
            self.logger.info(f"WandB run URL: {self.wandb.url}", context="WANDB")
            project_url = self.wandb.url.rsplit("/", 1)[0]
            self.logger.info(f"WandB project URL: {project_url}", context="WANDB")

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

    def csv_log_metrics(self):

        if self.metrics["Training"] is not None:
            self.metrics["Training"].to_csv(
                os.path.join(self.RUN_DIR, "training_metrics.csv")
            )
        if self.metrics["Validation"] is not None:
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
        step : int, optional
            Current training step.
        epoch : int, optional
            Current epoch number.
        batch_idx : int, optional
            Current batch index.
        dataloader_len : int, optional
            Length of the dataloader being used.
        extra_info : str, optional
            Additional information to display in the phase indicator.
        """
        # Define the phase indicator based on stage
        # if stage == "Training":
        #     epoch_batch_info = align(
        #         f"E {str(epoch+1)}/{self.epochs}", 8, "right"
        #     ) + align(f"B{str(batch_idx+1)}/{dataloader_len}", 8, "right")
        # elif stage == "Validation":
        #     epoch_batch_info = align(
        #         f"E {str(epoch+1)}/{self.epochs}", 8, "right"
        #     ) + align(f"B{str(batch_idx+1)}/{dataloader_len}", 8, "right")
        # elif "Test" in stage:
        #     test_idx = epoch
        # epoch_batch_info = f"E {int(test_idx)} " + align(
        #     f"B{str(batch_idx+1)}/{dataloader_len}", 8, "right"
        # )
        epoch_batch_info = align(
            f"E {str(epoch+1)}/{self.epochs} ", 10, "right"
        ) + align(f"B {str(batch_idx+1)}/{dataloader_len} ", 10, "left")
        if extra_info is not None:
            phase_indicator = f"[purple ]{extra_info}[/purple ]"  # + phase_indicator
        # Print header with run name and status information
        if "offline" in self.runname:
            printedrunname = "run"
        else:
            printedrunname = f'{self.runname.split("-")[0][0]}{self.runname.split("-")[1][0]}{self.runname.split("-")[2]}'
        metricstring = (
            align(f"{printedrunname}:", 6, "right")
            # + f"|"
            # + align(
            #     f"{phase_indicator}{self.step[f'{stage}_batch'] if stage != 'Test' else ''}",
            #     24,  # if "Test" in stage else 21,
            #     "center",
            # )
            # + f"|"
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
                ):
                    metrs += (
                        f"[yellow]{m[:4]}[/yellow]"
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
        with open(self.paths_file, mode="a") as file:
            file.write(f"{self.step[f'{phase}_batch']},{paths}\n")

    def create_all_images(
        self,
        framestack,
        warped,
        sim_matrix,
        source_pixels_matched,
        target_pixels_matched,
        true_pixels_matched,
        scores,
        batch_idx_match,
        triplets_dict,
        fundamental_pred,
        patch_size=None,
        topk=50,
        batch_idx=0,
    ):
        """
        Creates a dictionary of visualization images for training monitoring.

        This function generates visualization images for patch matches, pixel matches,
        epipolar geometry, and mined triplets from the current batch of data.

        Parameters:
        -----------
        framestack : torch.Tensor
            Tensor containing the source frames.
        warped : torch.Tensor
            Tensor containing the warped (target) frames.
        sim_matrix : torch.Tensor
            Similarity matrix between patches.
        source_pixels_matched : torch.Tensor
            Coordinates of matched pixels in the source image.
        target_pixels_matched : torch.Tensor
            Coordinates of matched pixels in the target image.
        true_pixels_matched : torch.Tensor
            Ground truth coordinates of matches in the target image.
        scores : torch.Tensor
            Confidence scores for the matches.
        batch_idx_match : torch.Tensor
            Batch indices for each match.
        triplets_dict : dict
            Dictionary containing triplet information (anchor, positive, negative indices).
        fundamental_pred : torch.Tensor
            Predicted fundamental matrices.
        patch_size : int
            Size of the patches used for matching.
        batch_idx : int, optional
            Batch index to visualize, defaults to 0.

        Returns:
        --------
        dict
            Dictionary of PIL image objects for visualization.
        """
        # Filter data for the specified batch index
        if patch_size is None:
            patch_size = self.patch_size
        batch_filter = batch_idx_match == batch_idx
        if triplets_dict is not None:
            batch_triplet_filter = triplets_dict["batch_indices"] == batch_idx
        TEST = triplets_dict is None
        # Create visualization dictionary
        visualization_dict = {
            "PatchMatches": viz.viewPatchMatches(
                img1=framestack[batch_idx, 0],
                img2=framestack[batch_idx, 1],
                similarity_matrix=sim_matrix[batch_idx],
                patch_size=patch_size,
                topk=topk,
                use_actual_topk=False,
            ),
            "PixelMatches": viz.viewComparePixelMatches(
                img1=framestack[batch_idx, 0],
                img2=framestack[batch_idx, 1],
                pts1=source_pixels_matched[batch_filter],
                pts2=target_pixels_matched[batch_filter],
                pts2_true=(
                    true_pixels_matched[batch_filter]
                    if true_pixels_matched is not None
                    else target_pixels_matched[batch_filter]
                ),
                scores=scores[batch_filter],
                topk=min(topk, len(source_pixels_matched[batch_filter])),
                use_actual_topk=True,
            ),
            "Epipolar": viz.viewEpipolarGeometry(
                img1=framestack[batch_idx, 0],
                img2=framestack[batch_idx, 1],
                pts1=source_pixels_matched[batch_filter],
                pts2=target_pixels_matched[batch_filter],
                scores=scores[batch_filter],
                F=fundamental_pred[batch_idx],
                topk=min(topk, len(source_pixels_matched[batch_filter])),
                use_actual_topk=True,
            ),
            "TripletsMined": (
                viz.viewTriplets(
                    framestack[batch_idx, 0],
                    framestack[batch_idx, 1],
                    anchor_indices=triplets_dict["anchor_indices"][
                        batch_triplet_filter
                    ],
                    positive_indices=triplets_dict["positive_indices"][
                        batch_triplet_filter
                    ],
                    negative_indices=triplets_dict["negative_indices"][
                        batch_triplet_filter
                    ],
                    patch_size=patch_size,
                    num_triplets=min(
                        topk, len(triplets_dict["anchor_indices"][batch_triplet_filter])
                    ),
                )
                if triplets_dict is not None
                else None
            ),
        }

        return visualization_dict

    def reinstantiate_model_from_checkpoint(self):
        """
        Reinstantiate the model from the latest checkpoint saved.
        """
        # Reinstantiate the model from the latest checkpoint
        self.model.fromArtifact(self.runname)
        self.logger.info(
            f"Model reinstantiated from checkpoint @ {self.runname}",
            context="GCLOUD",
        )
