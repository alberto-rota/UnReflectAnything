"""
Initialization utilities for the Trainer class.
Contains functions to initialize various components of the training pipeline.
"""

import torch
import torch.nn as nn
import os
import datetime
import json
import io
import contextlib
import re
import pandas as pd
import wandb as weightsandbiases
import torchvision
import cv2
from rich import print
from logger import get_logger
import optimization
logger = get_logger(__name__)


def device_and_directories():
    """Initialize device and create necessary directories"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create base directories
    if not os.path.exists("runs"):
        os.makedirs("runs")  # Create 'runs' directory if it doesn't exist
    runs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../runs")

    return {"device": device, "runs_dir": runs_dir}


def dataloaders(dataset, config):
    """Initialize datasets and dataloaders"""
    # Store dataset references
    training_ds = dataset["Training"]  # Training dataset
    validation_ds = dataset["Validation"]  # Validation dataset
    test_ds = dataset["Test"]  # Testing dataset
    workers = dataset["workers"]  # Number of workers for data loading

    # Dataloaders only for initialization purposes (samplers will be modified)
    training_dl = torch.utils.data.DataLoader(
        training_ds,
        batch_size=config["BATCH_SIZE"],
        shuffle=config["SHUFFLE"]
    )
    validation_dl = torch.utils.data.DataLoader(
        validation_ds,
        batch_size=config["BATCH_SIZE"],
        shuffle=config["SHUFFLE"]
    )

    return {
        "dataset": {
            "Training": training_ds,
            "Validation": validation_ds,
            "Test": test_ds,
        },
        "workers": workers,
        "training_dl": training_dl,
        "validation_dl": validation_dl,
    }


def dimensions(training_dl, config):
    """Extract dimensions from the training data"""
    # Extract frame dimensions
    input_shape = next(iter(training_dl))[
        "rgb"
    ].shape  # [batch_size, channels, height, width]
    sample_shape = next(iter(training_dl))["rgb"].shape[
        1:
    ]  # [channels, height, width]
    height = sample_shape[-2]  # Image height
    width = sample_shape[-1]  # Image width
    channels = 3  # Number of image channels
    batch_size = next(iter(training_dl))["rgb"].shape[0]  # Batch size


    return {
        "input_shape": input_shape,
        "sample_shape": sample_shape,
        "height": height,
        "width": width,
        "channels": channels,
        "batch_size": batch_size
    }


def hyperparameters(config):
    """Initialize training hyperparameters"""
    # Optimizer parameters
    momentum = config.get("MOMENTUM", 0)  # Momentum for optimizer, default to 0
    learning_rate = config.get(
        "LEARNING_RATE", 1e-3
    )  # Learning rate for optimizer, default to 1e-3
    weight_decay = config.get(
        "WEIGHT_DECAY", 0
    )  # Weight decay for optimizer, default to 0
    epochs = config.get("EPOCHS", 10)  # Number of training epochs, default to 10
    lastepoch = 0  # Initialize the last epoch

    # Training control parameters
    earlystopping_patience = config.get(
        "EARLY_STOPPING_PATIENCE", 5
    )  # Early stopping patience, default to 5 epochs
    actual_epoch_time = 0  # Initialize actual epoch time
    optimizer_bootstrap_name = config.get(
        "OPTIMIZER_BOOTSTRAP_NAME", "Adam"
    )  # Optimizer name for bootstrapping, default to 'Adam'
    optimizer_refining_name = config.get(
        "OPTIMIZER_REFINING_NAME", "Adam"
    )  # Optimizer name for refining, default to 'Adam'
    gradient_accumulation_steps = config.get(
        "GRADIENT_ACCUMULATION_STEPS", 1
    )  # Gradient accumulation steps, default to 1
    warmup_steps = config.get(
        "WARMUP_STEPS", 0
    )  # Warmup steps for learning rate, default to 0
    depth_scale_factor = config.get(
        "DEPTH_SCALE_FACTOR", 40
    )  # Depth scale factor, default to 40
    sift_patch_search_area = config.get(
        "REFINEMENT_AREA", 8
    )  # SIFT patch search area, default to 3
    switch_optimizer_epoch = config.get(
        "SWITCH_OPTIMIZER_EPOCH", 2
    )  # Epoch to switch optimizer, default to 2

    # Phase tracking
    in_swa_phase = False
    in_optswitch_phase = False

    # Other parameters
    logfreq_wandb = config.get("LOG_FREQ_WANDB", 1)
    logfreq_rerun = config.get("LOG_FREQ_RERUN", 1)

    return {
        "momentum": momentum,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "epochs": epochs,
        "lastepoch": lastepoch,
        "earlystopping_patience": earlystopping_patience,
        "actual_epoch_time": actual_epoch_time,
        "optimizer_bootstrap_name": optimizer_bootstrap_name,
        "optimizer_refining_name": optimizer_refining_name,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "warmup_steps": warmup_steps,
        "depth_scale_factor": depth_scale_factor,
        "sift_patch_search_area": sift_patch_search_area,
        "switch_optimizer_epoch": switch_optimizer_epoch,
        "in_swa_phase": in_swa_phase,
        "in_optswitch_phase": in_optswitch_phase,
        "logfreq_wandb": logfreq_wandb,
        "logfreq_rerun": logfreq_rerun,
    }


def optimizers(model, config):
    """Initialize optimizers and related components"""
    # Main optimizer
    optimizer_class = getattr(optimization, config.get("OPTIMIZER_BOOTSTRAP_NAME"))
    optimizer = optimizer_class(
        model.parameters(),
        lr=config.get("LEARNING_RATE"),
        weight_decay=config.get("WEIGHT_DECAY"),
    )

    # Gradient scaler for mixed precision
    scaler = torch.amp.GradScaler(
        "cuda" if torch.cuda.is_available() else "cpu", enabled=True
    )

    return {
        "optimizer": optimizer,
        "scaler": scaler,
    }


def loss_functions(config):
    """Initialize loss functions"""
    from losses import TripletLoss

    # Main loss function
    loss_fn = config.get("LOSS_FUNCTION", TripletLoss())
    if not isinstance(loss_fn, TripletLoss):
        loss_fn = TripletLoss()

    # Loss weights
    toplevel_loss_weights = config.get("TOPLEVEL_WEIGHTS_LOSS_FUN", [1.0, 1.0, 1.0])
    depth_loss_weights = config.get("WEIGHTS_DEPTH_LOSS_FUN", [0.85, 0.15])
    lossthresholds = config["LOSS_THRESHOLDS_WANDB"]

    return {
        "loss_fn": loss_fn,
        "toplevel_loss_weights": toplevel_loss_weights,
        "depth_loss_weights": depth_loss_weights,
        "lossthresholds": lossthresholds,
    }


def schedulers(optimizer, config, training_dl):
    """Initialize learning rate schedulers"""
    scheduler_config = config.get("LR_SCHEDULER")
    assert len(scheduler_config.keys()) == 2, "Only one scheduler (+OnPlateau)can be used at a time"
    
    if scheduler_config.get("ONPLATEAU"):
        onplateau_scheduler = scheduler_config.get("ONPLATEAU")
        # Plateau scheduler
        LRschedulerPlateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            "min",
            patience=onplateau_scheduler["PATIENCE"],
            factor=onplateau_scheduler["FACTOR"],
        )
    if scheduler_config.get("STEPWISE"):
        stepwise_scheduler = scheduler_config.get("STEPWISE")
        LRscheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config["EPOCHS"]
            * len(training_dl)
            // stepwise_scheduler["N_STEPS"],
            gamma=stepwise_scheduler["GAMMA"],
        )
    if scheduler_config.get("COSINE"):
        cosine_scheduler = scheduler_config.get("COSINE")
        batches_per_epoch = len(training_dl)  # int: number of batches per epoch
        n_epochs = config.get("EPOCHS")  # int: number of epochs
        n_peaks = cosine_scheduler.get("N_PERIODS", 1)  # int: number of cosine peaks
        T_max = (n_epochs * batches_per_epoch) // n_peaks // 2 - 1  # int: steps per peak
        cosine_scheduler = scheduler_config.get("COSINE")
        LRscheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=T_max  # T_max: int, number of steps per cosine cycle
        )
    if scheduler_config.get("EXPONENTIAL"):
        exponential_scheduler = scheduler_config.get("EXPONENTIAL")
        LRscheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=exponential_scheduler["GAMMA"],
        )
    
    return {
        "LRscheduler": LRscheduler,
        "LRschedulerPlateau": LRschedulerPlateau,
    }


def transforms(height, width):
    """Initialize image transformations"""
    imsavetransform = torchvision.transforms.Resize((256, 320), antialias=True)
    trimtransform = torchvision.transforms.Compose(
        [torchvision.transforms.Resize((height, width), antialias=True)]
    )

    return {
        "imsavetransform": imsavetransform,
        "trimtransform": trimtransform,
    }


def wandb(config, model=None, notes="", no_wandb=False):
    """Initialize Weights & Biases tracking"""
    if no_wandb:
        return {"wandb": None, "testtable": None}

    # Find existing run if specified
    logger.set_context("WANDB")

    run_id = None
    if "RUN" in config:
        try:
            resume_run = weightsandbiases.Api().runs(
                path=f'{weightsandbiases.api.default_entity}/{config.get("PROJECT", "MONO3D")}',
                filters={"display_name": config["RUN"]},
            )
            run_id = resume_run[0].id
            logger.info("Found run to resume:", run_id)

        except Exception:
            # logger.info("Creating new WandB run")
            resume_run = None
    else:
        resume_run = None

    # Initialize WandB
    stderr_capture = io.StringIO()
    with contextlib.redirect_stderr(stderr_capture):
        wandb_instance = weightsandbiases.init(
            entity=config.get("WANDB_ENTITY", "unreflectanything"),
            project=config.get("WANDB_PROJECT", "UnReflectAnything"),
            config=config,
            notes=notes,
            resume=("must" if resume_run is not None else "allow"),
            id=run_id,
            settings=weightsandbiases.Settings(code_dir="."),
        )

    # Extract run links from stderr
    captured_stderr = stderr_capture.getvalue()
    url_pattern = r"https?://[^\s]+"
    wandb_links = re.findall(url_pattern, captured_stderr)
    try:
        logger.info(
            f"Created run [yellow]{wandb_instance.name}[/] in project [orange1]{wandb_instance.project}[/]"
        )
        logger.info("[yellow]󰙨 Wandb RUN    [/]:", wandb_links[2])
        logger.info("[orange1]󱗼 Wandb PROJECT[/]:", wandb_links[1])
    except:
        pass

    # Define WandB metrics
    _define_wandb_metrics()

    # Initialize test table
    testtable = weightsandbiases.Table(
        columns=[
            "Video",
            "Batch",
            "Loss",
            "Precision",
            "Recall",
            "AUCPR",
            "Epipolar",
            "Fundamental",
            "Inliers",
            "MDistMean",
        ]
        )

    # Model Watcher
    wandb_instance.watch(model, log="all", log_freq=config["MODEL_WATCHER_FREQ_WANDB"])
    return {"wandb": wandb_instance, "testtable": testtable}


def _define_wandb_metrics():
    """Define metric structures for WandB"""
    # Step metrics
    weightsandbiases.define_metric("Step/batch")
    weightsandbiases.define_metric("Step/valbatch")
    weightsandbiases.define_metric("Step/lossissues")

    # Group metrics by step
    weightsandbiases.define_metric("Issues/*", step_metric="Step/lossissues")
    weightsandbiases.define_metric("Training/*", step_metric="Step/batch")
    weightsandbiases.define_metric("Gradients/*", step_metric="Step/batch")
    weightsandbiases.define_metric("Validation/*", step_metric="Step/valbatch")
    weightsandbiases.define_metric("HyperParameters/*", step_metric="Step/batch")

    # Epoch metrics
    weightsandbiases.define_metric("Step/epoch")
    weightsandbiases.define_metric("Training/epoch*", step_metric="Step/epoch")
    weightsandbiases.define_metric("Validation/epoch*", step_metric="Step/epoch")


def tracking_metrics():
    """Initialize step counters and metrics tracking"""
    # Metric trend tracking
    epoch_loss_trend_training = []
    epoch_loss_trend_validation = []

    # Summary metrics
    summary_train = []
    summary_val = []
    summary_test = []

    # Create dataframes for metrics
    metrics = {
        "Training": pd.DataFrame(),
        "Validation": pd.DataFrame(),
        "Test": pd.DataFrame(),
    }

    # Track loaded data
    loaded_paths = []
    startedat = datetime.datetime.now()

    return {
        "step": {
            "Training_batch": 0,
            "Validation_batch": 0,
            "Test_batch": 0,
            "epoch": 0,
            "idx": 0,
            "summary": 0,
        },
        "epoch_loss_trend_training": epoch_loss_trend_training,
        "epoch_loss_trend_validation": epoch_loss_trend_validation,
        "summary_train": summary_train,
        "summary_val": summary_val,
        "summary_test": summary_test,
        "metrics": metrics,
        "loaded_paths": loaded_paths,
        "startedat": startedat,
    }


def setup_run_directories(runs_dir, wandb_instance=None, savelocally=False):
    """Set up directories for storing run artifacts"""
    # Get run name from wandb or generate a new one
    runname = wandb_instance.name if wandb_instance is not None else None
    if runname is None or runname == "":
        runname = "offline_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create run directory structure
    run_dir = os.path.join(runs_dir, f"{runname}")
    os.makedirs(run_dir, exist_ok=True)

    paths_file = os.path.join(run_dir, "loadeddata.csv")

    models_dir = os.path.join(run_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    test_dir = os.path.join(run_dir, "tests")
    os.makedirs(os.path.join(test_dir), exist_ok=True)
    logger.set_context("SAVE")
    logger.info(" Run directory:", os.path.normpath(run_dir))
    return {
        "runname": runname,
        "RUN_DIR": run_dir,
        "paths_file": paths_file,
        "MODELS_DIR": models_dir,
        "TEST_DIR": test_dir,
        "savelocally": savelocally,
    }


def earlystopping(patience, models_dir, runname=None):
    """Initialize early stopping callback"""
    earlystopping = optimization.EarlyStopping(
        patience=patience,
        verbose=True,
        checkpointpath=models_dir,
        runname=runname,
    )

    return earlystopping


def matching_pipeline(config, model, device):
    """Initialize the matching pipeline and compute dimensions"""
    # Create matching pipeline
    matchingPipeline = MatchingPipeline(config, model, device=device)

    # Compute embedding dimensions
    embed_dim, seq_len = matchingPipeline.embed_dim, matchingPipeline.seq_len

    # Compute patch size
    height = matchingPipeline.height
    width = matchingPipeline.width
    patch_size = int((height * width / seq_len) ** 0.5)

    return {
        "matchingPipeline": matchingPipeline,
        "embed_dim": embed_dim,
        "seq_len": seq_len,
        "patch_size": patch_size,
    }


def save_hyperparameters_json(run_dir, config):
    """Save hyperparameters to disk"""
    hyperparams_path = os.path.join(run_dir, "hyperparams.json")

    # Ensure the file exists
    if not os.path.exists(hyperparams_path):
        with open(hyperparams_path, "x"):
            pass

    # Write the config
    with open(hyperparams_path, "w") as f:
        all_hyperparams = {"training": config}
        json.dump(all_hyperparams, f, indent=4, skipkeys=True, default=str)
