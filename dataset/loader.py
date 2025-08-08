"""
Dataset loading and configuration utilities.
"""

from typing import Any, Dict, List, Optional
import os
import torch
from torch.utils.data import DataLoader
from logger import get_logger
from utilities import get_hostname, detect_aval_cpus, coloredbar

from .base import Mono3D_Dataset
from .multi_dataset import MultiDataset
from .specialized import SCARED, CHOLEC80, GRASP
from .utils import split_videos

logger = get_logger(__name__).set_context("DATASET_LOADER")


def initialize_from_config(config: Dict[str, Any], inference: bool = False, verbose: bool = False) -> Dict[str, Any]:
    """
    Initialize datasets and dataloaders from configuration.

    Args:
        config: Configuration dictionary
        inference: Whether to initialize for inference (disabling augmentations)
        verbose: Whether to print verbose output

    Returns:
        Dictionary containing initialized datasets, dataloaders, and other components
    """
    # Initialize configuration parameters in global namespace for backward compatibility
    for key, value in config.items():
        globals()[key] = value

    # Setup device and compute resources
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.HOSTNAME = get_hostname()
    NUM_WORKERS = detect_aval_cpus()
    DATASET_ROOTDIR = os.environ.get("DATASET_ROOTDIR")

    # Get datasets configuration
    datasets_config = config.get("DATASETS", {})
    if not datasets_config:
        raise ValueError("No datasets specified in configuration")

    # Initialize empty lists for training and validation datasets
    training_datasets: List[Mono3D_Dataset] = []
    validation_datasets: List[Mono3D_Dataset] = []
    test_datasets: List[Mono3D_Dataset] = []
    all_dataset_names: List[str] = []
    dataset_info: Dict[str, Any] = {}

    # Process each dataset
    for dataset_name, dataset_config in datasets_config.items():
        all_dataset_names.append(dataset_name)

        # Get dataset path
        dataset_path = (
            os.path.join(DATASET_ROOTDIR, dataset_name)
            if DATASET_ROOTDIR
            and os.path.exists(os.path.join(DATASET_ROOTDIR, dataset_name))
            else dataset_config.get("PATH")
        )

        # Get dataset videos
        dataset_class = globals().get(dataset_name)
        if dataset_class is None:
            raise ValueError(f"No dataset class found for dataset {dataset_name}")

        all_videos = dataset_class.videonames()

        # Split videos into training, validation, and test
        test_videos = dataset_config.get("TEST_VIDEOS", [])
        train_val_split = dataset_config.get("TRAIN_VAL_SPLIT", 0.7)

        # Split video datasets
        training_videos, validation_videos = split_videos(
            all_videos, train_val_split, test_videos
        )
        # Create training dataset
        training_ds = dataset_class(
            path=dataset_path,
            name=dataset_name,
            vids=training_videos,
            height=IMAGE_HEIGHT,
            width=IMAGE_WIDTH,
            frameskip=dataset_config.get("FRAMESKIP", [1]),
            fps=dataset_config.get("FPS", 1),
            random_pose=dataset_config.get("RANDOM_POSE_TRAINING", False),
            random_pose_ranges=dataset_config.get("RANDOM_POSE_RANGES", []),
            # We force no augmentation when the dataset is instantiated, but we apply it with target_only at training time
            geometric_augmentation_prob=(
                0.0
                # dataset_config.get("AUGMENTATION_PROBABILITY").GEOMETRIC
                # if not inference
                # else 0.0
            ),
            color_augmentation_prob=(
                0.0
                # dataset_config.get("AUGMENTATION_PROBABILITY").COLOR
                # if not inference
                # else 0.0
            ),
            reverse_augmentation_prob=(
                0.0
                # dataset_config.get("AUGMENTATION_PROBABILITY").REVERSE
                # if not inference
                # else 0.0
            ),
            standstill_augmentation_prob=(
                0.0
                # dataset_config.get("AUGMENTATION_PROBABILITY").STANDSTILL
                # if not inference
                # else 0.0
            ),
            curriculum_factor=dataset_config.get("CURRICULUM_FACTOR", 1),
            device=DEVICE,
            with_frameskip=True,
            with_paths=True,
            as_euler=True,
            skip_order_check=False,
            verbose=verbose,
            fewframes=config.FEWFRAMES,
        )

        # Create validation dataset
        validation_ds = dataset_class(
            path=dataset_path,
            name=dataset_name,
            vids=validation_videos,
            height=IMAGE_HEIGHT,
            width=IMAGE_WIDTH,
            frameskip=dataset_config.get("FRAMESKIP", [1]),
            fps=dataset_config.get("FPS", 1),
            random_pose=dataset_config.get("RANDOM_POSE_TRAINING", False),
            random_pose_ranges=dataset_config.get("RANDOM_POSE_RANGES", []),
            curriculum_factor=dataset_config.get("CURRICULUM_FACTOR", 1),
            device=DEVICE,
            with_frameskip=True,
            with_paths=True,
            as_euler=True,
            skip_order_check=False,
            verbose=verbose,
            fewframes=config.FEWFRAMES,
        )

        # Create test dataset
        test_ds = dataset_class(
            path=dataset_path,
            name=dataset_name,
            vids=test_videos,
            height=IMAGE_HEIGHT,
            width=IMAGE_WIDTH,
            frameskip=dataset_config.get("FRAMESKIP", [1]),
            fps=dataset_config.get("FPS", 1),
            random_pose=False,
            curriculum_factor=dataset_config.get("CURRICULUM_FACTOR", 1),
            device=DEVICE,
            with_frameskip=True,
            with_paths=True,
            as_euler=True,
            skip_order_check=False,
            verbose=verbose,
            fewframes=config.FEWFRAMES,
        )

        # Add datasets to lists
        training_datasets.append(training_ds)
        validation_datasets.append(validation_ds)
        test_datasets.append(test_ds)

        # Store dataset info for reporting
        dataset_info[dataset_name] = {
            "path": dataset_path,
            "test_videos": test_videos,
        }

    # Create multi-datasets for training and validation
    training_ds = MultiDataset(training_datasets)
    validation_ds = MultiDataset(validation_datasets)
    test_ds = MultiDataset(test_datasets)

    # Create dataloaders
    training_dl, validation_dl, test_dl = [
        DataLoader(
            ds,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            drop_last=True,
            sampler=ds.sampler,
        )
        for ds in [training_ds, validation_ds, test_ds]
    ]

    # Print dataset information if requested
    if verbose:
        _print_dataset_summary(
            training_ds,
            validation_ds,
            test_ds,
            all_dataset_names,
            training_dl,
            validation_dl,
            test_dl,
            BATCH_SIZE,
        )

    # Return initialized components
    return {
        "dataset": {
            "Training": training_ds,
            "Validation": validation_ds,
            "Test": test_ds,
            "workers": NUM_WORKERS,
            "shuffle": SHUFFLE if "SHUFFLE" in globals() else True,
            "dataset": training_ds,
            "training_dl": training_dl,
            "validation_dl": validation_dl,
            "test_dl": test_dl,
            "paths": {
                ds_name: dataset_info[ds_name]["path"] for ds_name in all_dataset_names
            },
            "test_videos": {
                ds_name: dataset_info[ds_name]["test_videos"]
                for ds_name in all_dataset_names
            },
        },
        "config": config,
        "device": DEVICE,
    }


def _print_dataset_summary(
    training_ds,
    validation_ds,
    test_ds,
    all_dataset_names,
    training_dl,
    validation_dl,
    test_dl,
    batch_size,
):
    """
    Print a summary of the dataset configuration.

    Args:
        training_ds (MultiDataset): Training dataset
        validation_ds (MultiDataset): Validation dataset
        test_ds (MultiDataset): Test dataset
        all_dataset_names (list): List of all dataset names
        training_dl (DataLoader): Training dataloader
        validation_dl (DataLoader): Validation dataloader
        test_dl (DataLoader): Test dataloader
        batch_size (int): Batch size
    """
    logger.info(
        f"{training_ds.numvideos+validation_ds.numvideos+test_ds.numvideos} videos registered "
        f"[{training_ds.numframes+validation_ds.numframes+test_ds.numframes} total frames]"
    )
    logger.info(
        f"{len(training_ds.sampler)+len(validation_ds.sampler)+len(test_ds.sampler)} frames sampled "
    )

    # Print training dataset information
    frac_strings = []
    for ds_name in all_dataset_names:
        if ds_name in training_ds.fracframes:
            frac = training_ds.fracframes[ds_name] * 100
            frac_strings.append(f"{frac:.2f}% from {ds_name}")
    logger.info(
        f"[orange1]Training[/orange1]       [orange1]{len(training_dl)}[/orange1] batches of "
        f"{batch_size} samples [orange1] >>> {', '.join(frac_strings)}"
    )

    # Print validation dataset information
    frac_strings = []
    for ds_name in all_dataset_names:
        if ds_name in validation_ds.fracframes:
            frac = validation_ds.fracframes[ds_name] * 100
            frac_strings.append(f"{frac:.2f}% from {ds_name}")
    logger.info(
        f"[green]Validation[/green]     [green]{len(validation_dl)}[/green] batches of "
        f"{batch_size} samples [green] >>> {', '.join(frac_strings)}"
    )

    # Print test dataset information
    frac_strings = []
    for ds_name in all_dataset_names:
        if ds_name in test_ds.fracframes:
            frac = test_ds.fracframes[ds_name] * 100
            frac_strings.append(f"{frac:.2f}% from {ds_name}")
    logger.info(
        f"[cyan]Test[/cyan]           [cyan]{len(test_dl)}[/cyan] batches of "
        f"{batch_size} samples [cyan] >>> {', '.join(frac_strings)}"
    )

    # Print sample summary and split information
    training_ds.samplesummary()
    logger.info(
        "Splits: "
        + coloredbar(
            [len(training_dl), len(validation_dl), len(test_dl)],
            ["green", "orange3", "cyan"],
            50,
        )
    )
