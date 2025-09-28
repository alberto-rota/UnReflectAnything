"""
# Base Dataset Implementation

Base dataset implementation for monocular 3D camera pose estimation.

This module provides the core `Mono3D_Dataset` class that handles loading videos, frames, 
and camera poses, and provides methods for curriculum learning, augmentation, and various output formats.
"""

import os
import random
import json
import numpy as np
import torch
import torchvision
import torchvision.transforms as tvt
from torch.utils.data import Dataset
from tqdm import tqdm
from PIL import Image
from natsort import natsorted
from io import BytesIO
from google.cloud import storage

import projections as proj
import geometry
from utilities import closest_multiple
from utilities import generate_random_pose_tensor
from utilities import mat2quat
from dataset.utils import adapt_intrinsics_two_step

from logger import get_logger

logger = get_logger(__name__).set_context("DATASET")


class Mono3D_Dataset(Dataset):
    """
    # Mono3D Dataset Class

    Base dataset class for monocular 3D camera pose estimation.

    This dataset handles loading videos, frames, and camera poses, and provides
    methods for curriculum learning, augmentation, and various output formats.

    ## Features

    - **Multi-format Support**: Handles local filesystem and Google Cloud Storage
    - **Data Augmentation**: Built-in geometric, color, reverse, and standstill augmentations
    - **Curriculum Learning**: Progressive difficulty adjustment through frameskip curriculum
    - **Memory Optimization**: Optional preloading for faster training
    - **Flexible Output**: Configurable output formats (poses, intrinsics, depth, etc.)

    ## Dataset Structure

    Expects datasets organized as:
    ```
    dataset_root/
    ├── video1/
    │   ├── frame/
    │   │   ├── 000000.png
    │   │   ├── 000001.png
    │   │   └── ...
    │   └── poses_absolute/
    │       ├── 000000.json
    │       ├── 000001.json
    │       └── ...
    └── video2/
        └── ...
    ```

    ## Usage Example

    ```python
    # Basic usage
    dataset = Mono3D_Dataset(
        path="path/to/dataset",
        frameskip=[1, 2, 4],
        height=384,
        width=384
    )
    
    # With augmentation
    dataset = Mono3D_Dataset(
        path="path/to/dataset",
        geometric_augmentation_prob=0.5,
        color_augmentation_prob=0.3,
        curriculum_factor=2
    )
    
    # Get a sample
    sample = dataset[0]
    framestack = sample["framestack"]  # Shape: (2, 3, H, W)
    Ts2t = sample["Ts2t"]             # Shape: (4, 4) - transformation matrix
    ```

    ## Output Format

    Each sample returns a dictionary containing:

    - **framestack**: Tensor of shape `(2, 3, H, W)` containing source and target frames
    - **Ts2t**: Transformation matrix of shape `(4, 4)` from source to target pose
    - **intrinsics**: Camera intrinsic matrix of shape `(3, 3)` (if enabled)
    - **depth**: Depth maps of shape `(2, 1, H, W)` (if enabled and available)
    - **paths**: File paths for debugging (if enabled)
    - **frameskip**: Current frameskip value (if enabled)
    """

    def __init__(
        self,
        path=None,
        name=None,
        frameskip=1,
        height=384,
        width=384,
        original_width=1280,
        original_height=1024,
        backbone_patch_size=16,
        color_augmentation_prob=0.0,
        geometric_augmentation_prob=0.0,
        reverse_augmentation_prob=0.0,
        standstill_augmentation_prob=0.0,
        standardize=False,
        curriculum_factor=1,
        target_length=1,
        short=False,
        fewframes=False,
        fps=1,
        nvids=None,
        vids=None,
        exclude=None,
        device="cpu",
        as_euler=False,
        as_embedding=False,
        unit_translation=False,
        as_quat=False,
        with_fundamental=False,
        with_paths=False,
        with_frameskip=False,
        with_intrinsics=False,
        with_distortions=False,
        with_global_poses=False,
        with_depth=False,
        random_pose=False,
        random_pose_ranges=[],
        target_pose_only=False,
        transforms_only=False,
        skip_order_check=False,
        verbose=False,
        preload_in_memory=False,
        preload_transforms=tvt.Compose([]),
    ):
        """
        Initialize the Mono3D_Dataset for monocular 3D camera pose estimation.

        ## Parameters

        ### Data Source
        - **path** (`str`, optional): Base path to the dataset. If None, creates empty dataset.
        - **name** (`str`, optional): Dataset name. If None, inferred from path.
        - **vids** (`list`, optional): Specific videos to use. If None, uses all available.
        - **exclude** (`list`, optional): Videos to exclude from loading.
        - **nvids** (`int`, optional): Number of videos to use. If None, uses all available.

        ### Frame Selection
        - **frameskip** (`int` or `list`): Number of frames to skip between source and target.
          - If `int`: Fixed frameskip value
          - If `list`: Curriculum learning with progressive frameskip values
        - **fps** (`int`): Frames per second for sampling. Default: 1.
        - **target_length** (`int`): Length of target sequence. Default: 1.

        ### Image Processing
        - **height** (`int`): Target image height after resizing. Default: 384.
        - **width** (`int`): Target image width after resizing. Default: 384.
        - **original_width** (`int`): Original width of images. Default: 1280.
        - **original_height** (`int`): Original height of images. Default: 1024.
        - **backbone_patch_size** (`int`): Patch size for backbone network. Default: 16.

        ### Augmentation
        - **color_augmentation_prob** (`float`): Probability of color augmentation. Range: [0, 1].
        - **geometric_augmentation_prob** (`float`): Probability of geometric augmentation. Range: [0, 1].
        - **reverse_augmentation_prob** (`float`): Probability of reverse frame order. Range: [0, 1].
        - **standstill_augmentation_prob** (`float`): Probability of standstill augmentation. Range: [0, 1].
        - **standardize** (`bool`): Whether to standardize data. Default: False.

        ### Curriculum Learning
        - **curriculum_factor** (`int`): Factor for curriculum learning progression. Default: 1.
        - **short** (`bool`): Use small subset for quick testing. Default: False.
        - **fewframes** (`bool`): Use extremely small subset. Default: False.

        ### Output Configuration
        - **as_euler** (`bool`): Use Euler angles for poses. Default: True.
        - **as_quat** (`bool`): Use quaternions for poses. Default: False.
        - **as_embedding** (`bool`): Use embeddings. Default: False.
        - **unit_translation** (`bool`): Normalize translations to unit length. Default: False.
        - **with_fundamental** (`bool`): Include fundamental matrices. Default: True.
        - **with_paths** (`bool`): Include file paths in output. Default: True.
        - **with_frameskip** (`bool`): Include frameskip in output. Default: True.
        - **with_intrinsics** (`bool`): Include camera intrinsics. Default: True.
        - **with_distortions** (`bool`): Include distortion parameters. Default: False.
        - **with_global_poses** (`bool`): Include global poses. Default: True.
        - **with_depth** (`bool`): Include depth maps. Default: True.

        ### Advanced Options
        - **random_pose** (`bool`): Use random poses instead of real poses. Default: False.
        - **random_pose_ranges** (`list`): Ranges for random pose generation. Default: [].
        - **target_pose_only** (`bool`): Return only target pose. Default: False.
        - **transforms_only** (`bool`): Return only transformation matrices. Default: False.
        - **device** (`str`): Device for tensor operations. Default: "cpu".
        - **skip_order_check** (`bool`): Skip frame ordering validation. Default: False.
        - **verbose** (`bool`): Print verbose output. Default: False.
        - **preload_in_memory** (`bool`): Preload dataset in memory. Default: False.
        - **preload_transforms** (`torchvision.transforms`): Transforms for preloading. Default: Compose([]).

        ## Raises

        - **ValueError**: If invalid parameters are provided
        - **FileNotFoundError**: If dataset path doesn't exist
        - **RuntimeError**: If dataset loading fails

        ## Example

        ```python
        # Basic dataset with curriculum learning
        dataset = Mono3D_Dataset(
            path="/path/to/scared/dataset",
            frameskip=[1, 2, 4, 8],
            height=384,
            width=384,
            curriculum_factor=2,
            geometric_augmentation_prob=0.3
        )
        
        # Dataset for inference (no augmentation)
        dataset = Mono3D_Dataset(
            path="/path/to/dataset",
            frameskip=1,
            geometric_augmentation_prob=0.0,
            color_augmentation_prob=0.0,
            with_paths=True,
            with_intrinsics=True
        )
        ```
        """
        # Store configuration parameters
        self._initialize_parameters(
            path,
            name,
            frameskip,
            fps,
            device,
            curriculum_factor,
            as_euler,
            as_embedding,
            unit_translation,
            as_quat,
            with_fundamental,
            with_paths,
            with_frameskip,
            with_intrinsics,
            with_distortions,
            with_global_poses,
            with_depth,
            random_pose,
            random_pose_ranges,
            target_pose_only,
            transforms_only,
            preload_in_memory,
            preload_transforms,
            color_augmentation_prob,
            geometric_augmentation_prob,
            reverse_augmentation_prob,
            standstill_augmentation_prob,
            standardize,
            target_length,
        )

        # Set up image dimensions and transforms
        self._setup_transforms(
            height, width, original_height, original_width, backbone_patch_size
        )

        # Initialize data storage
        self._initialize_data_structures()

        # Handle GCS or local filesystem
        self._setup_storage_backend()

        # Load video data
        if path:
            self._load_videos(
                path, vids, exclude, short, fewframes, nvids, skip_order_check, verbose
            )

            # Preload dataset if requested
            if self.preload_in_memory and self.numframes > 0:
                if verbose:
                    logger.info(
                        f"Preloading {len(self.rgbpathlist)} frames into memory..."
                    )
                self._preload_dataset(verbose)
                if verbose:
                    logger.info(f"Preloading complete!")
        else:
            # Handle empty dataset case
            self._initialize_empty_dataset(verbose)

    def _initialize_parameters(
        self,
        path,
        name,
        frameskip,
        fps,
        device,
        curriculum_factor,
        as_euler,
        as_embedding,
        unit_translation,
        as_quat,
        with_fundamental,
        with_paths,
        with_frameskip,
        with_intrinsics,
        with_distortions,
        with_global_poses,
        with_depth,
        random_pose,
        random_pose_ranges,
        target_pose_only,
        transforms_only,
        preload_in_memory,
        preload_transforms,
        color_augmentation_prob,
        geometric_augmentation_prob,
        reverse_augmentation_prob,
        standstill_augmentation_prob,
        standardize,
        target_length,
    ):
        """
        Initialize and store all dataset parameters.

        Args:
            Multiple parameters from the constructor.
        """
        # Core parameters
        self.DEVICE = device
        self.name = path.split("/")[-1] if name is None else name
        self.fps = max(1, fps)  # Ensure fps is at least 1

        # Data format options
        self.as_euler = as_euler
        self.as_embedding = as_embedding
        self.unit_translation = unit_translation
        self.as_quat = as_quat

        # Output content flags
        self.with_paths = with_paths
        self.with_frameskip = with_frameskip
        self.with_intrinsics = with_intrinsics
        self.with_distortions = with_distortions
        self.with_fundamental = with_fundamental
        self.with_global_poses = with_global_poses
        self.with_depth = with_depth
        self.transforms_only = transforms_only
        self.target_pose_only = target_pose_only
        self.random_pose = random_pose
        if self.random_pose:
            self.random_pose_ranges = random_pose_ranges
        else:
            self.random_pose_ranges = []

        # Augmentation parameters
        self.color_augmentation_prob = color_augmentation_prob
        self.geometric_augmentation_prob = geometric_augmentation_prob
        self.reverse_augmentation_prob = reverse_augmentation_prob
        self.standstill_augmentation_prob = standstill_augmentation_prob

        # Learning parameters
        self.standardize = standardize
        self.target_length = target_length
        self.curriculum_factor = curriculum_factor

        # Preloading options
        self.preload_in_memory = preload_in_memory
        self.preload_transforms = preload_transforms

        # Frameskip configuration
        self.manual_frameskip = False
        self._setup_frameskip(frameskip)

        # Initialize pose converter

        # Check if path is a GCS path
        self.is_gcs = path.startswith("gs://") if path else False

    def _setup_frameskip(self, frameskip):
        """
        Set up frameskip and curriculum learning parameters.

        Args:
            frameskip (int or list): Number of frames to skip between source and target.
        """
        # Convert single frameskip to list if needed
        if isinstance(frameskip, int):
            frameskip = [frameskip]

        # Sort frameskips in descending order
        self.frameskip_set = sorted(frameskip, reverse=True)
        self.frameskip_curriculum_step = 0

        # Set up curriculum learning frameskips
        self.frameskip_set_curriculum = self.frameskip_set + (
            self.curriculum_factor - 1
        ) * [self.frameskip_set[self.frameskip_curriculum_step]]

    def _setup_transforms(
        self, height, width, original_height, original_width, backbone_patch_size
    ):
        """
        Set up image dimensions and transformation parameters.

        Args:
            height (int): Target image height
            width (int): Target image width
            original_height (int): Original image height
            original_width (int): Original image width
            backbone_patch_size (int): Patch size for the backbone network
        """
        # Store original dimensions
        self.original_height = original_height
        self.original_width = original_width
        self.aspect_ratio = original_width / original_height

        # Calculate backbone dimensions (multiples of patch_size)
        self.backbone_width = closest_multiple(width, backbone_patch_size, "inf")
        self.backbone_height = closest_multiple(height, backbone_patch_size, "inf")

        # Final dimensions for the dataset
        self.width = self.backbone_width
        self.height = self.backbone_height

        # Create resize transform
        self.resize_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(
                    (
                        self.backbone_height,
                        int(self.backbone_width * self.aspect_ratio),
                    ),
                    antialias=True,
                ),
            ]
        )

    def _initialize_data_structures(self):
        """Initialize all data storage structures used by the dataset."""
        # Path lists
        self.rgbpathlist = []
        self.pathlist = []
        self.depthpathlist = [] if self.with_depth else None

        # Pose and calibration lists
        self.poseslist = []
        self.intrinsicslist = []
        self.distortionslist = []
        self.Tlist = []
        self.Tinvlist = []

        # Dataset organization
        self.sampler = []

        # Preloading caches
        self.frame_cache = {} if self.preload_in_memory else None
        self.depth_cache = {} if self.preload_in_memory and self.with_depth else None
        self.embedding_cache = (
            {} if self.preload_in_memory and self.as_embedding else None
        )

    def _setup_storage_backend(self):
        """Set up the storage backend (GCS or local filesystem)."""
        if self.is_gcs:
            # Initialize GCS client
            self.storage_client = storage.Client()

            # Parse bucket name and prefix from path
            path_parts = self.name.split("/")
            bucket_name = path_parts[2]
            self.bucket = self.storage_client.bucket(bucket_name)
            self.gcs_prefix = "/".join(path_parts[3:])

    def _initialize_empty_dataset(self, verbose):
        """
        Initialize an empty dataset when no path is provided.

        Args:
            verbose (bool): Whether to print verbose output
        """
        self.numframes = 0
        self.numvideos = 0
        self.sampler = []
        self.Tlist = []
        self.pathlist = []
        self.Tinvlist = []

        if verbose:
            logger.info(f": [orange3]No videos loaded[/orange3]")

    def _load_videos(
        self, path, vids, exclude, short, fewframes, nvids, skip_order_check, verbose
    ):
        """
        Load videos from the specified path.

        Args:
            path (str): Path to the dataset
            vids (list): Specific videos to use
            exclude (list): Videos to exclude
            short (bool): Whether to use only a single video
            fewframes (bool): Whether to use a very small subset of a single video
            nvids (int): Number of videos to use
            skip_order_check (bool): Whether to skip the order check
            verbose (bool): Whether to print verbose output
        """
        # Load exclude list if it exists
        self.excluded = self._load_exclude_list(path)

        # Get list of video folders
        videofolders = self._list_video_folders(path)

        # Filter video folders based on parameters
        videofolders = self._filter_video_folders(
            videofolders, vids, exclude, short, fewframes, nvids
        )

        if verbose and path:
            logger.info(f"Loading {self.name}: ", end="")

        # Initialize counters
        frame_count = 0
        video_count = -1  # Start at -1 to handle empty list case

        # Process each video folder
        for video_count, video_folder in enumerate(natsorted(videofolders)):
            frames_in_video = self._load_video_data(
                path, video_folder, fewframes, verbose
            )
            frame_count += frames_in_video

            if verbose:
                logger.info(f"{video_folder} ", end="")

        # If videos were found, process the loaded frames and poses
        if video_count >= 0:
            self._process_loaded_data(
                frame_count, video_count, skip_order_check, verbose
            )
        else:
            self._initialize_empty_dataset(verbose)

    def _load_exclude_list(self, path):
        """
        Load the exclude list from the dataset path.

        Args:
            path (str): Path to the dataset

        Returns:
            list: List of excluded files
        """
        if path and os.path.exists(os.path.join(path, "exclude.json")):
            return json.load(open(os.path.join(path, "exclude.json")))
        return []

    def _list_video_folders(self, path):
        """
        List all video folders in the given path.

        Args:
            path (str): Path to the dataset

        Returns:
            list: List of video folder names
        """
        if not path:
            return []

        if self.is_gcs:
            # List all blobs with the prefix
            blobs = list(self.bucket.list_blobs(prefix=self.gcs_prefix))

            # Get unique video folders
            videofolders = set()
            for blob in blobs:
                # Get the first folder after the prefix
                rel_path = blob.name[len(self.gcs_prefix) :].strip("/")
                if rel_path:
                    folder = rel_path.split("/")[0]
                    if folder and "IGNORE" not in folder.upper():
                        videofolders.add(folder)
            return list(videofolders)
        else:
            # List folders from local filesystem
            return [
                vf
                for vf in os.listdir(path)
                if os.path.isdir(os.path.join(path, vf))
                and "IGNORE" not in os.path.join(path, vf).upper()
            ]

    def _filter_video_folders(
        self, videofolders, vids, exclude, short, fewframes, nvids
    ):
        """
        Filter video folders based on specified criteria.

        Args:
            videofolders (list): List of video folders
            vids (list): Specific videos to include
            exclude (list): Videos to exclude
            short (bool): Whether to use only one video
            fewframes (bool): Whether to use only one video and limit frames
            nvids (int): Maximum number of videos to use

        Returns:
            list: Filtered list of video folders
        """
        # Filter by specific video list
        if vids is not None:
            videofolders = [vf for vf in videofolders if vf in vids]

        # Filter out excluded videos
        if exclude is not None:
            videofolders = [vf for vf in videofolders if vf not in exclude]

        # Take only the first video for short/fewframes mode
        if short or fewframes:
            videofolders = [videofolders[0]] if videofolders else []

        # Limit number of videos
        elif nvids is not None:
            videofolders = videofolders[:nvids]

        return videofolders

    def _load_video_data(self, path, video_folder, fewframes, verbose):
        """
        Load data for a single video folder.

        Args:
            path (str): Base path to dataset
            video_folder (str): Name of the video folder
            fewframes (bool): Whether to limit the number of frames
            verbose (bool): Whether to print verbose output

        Returns:
            int: Number of frames loaded from this video
        """
        frames_in_video = 0
        path = path.rstrip("/") if path else ""

        if self.is_gcs:
            # Handle GCS path
            return self._load_video_from_gcs(path, video_folder, fewframes)
        else:
            # Handle local filesystem path
            return self._load_video_from_local(path, video_folder, fewframes)

    def _load_video_from_gcs(self, path, video_folder, fewframes):
        """
        Load video data from Google Cloud Storage.

        Args:
            path (str): Base path to dataset
            video_folder (str): Name of the video folder
            fewframes (bool): Whether to limit the number of frames

        Returns:
            int: Number of frames loaded
        """
        frames_in_video = 0
        video_path = f"{path}/{video_folder}"

        # List frame files
        frame_path = f"{video_path}/frame"
        frames = [
            f
            for f in self.list_gcs_files(frame_path)
            if f.endswith((".png", ".jpg", ".jpeg", ".pt"))
        ]

        # List pose files
        pose_path = f"{video_path}/poses_absolute"
        poses = [f for f in self.list_gcs_files(pose_path) if f.endswith((".json"))]

        # List depth files if needed
        depth_files = []
        if self.with_depth:
            depth_path = f"{video_path}/depth"
            depth_files = [
                f
                for f in self.list_gcs_files(depth_path)
                if f.endswith((".png", ".jpg", ".jpeg", ".pt"))
            ]

        # Process matched frame and pose files
        for frame_file, pose_file in zip(natsorted(frames), natsorted(poses)):
            # Add frame path to list
            self.rgbpathlist.append(frame_file)

            # Add depth path if requested
            if self.with_depth:
                frame_name = os.path.basename(frame_file)
                matching_depth = [
                    d for d in depth_files if os.path.basename(d) == frame_name
                ]
                if matching_depth:
                    self.depthpathlist.append(matching_depth[0])
                else:
                    self.depthpathlist.append(None)

            # Load and process pose data
            with open(
                os.path.join(video_path, "poses_absolute", pose_file), "r"
            ) as pfile:
                posejson = json.load(pfile)

            # Store pose and intrinsics
            self.poseslist.append(
                torch.tensor(posejson["camera-pose"], dtype=torch.float32)
            )
            self.intrinsicslist.append(
                torch.tensor(posejson["camera-calibration"]["KL"], dtype=torch.float32)
            )
            self.pathlist.append(os.path.join(video_path, "poses_absolute", pose_file))

            frames_in_video += 1

            # Limit frames if fewframes is True
            if fewframes and frames_in_video >= 100:
                break

        return frames_in_video

    def _load_video_from_local(self, path, video_folder, fewframes):
        """
        Load video data from local filesystem.

        Args:
            path (str): Base path to dataset
            video_folder (str): Name of the video folder
            fewframes (bool): Whether to limit the number of frames

        Returns:
            int: Number of frames loaded
        """
        frames_in_video = 0
        video_path = os.path.join(path, video_folder)

        # Check if required directories exist
        frame_dir = os.path.join(video_path, "rgb")
        pose_dir = os.path.join(video_path, "poses_absolute")
        depth_dir = os.path.join(video_path, "depth") if self.with_depth else None

        if os.path.exists(frame_dir) and os.path.exists(pose_dir):
            # Get sorted frames and poses
            frames = natsorted(os.listdir(frame_dir))
            poses = natsorted(os.listdir(pose_dir))

            # Get sorted depth files if needed
            depths = []
            if self.with_depth and depth_dir and os.path.exists(depth_dir):
                depths = natsorted(os.listdir(depth_dir))

            # Process matched frame and pose files
            for frame_file, pose_file in zip(frames, poses):
                # Add frame path to list
                self.rgbpathlist.append(os.path.join(video_path, "rgb", frame_file))

                # Add depth path if requested
                if self.with_depth:
                    if frame_file in depths:
                        self.depthpathlist.append(os.path.join(depth_dir, frame_file))
                    else:
                        self.depthpathlist.append(None)

                # Load and process pose data
                with open(
                    os.path.join(video_path, "poses_absolute", pose_file), "r"
                ) as pfile:
                    posejson = json.load(pfile)

                # Store pose, intrinsics, and distortions
                self.poseslist.append(
                    torch.tensor(posejson["camera-pose"], dtype=torch.float32)
                )
                self.intrinsicslist.append(
                    torch.tensor(
                        posejson["camera-calibration"]["KL"], dtype=torch.float32
                    )
                )
                self.distortionslist.append(
                    torch.tensor(
                        posejson["camera-calibration"]["DL"], dtype=torch.float32
                    )
                )
                self.pathlist.append(
                    os.path.join(video_path, "poses_absolute", pose_file)
                )

                frames_in_video += 1

                # Limit frames if fewframes is True
                if fewframes and frames_in_video >= 100:
                    break

        return frames_in_video

    def _process_loaded_data(self, frame_count, video_count, skip_order_check, verbose):
        """
        Process loaded data to compute transformation matrices and check ordering.

        Args:
            frame_count (int): Total number of frames loaded
            video_count (int): Number of videos loaded
            skip_order_check (bool): Whether to skip frame order checking
            verbose (bool): Whether to print verbose information
        """
        # Natural sort the RGB paths
        self.rgbpathlist = natsorted(self.rgbpathlist)

        # Check if frames are ordered correctly
        self.order_check = False
        if not skip_order_check and self.rgbpathlist and self.pathlist:
            self.order_check = self._check_frame_ordering()

        # If ordering is correct (or check is skipped), process the data
        if self.order_check or skip_order_check:
            # Store dataset statistics
            self.numvideos = video_count + 1
            self.numframes = frame_count

            # Store transformation matrices
            self.Tlist = list(self.poseslist)

            # Compute inverse transformations
            self.Tinvlist = [
                T if torch.all(T == -1) else torch.linalg.inv(T) for T in self.Tlist
            ]

            # Set up sampler
            self.sampler = list(
                range(
                    random.randint(
                        max(self.frameskip_set), max(self.frameskip_set) + self.fps - 1
                    ),
                    len(self),
                    self.fps,
                )
            )

            # Print results
            if verbose and not skip_order_check:
                logger.info(
                    f": [green]Loaded {self.numframes} frames - ORDER CHECK PASSED[/green]"
                )
            elif verbose and skip_order_check:
                logger.info(
                    f": [yellow]Loaded {self.numframes} frames - ORDER CHECK SKIPPED[/yellow]"
                )
        else:
            # Print error if order check failed
            if verbose:
                logger.info(
                    f": [red]Loaded {self.numframes} frames - FOUND ERRORS IN FRAME ORDERING[/red]"
                )

    def _check_frame_ordering(self):
        """
        Check if the frames are ordered correctly.

        Returns:
            bool: True if frames are ordered correctly, False otherwise
        """
        # Extract video and frame numbers for RGB files
        rgb_vid_list = np.array(
            [int(path.split("/")[-3][1:]) for path in self.rgbpathlist]
        )
        rgb_fnum_list = np.array(
            [
                int(
                    path.split("/")[-1]
                    .replace(".png", "")
                    .replace(".pt", "")
                    .replace(".jpg", "")
                )
                for path in self.rgbpathlist
            ]
        )

        # Check changes in video and frame numbers for RGB
        rgb_vid_list_diff = np.diff(rgb_vid_list) != 0
        rgb_fnum_list_diff = np.diff(rgb_fnum_list) != 1

        # Extract video and frame numbers for pose files
        poses_vid_list = np.array(
            [int(path.split("/")[-3][1:]) for path in self.pathlist]
        )
        poses_fnum_list = np.array(
            [int(path.split("/")[-1].replace(".json", "")) for path in self.pathlist]
        )

        # Check changes in video and frame numbers for poses
        poses_vid_list_diff = np.diff(poses_vid_list) != 0
        poses_fnum_list_diff = np.diff(poses_fnum_list) != 1

        # Check that both RGB and pose files have the same pattern of changes
        return np.all(
            np.logical_and(
                # Video changes should match
                np.logical_not(np.logical_xor(rgb_vid_list_diff, poses_vid_list_diff)),
                # Frame changes should match
                np.logical_not(
                    np.logical_xor(rgb_fnum_list_diff, poses_fnum_list_diff)
                ),
            )
        )

    def _preload_dataset(self, verbose=False):
        """Preload the entire dataset into memory."""
        if verbose:
            iterator = tqdm(
                enumerate(self.rgbpathlist),
                total=len(self.rgbpathlist),
                desc="Preloading frames",
            )
        else:
            iterator = enumerate(self.rgbpathlist)

        for idx, path in iterator:
            # Load image
            if self.is_gcs:
                # Remove gs:// prefix and bucket name
                blob_name = "/".join(path.split("/")[3:])
                blob = self.bucket.blob(blob_name)
                # Download to memory
                image_bytes = BytesIO(blob.download_as_bytes())
                img = Image.open(image_bytes)
                img_tensor = tvt.ToTensor()(img)
            else:
                img = Image.open(path)
                img_tensor = tvt.ToTensor()(img)

            # Apply any additional preload transformations
            if self.preload_transforms:
                img_tensor = self.preload_transforms(img_tensor)

            # Store in cache
            self.frame_cache[path] = img_tensor

            # Load embeddings if needed
            if self.as_embedding:
                emb_path = (
                    path.replace("rgb", "embeddings")
                    .replace(".png", ".pt")
                    .replace(".jpg", ".pt")
                )
                try:
                    emb = torch.stack(
                        torch.load(emb_path, weights_only=True, map_location="cpu")
                    ).squeeze(1)
                    self.embedding_cache[path] = emb
                except Exception as e:
                    if verbose:
                        logger.warning(f"Could not load embedding for {path}: {e}")

            # Load depth maps if needed
            if self.with_depth and self.depthpathlist:
                depth_path = self.depthpathlist[idx]
                if depth_path is not None:
                    try:
                        depth_tensor = self.load_depth(depth_path)
                        if self.preload_transforms:
                            depth_tensor = self.preload_transforms(depth_tensor)
                        self.depth_cache[depth_path] = depth_tensor
                    except Exception as e:
                        if verbose:
                            logger.warning(
                                f"Could not load depth for {depth_path}: {e}"
                            )

    def load_frame(self, path):
        """Load a frame from path or cache."""
        if self.preload_in_memory and path in self.frame_cache:
            return self.frame_cache[path]

        # If not in cache or not preloading, load from disk/GCS
        if self.is_gcs:
            # Remove gs:// prefix and bucket name
            blob_name = "/".join(path.split("/")[3:])
            blob = self.bucket.blob(blob_name)
            # Download to memory
            image_bytes = BytesIO(blob.download_as_bytes())
            img = Image.open(image_bytes)
            return tvt.ToTensor()(img)
        else:
            img = Image.open(path)
            return tvt.ToTensor()(img)

    def load_embedding(self, path):
        """Load an embedding from path or cache."""
        emb_path = (
            path.replace("rgb", "embeddings")
            .replace(".png", ".pt")
            .replace(".jpg", ".pt")
        )

        if (
            self.preload_in_memory
            and self.embedding_cache
            and path in self.embedding_cache
        ):
            return self.embedding_cache[path]

        # If not in cache or not preloading, load from disk
        return torch.stack(
            torch.load(emb_path, weights_only=True, map_location="cpu")
        ).squeeze(1)

    def step_frameskip_curriculum(self):
        if self.frameskip_curriculum_step >= len(self.frameskip_set) - 1:
            self.frameskip_curriculum_step = len(self.frameskip_set) - 1
            return

        self.frameskip_curriculum_step += 1
        self.frameskip_set_curriculum = self.frameskip_set + (
            self.curriculum_factor - 1
        ) * [self.frameskip_set[self.frameskip_curriculum_step]]

    def load_depth(self, path):
        """Load a depth map from path or cache."""
        if path is None:
            # Return zero tensor with same dimensions as RGB images
            return torch.zeros(3, self.height, self.width)

        if self.preload_in_memory and path in self.depth_cache:
            return self.depth_cache[path]

        # If not in cache or not preloading, load from disk/GCS
        if self.is_gcs:
            # Remove gs:// prefix and bucket name
            blob_name = "/".join(path.split("/")[3:])
            blob = self.bucket.blob(blob_name)
            # Download to memory
            image_bytes = BytesIO(blob.download_as_bytes())
            depth_img = Image.open(image_bytes)
            return tvt.ToTensor()(depth_img)
        else:
            depth_img = Image.open(path)
            return tvt.ToTensor()(depth_img)

    def reset_sampler(self):
        """Reset the sampler to the beginning of the dataset."""
        self.sampler = list(
            range(
                random.randint(
                    max(self.frameskip_set), max(self.frameskip_set) + self.fps - 1
                ),
                len(self),
                self.fps,
            )
        )

    def __getitem__(self, idx):
        """
        Get a sample from the dataset at the specified index.

        This method handles selecting frames based on curriculum learning (if applicable),
        ensuring frames are sequential, applying geometric and color augmentations,
        and preparing the output dictionary.

        Args:
            idx (int): Index of the frame to retrieve

        Returns:
            dict: Dictionary containing the requested data, or None if there was an error
        """
        # Validate and adjust index
        idx = self._validate_index(idx)

        # Select frameskip based on curriculum settings
        self._select_frameskip()
        # Adjust index if needed and check frame validity
        idx = self._ensure_valid_frame_pair(idx)
        if idx is None:
            return None

        # Check for excluded frames
        idx = self._check_excluded_frames(idx)
        if idx is None:
            return None

        # Handle target-pose-only mode
        if self.target_pose_only:
            return self._prepare_target_pose_only_output(idx)

        # Get the transformation matrix between source and target frames
        Ts2t = self._get_transformation_matrix(idx)

        # If transforms_only mode is enabled, return only the transformation data
        if self.transforms_only:
            return self._prepare_transforms_only_output(idx, Ts2t)

        # Load frames
        framestack, embeddings, depthstack = self._load_frame_pair(idx)

        # Apply augmentations
        # framestack, Ts2t = self._apply_augmentations(framestack, Ts2t)

        # Apply the same augmentations to depthstack if it exists
        # if depthstack is not None and self.geometric_augmentation_prob > 0:
        #     # Only apply geometric augmentations to depth maps
        #     depthstack, _ = aug.geometric_augmentation(
        #         depthstack, Ts2t, self.geometric_augmentation_prob
        #     )

        # Prepare output dictionary with all requested data
        return self._prepare_full_output(idx, framestack, Ts2t, embeddings, depthstack)

    def __len__(self):
        return len(self.rgbpathlist)

    def _validate_index(self, idx):
        """
        Validate and adjust the index to ensure it's within bounds.

        Args:
            idx (int): Original index

        Returns:
            int: Adjusted index within valid bounds
        """
        if idx >= len(self):
            return len(self) - 1
        return idx

    def _select_frameskip(self):
        """
        Select the frameskip value based on curriculum settings.
        """
        if not self.manual_frameskip:
            self.frameskip = random.choice(self.frameskip_set_curriculum)

    def _ensure_valid_frame_pair(self, idx):
        """
        Ensure that the source and target frames form a valid pair.

        This checks that the frames are from the same video and are sequential.

        Args:
            idx (int): Target frame index

        Returns:
            int: Adjusted index, or None if no valid pair can be found
        """
        # Ensure index is at least frameskip
        if idx < self.frameskip:
            idx = self.frameskip

        # Ensure source and target are from the same video
        source_idx = idx
        target_idx = idx - self.frameskip

        sourcevid = self.rgbpathlist[source_idx].split("/")[-3]
        targetvid = self.rgbpathlist[target_idx].split("/")[-3]

        if sourcevid != targetvid:
            # Skip to the next frameskip if frames are from different videos
            idx += self.frameskip
            if idx >= len(self):
                idx = len(self) - 1

            # Try again with the new index
            return self._ensure_valid_frame_pair(idx)

        # Check if the frames are sequential
        if not self._check_sequential(source_idx, target_idx):
            logger.error(
                f"Trying to load frames {self.rgbpathlist[source_idx]} and "
                f"{self.rgbpathlist[target_idx]} which are not sequential"
            )
            return None

        return idx

    def _check_excluded_frames(self, idx):
        """
        Check if any frame in the sequence is excluded.

        Args:
            idx (int): Target frame index

        Returns:
            int: Adjusted index, or None if no valid pair can be found
        """
        loading_excluded = True
        while loading_excluded:
            loading_excluded = False
            provisional_paths = [
                self.pathlist[idx - self.frameskip + i]
                for i in range(self.frameskip + 1)
            ][::-1]

            for path in provisional_paths:
                if path in self.excluded:
                    # Skip to the next frameskip if any frame is excluded
                    idx += self.frameskip
                    if idx >= len(self):
                        idx = len(self) - 1
                    loading_excluded = True
                    break

        return idx

    def _check_sequential(self, sourceidx, targetidx):
        """
        Checks if the source and target indices correspond to sequential frames in the same video.

        This is done by comparing video identifiers and frame numbers extracted from the file paths
        stored in `rgbpathlist` and `pathlist`.

        Args:
            sourceidx (int): Index of the source frame.
            targetidx (int): Index of the target frame.

        Returns:
            bool: True if the source and target frames are sequential and belong to the same video, False otherwise.
        """
        # Extract video and frame identifiers for the source and target from both rgb and data paths
        sourcev_rgb = self.rgbpathlist[sourceidx].split("/")[-3][1:]
        targetv_rgb = self.rgbpathlist[targetidx].split("/")[-3][1:]
        sourcev_pose = self.pathlist[sourceidx].split("/")[-3][1:]
        targetv_pose = self.pathlist[targetidx].split("/")[-3][1:]

        sourcef_rgb = (
            self.rgbpathlist[sourceidx]
            .split("/")[-1]
            .replace(".png", "")
            .replace(".pt", "")
            .replace(".jpg", "")
        )
        sourcef_pose = self.pathlist[sourceidx].split("/")[-1].replace(".json", "")
        targetf_rgb = (
            self.rgbpathlist[targetidx]
            .split("/")[-1]
            .replace(".png", "")
            .replace(".pt", "")
            .replace(".jpg", "")
        )
        targetf_pose = self.pathlist[targetidx].split("/")[-1].replace(".json", "")

        # Check if the source and target frames are from the same video
        samevideo = sourcev_rgb == targetv_rgb == sourcev_pose == targetv_pose

        # Check if the frame numbers are sequential according to the frameskip
        correctframe = (
            sourcef_rgb == sourcef_pose
            and targetf_rgb == targetf_pose
            and (int(sourcef_rgb) - self.frameskip == int(targetf_rgb))
            and (int(sourcef_pose) - self.frameskip == int(targetf_pose))
        )

        return samevideo and correctframe

    def _prepare_target_pose_only_output(self, idx):
        """
        Prepare output dictionary when only target pose is needed.

        Args:
            idx (int): Target frame index

        Returns:
            dict: Dictionary containing the target pose
        """
        batch_data_dict = {}

        # Add target pose in requested format
        if self.as_euler:
            batch_data_dict["Tt"] = geometry.mat2euler(self.Tlist[idx])
        elif self.as_quat:
            batch_data_dict["Tt"] = mat2quat(self.Tlist[idx])
        else:
            batch_data_dict["Tt"] = self.Tlist[idx]

        # Add paths if requested
        if self.with_paths:
            pathlist_ofbatch = [self.pathlist[idx], self.pathlist[idx - self.frameskip]]
            batch_data_dict["paths"] = pathlist_ofbatch[::-1]

        return batch_data_dict

    def _get_transformation_matrix(self, idx):
        """
        Get the transformation matrix between source and target frames.

        Args:
            idx (int): Target frame index

        Returns:
            torch.Tensor: Transformation matrix
        """
        # Get transformation matrix based on mode (random or from data)
        if not self.random_pose:
            Ts2t = torch.matmul(self.Tinvlist[idx - self.frameskip], self.Tlist[idx])
        else:
            Ts2t_euler_random = generate_random_pose_tensor(
                translation_minmax=[
                    (-self.random_pose_ranges[0], self.random_pose_ranges[0]),
                    (-self.random_pose_ranges[0], self.random_pose_ranges[0]),
                    (-self.random_pose_ranges[0], self.random_pose_ranges[0]),
                ],
                euler_minmax=[
                    (-self.random_pose_ranges[1], self.random_pose_ranges[1]),
                    (-self.random_pose_ranges[1], self.random_pose_ranges[1]),
                    (-self.random_pose_ranges[1], self.random_pose_ranges[1]),
                ],
                angle_unit="degrees",
            )[0]
            Ts2t = geometry.euler2mat(Ts2t_euler_random)

        # Normalize translation if requested
        if self.unit_translation:
            Ts2t[:3, -1] /= Ts2t[:3, -1].norm()

        # Convert to requested format
        if self.as_euler:
            Ts2t = geometry.mat2euler(Ts2t)
        elif self.as_quat:
            Ts2t = mat2quat(Ts2t)

        return Ts2t

    def _prepare_transforms_only_output(self, idx, Ts2t):
        """
        Prepare output dictionary when only transformation data is needed.

        Args:
            idx (int): Target frame index
            Ts2t (torch.Tensor): Transformation matrix

        Returns:
            dict: Dictionary containing the transformation data
        """
        batch_data_dict = {"Ts2t": Ts2t}

        # Add paths if requested
        if self.with_paths:
            pathlist_ofbatch = [self.pathlist[idx], self.pathlist[idx - self.frameskip]]
            batch_data_dict["paths"] = pathlist_ofbatch[::-1]

        return batch_data_dict

    def _load_frame_pair(self, idx):
        """
        Load the source and target frames.

        Args:
            idx (int): Target frame index

        Returns:
            tuple: (framestack, embeddings, depthstack) where:
                - framestack is a torch.Tensor containing the frames
                - embeddings is a torch.Tensor containing embeddings or None
                - depthstack is a torch.Tensor containing depth maps or None
        """
        # Load target frame
        target = self.load_frame(self.rgbpathlist[idx])

        # Load source frame
        source = self.load_frame(self.rgbpathlist[idx - self.frameskip])

        # Load embeddings if needed
        embeddings = None
        if self.as_embedding:
            if self.preload_in_memory and self.embedding_cache:
                targetemb = self.embedding_cache.get(self.rgbpathlist[idx])
                sourceemb = self.embedding_cache.get(
                    self.rgbpathlist[idx - self.frameskip]
                )
            else:
                targetemb = self.load_embedding(self.rgbpathlist[idx])
                sourceemb = self.load_embedding(self.rgbpathlist[idx - self.frameskip])
            embeddings = torch.stack([sourceemb] + [targetemb])

        # Load depth maps if needed
        depthstack = None
        if self.with_depth and self.depthpathlist:
            # Get corresponding depth maps
            target_depth_path = (
                self.depthpathlist[idx] if idx < len(self.depthpathlist) else None
            )
            source_depth_path = (
                self.depthpathlist[idx - self.frameskip]
                if idx - self.frameskip < len(self.depthpathlist)
                else None
            )

            # Load depth maps
            # We add the .mean(0) to ensure that the depth maps are interpreted as grayscale
            target_depth = self.load_depth(target_depth_path).mean(0, keepdim=True)
            source_depth = self.load_depth(source_depth_path).mean(0, keepdim=True)
            depthstack = torch.stack([source_depth] + [target_depth])

        # Combine source and target into a single tensor
        framestack = torch.stack([source] + [target])

        # Adjust dimensions to match selected backbone
        framestack = self.resize_transform(framestack)

        # Apply same transformations to depthstack if it exists
        if depthstack is not None:
            depthstack = self.resize_transform(depthstack)

        # Crop to final dimensions
        if self.height == self.width:
            # For square output, use minimum dimension
            cropdim = min(self.height, self.width)
            framestack = tvt.CenterCrop((cropdim, cropdim))(framestack)
            if depthstack is not None:
                depthstack = tvt.CenterCrop((cropdim, cropdim))(depthstack)
        else:
            # For rectangular output, use specified dimensions
            framestack = tvt.CenterCrop((self.height, self.width))(framestack)
            if depthstack is not None:
                depthstack = tvt.CenterCrop((self.height, self.width))(depthstack)

        return framestack, embeddings, depthstack

    def _prepare_full_output(self, idx, framestack, Ts2t, embeddings, depthstack):
        """Prepare the full output dictionary with all required data.

        Args:
            idx: Index of the sample
            framestack: Stack of frames/images
            Ts2t: Transformation matrices
            embeddings: Pre-computed embeddings if any
            depthstack: Stack of depth maps if available

        Returns:
            Dictionary containing all processed data for the model
        """
        # Basic output that's always included
        output = {"idx": idx, "framestack": framestack, "Ts2t": Ts2t}

        # Add embeddings if available
        if embeddings is not None:
            output["embeddings"] = embeddings

        # Add depth information if available
        if depthstack is not None and self.with_depth:
            output["depthstack"] = depthstack

        # Add paths if requested
        if self.with_paths:
            # Return a tuple of (source_path, target_path) directly
            source_path = self.pathlist[idx - self.frameskip]
            target_path = self.pathlist[idx]
            output["paths"] = (source_path, target_path)

        # Add frameskip if requested
        if self.with_frameskip:
            output["frameskip"] = self.frameskip

        # Add intrinsics if requested
        if self.with_intrinsics:
            # Get intrinsics matrix
            K = self.intrinsicslist[idx]

            # Check if intrinsics matrix contains negative values
            if torch.any(K < 0):
                # Generate sensible intrinsics based on image dimensions
                # Common values: focal length ~= image_dimension, principal point at image center
                fx = fy = max(
                    self.width, self.height
                )  # Reasonable focal length estimate
                cx, cy = (
                    self.width / 2,
                    self.height / 2,
                )  # Principal point at image center

                # Create new intrinsics matrix with sensible values
                K = torch.tensor(
                    [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
                    dtype=torch.float32,
                    device=K.device,
                )

            # Adapt intrinsics to correct dimensions
            output["intrinsics"] = adapt_intrinsics_two_step(
                K=K,
                orig_width=self.original_width,
                orig_height=self.original_height,
                backbone_width=self.backbone_width,
                backbone_height=self.backbone_height,
                final_width=self.width,
                final_height=self.height,
            )

        # Add distortions if requested
        if (
            self.with_distortions
            and hasattr(self, "distortionslist")
            and len(self.distortionslist) > idx
        ):
            output["distortions"] = self.distortionslist[idx]

        # Add fundamental matrix if requested
        if self.with_fundamental:
            # Only compute fundamental matrix if Ts2t is a transformation matrix
            K = (
                output.get("intrinsics")
                if "intrinsics" in output
                else adapt_intrinsics_two_step(
                    K=self.intrinsicslist[idx],
                    orig_width=self.original_width,
                    orig_height=self.original_height,
                    backbone_width=self.backbone_width,
                    backbone_height=self.backbone_height,
                    final_width=self.width,
                    final_height=self.height,
                )
            )
            F = self.pose2fund(geometry.euler2mat(Ts2t) if self.as_euler else Ts2t, K)
            output["fundamental"] = F

        # Add global poses if requested
        if self.with_global_poses:
            output["Ts"] = self.Tlist[idx - self.frameskip]
            output["Tt"] = self.Tlist[idx]

        return output
    


class SCARED(Mono3D_Dataset):
    """
    # SCARED Dataset Class

    SCARED (Surgical Computer Vision) dataset class for monocular 3D camera pose estimation.

    Extends the base `Mono3D_Dataset` with SCARED-specific configurations including
    original image dimensions and depth map support.

    ## Dataset Information

    - **Original Dimensions**: 1280×1024 pixels
    - **Depth Support**: Yes (depth maps available)
    - **Video Count**: 34 videos (v1 to v34)
    - **Domain**: Surgical computer vision
    - **Use Case**: Monocular visual odometry in surgical environments

    ## Default Configuration

    - `original_width`: 1280
    - `original_height`: 1024
    - `with_depth`: True

    ## Usage Example

    ```python
    from dataset import SCARED
    
    # Basic usage with default settings
    scared_dataset = SCARED(path="/path/to/scared/dataset")
    
    # With custom configuration
    scared_dataset = SCARED(
        path="/path/to/scared/dataset",
        frameskip=[1, 2, 4, 8],
        height=384,
        width=384,
        geometric_augmentation_prob=0.3,
        curriculum_factor=2
    )
    
    # Get available video names
    video_names = SCARED.videonames()
    print(f"Available videos: {video_names}")
    ```

    ## Video Naming Convention

    Videos are named as `v1`, `v2`, ..., `v34` following the SCARED dataset structure.
    """

    def __init__(self, **kwargs):
        """
        Initialize the SCARED dataset.

        ## Parameters

        All parameters are passed through to the parent `Mono3D_Dataset` class.
        The following parameters are set by default for SCARED:

        - **original_width** (`int`): 1280 (original image width)
        - **original_height** (`int`): 1024 (original image height)
        - **with_depth** (`bool`): True (depth maps are available)

        ## Example

        ```python
        # Initialize with default SCARED settings
        dataset = SCARED(path="/path/to/scared")
        
        # Override default settings
        dataset = SCARED(
            path="/path/to/scared",
            original_width=640,  # Override default
            with_depth=False     # Disable depth loading
        )
        ```
        """
        params = {
            "original_width": 1280,
            "original_height": 1024,
            "with_depth": True,
        }
        params.update(kwargs)
        super().__init__(**params)

    @staticmethod
    def videonames():
        """
        Get the list of video names for the SCARED dataset.

        ## Returns

        `list`: List of video names from v1 to v34.

        ## Example

        ```python
        video_names = SCARED.videonames()
        print(video_names)  # ['v1', 'v2', ..., 'v34']
        ```
        """
        return [f"v{i}" for i in range(1, 35)]
