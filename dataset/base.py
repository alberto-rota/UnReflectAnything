"""
Base dataset implementation for monocular 3D camera pose estimation.
"""

import os
import random
import json
from typing import Any, Dict, List, Optional, Union
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

import geometry
import dataset.augmentation as aug
from utilities import closest_multiple
from utilities import generate_random_pose_tensor
from utilities import mat2quat
from dataset.utils import adapt_intrinsics_two_step

from logger import get_logger

logger = get_logger(__name__).set_context("DATASET")


class Mono3D_Dataset(Dataset):
    """
    Base dataset class for monocular 3D camera pose estimation.

    This dataset handles loading videos, frames, and camera poses, and provides
    methods for curriculum learning, augmentation, and various output formats.
    """

    def __init__(
        self,
        path: Optional[str] = None,
        name: Optional[str] = None,
        frameskip: Union[int, List[int]] = 1,
        height: int = 384,
        width: int = 384,
        original_width: int = 1280,
        original_height: int = 1024,
        backbone_patch_size: int = 16,
        color_augmentation_prob: float = 0.0,
        geometric_augmentation_prob: float = 0.0,
        reverse_augmentation_prob: float = 0.0,
        standstill_augmentation_prob: float = 0.0,
        standardize: bool = False,
        curriculum_factor: int = 1,
        target_length: int = 1,
        short: bool = False,
        fewframes: bool = False,
        fps: int = 1,
        nvids: Optional[int] = None,
        vids: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
        device: str = "cpu",
        as_euler: bool = True,
        as_embedding: bool = False,
        unit_translation: bool = False,
        as_quat: bool = False,
        with_fundamental: bool = True,
        with_paths: bool = True,
        with_frameskip: bool = True,
        with_intrinsics: bool = True,
        with_distortions: bool = False,
        with_global_poses: bool = True,
        with_depth: bool = True,
        random_pose: bool = False,
        random_pose_ranges: List[float] = None,
        target_pose_only: bool = False,
        transforms_only: bool = False,
        skip_order_check: bool = False,
        verbose: bool = False,
        preload_in_memory: bool = False,
        preload_transforms: tvt.Compose = None,
    ) -> None:
        """
        Initialize the Mono3D_Dataset for monocular 3D camera pose estimation.

        Args:
            path: Base path to the dataset
            name: Dataset name (if None, inferred from path)
            frameskip: Number of frames to skip between source and target
            height: Target image height after resizing
            width: Target image width after resizing
            original_width: Original width of the images
            original_height: Original height of the images
            backbone_patch_size: Patch size for the backbone network
            color_augmentation_prob: Probability of applying color augmentation
            geometric_augmentation_prob: Probability of applying geometric augmentation
            reverse_augmentation_prob: Probability of applying reverse augmentation
            standstill_augmentation_prob: Probability of applying standstill augmentation
            standardize: Whether to standardize the data
            curriculum_factor: Factor for curriculum learning
            target_length: Length of the target sequence
            short: Whether to use only a small part of the dataset
            fewframes: Whether to use an extremely small part of the dataset
            fps: Frames per second for sampling
            nvids: Number of videos to use (if None, use all)
            vids: Specific videos to use (if None, use all)
            exclude: Videos to exclude
            device: Device to use ("cpu" or "cuda")
            as_euler: Whether to use Euler angles for poses
            as_embedding: Whether to use embeddings
            unit_translation: Whether to normalize translations to unit length
            as_quat: Whether to use quaternions for poses
            with_fundamental: Whether to include fundamental matrices
            with_paths: Whether to include paths in the output
            with_frameskip: Whether to include frameskip in the output
            with_intrinsics: Whether to include intrinsics in the output
            with_distortions: Whether to include distortions in the output
            with_global_poses: Whether to include global poses in the output
            with_depth: Whether to include depth maps in the output
            random_pose: Whether to use random poses
            random_pose_ranges: Ranges for random pose generation
            target_pose_only: Whether to only include the target pose
            transforms_only: Whether to only include the transforms
            skip_order_check: Whether to skip the order check
            verbose: Whether to print verbose output
            preload_in_memory: Whether to preload the dataset in memory
            preload_transforms: Additional transforms to apply during preloading
        """
        if random_pose_ranges is None:
            random_pose_ranges = []
        if preload_transforms is None:
            preload_transforms = tvt.Compose([])

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
                    logger.info("Preloading complete!")
        else:
            # Handle empty dataset case
            self._initialize_empty_dataset(verbose)

    def _initialize_parameters(
        self,
        path: Optional[str],
        name: Optional[str],
        frameskip: Union[int, List[int]],
        fps: int,
        device: str,
        curriculum_factor: int,
        as_euler: bool,
        as_embedding: bool,
        unit_translation: bool,
        as_quat: bool,
        with_fundamental: bool,
        with_paths: bool,
        with_frameskip: bool,
        with_intrinsics: bool,
        with_distortions: bool,
        with_global_poses: bool,
        with_depth: bool,
        random_pose: bool,
        random_pose_ranges: List[float],
        target_pose_only: bool,
        transforms_only: bool,
        preload_in_memory: bool,
        preload_transforms: tvt.Compose,
        color_augmentation_prob: float,
        geometric_augmentation_prob: float,
        reverse_augmentation_prob: float,
        standstill_augmentation_prob: float,
        standardize: bool,
        target_length: int,
    ) -> None:
        """
        Initialize dataset parameters.

        Args:
            path: Base path to the dataset
            name: Dataset name
            frameskip: Frame skip configuration
            fps: Frames per second
            device: Device to use
            curriculum_factor: Curriculum learning factor
            as_euler: Whether to use Euler angles
            as_embedding: Whether to use embeddings
            unit_translation: Whether to normalize translations
            as_quat: Whether to use quaternions
            with_fundamental: Whether to include fundamental matrices
            with_paths: Whether to include paths
            with_frameskip: Whether to include frameskip
            with_intrinsics: Whether to include intrinsics
            with_distortions: Whether to include distortions
            with_global_poses: Whether to include global poses
            with_depth: Whether to include depth maps
            random_pose: Whether to use random poses
            random_pose_ranges: Random pose ranges
            target_pose_only: Whether to only include target pose
            transforms_only: Whether to only include transforms
            preload_in_memory: Whether to preload in memory
            preload_transforms: Preload transforms
            color_augmentation_prob: Color augmentation probability
            geometric_augmentation_prob: Geometric augmentation probability
            reverse_augmentation_prob: Reverse augmentation probability
            standstill_augmentation_prob: Standstill augmentation probability
            standardize: Whether to standardize data
            target_length: Target sequence length
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
        # self.pose2fund = proj.Pose2Fundamental()

        # Check if path is a GCS path
        self.is_gcs = path.startswith("gs://") if path else False

    def _setup_frameskip(self, frameskip: Union[int, List[int]]) -> None:
        """
        Set up frameskip and curriculum learning parameters.

        Args:
            frameskip: Number of frames to skip between source and target.
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
        self,
        height: int,
        width: int,
        original_height: int,
        original_width: int,
        backbone_patch_size: int,
    ) -> None:
        """
        Set up image dimensions and transformation parameters.

        Args:
            height: Target image height
            width: Target image width
            original_height: Original image height
            original_width: Original image width
            backbone_patch_size: Patch size for the backbone network
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

    def _initialize_data_structures(self) -> None:
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
            logger.info(": [orange3]No videos loaded[/orange3]")

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
        frame_dir = os.path.join(video_path, "frame")
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
                self.rgbpathlist.append(os.path.join(video_path, "frame", frame_file))

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
                    path.replace("frame", "embeddings")
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
            path.replace("frame", "embeddings")
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

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        """
        Get a sample from the dataset at the specified index.

        This method handles selecting frames based on curriculum learning (if applicable),
        ensuring frames are sequential, applying geometric and color augmentations,
        and preparing the output dictionary.

        Args:
            idx: Index of the frame to retrieve

        Returns:
            Dictionary containing the requested data, or None if there was an error
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

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.rgbpathlist)

    def _validate_index(self, idx: int) -> int:
        """
        Validate and adjust the index to ensure it's within bounds.

        Args:
            idx: Original index

        Returns:
            Adjusted index within valid bounds
        """
        if idx >= len(self):
            return len(self) - 1
        return idx

    def _select_frameskip(self) -> None:
        """
        Select the frameskip value based on curriculum settings.
        """
        if not self.manual_frameskip:
            self.frameskip = random.choice(self.frameskip_set_curriculum)

    def _ensure_valid_frame_pair(self, idx: int) -> Optional[int]:
        """
        Ensure that the source and target frames form a valid pair.

        This checks that the frames are from the same video and are sequential.

        Args:
            idx: Target frame index

        Returns:
            Adjusted index, or None if no valid pair can be found
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
            target_depth = (
                self.load_depth(target_depth_path).float().mean(0, keepdim=True)
            )
            source_depth = (
                self.load_depth(source_depth_path).float().mean(0, keepdim=True)
            )
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

    def _apply_augmentations(self, framestack, Ts2t):
        """
        Apply various augmentations to the framestack and update the transformation accordingly.

        Args:
            framestack (torch.Tensor): Stack of frames
            Ts2t (torch.Tensor): Transformation matrix

        Returns:
            tuple: (framestack, Ts2t) after augmentations
        """
        # Apply various augmentations with their respective probabilities
        framestack, Ts2t = aug.color_augmentation(
            framestack, Ts2t, self.color_augmentation_prob, target_only=True
        )
        framestack, Ts2t = aug.reverse_augmentation(
            framestack, Ts2t, self.reverse_augmentation_prob
        )
        framestack, Ts2t = aug.geometric_augmentation(
            framestack, Ts2t, self.geometric_augmentation_prob
        )
        framestack, Ts2t = aug.standstill_augmentation(
            framestack, Ts2t, self.standstill_augmentation_prob
        )

        return framestack, Ts2t

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
            # F = self.pose2fund(geometry.euler2mat(Ts2t) if self.as_euler else Ts2t, K)
            # output["fundamental"] = F

        # Add global poses if requested
        if self.with_global_poses:
            output["Ts"] = self.Tlist[idx - self.frameskip]
            output["Tt"] = self.Tlist[idx]

        return output
