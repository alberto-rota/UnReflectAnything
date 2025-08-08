import torch
import torch.nn as nn
import torch.nn.functional as F

from utilities import *
import networks.depth_decoding as depth_decoding
from .tsdf import TSDistanceFeatureColorVolume
from utilities.dev_utils import embedding2color
from utilities.visualization import log_rerun_camera

import copy
import warnings
import logging
torch.autograd.set_detect_anomaly(True)

from pipelines.features.featureextractor import FeatureExtractor

logger = logging.getLogger(__name__)


class DepthPipeline:
    def __init__(
        self,
        config,
        model=None,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        to_rerun=False,  # Add rerun logging capability
    ):
        ### INITIALIZATIONS
        self.config = config
        self.device = device
        self.to_rerun = to_rerun

        ### MODULES
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            if isinstance(model, DepthEstimator):  # If model passed directly, use it
                self.model = model.to(self.device)
            elif isinstance(model, str) and model != "":  # If model passed as a name str, load checkpoint
                self.model = DepthEstimator(shared_key="asdjnasljkn").to(self.device)
                self.model.fromArtifact(model)
            elif (
                config.get("RUN") is None
                or config.get("RUN") == ""
                or model == ""
                or model is None
            ):  # If no model or model name provided, initialize from scratch.
                self.model = DepthEstimator(
                    backbone_brand=config.get("BACKBONE_BRAND", "intel"),
                    size=config.get("BACKBONE_SIZE", "beit-base-384"),
                    image_height=config.IMAGE_HEIGHT,
                    image_width=config.IMAGE_WIDTH,
                    depth_scale_factor=config.DEPTH_SCALE_FACTOR,
                    depth_bias_factor=config.DEPTH_BIAS_FACTOR,
                    shared_key=None,  # New parameter for sharing feature extractor
                ).to(self.device)

        ### DIMENSIONS
        self.height, self.width, self.batch_size = (
            config.IMAGE_HEIGHT,
            config.IMAGE_WIDTH,
            config.BATCH_SIZE,
        )
        
    def inversedepth(
        self, 
        image, 
    ):
        """
        Estimate inversedepth for a single image.
        
        Args:
            image: Input image tensor [B, C, H, W]
            refine: Whether to refine the depth map (default=True)
            scale_factor: Optional scale factor to apply to the depth map
            
        Returns:
            Depth map tensor [B, 1, H, W]\
        """
        # Convert image to appropriate format if needed
        if isinstance(image, torch.Tensor):
            if len(image.shape) == 3:  # [C, H, W]
                image = image.unsqueeze(0)  # [1, C, H, W]
            
        # Get raw depth estimate from model
        with torch.no_grad():
            depth_map = self.model.depth(image)  # [B, 1, H, W]
            
        return depth_map

    def depth(
        self,
        image,
    ):        
        """
        Estimate depth for a single image.
        
        Args:
            image: Input image tensor [B, C, H, W]
            refine: Whether to refine the depth map (default=True)
            scale_factor: Optional scale factor to apply to the depth map
            
        Returns:
            Depth map tensor [B, 1, H, W]
        """
        return 1 / self.inversedepth(image)
    
    def fuse_depth(
        self,
        dataset,
        vol_bnds=[(-100, 100), (-100, 100), (-100, 100)],  # [(xmin,xmax), (ymin,ymax), (zmin,zmax)]
        voxel_size=0.5,
        margin=40,
        feature_vector_length=1024,
        patch_size=16,
        depth_scale_factor=None,
        depth_bias_factor=None,
        max_frames=None,
        **kwargs
    ):
        """
        Fuse depth and features from dataset into TSDF volume using GPU-optimized operations.
        
        Args:
            dataset: PyTorch dataset with samples containing 'framestack', 'intrinsics', 'Ts'
            vol_bnds: List of 3 tuples defining volume bounds [(xmin,xmax), (ymin,ymax), (zmin,zmax)]
            voxel_size: Size of each voxel in meters
            margin: Truncation margin for TSDF
            feature_vector_length: Dimension of feature vectors to store
            patch_size: Patch size for feature extraction
            depth_scale_factor: Scale factor for depth normalization  
            depth_bias_factor: Bias factor for depth normalization
            max_frames: Maximum number of frames to process (None for all)
            
        Returns:
            tsd: TSDistanceFeatureColorVolume object containing fused data
        """
        # Initialize TSDF volume - shape: [X, Y, Z] voxels
        tsd = TSDistanceFeatureColorVolume(
            vol_bnds=vol_bnds,
            voxel_size=voxel_size,
            margin=margin,
            feature_vector_length=feature_vector_length,
            patch_size=patch_size,
        )
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, sampler=dataset.sampler)

        if depth_scale_factor is None:
            depth_scale_factor = self.config.DEPTH_SCALE_FACTOR
        if depth_bias_factor is None:
            depth_bias_factor = self.config.DEPTH_BIAS_FACTOR
        
        # Initialize PCA for feature visualization if using rerun
        pca = None
        origin = None
        
        if self.to_rerun:
            try:
                import rerun as rr
                rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN)
            except (ImportError, RuntimeError) as e:
                logger.warning(f"Rerun not available: {e}. Disabling rerun logging.")
                self.to_rerun = False

        # Process dataset - vectorized operations where possible
        for s, sample in enumerate(dataloader): 
            if max_frames is not None and s >= max_frames:
                break
                
            # Set origin from first frame for consistent coordinate system
            if s == 0:
                origin = sample["Tt"][0].to(self.device)
            
            # Move all tensors to device in batch - shape: [B, ...]
            sample = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                     for k, v in sample.items()}
            
            # Extract depth estimation - shape: [B, 1, H, W]
            with torch.no_grad():
                sample["depthnorm"] = self.depth(sample["framestack"])
                sample["depthnorm"] = sample["depthnorm"] * depth_scale_factor + depth_bias_factor
                
                # Extract features - shape: [B, E, H', W'] where E=feature_vector_length
                features_full = self.model.embed(sample["framestack"])
                sample["features"] = features_full[0:1] if len(features_full.shape) == 4 else features_full.unsqueeze(0)
            
            # Initialize PCA from first frame for consistent color mapping
            if s == 0 and self.to_rerun:
                em, pca = embedding2color(sample["features"])
            elif self.to_rerun:
                em = embedding2color(sample["features"], pca=pca)
            
            # Get camera parameters - shapes: K:[3,3], T:[4,4]  
            K = sample["intrinsics"][0]
            T = torch.inverse(torch.inverse(origin) @ sample["Ts"][0])
            
            # TSDF integration - vectorized operations on GPU
            tsd.integrate(
                frame=sample["framestack"][0, 0],     # [3, H, W] RGB image
                depthmap=sample["depthnorm"][0].unsqueeze(0),  # [1, H, W] depth map
                K=K,                                   # [3, 3] intrinsics
                T=T,                                   # [4, 4] pose transformation
                featuremap=sample["features"][0],      # [E, H', W'] feature map
                weights=1.0,
            )
            
            # Optional rerun logging for visualization
            if self.to_rerun:
                # Get mesh for visualization - GPU operations
                verts_world, faces, norms, colors = tsd.get_mesh(
                    threshold=0.05,
                    include_features=False,
                    feature_as_colors=True,
                    pca=pca
                )
                
                # Log camera pose
                log_rerun_camera(K, T, entity=f"/camera_{s}")
                
                # Log 3D mesh - shape: verts_world:[N,3], faces:[M,3]
                rr.log(
                    "/3d",
                    rr.Mesh3D(
                        vertex_positions=verts_world,
                        triangle_indices=faces,
                        vertex_normals=norms,
                        vertex_colors=colors,
                    ),
                )
                
                # Log images - shapes: [H,W,3] for display
                rr.log("/rgb", rr.Image(sample["framestack"][0, 0].permute(1, 2, 0) * 255))
                rr.log("/features", rr.Image(em[0].permute(1, 2, 0) * 255))
                rr.log("/depth", rr.DepthImage(sample["depthnorm"][0:1].permute(1, 2, 0)))
            
            # Progress logging
            if s % 10 == 0:
                logger.info(f"Processed frame {s}")
                
        return tsd

class DepthEstimator(FeatureExtractor):
    def __init__(
        self,
        backbone_brand="intel",
        size="beit-base-384",
        image_height=384,
        image_width=384,
        depth_scale_factor=20,
        depth_bias_factor=40,
        shared_key=None,  # New parameter for sharing feature extractor
    ):
        super().__init__(backbone_brand=backbone_brand, size=size, shared_key=shared_key)

        # Create the depth decoder (DPT )
        self.depthpredictor = depth_decoding.DPT_Predictor(
            backbone_brand=backbone_brand,
            size=size,
            out_h=image_height,
            out_w=image_width,
        )
        self.depth_scale_factor = depth_scale_factor
        self.depth_bias_factor = depth_bias_factor
        
    def forward(self, x):
        """
        Args:
            x: Input image tensor [B, C, H, W]
        Returns:
            Depth map tensor [B, 1, H, W]
        """
        # Extract DINO features using FeatureExtractor
        if isinstance(x, torch.Tensor):
            features = self.extract_features(x)  # list of [B, N+1, C]
        elif isinstance(x, list):
            features = x
        else:
            raise ValueError(f"Invalid input type: {type(x)}")
        # Decode features to get depth map
        inverse_depth = self.depthpredictor(features, self.backbone_out_indices)
        # Optionally: Apply sigmoid and invert (1/depth for metric depth)
        depth = (1.0 / inverse_depth)* self.depth_scale_factor + self.depth_bias_factor
        return depth

    def inverse_depth(self, x):
        """
        Compute inverse depth map for a given input
        Args:
            x: Input image tensor [B, C, H, W]
        Returns:
            Inverse depth map tensor [B, 1, H, W]
        """
        return self.forward(x)
    
    def depth_from_features(self, features):
        pass
        
    def depth(self, x):
        """
        Compute depth map for a given input
        Args:
            x: Input image tensor [B, C, H, W]
        Returns:
            Depth map tensor [B, 1, H, W]
        """
        return 1/ self.forward(x)
    
    def fromArtifact(
        self,
        model_name,
        bucket="alberto-bucket",
        device="cuda",
    ):
        """
        Loads a PyTorch model from Google Cloud Storage or local checkpoint

        Args:
            bucket (str): Name of the GCS bucket
            model_name (str): Path to the model file in the bucket
            device (str): Device to load the model on ('cuda' or 'cpu')

        Returns:
            torch.nn.Module: Loaded model, or None if loading failed
        """
        # Try downloading from GCS first
        try:
            local_path = download_from_gcs(model_name=model_name, bucket_name=bucket)
        except Exception as e:
            logger.warning(f"Failed to download from GCS: {str(e)}")
            local_path = None

        # If GCS download fails, check local checkpoint directory
        if local_path is None:
            local_path = os.path.join("checkpoints", "weights_best.pt")
            if not os.path.exists(local_path):
                raise FileNotFoundError(
                    f"Model not found in GCS bucket {bucket} or locally at {local_path}"
                )

        # Load the model state dict
        loaded_dict = torch.load(local_path, map_location=device, weights_only=True)
        self.load_state_dict(loaded_dict)

        # Clean up temporary file if it exists
        if os.path.dirname(local_path) == tempfile.gettempdir():
            os.remove(local_path)
