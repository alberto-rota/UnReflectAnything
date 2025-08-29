import torch
import torch.nn as nn

from utilities import *
from networks.base import MONO3DModel
import networks.backbones as backbones
import networks.depth_decoding as depth_decoding
import copy

from pipelines.matching.helpers import *

# Global registry for sharing feature extractors
_SHARED_EXTRACTORS = {}

class FeatureExtractor(MONO3DModel):
    def __init__(
        self,
        backbone_brand="intel",
        size="beit-base-384",
        shared_key=None,  # New parameter for sharing
    ):
        super(FeatureExtractor, self).__init__()
        self.backbone_brand = backbone_brand
        self.size = size
        self.shared_key = shared_key

        # Check if we should use a shared extractor
        if shared_key is not None and shared_key in _SHARED_EXTRACTORS:
            # Use shared components
            shared_extractor = _SHARED_EXTRACTORS[shared_key]
            self.backbone = shared_extractor.backbone
            self.backbone_out_indices = shared_extractor.backbone_out_indices  
            self.lastvitlayer = shared_extractor.lastvitlayer
            self._is_shared = True
        else:
            # Create new components
            self.backbone = getattr(backbones, f"DINOv2_{backbone_brand.capitalize()}")(size=size)        
            self.backbone_out_indices = self.backbone.model.config.to_dict()[
                (
                    "out_indices"
                    if "swin" in size or "beit" in size or "facebook" in backbone_brand
                    else "backbone_out_indices"
                )
            ]
            self.lastvitlayer = copy.deepcopy(self.backbone.model.encoder.layer[-1])
            for p in self.lastvitlayer.parameters():
                p.requires_grad = True
            self._is_shared = False
            
            # Register this extractor if shared_key is provided
            if shared_key is not None:
                _SHARED_EXTRACTORS[shared_key] = self
        
    def extract_features(self, framestack):
        """
        Args:
            framestack: torch.Tensor, shape [B, 3, H, W] or [B, S, 3, H, W] where S is sequence length
        Returns:
            features: list of torch.Tensor, DINO features at each layer
            - For [B, 3, H, W] input: list of [B, N+1, C] tensors
            - For [B, S, 3, H, W] input: list of [B, S, N+1, C] tensors
        """
        # Handle sequence input by reshaping to batch
        if len(framestack.shape) == 5:  # [B, S, 3, H, W]
            B, S = framestack.shape[:2]
            framestack = framestack.reshape(-1, *framestack.shape[2:])  # [B*S, 3, H, W]
            
        # Get features from backbone
        features = self.backbone(framestack)[
            "feature_maps" if "swin" in self.size or "beit" in self.size else "hidden_states"
        ]
        
        # Reshape back to sequence if needed
        if len(framestack.shape) == 5:
            features = [f.reshape(B, S, *f.shape[1:]) for f in features]  # [B, S, N+1, C]
            
        return features  # list of [B, N+1, C] or [B, S, N+1, C]

    def _compute_depth(self, source_features_dino):
        """Compute depth maps from source features."""
        depthstack = self.depthpredictor(
            source_features_dino,
            self.backbone_out_indices,
        )
        return depthstack.view(-1, 1, depthstack.shape[-2], depthstack.shape[-1])

    def _extract_embeddings(self, features, loftr_shape=True):
        """Extract embeddings for a single image."""
        if loftr_shape:
            return self.patchsize_resampler(embedding2chw(features[-1][:, 1:]))
        else:
            return embedding2chw(features[-1][:, 1:])
        
    def _extract_embeddings_multires(self, features, loftr_shape=True):
        """Extract embeddings for a single image."""
        if loftr_shape:
            return self.patchsize_resampler(embedding2chw(features[:, 1:]))
        else:
            return features

    def depth(self, framestack):
        """Inference-only method to return depth maps for one or more images or frame stacks."""

        source_features_dino = self.extract_features(
            framestack[:, 0] if len(framestack.shape) == 5 else framestack
        )
        return self._compute_depth(source_features_dino)

    def embed(self, *frames, mode="chw", multires=False, return_cls=False):
        """Inference-only method to return embeddings for one or more frames.
        
        Args:
            *frames: One or more input frames
            mode (str): Output format - 'chw' for feature maps or 'seq' for sequence
            multires (bool): Whether to return multi-resolution features
            return_cls (bool): If True and mode='chw', also return CLS tokens
            
        Returns:
            If mode='chw':
                - If return_cls=False: List of feature maps [B,E,H,W] 
                - If return_cls=True: Tuple of (feature maps [B,E,H,W], CLS tokens [B,E])
            If mode='seq':
                - List of sequence features [B,N,E]
        """
        assert mode in ["chw", "seq"], "Invalid mode. Must be 'chw' or 'seq'."
        
        if multires:
            features = [self.extract_features(image) for image in frames]
            embs = [self._extract_embeddings_multires(feat, False) for feat in features]
            if len(embs) == 1:
                embs = embs[0]
                
            if mode == "chw":
                embs_spatial = [embedding2chw(emb[:,1:]) for emb in embs]
                if return_cls:
                    cls_tokens = [emb[:,0].unsqueeze(1) for emb in embs] # [B,E]
                    if len(cls_tokens) == 1:
                        cls_tokens = cls_tokens[0]
                    return embs_spatial, cls_tokens
                return embs_spatial
            return embs
            
        else:
            features = [self.extract_features(image) for image in frames]
            embs = [self._extract_embeddings(feat, False) for feat in features]
            if len(embs) == 1:
                embs = embs[0]
                
            if mode == "chw" and return_cls:
                # For non-multires, embs are already in CHW format
                # Need to extract CLS tokens from original features
                cls_tokens = [feat[-1][:,0] for feat in features] # [B,E] 
                if len(cls_tokens) == 1:
                    cls_tokens = cls_tokens[0]
                return embs, cls_tokens
                
            return embs

    def fromArtifact(
        self,
        model_name,
        bucket=None,
        device="cuda",
    ):
        """
        Loads a PyTorch model from Google Cloud Storage or local checkpoint

        Args:
            bucket (str, optional): Name of the GCS bucket. If None, uses GCS_BUCKET_NAME environment variable
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