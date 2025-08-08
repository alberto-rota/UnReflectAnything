# -------------------------------------------------------------------------------------------------#

""" Copyright (c) 2024 Asensus Surgical """

""" Code Developed by: Alberto Rota """
""" Supervision: Uriya Levy, Gal Weizman, Stefano Pomati """


# -------------------------------------------------------------------------------------------------#
from dataclasses import dataclass
from typing import Dict, Tuple, Union, Optional
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms.functional as TF
from torch.nn import functional as F


class PreprocessorConfig:
    """A flexible configuration class that can handle arbitrary attributes.

    Attributes are dynamically set from the initialization dictionary.
    Can be initialized directly or from a dictionary using from_dict().

    Example:
        config_dict = {
            "model_name": "resnet50",
            "batch_size": 32,
            "learning_rate": 0.001,
            "image_size": {"height": 512, "width": 512}
        }
        config = PreprocessorConfig.from_dict(config_dict)
        # Access attributes
        print(config.model_name)  # "resnet50"
        print(config.image_size)  # {"height": 512, "width": 512}
    """

    def __init__(self, configdict):
        """Initialize configuration with arbitrary keyword arguments.

        Args:
            configdict: Arbitrary keyword arguments that will become attributes.
        """
        for key, value in configdict.items():
            setattr(self, key, value)

    def to_dict(self) -> dict:
        """Convert configuration to dictionary.

        Returns:
            dict: Dictionary containing all attributes and their values.

        Example:
            config_dict = config.to_dict()
        """
        return {key: value for key, value in vars(self).items()}

    def update(self, update_dict: dict) -> None:
        """Update configuration with new values.

        Args:
            update_dict (dict): Dictionary of parameters to update.
                Can include new attri/home/ffati/DATA/PDS/pth/test/0
        Example:
            config.update({"new_param": 42, "hidden_size": 1024})
        """
        for key, value in update_dict.items():
            setattr(self, key, value)

    def __repr__(self) -> str:
        """String representation of the configuration.

        Returns:
            str: Formatted string showing all attributes and their values.
        """
        attrs = [f"{key}={repr(value)}" for key, value in vars(self).items()]
        return "PreprocessorConfig(\n    " + ",\n    ".join(attrs) + "\n)"

    def __eq__(self, other: object) -> bool:
        """Check equality with another PreprocessorConfig instance.

        Args:
            other (object): Another PreprocessorConfig instance to compare with.

        Returns:
            bool: True if all attributes and values are equal.
        """
        if not isinstance(other, PreprocessorConfig):
            return NotImplemented
        return vars(self) == vars(other)


class BackbonePreprocessor:
    """
    PyTorch implementation of DPT Image Processor for GPU compatibility.
    Assumes input images are already 384x384.

    This processor handles image preprocessing operations including normalization
    and padding, matching the HuggingFace BackbonePreprocessor behavior.
    All operations are performed on GPU when available.
    """

    def __init__(self, slow_preprocessor_fn):
        """
        Initialize the DPT Image Processor.

        Args:
            config (Optional[ProcessorConfig]): Configuration for the processor.
                If None, default settings will be used.
        """
        self.config = PreprocessorConfig(slow_preprocessor_fn.to_dict())

    @staticmethod
    def _ensure_tensor(image: Union[torch.Tensor, Image.Image]) -> torch.Tensor:
        """
        Ensure input is a torch tensor with shape [C, H, W] or [B, C, H, W].

        Args:
            image: Input image as PIL Image or torch.Tensor

        Returns:
            torch.Tensor: Processed image tensor
        """
        if isinstance(image, Image.Image):
            # Convert PIL Image to tensor [C, H, W]
            image = TF.to_tensor(image)
        elif isinstance(image, torch.Tensor):
            if image.dim() == 3:
                pass  # Already in [C, H, W] format
            elif image.dim() == 4:
                pass  # Already in [B, C, H, W] format
            else:
                raise ValueError(
                    f"Tensor must have 3 or 4 dimensions, got {image.dim()}"
                )
        else:
            raise TypeError(
                f"Input must be PIL Image or torch.Tensor, got {type(image)}"
            )
        return image

    def resize(self, image: torch.Tensor) -> torch.Tensor:
        """
        Resize image to target size.

        Args:
            image (torch.Tensor): Input image tensor [B, C, H, W] or [C, H, W]

        Returns:
            torch.Tensor: Resized image tensor
        """
        if not self.config.do_resize:
            return image

        target_size = (self.config.size["height"], self.config.size["width"])

        # Handle both batched and unbatched inputs
        unbatched = image.dim() == 3
        if unbatched:
            image = image.unsqueeze(0)

        # Resize using bilinear interpolation
        # Input: [B, C, H, W], Output: [B, C, target_h, target_w]
        resized = F.interpolate(
            image, size=target_size, mode="bilinear", align_corners=False
        )

        return resized.squeeze(0) if unbatched else resized

    def normalize(self, image: torch.Tensor) -> torch.Tensor:
        """
        Normalize image using mean and standard deviation.

        Args:
            image (torch.Tensor): Input image tensor [B, C, H, W] or [C, H, W]

        Returns:
            torch.Tensor: Normalized image tensor
        """
        if not self.config.do_normalize:
            return image

        mean = torch.tensor(self.config.image_mean, device=image.device)
        std = torch.tensor(self.config.image_std, device=image.device)

        # Handle both batched and unbatched inputs
        if image.dim() == 3:
            mean = mean.view(-1, 1, 1)
            std = std.view(-1, 1, 1)
        else:  # dim == 4
            mean = mean.view(1, -1, 1, 1)
            std = std.view(1, -1, 1, 1)

        return (image - mean) / std

    def rescale(self, image: torch.Tensor) -> torch.Tensor:
        """
        Rescale image values by rescale_factor.

        Args:
            image (torch.Tensor): Input image tensor

        Returns:
            torch.Tensor: Rescaled image tensor
        """
        if not self.config.do_rescale:
            return image

        return image * self.config.rescale_factor

    def pad(self, image: torch.Tensor) -> torch.Tensor:
        """
        Pad image to ensure dimensions are multiples of ensure_multiple_of.

        Args:
            image (torch.Tensor): Input image tensor

        Returns:
            torch.Tensor: Padded image tensor
        """
        if not self.config.do_pad or self.config.ensure_multiple_of is None:
            return image

        multiple = self.config.ensure_multiple_of

        # Calculate padding
        h, w = image.shape[-2:]
        pad_h = (multiple - h % multiple) % multiple
        pad_w = (multiple - w % multiple) % multiple

        # Pad image
        padding = (0, pad_w, 0, pad_h)  # left, right, top, bottom
        return F.pad(image, padding, mode="reflect")

    def __call__(self, image: Union[torch.Tensor, Image.Image]) -> torch.Tensor:
        """
        Process the input image through all configured transformations.

        Args:
            image: Input image as PIL Image or torch.Tensor

        Returns:
            torch.Tensor: Processed image tensor
        """
        # Convert to tensor if needed
        image = self._ensure_tensor(image)

        # Move to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        image = image.to(device)

        # Apply transformations in order
        image = self.rescale(image)
        image = self.resize(image)
        image = self.normalize(image)
        image = self.pad(image)

        return image
