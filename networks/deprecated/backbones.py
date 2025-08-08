import torch
import torch.nn as nn
import transformers
from utilities import *
from networks.base import MONO3DModel
from transformers import logging


class Backbone_Depth(MONO3DModel):
    def __init__(self, size="large"):
        """
        Initializes a DPT_DepthEstimator object.

        Args:
            pretrained_backbone (bool): Whether to use a pretrained backbone. Defaults to False.
            pretrained_neck (bool): Whether to use a pretrained neck. Defaults to False.
            pretrained_head (bool): Whether to use a pretrained head. Defaults to False.
        """
        super(Backbone_Depth, self).__init__()
        self.model = transformers.DPTForDepthEstimation.from_pretrained(
            f"Intel/dpt-{size}"
        )
        self.preprocess = transformers.AutoImageProcessor.from_pretrained(
            f"Intel/dpt-{size}",
            do_rescale=False,
        )

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the DPT_DepthEstimator.

        Args:
            source (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        h, w = source.shape[-2:]
        source = self.preprocess(images=source, return_tensors="pt")["pixel_values"].to(
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        depthmap = self.model(source)  # ["last_hidden_state"].unsqueeze(1)
        # return hiddens.last_hidden_state
        return resizeTransform(h, w)(depthmap.predicted_depth)


class SWIN_backbone(nn.Module):
    def __init__(
        self, config=None, frozen: bool = True, size="base", output_hidden_states=False
    ):
        """
        Initializes a DINO_backbone object.

        Args:
            pretrained (bool): Whether to load the pretrained DINO model or not. Defaults to True.
        """
        super(SWIN_backbone, self).__init__()
        self.frozen = frozen
        self.size = size
        self.output_hidden_states = output_hidden_states

        self.processor = transformers.AutoImageProcessor.from_pretrained(
            f"microsoft/swinv2-{self.size}-patch4-window16-256",
            do_rescale=False,
        )
        self.processor.do_rescale = False
        if self.size in ["tiny", "small", "base", "large"]:
            self.swin = transformers.AutoModel.from_pretrained(
                f"microsoft/swinv2-{self.size}-patch4-window16-256",
                output_hidden_states=self.output_hidden_states,
            )
            if frozen:
                for param in self.swin.parameters():
                    param.requires_grad = False
        else:
            self.swin = transformers.Swinv2Model(config)

    def get_preprocessor(self):
        return self.processor

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the DINO_backbone.

        Args:
            source (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        swin_hidden = self.swin(
            self.processor(images=source, return_tensors="pt")["pixel_values"].to(
                source.device
            )
        )
        if self.output_hidden_states:
            return swin_hidden.hidden_states
        return swin_hidden.last_hidden_state


class SWIN_backbone(nn.Module):
    def __init__(
        self, config=None, frozen: bool = True, size="base", output_hidden_states=False
    ):
        """
        Initializes a DINO_backbone object.

        Args:
            pretrained (bool): Whether to load the pretrained DINO model or not. Defaults to True.
        """
        super(SWIN_backbone, self).__init__()
        self.frozen = frozen
        self.size = size
        self.output_hidden_states = output_hidden_states

        self.processor = transformers.AutoImageProcessor.from_pretrained(
            f"microsoft/swinv2-{self.size}-patch4-window16-256",
            do_rescale=False,
        )
        self.processor.do_rescale = False
        if self.size in ["tiny", "small", "base", "large"]:
            self.swin = transformers.AutoModel.from_pretrained(
                f"microsoft/swinv2-{self.size}-patch4-window16-256",
                output_hidden_states=self.output_hidden_states,
            )
            if frozen:
                for param in self.swin.parameters():
                    param.requires_grad = False
        else:
            self.swin = transformers.Swinv2Model(config)

    def get_preprocessor(self):
        return self.processor

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the DINO_backbone.

        Args:
            source (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        swin_hidden = self.swin(
            self.processor(images=source, return_tensors="pt")["pixel_values"].to(
                source.device
            )
        )
        if self.output_hidden_states:
            return swin_hidden.hidden_states
        return swin_hidden.last_hidden_state
