

import torch
import torch.nn as nn

import matplotlib.pyplot as plt

from utilities import *
import projections as proj
from rich import print
import inspect
import copy
import sys

import importlib
import networks.layers_pose

importlib.reload(networks.layers_pose)
from networks.layers_pose import *

# -------------------------------------------------------------------------------------------------#
# READERS
# -------------------------------------------------------------------------------------------------#


class ProjectReader(nn.Module):
    """
    A neural network module that applies a linear transformation to the input
    sequence and optionally returns the first element of the sequence.

    Args:
        sequence_length (int): The length of the input sequence.

    Attributes:
        projector (nn.Linear): A linear layer that transforms the input sequence.
    """

    def __init__(self, sequence_length: int):
        super(ProjectReader, self).__init__()
        self.projector = nn.Linear(sequence_length + 1, sequence_length)

    def forward(self, source: torch.Tensor, return_cls: bool = False) -> torch.Tensor:
        """
        Forward pass of the ProjectReader.

        Args:
            source (torch.Tensor): The input tensor of shape (..., sequence_length + 1).
            return_cls (bool, optional): Whether to return the first element of the sequence along with the projection. Default is False.

        Returns:
            torch.Tensor: If return_cls is False, returns the projected tensor of shape (..., sequence_length).
                          If return_cls is True, returns a tuple of the first element of the sequence and the projected tensor.
        """
        projectedsource = self.projector(source)  # --> [..., sequence_length]
        if return_cls:
            return source[..., 0], projectedsource  # --> [...], [..., sequence_length]
        return projectedsource


class IgnoreReader(nn.Module):
    """
    A neural network module that optionally returns the first element of the sequence
    and the rest of the sequence, or just the rest of the sequence.

    """

    def __init__(self):
        super(IgnoreReader, self).__init__()

    def forward(self, source: torch.Tensor, return_cls: bool = False) -> torch.Tensor:
        """
        Forward pass of the IgnoreReader.

        Args:
            source (torch.Tensor): The input tensor of shape (..., sequence_length+1).
            return_cls (bool, optional): Whether to return the first element of the sequence along with the rest. Default is False.

        Returns:
            torch.Tensor: If return_cls is False, returns the rest of the sequence of shape (..., sequence_length - 1).
                          If return_cls is True, returns a tuple of the first element of the sequence and the rest of the sequence.
        """
        if return_cls:
            return (
                source[..., 0],
                source[..., 1:],
            )  # --> [embed_dim, 1], [embed_dim, sequence_length]
        return source[..., 1:]  # --> [embed_dim, sequence_length]


class CLSReader(nn.Module):
    """
    A neural network module that returns the first element of the sequence as a tensor.
    """

    def __init__(self):
        super(CLSReader, self).__init__()

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the CLSReader.

        Args:
            source (torch.Tensor): The input tensor of shape (..., sequence_length+1).

        Returns:
            torch.Tensor: The first element of the sequence as a tensor of shape (..., 1).
        """
        return source[..., 0].unsqueeze(-1)  # --> [embed_dim, 1]


# -------------------------------------------------------------------------------------------------#
# COMBINERS
# -------------------------------------------------------------------------------------------------#
class CatCombiner(nn.Module):
    """
    A neural network module that concatenates the first and last frames of a framestack.
    """

    def __init__(self):
        super(CatCombiner, self).__init__()

    def forward(self, framestack: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the CatCombiner.

        Args:
            framestack (torch.Tensor): The input tensor of shape (batch_size, num_frames, sequence_length, feature_dim).

        Returns:
            torch.Tensor: The concatenated tensor of shape (batch_size, sequence_length * 2, feature_dim).
        """
        source = framestack[:, 0, ...]  # --> [batch_size, sequence_length, feature_dim]
        target = framestack[
            :, -1, ...
        ]  # --> [batch_size, sequence_length, feature_dim]
        return torch.cat(
            [source, target], dim=1
        )  # --> [batch_size, sequence_length * 2, feature_dim]


class LinearCombiner(nn.Module):
    """
    A neural network module that combines the first and last frames of a framestack using a linear layer.

    Args:
        sequence_length (int): The length of the input sequence.

    Attributes:
        linear (nn.Linear): A linear layer that transforms the concatenated source and target frames.
    """

    def __init__(self, sequence_length: int):
        super(LinearCombiner, self).__init__()
        self.linear = nn.Linear(sequence_length * 2, sequence_length)

    def forward(self, framestack: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the LinearCombiner.

        Args:
            framestack (torch.Tensor): The input tensor of shape (batch_size, num_frames, sequence_length, feature_dim).

        Returns:
            torch.Tensor: The transformed tensor of shape (batch_size, sequence_length, feature_dim).
        """
        source = framestack[:, 0, ...]  # --> [batch_size, sequence_length, feature_dim]
        target = framestack[
            :, -1, ...
        ]  # --> [batch_size, sequence_length, feature_dim]
        source = source.permute(
            0, 2, 1
        )  # --> [batch_size, feature_dim, sequence_length]
        target = target.permute(
            0, 2, 1
        )  # --> [batch_size, feature_dim, sequence_length]
        return self.linear(torch.cat([source, target], dim=2)).permute(
            0, 2, 1
        )  # --> [batch_size, sequence_length, feature_dim]


class MHACombiner(nn.Module):
    """
    A neural network module that combines the first and last frames of a framestack using multi-head attention.

    Args:
        embed_dim (int, optional): The embedding dimension. Default is 384.
        num_heads (int, optional): The number of attention heads. Default is 4.
        dropout (float, optional): The dropout rate. Default is 0.0.
    """

    def __init__(
        self,
        embed_dim: int = 384,
        sequence_length: int = 1024,
        num_heads: int = 4,
        dropout: float = 0.0,
    ):
        super(MHACombiner, self).__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.conv1_atn = nn.Conv2d(
            in_channels=sequence_length, out_channels=256, kernel_size=2, padding="same"
        )
        self.conv2_atn = nn.Conv2d(
            in_channels=256, out_channels=128, kernel_size=2, padding="same"
        )
        self.conv3_atn = nn.Conv2d(
            in_channels=128, out_channels=32, kernel_size=2, padding="same"
        )
        self.gelu = nn.GELU()

    def forward(self, framestack: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MHACombiner.

        Args:
            framestack (torch.Tensor): The input tensor of shape (batch_size, num_frames, sequence_length, feature_dim).
            atn_weights (bool, optional): Whether to return attention weights. Default is False.

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, sequence_length, feature_dim).
        """
        source = framestack[:, 0, ...]  # --> [batch_size, sequence_length, feature_dim]
        target = framestack[
            :, -1, ...
        ]  # --> [batch_size, sequence_length, feature_dim]
        atn = self.mha(source, target, target)[
            1
        ]  # --> [batch_size, sequence_length, sequence_length]
        atn_convinput = atn.reshape(
            atn.shape[0], -1, int(atn.shape[-1] ** 0.5), int(atn.shape[-1] ** 0.5)
        )  # --> [batch_size, sequence_length, num_patches_v, num_patches_h]
        atn_conv1 = self.gelu(self.conv1_atn(atn_convinput))
        atn_conv2 = self.gelu(self.conv2_atn(atn_conv1))
        atn_conv3 = self.gelu(self.conv3_atn(atn_conv2))
        return atn_conv3.reshape(
            atn.shape[0], 32, -1
        )  # --> [batch_size, sequence_length, feature_dim]


class SelfSeqLenMHACombiner(nn.Module):
    """
    A neural network module that combines frames within a framestack using self-attention.

    Args:
        embed_dim (int, optional): The embedding dimension. Default is 384.
        num_heads (int, optional): The number of attention heads. Default is 4.
        dropout (float, optional): The dropout rate. Default is 0.0.
    """

    def __init__(self, embed_dim: int = 384, num_heads: int = 4, dropout: float = 0.0):
        super(SelfSeqLenMHACombiner, self).__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )

    def forward(self, framestack: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the SelfSeqLenMHACombiner.

        Args:
            framestack (torch.Tensor): The input tensor of shape (batch_size, num_frames, sequence_length, feature_dim).

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, num_frames * sequence_length, feature_dim).
        """
        selfstack = framestack.reshape(
            framestack.shape[0], -1, framestack.shape[-1]
        )  # --> [batch_size, num_frames * sequence_length, feature_dim]
        return self.mha(selfstack, selfstack, selfstack)[
            0
        ]  # --> [batch_size, num_frames * sequence_length, feature_dim]


class SelfEmbedMHACombiner(nn.Module):
    """
    A neural network module that combines frames within a framestack using self-attention with embedding concatenation.

    Args:
        embed_dim (int, optional): The embedding dimension. Default is 384.
        num_heads (int, optional): The number of attention heads. Default is 4.
        dropout (float, optional): The dropout rate. Default is 0.0.
        num_frames (int, optional): The number of frames in the input. Default is 4.
    """

    def __init__(
        self,
        embed_dim: int = 384,
        num_heads: int = 4,
        dropout: float = 0.0,
        num_frames: int = 4,
    ):
        super(SelfEmbedMHACombiner, self).__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim * num_frames,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, framestack: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the SelfEmbedMHACombiner.

        Args:
            framestack (torch.Tensor): The input tensor of shape (batch_size, num_frames, sequence_length, feature_dim).

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, num_frames * sequence_length, feature_dim).
        """
        selfstack = framestack.reshape(
            framestack.shape[0], framestack.shape[-2], -1
        )  # --> [batch_size, sequence_length, num_frames * feature_dim]
        return self.mha(selfstack, selfstack, selfstack)[
            0
        ]  # --> [batch_size, sequence_length, num_frames * feature_dim]


class CatEmbedMHACombiner(nn.Module):
    """
    A neural network module that combines the first frame with the rest of the frames in a framestack using concatenation and multi-head attention.

    Args:
        embed_dim (int, optional): The embedding dimension. Default is 384.
        num_frames (int, optional): The number of frames in the input. Default is 4.
        num_heads (int, optional): The number of attention heads. Default is 4.
        dropout (float, optional): The dropout rate. Default is 0.0.
    """

    def __init__(
        self,
        embed_dim: int = 384,
        num_frames: int = 4,
        num_heads: int = 4,
        dropout: float = 0.0,
    ):
        super(CatEmbedMHACombiner, self).__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            kdim=embed_dim * num_frames,
            vdim=embed_dim * num_frames,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, framestack: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass of the CatEmbedMHACombiner.

        Args:
            framestack (torch.Tensor): The input tensor of shape (batch_size, num_frames, sequence_length, feature_dim).

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, sequence_length, feature_dim).
        """
        target = framestack[:, 0, ...]  # --> [batch_size, sequence_length, feature_dim]
        sourcestack = framestack[
            :, 1:, ...
        ]  # --> [batch_size, num_frames - 1, sequence_length, feature_dim]
        sourcestack = sourcestack.reshape(
            sourcestack.shape[0], sourcestack.shape[-2], -1
        )  # --> [batch_size, sequence_length, (num_frames - 1) * feature_dim]
        framestack = framestack.reshape(
            framestack.shape[0], framestack.shape[-2], -1
        )  # --> [batch_size, sequence_length, num_frames * feature_dim]
        if "need_weights" in kwargs and kwargs["need_weights"]:
            return self.mha(target, framestack, framestack)
        return self.mha(target, framestack, framestack)[
            0
        ]  # --> [batch_size, sequence_length, feature_dim]


class CatSeqLenMHACombiner(nn.Module):
    """
    A neural network module that combines the first frame with the rest of the frames in a framestack using concatenation and multi-head attention,
    preserving the sequence length dimension.

    Args:
        embed_dim (int, optional): The embedding dimension. Default is 384.
        num_heads (int, optional): The number of attention heads. Default is 4.
        dropout (float, optional): The dropout rate. Default is 0.0.
    """

    def __init__(self, embed_dim: int = 384, num_heads: int = 4, dropout: float = 0.0):
        super(CatSeqLenMHACombiner, self).__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )

    def forward(self, framestack: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the CatSeqLenMHACombiner.

        Args:
            framestack (torch.Tensor): The input tensor of shape (batch_size, num_frames, sequence_length, feature_dim).

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, sequence_length, feature_dim).
        """
        target = framestack[:, 0, ...]  # --> [batch_size, sequence_length, feature_dim]
        sourcestack = framestack[
            :, 1:, ...
        ]  # --> [batch_size, num_frames - 1, sequence_length, feature_dim]
        sourcestack = sourcestack.reshape(
            sourcestack.shape[0], -1, sourcestack.shape[-1]
        )  # --> [batch_size, (num_frames - 1) * sequence_length, feature_dim]
        framestack = framestack.reshape(
            framestack.shape[0], -1, framestack.shape[-1]
        )  # --> [batch_size, num_frames * sequence_length, feature_dim]

        return self.mha(target, framestack, framestack)[
            0
        ]  # --> [batch_size, sequence_length, feature_dim]


class FeatureFusionBlock(nn.Module):
    def __init__(
        self,
        stages: int = 4,
        embed_dim: int = 384,
        sequence_length: int = 256,
    ):
        super(FeatureFusionBlock, self).__init__()
        self.stages = stages
        self.resconv = nn.ModuleList(
            [
                nn.Sequential(
                    ResConv2d(
                        in_channels=embed_dim,
                        out_channels=embed_dim,
                        kernel_size=3,
                    ),
                    nn.BatchNorm2d(num_features=embed_dim),
                )
                for i in range(stages)
            ]
        )

    def forward(self, hiddens):
        hiddens_fused = torch.zeros_like(hiddens[0])
        for i in range(len(self.resconv)):
            hiddens_fused = self.resconv[i](hiddens_fused + hiddens[i])

        return hiddens_fused
