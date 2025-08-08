# -------------------------------------------------------------------------------------------------#

""" Copyright (c) 2024 Asensus Surgical """

""" Code Developed by: Alberto Rota """
""" Supervision: Uriya Levy, Gal Weizman, Stefano Pomati """

# -------------------------------------------------------------------------------------------------#
# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

from utilities import *
from rich import print
import inspect
import copy
import math
import sys

import importlib
import networks.layers_pose

importlib.reload(networks.layers_pose)
from networks.layers_pose import *

import networks.backbones

importlib.reload(networks.backbones)
from networks.backbones import *
import networks.combiners_readers_pose

importlib.reload(networks.combiners_readers_pose)
from networks.combiners_readers_pose import *
import networks.heads_pose

importlib.reload(networks.heads_pose)
from networks.heads_pose import *


class ViT_PoseEstimator(nn.Module):
    def __init__(
        self,
        stack_shape=(4, 3, 512, 640),
        backbone: str = "dino-small",
        reader: str = "ignore",
        combiner: str = "diff",
        regressor: str = "single_step",
        num_heads: int = 8,
        frozen: bool = True,
        batchnorm: bool = True,
        dropout: float = 0.1,
        size: str = None,
        *args,
        **kwargs,
    ):
        """
        Initializes the ViT-based Pose Estimator.

        Args:
            stack_shape (Tuple[int, int, int, int]): The shape of the input data stack.
            backbone (str): The backbone model to use for feature extraction.
            combiner (str): The method used for combining tokens.
            num_heads (int): The number of attention heads.
            reader (str): The method used for reading the input stack.
            regressor (str): The type of regressor to use.
            frozen (bool): Whether the backbone model should be frozen.
            batchnorm (bool): Whether to use batch normalization.
            dropout (float): The dropout rate.
            size (str): The size variant of the backbone model.

        Additional *args and **kwargs are passed to the PoseEstimator base class.
        """
        super(ViT_PoseEstimator, self).__init__()

        # Data initialization
        backbone, size = backbone.split("-")
        self.backbone = backbone
        self.size = size
        self.frozen = frozen
        self.reader = reader
        self.combiner = combiner
        self.regressor = regressor
        self.num_heads = num_heads
        self.stack_shape = stack_shape
        self.batchnorm = batchnorm
        self.dropout = dropout

        # Initilizes a random virtual vector with the same shape as the input stack.
        # Will be used in __init__ to initialize the model with the proper layer shapes and parameters
        self.x = torch.rand(stack_shape).unsqueeze(0)
        self.input_x = self.x

        # Initializes backbone based on the selected model and sizw
        self.backbone_net = getattr(
            networks.backbones, f"{self.backbone.upper()}_backbone"
        )(
            config=None,
            size=self.size,
            frozen=self.frozen,
        )
        # Updating self.x
        self.x = torch.stack([self.backbone_net(frame) for frame in self.x])
        # Calculating the embedding dimension, sequence length and stack size
        self.embed_dim = self.x.shape[-1]
        self.sequence_length = self.x.shape[-2]
        self.stacksize = self.x.shape[1]
        # Initializing the reader layer, which deals with the [CLS] token
        self.readers = {
            "project": ProjectReader(self.sequence_length - 1),
            "ignore": IgnoreReader(),
            "cls": CLSReader(),
        }
        self.read = self.readers[self.reader]
        if (
            self.backbone == "swin"
        ):  # Swin Transformers do not output a [CLS] token,, so no reading is necessary
            self.read = nn.Identity()
        # Updating self.x after reading
        self.x = torch.stack(
            [self.read(frame.permute(0, 2, 1)).permute(0, 2, 1) for frame in self.x]
        )
        # Updating the embedding dimension and sequence length after reading
        self.embed_dim = self.x.shape[-1]
        self.sequence_length = self.x.shape[-2]
        self.stacksize = self.x.shape[1]

        # Combines tokens embedded from the source and target frames
        self.combiners = {
            "cat": CatCombiner(),
            "linear": LinearCombiner(
                sequence_length=self.sequence_length
            ),  # Linear combination
            "mha": MHACombiner(
                sequence_length=self.sequence_length,
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                dropout=0.0,
            ),  # Standard Multi-Head Attention
            "catseqlen_mha": CatSeqLenMHACombiner(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                dropout=0.0,
            ),  # MHA on the concatenation along the SequenceLength dimension
            "catembed_mha": CatEmbedMHACombiner(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                dropout=0.0,
                num_frames=self.x.shape[1],
            ),  # MHA on the concatenation along the Embedding dimension
            "selfseqlen_mha": SelfSeqLenMHACombiner(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                dropout=0.0,
            ),  # SelfAttention after concatenation along the SequenceLength dimension
            "selfembed_mha": SelfEmbedMHACombiner(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                dropout=0.0,
                num_frames=self.x.shape[1],
            ),  # SelfAttention after concatenation along the Embedding dimension
        }
        self.combine = self.combiners[self.combiner]
        # Dropout layer after the concatenation
        self.combiner_dropout = nn.Dropout(p=self.dropout)
        # Uodating self.x after combining
        self.x = self.combine(self.x)
        self.embed_dim = self.x.shape[-1]
        self.sequence_length = self.x.shape[-2]
        # Reassembling: Stacks the tokens back to the original spatial organization
        self.reassemble = Reassemble()

        # Prediction Heads, output the 6-DoF pose
        # Merged Heads will output the 6-DoF pose in a single tensor
        # Split Heads will output the 6-DoF pose in two separate 3-DoF tensors
        heads = {
            "fc_split": TwoHeads(
                [
                    RegressionHead_FC(
                        embed_dim=self.embed_dim,
                        sequence_length=self.sequence_length,
                        output_dim=3,
                        dropout=self.dropout,
                    )
                    for _ in range(2)
                ]
            ),
            "mlp_split": TwoHeads(
                [
                    RegressionHead_MLP(
                        embed_dim=self.embed_dim,
                        sequence_length=self.sequence_length,
                        output_dim=3,
                        dropout=self.dropout,
                    )
                    for _ in range(2)
                ]
            ),
            "fc_merged": RegressionHead_FC(
                embed_dim=self.embed_dim,
                sequence_length=self.sequence_length,
                output_dim=6,
                dropout=self.dropout,
            ),
            "mlp_merged": RegressionHead_MLP(
                embed_dim=self.embed_dim,
                sequence_length=self.sequence_length,
                output_dim=6,
                dropout=self.dropout,
            ),
        }
        self.head = heads[self.regressor]

        self.bn = {
            True: nn.BatchNorm1d(num_features=self.embed_dim),
            False: nn.Identity(),
        }
        self.bn = self.bn[batchnorm]

    def forward(self, framestack: torch.Tensor) -> torch.Tensor:

        # FEATURE EXTRACTION
        framestack_tks = torch.stack([self.backbone_net(frame) for frame in framestack])
        framestack_tksp = torch.stack(
            [
                self.read(frame.permute(0, 2, 1)).permute(0, 2, 1)
                for frame in framestack_tks
            ]
        )
        # --> B x S x L x (E+1)

        # COMBINE: Combines the source and target tokens
        combined = self.combine(framestack_tksp)
        combined = self.combiner_dropout(combined)
        # --> B x S x L x E

        # BATCHNORM: If selected
        combined = self.bn(combined.permute(0, 2, 1)).permute(0, 2, 1)
        # --> B x S x L x E

        odom = self.head(combined)
        # --> B x 6

        return odom

    def preprocess(self, source: torch.Tensor) -> torch.Tensor:
        return self.backbone_net.processor(images=source, return_tensors="pt")[
            "pixel_values"
        ]


class CrossTransformer_PoseEstimator(nn.Module):
    def __init__(
        self,
        image_size=(3, 512, 640),
        backbone: str = "dino-small",
        reader: str = "ignore",
        regressor: str = "mlp",
        frozen: bool = True,
        batchnorm: bool = True,
        layernorm: bool = True,
        dropout: float = 0.1,
        conv_groups: int = 8,
        size: str = None,
        *args,
        **kwargs,
    ):
        super(CrossTransformer_PoseEstimator, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        backbone, size = backbone.split("-")
        # assert backbone.lower() in ["dino", "swin"]
        # assert size.lower() in ["tiny", "small", "base", "large"]
        # assert reader.lower() in ["ignore", "project"]
        # assert regressor.lower() in ["fc_split", "mlp_split"]

        self.x = torch.rand(image_size).unsqueeze(0)
        self.input_x = self.x
        self.backbone = backbone
        self.size = size
        self.frozen = frozen
        self.reader = reader
        self.regressor = regressor
        self.image_size = image_size
        self.batchnorm = batchnorm
        self.layernorm = layernorm
        self.dropout = dropout
        self.conv_groups = conv_groups

        # Bacbone
        self.backbone_net = getattr(
            networks.backbones_pose, f"{self.backbone.upper()}_backbone"
        )(
            config=None,
            size=self.size,
            frozen=self.frozen,
            output_hidden_states=True,
        )
        self.x = self.backbone_net(self.x)

        # Embed dimension
        self.embed_dim = self.x[-1].shape[-1]
        # Sequence length should be the one after the reading operation, hence the -1
        self.sequence_length = self.x[-1].shape[-2] - 1

        self.hidden_states = (
            len(self.x)
            if isinstance(self.x, list) or (isinstance(self.x, tuple))
            else 1
        )

        # Which hidden states are selected depends on which backbone is used
        selected_hidden_states = {
            "dino": np.arange(0, self.hidden_states, 3),
            "swin": np.arange(0, self.hidden_states, 1),
        }

        # Stage Selector
        self.stage_selector = Hidden_State_Selector(
            selected_hidden_states=selected_hidden_states[self.backbone],
            reader=self.reader if self.backbone == "dino" else "noread",
        )

        self.x = self.stage_selector(self.x)
        self.hidden_states = (
            len(self.x)
            if isinstance(self.x, list) or (isinstance(self.x, tuple))
            else 1
        )
        self.multistagecombiner = nn.ModuleList(
            [
                CrossMultiHeadAttention(
                    embed_dim=stage.shape[-1],
                    attention_dim=8,
                    sequence_length=stage.shape[-2],
                    layer_norm=self.layernorm,
                    num_heads=1,
                    dropout=self.dropout,
                )
                for stage in self.x
            ]
        )
        self.x = [
            self.multistagecombiner[i](self.x[i], self.x[i])
            for i in range(self.hidden_states)
        ]
        self.embed_dim = self.x[-1].shape[-1]
        self.sequence_length = self.x[-1].shape[-2]
        self.reassemble = nn.ModuleList(
            [
                Reassemble(),
            ]
            * self.hidden_states
        )

        smallest_spatial_dim = int(math.sqrt(self.x[-1].shape[-2]))
        self.resize = nn.AdaptiveAvgPool2d((smallest_spatial_dim, smallest_spatial_dim))
        if self.conv_groups == "depthwise":
            self.conv_groups = self.x[0].shape[-1]
        self.adjust_channels = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=stage.shape[-1],
                    out_channels=self.embed_dim,
                    kernel_size=2,
                    groups=self.conv_groups,
                    padding="same",
                )
                for stage in self.x
            ]
        )
        self.featurefusion = FeatureFusionBlock(
            stages=len(self.x),
            embed_dim=self.embed_dim,
        )

        batchnormdict = {
            True: nn.BatchNorm2d(num_features=self.embed_dim),
            False: nn.Identity(),
        }
        self.bn = nn.ModuleList(
            [batchnormdict[batchnorm] for _ in range(self.hidden_states)]
        )

        # self.gap = GlobalAveragePooling2d()
        heads = {
            "fc_split": TwoHeads(
                [
                    RegressionHead_FC(
                        embed_dim=self.embed_dim,
                        sequence_length=self.sequence_length,
                        output_dim=3,
                        dropout=self.dropout,
                    )
                    for _ in range(2)
                ]
            ),
            "mlp_split": TwoHeads(
                [
                    RegressionHead_MLP(
                        embed_dim=self.embed_dim,
                        sequence_length=self.sequence_length,
                        output_dim=3,
                        dropout=self.dropout,
                    )
                    for _ in range(2)
                ]
            ),
            "fc_merged": RegressionHead_FC(
                embed_dim=self.embed_dim,
                sequence_length=self.sequence_length,
                output_dim=6,
                dropout=self.dropout,
            ),
            "mlp_merged": RegressionHead_MLP(
                embed_dim=self.embed_dim,
                sequence_length=self.sequence_length,
                output_dim=6,
                dropout=self.dropout,
            ),
        }
        self.head = heads[self.regressor]

    def forward(self, source, target):

        source_tks = self.backbone_net(source)
        target_tks = self.backbone_net(target)
        # --> List of S' elements of shape B x L x E+1

        source_tks_selected_read = self.stage_selector(source_tks)
        target_tks_selected_read = self.stage_selector(target_tks)
        # --> List of S elements of shape B x L x E

        tks_combined = [
            self.multistagecombiner[i](
                source_tks_selected_read[i], target_tks_selected_read[i]
            )
            for i in range(self.hidden_states)
        ]
        # --> List of S elements of shape B x L x E

        tks_reassembled = [
            self.reassemble[i](tks_combined[i]) for i in range(self.hidden_states)
        ]
        # --> List of S elements of shape B x E x W x H
        tks_resized = [
            self.resize(tks_reassembled[i]) for i in range(self.hidden_states)
        ]
        # --> List of S elements of shape B x E x W x H (smallest W and H in the set)
        tks_adjusted = [
            self.adjust_channels[i](tks_resized[i]) for i in range(self.hidden_states)
        ]
        tks_adjusted = [self.bn[i](tks_adjusted[i]) for i in range(self.hidden_states)]
        # --> List of S elements of shape B x E x W x H (smallest W and H in the set, consistent E channels)
        fused_featuremaps = self.featurefusion(tks_adjusted)
        # --> B x E x W x H
        # flat_features = self.gap(fused_featuremaps)
        flat_features = fused_featuremaps
        # --> B x E x 1 x 1
        # The features extracted are then passed to a regressor
        odom = self.head(flat_features)
        return odom

    def preprocess(self, source: torch.Tensor) -> torch.Tensor:
        return self.backbone_net.processor(images=source, return_tensors="pt")[
            "pixel_values"
        ]
