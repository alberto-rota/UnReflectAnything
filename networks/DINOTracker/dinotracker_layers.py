import torch
import torch.nn as nn
import transformers
import torch.nn.functional as nnf
import numpy as np


class DeltaDINO(nn.Module):
    def __init__(
        self,
        channels=[3, 64, 128, 256, 1024],
        dilations=[1, 1, 1, 2],
        kernel_size=5,
        down_stride=2,
        padding_mode="reflect",
        downsample_layers=[True, True, True, False],
        vit_stride=7,
        vit_features_shape=(1, 1, 1, 1),
        vit_features_dtype=torch.float32,
        vit_features_device=torch.device("cuda"),
    ):
        super(DeltaDINO, self).__init__()

        self.downsample_layers = downsample_layers
        self.vit_stride = vit_stride
        self.down_stride = down_stride
        self.vit_features_shape = vit_features_shape
        self.vit_features_dtype = vit_features_dtype
        self.vit_features_device = vit_features_device
        # create layers
        self.layers_list = []
        for i in range(len(channels) - 1):
            is_last_layer = i == len(channels) - 2
            dilation = dilations[i]
            padding = (kernel_size + ((kernel_size - 1) * (dilation - 1))) // 2
            conv_layer = nn.Conv2d(
                channels[i],
                channels[i + 1],
                kernel_size=kernel_size,
                stride=1,
                dilation=dilation,
                padding=padding,
                padding_mode=padding_mode,
            )
            # zero init
            if is_last_layer:
                conv_layer.weight.data = torch.zeros_like(conv_layer.weight.data).to(
                    conv_layer.weight.data.device
                )
                conv_layer.bias.data = torch.zeros_like(conv_layer.bias.data).to(
                    conv_layer.bias.data.device
                )

            self.layers_list.append(conv_layer)
            # self.layers_list.append(nn.BatchNorm2d(channels[i + 1]))
            if is_last_layer:
                # initialize gamma of batch norm to inital_gamma
                self.layers_list[-1].weight.data.fill_(0.05)
            if not is_last_layer:
                self.layers_list.append(nn.ReLU())
            if self.downsample_layers[i]:
                self.layers_list.append(
                    antialiased_cnns.BlurPool(channels[i + 1], stride=down_stride)
                )

        self.layers = torch.nn.ModuleList(self.layers_list)

    def get_total_stride(self):
        # assumes that model does not contain upsampling layers
        n_down = sum(self.downsample_layers)
        return self.down_stride**n_down

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        cnn_stride = self.get_total_stride()
        x = align_cnn_vit_features(
            vit_features_bchw_shape=self.vit_features_shape,
            vit_features_dtype=self.vit_features_dtype,
            vit_features_device=self.vit_features_device,
            cnn_features_bchw=x,
            cnn_stride=cnn_stride,
            vit_stride=self.vit_stride,
        )

        return x


def align_cnn_vit_features(
    vit_features_bchw_shape: torch.Tensor,
    vit_features_dtype: torch.dtype,
    vit_features_device: torch.device,
    cnn_features_bchw: torch.Tensor,
    vit_patch_size: int = 14,
    vit_stride: int = 7,
    cnn_stride: int = 8,
) -> torch.Tensor:
    """
    Assumptions:
    1. CNN layers are fully padded, thus the feature in the top left corner is centered at the [0, 0] pixel in the image.
    2. ViT patch embed layer has no padding, thus the feature in the top left corner is centered at [vit_patch / 2, vit_patch / 2].
    3. Feature and pixel positions are based on square pixels and refer to the center of the square
       (hence `align_corners=True` in grid_sample)
    :param vit_features_bchw: input ViT features (device and dtype will be set according th them)
    :param cnn_features_bchw: input CNN features to be aligned to ViT features
    :param vit_patch_size:
    :param vit_stride:
    :param cnn_stride:
    :return: CNN features sampled at ViT grid positions
    """
    with torch.no_grad():
        dtype = vit_features_dtype
        device = vit_features_device

        # number of features (ViT/CNN) we got
        v_sz = vit_features_bchw_shape[-2:]
        c_sz = cnn_features_bchw.shape[-2:]

        # pixel position of the bottom right feature
        c_br = [(s_ - 1) * cnn_stride for s_ in c_sz]

        # pixel locations of ViT features
        vit_x = (
            torch.arange(v_sz[1], dtype=dtype, device=device) * vit_stride
            + vit_patch_size / 2.0
        )
        vit_y = (
            torch.arange(v_sz[0], dtype=dtype, device=device) * vit_stride
            + vit_patch_size / 2.0
        )
        # map pixel locations to CNN feature locations in [-1, 1] scaled interval

        vit_grid_x, vit_grid_y = torch.meshgrid(
            -1.0 - (1.0 / c_br[1]) + (2.0 * vit_x / c_br[1]),
            -1 - (1.0 / c_br[0]) + (2.0 * vit_y / c_br[0]),
            indexing="xy",
        )
        grid = torch.stack([vit_grid_x, vit_grid_y], dim=-1)[None, ...].expand(
            vit_features_bchw_shape[0], -1, -1, -1
        )
    grid.requires_grad_(
        False
    )  # do not propagate gradients to the grid, only to the sampled features.
    aligned_cnn_features = nnf.grid_sample(
        cnn_features_bchw,
        grid=grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return aligned_cnn_features


def dup_indexes(arr):
    unique, counts = np.unique(arr, return_counts=True)
    duplicates = unique[counts > 1]
    return [np.where(arr == dup)[0].tolist() for dup in duplicates]


class DeltaDINO_CNN_Intel(nn.Module):
    def __init__(self, size="large"):
        super(DeltaDINO_CNN_Intel, self).__init__()

        for available_sizes in size2embeddim:
            if available_sizes in size:
                size = available_sizes

        output_channels = size2embeddim[size]

        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=0
        )
        self.conv2 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=0
        )
        self.conv3 = nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=0
        )
        self.conv4 = nn.Conv2d(
            in_channels=256,
            out_channels=512,
            kernel_size=3,
            stride=2,
            padding=0,
        )
        self.conv5 = nn.Conv2d(
            in_channels=512,
            out_channels=output_channels,
            kernel_size=3,
            stride=1,
            padding=0,
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))

        x = self.relu(self.conv2(x))

        x = self.relu(self.conv3(x))

        x = self.relu(self.conv4(x))

        x = self.conv5(x)

        return x


class DeltaDINO_CNN_Facebook(nn.Module):
    def __init__(self, size="large"):
        super(DeltaDINO_CNN_Facebook, self).__init__()

        for available_sizes in size2embeddim:
            if available_sizes in size:
                size = available_sizes

        output_channels = size2embeddim[size]

        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=64, kernel_size=7, stride=3, padding=0
        )
        self.conv2 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=0
        )
        self.conv3 = nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=0
        )
        self.conv4 = nn.Conv2d(
            in_channels=256,
            out_channels=512,
            kernel_size=3,
            stride=2,
            padding=0,
        )
        self.conv5 = nn.Conv2d(
            in_channels=512,
            out_channels=output_channels,
            kernel_size=2,
            stride=1,
            padding=0,
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))

        x = self.relu(self.conv2(x))

        x = self.relu(self.conv3(x))

        x = self.relu(self.conv4(x))

        x = self.conv5(x)

        return x

    def forward(self, x):
        y1 = self.relu(self.conv1(x))

        y2 = self.relu(self.conv2(y1))

        y3 = self.relu(self.conv3(y2))

        y4 = self.relu(self.conv4(y3))

        y5 = self.conv5(y4)

        return y5
