
# %%
import torch
import torch.nn as nn
import transformers

import matplotlib.pyplot as plt

from utilities import *
import projections as proj
from rich import print
import inspect


########################################################################################################
# TRANSFORMER-BASED MODELS
########################################################################################################


class ViT_DepthEstimator_Baseline(nn.Module):
    """
    A baseline depth estimator module.
    """

    # ---------------------------------------------------------------------------------------------------------------
    # BASE class ViT_FOR WHOLE DEPTH ESTIMATOR. CONTAINS ALL AUXILIARY FUNCTIONS
    # ---------------------------------------------------------------------------------------------------------------

    def __init__(self):
        super(ViT_DepthEstimator_Baseline, self).__init__()

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the depth estimator module.

        Args:
            source (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        raise NotImplementedError(
            "Depth Estimator Module must implement a forward pass"
        )

    def parameters_summary(self, verbose: bool = False) -> dict:
        """
        Prints a summary of the model's parameters.

        Args:
            verbose (bool, optional): Whether to print detailed information about each parameter. Defaults to False.

        Returns:
            dict: A dictionary containing the number of trainable, untrainable, and total parameters.
        """
        for name, parameter in self.named_parameters():
            params = parameter.numel()
            if verbose:
                print(f"{name} : [blue]{params}[/blue]")
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        untrainable = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        total = sum(p.numel() for p in self.parameters())

        print(f"[green]TRAINABLE Parameters: {trainable} ~{millify(trainable)}[/green]")
        print(
            f"[orange3]UNTRAINABLE Parameters: {untrainable} ~{millify(untrainable)}[/orange3]"
        )
        print(f"[cyan]TOTAL Parameters: {total} [~{millify(total)}][/cyan]")
        print(coloredbar([untrainable, trainable], ["orange3", "green"], 50))

        return {"trainable": trainable, "untrainable": untrainable, "total": total}

    def backwardpass_grad_check(self, loss: torch.Tensor) -> bool:
        """
        Checks if the gradients of the model's parameters are finite after a backward pass.

        Args:
            loss (torch.Tensor): The loss tensor.

        Returns:
            bool: True if all gradients are finite, False otherwise.
        """
        loss.backward()
        for name, param in self.named_parameters():
            if param.grad is not None:
                if param.grad.norm() == 0:
                    return False
        return True

    def forwardpass_shape_check(
        self,
        source: torch.Tensor,
        depthmap_pred: torch.Tensor,
        warped: torch.Tensor,
        target: torch.Tensor,
    ) -> bool:
        """
        Checks if the shapes of the input tensors are compatible with the forward pass of the model.

        Args:
            source (torch.Tensor): The source image tensor.
            depthmap_pred (torch.Tensor): The predicted depth map tensor.
            warped (torch.Tensor): The warped image tensor.
            target (torch.Tensor): The target image tensor.

        Returns:
            bool: True if the shapes are compatible, False otherwise.
        """
        source, depthmap_pred, warped, target = (
            source.shape,
            depthmap_pred.shape,
            warped.shape,
            target.shape,
        )

        if source[0] == depthmap_pred[0] == warped[0] == target[0]:  # Batch Size
            if source[-1] == depthmap_pred[-1] == warped[-1] == target[-1]:  # Width
                if (
                    source[-2] == depthmap_pred[-2] == warped[-2] == target[-2]
                ):  # Height
                    if source[-3] == warped[-3] == target[-3] == 3:  # Channels RGB
                        if depthmap_pred[-3] == 1:
                            return True
                        else:
                            print(
                                "Depthmap prediction must be 1-channel (-3 dimension must be 1)"
                            )
                    else:
                        print(
                            "Source, warped and target must be RGB images (-3 dimension must be 3)"
                        )
                else:
                    print(
                        "Source, warped and target must have the same height (-2 dimension must coincide)"
                    )
            else:
                print(
                    "Source, warped and target must have the same width (-1 dimension must coincide)"
                )
        else:
            print(
                "Source, warped and target must have the same batch size (0 dimension must coincide)"
            )
        return False

    def __str__(self) -> str:
        """
        Returns a string representation of the model.

        Returns:
            str: The string representation of the model.
        """
        params = inspect.signature(self.__init__).parameters
        params = {
            k: v.default
            for k, v in params.items()
            if v.default is not inspect.Parameter.empty
        }
        modelstr = f"{self.__class__.__name__}(\n"
        for k, v in params.items():
            if hasattr(self, k):
                modelstr += f"    {k}: {getattr(self,k)} \n".replace("", "")
        modelstr += ")"
        return modelstr


class ViT_PoseEstimator_Baseline(nn.Module):
    """
    A baseline depth estimator module.
    """

    def __init__(self):
        super(ViT_DepthEstimator_Baseline, self).__init__()

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the depth estimator module.

        Args:
            source (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        raise NotImplementedError(
            "Depth Estimator Module must implement a forward pass"
        )

    def parameters_summary(self, verbose: bool = False) -> dict:
        """
        Prints a summary of the model's parameters.

        Args:
            verbose (bool, optional): Whether to print detailed information about each parameter. Defaults to False.

        Returns:
            dict: A dictionary containing the number of trainable, untrainable, and total parameters.
        """
        for name, parameter in self.named_parameters():
            params = parameter.numel()
            if verbose:
                print(f"{name} : [blue]{params}[/blue]")
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        untrainable = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        total = sum(p.numel() for p in self.parameters())

        print(f"[green]TRAINABLE Parameters: {trainable}[/green]")
        print(f"[orange3]UNTRAINABLE Parameters: {untrainable}[/orange3]")
        print(f"[cyan]TOTAL Parameters: {total}[/cyan]")
        print(coloredbar([untrainable, trainable], ["orange3", "green"], 50))

        return {"trainable": trainable, "untrainable": untrainable, "total": total}

    def backwardpass_grad_check(self, loss: torch.Tensor) -> bool:
        """
        Checks if the gradients of the model's parameters are finite after a backward pass.

        Args:
            loss (torch.Tensor): The loss tensor.

        Returns:
            bool: True if all gradients are finite, False otherwise.
        """
        loss.backward()
        for name, param in self.named_parameters():
            if param.grad is not None:
                if param.grad.norm() == 0:
                    return False
        return True

    def forwardpass_shape_check(
        self,
        source: torch.Tensor,
        depthmap_pred: torch.Tensor,
        warped: torch.Tensor,
        target: torch.Tensor,
    ) -> bool:
        """
        Checks if the shapes of the input tensors are compatible with the forward pass of the model.

        Args:
            source (torch.Tensor): The source image tensor.
            depthmap_pred (torch.Tensor): The predicted depth map tensor.
            warped (torch.Tensor): The warped image tensor.
            target (torch.Tensor): The target image tensor.

        Returns:
            bool: True if the shapes are compatible, False otherwise.
        """
        source, depthmap_pred, warped, target = (
            source.shape,
            depthmap_pred.shape,
            warped.shape,
            target.shape,
        )

        if source[0] == depthmap_pred[0] == warped[0] == target[0]:  # Batch Size
            if source[-1] == depthmap_pred[-1] == warped[-1] == target[-1]:  # Width
                if (
                    source[-2] == depthmap_pred[-2] == warped[-2] == target[-2]
                ):  # Height
                    if source[-3] == warped[-3] == target[-3] == 3:  # Channels RGB
                        if depthmap_pred[-3] == 1:
                            return True
                        else:
                            print(
                                "Depthmap prediction must be 1-channel (-3 dimension must be 1)"
                            )
                    else:
                        print(
                            "Source, warped and target must be RGB images (-3 dimension must be 3)"
                        )
                else:
                    print(
                        "Source, warped and target must have the same height (-2 dimension must coincide)"
                    )
            else:
                print(
                    "Source, warped and target must have the same width (-1 dimension must coincide)"
                )
        else:
            print(
                "Source, warped and target must have the same batch size (0 dimension must coincide)"
            )
        return False

    def __str__(self) -> str:
        """
        Returns a string representation of the model.

        Returns:
            str: The string representation of the model.
        """
        params = inspect.signature(self.__init__).parameters
        params = {
            k: v.default
            for k, v in params.items()
            if v.default is not inspect.Parameter.empty
        }
        modelstr = f"{self.__class__.__name__}(\n"
        for k, v in params.items():
            modelstr += f"    {k}: {v} \n".replace("", "")
        modelstr += ")"
        return modelstr


# ---------------------------------------------------------------------------------------------------------------


class ViT_DINO_backbone(nn.Module):
    def __init__(self, size: str = "small", pretrained: bool = True):
        """
        Initializes a DINO_backbone object.

        Args:
            pretrained (bool): Whether to load the pretrained DINO model or not. Defaults to True.
        """
        super(ViT_DINO_backbone, self).__init__()
        self.size = size
        self.pretrained = pretrained
        self.processor = transformers.AutoImageProcessor.from_pretrained(
            f"facebook/dinov2-{self.size}", do_rescale=False
        )
        if pretrained:
            self.dino = transformers.AutoModel.from_pretrained(
                f"facebook/dinov2-{self.size}"
            )
            self.dino.config.output_hidden_states = True
            for param in self.dino.parameters():
                param.requires_grad = False
        else:
            backboneconfig = transformers.Dinov2Config(
                output_hidden_states=True, out_indices=[3, 6, 9, 12]
            )
            self.dino = transformers.Dinov2Model(backboneconfig)

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the DINO_backbone.

        Args:
            source (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        return self.dino(
            self.processor(images=source, return_tensors="pt")["pixel_values"].to(
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        )


class ViT_Depth_RangeBiaser(nn.Module):
    def __init__(self, preact: str = "relu"):
        super(ViT_Depth_RangeBiaser, self).__init__()
        """
        Initializes a ViT_Depth_RangeBiaser object.

        Args:
            preact (str): The activation function to apply prior to the Conv. Defaults to "gelu".
        """
        assert preact.lower() in ["gelu", "relu", "sigmoid", "identity"]
        self.rescaler = nn.Conv2d(
            in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0
        )
        self.activations = nn.ModuleDict(
            [
                ["gelu", nn.GELU()],
                ["relu", nn.ReLU()],
                ["sigmoid", nn.Sigmoid()],
                ["identity", nn.Identity()],
            ]
        )
        self.preact = self.activations[preact.lower()]

    def forward(self, depthmap: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ViT_Depth_RangeBiaser.

        Args:
            depthmap (torch.Tensor): The input depth map tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        depthmap = self.preact(depthmap)
        depthmap = self.rescaler(depthmap)
        return depthmap

    def getparams(self):
        return self.rescaler.weight.item(), self.rescaler.bias.item()

    def wandb_log(self, wandb):
        wandb.log(
            {
                "other/depth_range": self.rescaler.weight.item(),
                "other/depth_bias": self.rescaler.bias.item(),
            }
        )


class ViT_DepthEstimator(ViT_DepthEstimator_Baseline):
    def __init__(
        self, pretrained_backbone: bool = True, pretrained_decoder: bool = False
    ):
        """
        Initializes a DepthEstimator object.

        Args:
            pretrained_backbone (bool): Whether to use a pretrained backbone. Defaults to True.
            pretrained_decoder (bool): Whether to use a pretrained decoder. Defaults to False.
        """
        super(ViT_DepthEstimator, self).__init__()
        self.pretrained_backbone = pretrained_backbone
        self.pretrained_decoder = pretrained_decoder
        self.backbone = ViT_DINO_backbone(pretrained=pretrained_backbone)
        if pretrained_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        dptconfig = transformers.DPTConfig(
            hidden_size=self.backbone.dino.config.hidden_size,
            num_hidden_layers=self.backbone.dino.config.num_hidden_layers,
            backbone_out_indices=self.backbone.dino.config.out_indices,
            num_attention_heads=self.backbone.dino.config.num_attention_heads,
            patch_size=self.backbone.dino.config.patch_size,
        )
        self.dptdecoder = transformers.DPTForDepthEstimation(dptconfig)
        # if pretrained_decoder:
        #     for param in self.dptdecoder.parameters():
        #         param.requires_grad = False
        self.neck = self.dptdecoder.neck
        self.head = self.dptdecoder.head
        self.rangebiaser = ViT_Depth_RangeBiaser()

    def forward(
        self, source: torch.Tensor, return_hidden_state: bool = False
    ) -> torch.Tensor:
        """
        Forward pass of the DepthEstimator.

        Args:
            source (torch.Tensor): The input tensor.
            return_hidden_state (bool): Whether to return the hidden state. Defaults to False.

        Returns:
            torch.Tensor: The output tensor.
        """
        dinoout = self.backbone(source)
        dino_hidden_states = list(dinoout.hidden_states)
        HIDDEN_LEVELS_CONNECTED = [5, 11, 17, 23]
        hidden_states_for_dpt = [
            dino_hidden_states[depth] for depth in HIDDEN_LEVELS_CONNECTED
        ]
        neckout = self.neck(hidden_states_for_dpt)
        headout = self.head(neckout)
        depthmap = self.rangebiaser(headout.unsqueeze(1))
        if return_hidden_state:
            return transform_resize_original()(depthmap), dino_hidden_states
        return transform_resize_original()(depthmap)


class ViT_DPT_DepthEstimator(ViT_DepthEstimator_Baseline):
    def __init__(
        self,
        backbone_size: str = "large",
        pretrained_backbone: bool = False,
        pretrained_neck: bool = False,
        pretrained_head: bool = False,
    ):
        """
        Initializes a DPT_DepthEstimator object.

        Args:
            pretrained_backbone (bool): Whether to use a pretrained backbone. Defaults to False.
            pretrained_neck (bool): Whether to use a pretrained neck. Defaults to False.
            pretrained_head (bool): Whether to use a pretrained head. Defaults to False.
        """
        self.pretrained_backbone = pretrained_backbone
        self.pretrained_neck = pretrained_neck
        self.pretrained_head = pretrained_head
        super(ViT_DPT_DepthEstimator, self).__init__()
        self.model = transformers.DPTForDepthEstimation.from_pretrained(
            f"Intel/dpt-{backbone_size}"
        )
        if pretrained_backbone:
            for param in self.model.dpt.parameters():
                param.requires_grad = False
        if pretrained_neck:
            for param in self.model.neck.parameters():
                param.requires_grad = False
        if pretrained_head:
            for param in self.model.head.parameters():
                param.requires_grad = False
        self.preprocess = transformers.AutoImageProcessor.from_pretrained(
            f"Intel/dpt-{backbone_size}",
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
        depthmap = self.model(source)["predicted_depth"].unsqueeze(1)
        return transform_resize_original(h, w)(depthmap)


class ViT_DINODPT_DepthEstimator_PRETRAINED(ViT_DepthEstimator_Baseline):
    def __init__(self):
        """
        Initializes a DINODPT_DepthEstimator_PRETRAINED object.
        """
        super(ViT_DINODPT_DepthEstimator_PRETRAINED, self).__init__()
        self.backbone = ViT_DINO_backbone(pretrained=True)
        dptwhole = transformers.DPTForDepthEstimation.from_pretrained("Intel/dpt-large")
        self.neck = dptwhole.neck
        self.head = dptwhole.head
        self.rangebiaser = ViT_Depth_RangeBiaser()

    def forward(
        self, source: torch.Tensor, return_hidden_state: bool = False
    ) -> torch.Tensor:
        """
        Forward pass of the DINODPT_DepthEstimator_PRETRAINED.

        Args:
            source (torch.Tensor): The input tensor.
            return_hidden_state (bool): Whether to return the hidden state. Defaults to False.

        Returns:
            torch.Tensor: The output tensor.
        """
        dinoout = self.backbone(source)
        dino_hidden_states = list(dinoout.hidden_states)
        HIDDEN_LEVELS_CONNECTED = [5, 11, 17, 23]
        hidden_states_for_dpt = [
            dino_hidden_states[depth] for depth in HIDDEN_LEVELS_CONNECTED[::-1]
        ]
        neckout = self.neck(hidden_states_for_dpt)
        headout = self.head(neckout)
        depthmap = headout.unsqueeze(1)
        # depthmap = self.rangebiaser(headout.unsqueeze(1))

        if return_hidden_state:
            return transform_resize_original()(depthmap), dino_hidden_states
        return transform_resize_original()(depthmap)


# ---------------------------------------------------------------------------------------------------------------
# TRANSFORMS
# ---------------------------------------------------------------------------------------------------------------
def transform_preprocess(height=224, width=224) -> torchvision.transforms.Compose:
    """
    Define the preprocessing transformation for DINOv2.

    Returns:
    torchvision.transforms.Compose: A composition of torchvision transformations.
    """
    assert height % 16 == 0 and width % 16 == 0
    return torchvision.transforms.Compose(
        [
            lambda x: 255.0 * x,  # Discard alpha component and scale by 255
            torchvision.transforms.Normalize(
                mean=(123.675, 116.28, 103.53),
                std=(58.395, 57.12, 57.375),
            ),
            torchvision.transforms.Resize((height, width), antialias=True),
        ]
    )


def transform_postprocess() -> torchvision.transforms.Compose:
    """
    Define the postprocessing transformation for DINOv2

    Returns:
    torchvision.transforms.Compose: A composition of inverse torchvision transformations.
    """
    return torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((512, 640), antialias=True),
            torchvision.transforms.Normalize(
                mean=(0.0, 0.0, 0.0), std=(1 / 58.395, 1 / 57.12, 1 / 57.375)
            ),
            torchvision.transforms.Normalize(
                mean=(-123.675, -116.28, -103.53), std=(1.0, 1.0, 1.0)
            ),
            lambda x: 1 / 255.0 * x,
        ]
    )


def transform_postprocess_vec() -> torchvision.transforms.Compose:
    """
    Define the postprocessing transformation for DINOv2

    Returns:
    torchvision.transforms.Compose: A composition of inverse torchvision transformations.
    """
    return torchvision.transforms.Compose(
        [
            # torchvision.transforms.Resize((512,640)),
            torchvision.transforms.Normalize(
                mean=(0.0, 0.0, 0.0), std=(1 / 58.395, 1 / 57.12, 1 / 57.375)
            ),
            torchvision.transforms.Normalize(
                mean=(-123.675, -116.28, -103.53), std=(1.0, 1.0, 1.0)
            ),
            lambda x: 1 / 255.0 * x,
        ]
    )


def transform_resize_original(height=512, width=640) -> torchvision.transforms.Compose:
    """
    Define the postprocessing transformation for DINOv2

    Returns:
    torchvision.transforms.Compose: A composition of inverse torchvision transformations.
    """
    return torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((height, width), antialias=True),
        ]
    )


########################################################################################################
# CONVOLUTIONAL-BASED MODELS
########################################################################################################


class ConvBlock(nn.Module):
    # Conv/BN/ELU : Keeps data spatial dimension unchanged, increases number of channels
    def __init__(self, input_shape, output_channels, kernel=3, stride=1):
        super(ConvBlock, self).__init__()

        input_c, input_h, input_w = input_shape
        self.conv = nn.Conv2d(
            in_channels=input_c,
            out_channels=output_channels,
            kernel_size=kernel,
            stride=stride,
            padding="same",
        )
        self.bn = nn.BatchNorm2d(output_channels)
        self.ELU = nn.ELU(inplace=True)

    def forward(self, x):
        y = self.conv(x)
        y = self.bn(y)
        y = self.ELU(y)
        return y


class ConvBlockDown(nn.Module):
    # Conv/BN/ELU : Halves the spatial dimension, increases number of channels
    def __init__(self, input_shape, output_channels, kernel=3, stride=1):
        super(ConvBlockDown, self).__init__()

        input_c, input_h, input_w = input_shape
        self.conv = nn.Conv2d(
            in_channels=input_c,
            out_channels=output_channels,
            kernel_size=kernel,
            stride=stride,
            padding="same",
        )
        self.bn = nn.BatchNorm2d(output_channels)

        # Downsampling with Convolutional with K=1 and Stride = 2
        self.conv_down = nn.Conv2d(
            in_channels=output_channels,
            out_channels=output_channels,
            kernel_size=1,
            stride=2,
        )

        self.ELU = nn.ELU(inplace=True)

    def forward(self, x):
        y = self.conv(x)
        y = self.conv_down(y)
        y = self.bn(y)
        y = self.ELU(y)
        return y


class ConvBlockUp(nn.Module):
    # Conv/BN/ELU : Halves the spatial dimension, increases number of channels
    def __init__(self, input_shape, output_channels, kernel=3, stride=1):
        super(ConvBlockUp, self).__init__()

        input_c, input_h, input_w = input_shape
        self.conv = nn.Conv2d(
            in_channels=input_c,
            out_channels=output_channels,
            kernel_size=kernel,
            stride=stride,
            padding="same",
        )
        self.bn = nn.BatchNorm2d(output_channels)

        # Downsampling with Convolutional with K=1 and Stride = 2
        self.conv_up = nn.ConvTranspose2d(
            in_channels=output_channels,
            out_channels=output_channels,
            kernel_size=2,
            stride=2,
        )

        self.ELU = nn.ELU(inplace=True)

    def forward(self, x):
        y = self.conv(x)
        y = self.conv_up(y)
        y = self.bn(y)
        y = self.ELU(y)
        return y


# TODO: Predicts 1-channel depth fron N channels at given scale
class Depth_SubscalePredictor(nn.Module):
    def __init__(self, in_channels, depth_lower=4, depth_upper=12):
        super(Depth_SubscalePredictor, self).__init__()
        self.ups = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.pred = nn.Conv2d(in_channels, 1, kernel_size=5, stride=1, padding="same")

        self.depth_lower = depth_lower
        self.depth_upper = depth_upper

        self.blur = torchvision.transforms.GaussianBlur(21, sigma=1)
        self.rescale = proj.DepthRescale(self.depth_lower, self.depth_upper)

    def forward(self, x):
        x_ups = self.ups(x)
        depthmap = self.pred(x_ups)
        depthmap = self.blur(depthmap)
        depthmap = proj.depth_rescale(depthmap, self.depth_lower, self.depth_upper)
        depthmap = self.rescale(depthmap)

        return depthmap


class Depth_Predictor(nn.Module):
    def __init__(self, in_channels, depth_lower=4, depth_upper=12):
        super(Depth_Predictor, self).__init__()
        # self.ups = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.pred = nn.Conv2d(in_channels, 1, kernel_size=5, stride=1, padding="same")

        self.depth_lower = depth_lower
        self.depth_upper = depth_upper

        self.blur = torchvision.transforms.GaussianBlur(21, sigma=1)
        self.rescale = proj.DepthRescale(self.depth_lower, self.depth_upper)

    def forward(self, x):
        # x_ups = self.ups(x)
        depthmap = self.pred(x)
        depthmap = self.blur(depthmap)
        depthmap = self.rescale(depthmap)
        # print(self.depth_lower)
        # print(self.depth_upper,self.depth.upper.requires_grad(),self.depth_upper.grad())
        return depthmap


class CONV_DepthNet_Encoder(nn.Module):
    def __init__(
        self,
        scales=4,
        img_shape=(3, 512, 640),
        FACTOR=16,
        depth_lower=4,
        depth_upper=12,
    ):
        super(CONV_DepthNet_Encoder, self).__init__()
        self.height, self.width = img_shape[1:]
        self.scales = range(scales)  # Iterator for the dencoder/decoder scales

        self.shape = img_shape
        self.input_conv = nn.Conv2d(
            3, FACTOR * 2, kernel_size=3, stride=1, padding="same"
        )
        blocks, downsamplers = [], []

        for s in self.scales:
            c = FACTOR * (s + 2)
            w = self.width // (2**s)
            h = self.height // (2**s)
            downsamplers.append(ConvBlockDown(input_shape=(c, h, w), output_channels=c))
            blocks.append(
                ConvBlock(
                    input_shape=(c, h, w),
                    output_channels=FACTOR * (s + 3),
                    kernel=3,
                    stride=1,
                )
            )

        downsamplers.append(ConvBlockDown(input_shape=(c, h, w), output_channels=c))
        self.blocks = nn.ModuleList(blocks[:-1])  # last conv block in not used
        self.downsamplers = nn.ModuleList(downsamplers)

    def summary(self):
        pipe = "\n\t|" * 2 + "\n\tV"
        x = torch.zeros(1, *self.shape).to(
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        print(f"[red] Input: {self.shape} {pipe}[/red]")
        x = self.input_conv(x)
        print(f"[blue]> Conv: {tuple(x.shape)}[/blue]")

        skips = [x]
        print(
            f"[yellow]------------------------------------------> Skip Output: {tuple(x.shape)}[/yellow]"
        )

        for b in range(len(self.blocks)):
            x = self.downsamplers[b](x)
            print(f"[green]> Down: {tuple(x.shape)}[/green]")
            x = self.blocks[b](x)
            print(f"[blue]> Conv: {tuple(x.shape)}[/blue]")

            skips.append(x)
            print(
                f"[yellow]------------------------------------------> Skip Output: {tuple(x.shape)}[/yellow]"
            )

        features = self.downsamplers[-1](x)
        print(f"[green]> Down: {tuple(features.shape)}[/green]")

        print(f"[bright_cyan] {pipe}\nOutput: {tuple(features.shape)} [/bright_cyan]")

    def parameters_summary(self):
        total_params = 0
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            params = parameter.numel()
            print([name, params])
            total_params += params
        print(f"Total Trainable Params: {total_params}")
        return total_params

    def forward(self, inp):
        x = self.input_conv(inp)
        skips = [x]
        for b in range(len(self.blocks)):
            x = self.downsamplers[b](x)
            x = self.blocks[b](x)
            skips.append(x)
        features = self.downsamplers[-1](x)
        return features, skips


class CONV_DepthNet_Decoder(nn.Module):
    def __init__(
        self,
        scales=4,
        features_shape=(80, 32, 40),
        FACTOR=16,
        depth_lower=4,
        depth_upper=12,
    ):
        super(CONV_DepthNet_Decoder, self).__init__()
        self.scales = range(scales)  # Iterator for the dencoder/decoder scales
        self.channels, self.height, self.width = features_shape
        self.FACTOR = FACTOR
        self.features_shape = features_shape

        blocks, upsamplers, predictors = [], [], []
        scales_dec = self.scales[::-1]
        for s in self.scales:
            S = scales_dec[s] + 2
            h = self.height * (2**s)
            w = self.width * (2**s)
            c = self.channels - (FACTOR * s)
            predictors.append(Depth_SubscalePredictor(c, depth_lower, depth_upper))
            upsamplers.append(ConvBlockUp(input_shape=(c, h, w), output_channels=c))
            blocks.append(
                ConvBlock(
                    input_shape=(2 * S * FACTOR + 1, h * 2, w * 2),
                    output_channels=FACTOR * (S - 1),
                    kernel=3,
                    stride=1,
                )
            )

        self.blocks = nn.ModuleList(blocks)
        self.upsamplers = nn.ModuleList(upsamplers)
        self.predictors = nn.ModuleList(predictors)

        self.output = Depth_Predictor(FACTOR, depth_lower, depth_upper)

    def summary(self):
        pipe = "\n\t|" * 2 + "\n\tV"
        x = torch.zeros(1, *self.features_shape).to(
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        skips = [
            torch.zeros(
                1,
                self.FACTOR * (s + 2),
                self.height * (2 ** (self.scales[-s - 1] + 1)),
                self.width * (2 ** (self.scales[-s - 1] + 1)),
            ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            for s in self.scales[::-1]
        ]
        skips = skips[::-1]
        # print([s.shape for s in skips])
        print(f"[red] Input: {tuple(x.shape)} {pipe}[/red]")

        depths_subscales = []
        for b in range(len(self.blocks)):
            depth_subscale = self.predictors[b](x)
            x = self.upsamplers[b](x)
            print(f"[green]> Up: {tuple(x.shape)}[/green]")

            depths_subscales.append(x)
            print(
                f"[medium_purple1]------------------------------------------> Depth Subscale: {tuple(x.shape)}[/medium_purple1]"
            )
            print(
                f"[medium_purple1]<----------------------------------------------------------------[/medium_purple1]"
            )

            print(
                f"[yellow]<------------------------------------------ Skip Input: {tuple(x.shape)}[/yellow]"
            )

            x = torch.cat([x, depth_subscale, skips[-b - 1]], dim=1)
            x = self.blocks[b](x)
            print(f"[blue]> Conv: {tuple(x.shape)}[/blue]")

        depth = self.output(x)
        print(f"[bright_cyan] {pipe}\nOutput: {tuple(depth.shape)} [/bright_cyan]")

    def parameters_summary(self, verbose=False):
        total_params = 0
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            params = parameter.numel()
            if verbose:
                print([name, params])
            total_params += params
        print(f"Total Trainable Params: {total_params}")
        return total_params

    def forward(self, features, skips):
        x = features
        depths_subscales = []
        for b in range(len(self.blocks)):
            depth_subscale = self.predictors[b](x)
            x = self.upsamplers[b](x)
            depths_subscales.append(x)
            x = torch.cat([x, depth_subscale, skips[-b - 1]], dim=1)
            x = self.blocks[b](x)
        depth = self.output(x)
        return depth  # , tuple(depths_subscales)


class CONV_DepthNet(nn.Module):
    def __init__(
        self,
        scales=4,
        img_shape=(3, 512, 640),
        depth_shape=(1, 512, 640),
        factor=16,
        depth_lower=40,
        depth_upper=120,
    ):
        super(CONV_DepthNet, self).__init__()

        self.hyperparameters = {
            "scales": scales,
            "factor": factor,
            "depth_lower": depth_lower,
            "depth_upper": depth_upper,
        }

        self.depth_lower = nn.Parameter(TTensor([depth_lower]), requires_grad=True)
        self.depth_upper = nn.Parameter(TTensor([depth_upper]), requires_grad=True)
        self.encoder = DepthNet_Encoder(
            scales=scales,
            img_shape=img_shape,
            FACTOR=factor,
            depth_lower=self.depth_lower.item(),
            depth_upper=self.depth_upper.item(),
        )

        features_shape = tuple(
            self.encoder(torch.zeros((1, *img_shape)))[0].shape[1:]
        )  # Quick forward pass tp get features shape
        # print(features_shape)
        self.decoder = DepthNet_Decoder(
            scales=scales,
            features_shape=features_shape,
            FACTOR=factor,
            depth_lower=self.depth_lower.item(),
            depth_upper=self.depth_upper.item(),
        )

        self.tboard = SummaryWriter("tboard/run")
        self.tboard.add_graph(self, torch.randn(1, *img_shape))

    def grads_norm(self):
        grads = [
            param.grad.detach().flatten()
            for param in self.parameters()
            if param.grad is not None
        ]
        return torch.cat(grads).norm()

    def weights_norm(self):
        params = [
            param.detach().flatten() for param in self.parameters() if param is not None
        ]
        return torch.cat(params).norm()

    def summary(self):
        print("<----- DEPTHNET ----->")
        self.encoder.summary()
        print("||||||||||")
        print("BOTTLENECK")
        print("||||||||||")
        self.decoder.summary()

    def parameters_summary(self, verbose=False):
        for name, parameter in self.named_parameters():
            params = parameter.numel()
            if verbose:
                print(f"{name} : [blue]{params}[/blue]")
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        untrainable = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        total = sum(p.numel() for p in self.parameters())

        print("> PARAMETERS")
        print(f"[green]TRAINABLE Parameters: {trainable}[/green]")
        print(f"[blue]UNTRAINABLE Parameters: {untrainable}[/blue]")
        print(f"[cyan]TOTAL Parameters: {total}[/cyan]")

        return {"trainable": trainable, "untrainable": untrainable, "total": total}

    def forward(self, x):
        features, skips = self.encoder(x)
        depth = self.decoder(features, skips)
        return depth

    def backwardpass_grad_check(self, loss):
        loss.backward()
        for name, param in self.named_parameters():
            if param.grad is not None:
                if torch.isfinite(param.grad).all():
                    return True
                else:
                    print(f"Gradient for {name} is not finite")
            else:
                print(f"Gradient for {name} is None")
        return False

    def forwardpass_shape_check(self, source, depthmap_pred, warped, target):
        source, depthmap_pred, warped, target = (
            source.shape,
            depthmap_pred.shape,
            warped.shape,
            target.shape,
        )

        if source[0] == depthmap_pred[0] == warped[0] == target[0]:  # Batch Size
            if source[-1] == depthmap_pred[-1] == warped[-1] == target[-1]:  # Width
                if (
                    source[-2] == depthmap_pred[-2] == warped[-2] == target[-2]
                ):  # Height
                    if source[-3] == warped[-3] == target[-3] == 3:  # Channels RGB
                        if depthmap_pred[-3] == 1:
                            return True
                        else:
                            print(
                                "Depthmap prediction must be 1-channel (-3 dimension must be 1)"
                            )
                    else:
                        print(
                            "Source, warped and target must be RGB images (-3 dimension must be 3)"
                        )
                else:
                    print(
                        "Source, warped and target must have the same height (-2 dimension must coincide)"
                    )
            else:
                print(
                    "Source, warped and target must have the same width (-1 dimension must coincide)"
                )
        else:
            print(
                "Source, warped and target must have the same batch size (0 dimension must coincide)"
            )
        return False
