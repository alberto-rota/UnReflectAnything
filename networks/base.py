# -------------------------------------------------------------------------------------------------#

""" Copyright (c) 2024 Asensus Surgical """

""" Code Developed by: Alberto Rota """
""" Supervision: Uriya Levy, Gal Weizman, Stefano Pomati """

# -------------------------------------------------------------------------------------------------#
import torch
import torch.nn as nn
from utilities import *
from utilities import *
import projections as proj
from rich import print

from rich.table import Table
from rich.console import Console
from rich import box


class MONO3DModel(nn.Module):
    """
    A baseline depth estimator module.
    """

    def __init__(self):
        super(MONO3DModel, self).__init__()

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

        print(
            f"[green]TRAINABLE Parameters: {trainable} [~{millify(trainable)}][/green]"
        )
        print(
            f"[orange3]UNTRAINABLE Parameters: {untrainable} [~{millify(untrainable)}][/orange3]"
        )
        print(f"[cyan]TOTAL Parameters: {total} [~{millify(total)}][/cyan]")
        print(coloredbar([untrainable, trainable], ["orange3", "green"], 50))

        return {"trainable": trainable, "untrainable": untrainable, "total": total}

    def backwardpass_grad_check(self, loss, verbosity: int = 0) -> bool:
        """
        Checks if the gradients of the model's parameters are finite after a backward pass.

        Args:
            loss (torch.Tensor): The loss tensor.
            verbosity (int): Level of verbosity for output (0, 1, or 2).

        Returns:
            bool: True if all gradients are finite, False otherwise.
        """
        from rich.console import Console
        from rich.table import Table

        console = Console()
        assert verbosity in [0, 1, 2]
        gradsok = True
        allweights, allgrads, allshapes, allnames = [], [], [], []
        loss.backward()
        dummy_opt = torch.optim.SGD(self.parameters(), lr=1e-3)
        dummy_opt.step()

        frozen = []
        reqgrad_ok = []
        reqgrad_gnone = []
        reqgrad_gzero = []
        total_num_layers = 0
        healthy_layers = 0

        # Collect data for the table
        layers_info = []

        for name, param in self.named_parameters():
            total_num_layers += 1
            layer_info = {
                "name": name,
                "requires_grad": param.requires_grad,
                "grad_status": "",
                "weight_norm": None,
                "grad_norm": None,
                "shape": tuple(param.shape),
            }
            if param.requires_grad:
                layer_info["weight_norm"] = param.norm().item()
                if param.grad is None:
                    layer_info["grad_status"] = "Grad None"
                    reqgrad_gnone.append(name)
                    gradsok = False
                elif param.grad.norm().item() == 0:
                    layer_info["grad_status"] = "Zero Grad"
                    reqgrad_gzero.append(name)
                    gradsok = False
                    layer_info["grad_norm"] = param.grad.norm().item()
                else:
                    layer_info["grad_status"] = "OK"
                    reqgrad_ok.append(name)
                    healthy_layers += 1
                    layer_info["grad_norm"] = param.grad.norm().item()
                    allweights.append(param.norm().cpu().detach())
                    allgrads.append(param.grad.norm().cpu().detach())
                    allshapes.append(tuple(param.shape))
                    allnames.append(name)
            else:
                layer_info["grad_status"] = "Frozen"
                frozen.append(name)
                healthy_layers += 1
            layers_info.append(layer_info)

        same_weights = dup_indexes(torch.tensor(allweights))
        duplicate_layers = []
        for sw in same_weights:
            layer_a = {
                "name": allnames[sw[0]],
                "weight_norm": allweights[sw[0]].item(),
                "grad_norm": allgrads[sw[0]].item(),
                "shape": allshapes[sw[0]],
            }
            layer_b = {
                "name": allnames[sw[1]],
                "weight_norm": allweights[sw[1]].item(),
                "grad_norm": allgrads[sw[1]].item(),
                "shape": allshapes[sw[1]],
            }
            duplicate_layers.append((layer_a, layer_b))

        # Build and display tables
        if verbosity >= 1:
            # Detailed layers table
            table = Table(title="Layer Gradient Check", title_justify="left")
            table.add_column("Layer Name", style="cyan", no_wrap=True)
            table.add_column("Requires Grad", style="magenta")
            table.add_column("Grad Status", style="yellow")
            table.add_column("Weight Norm", justify="right")
            table.add_column("Grad Norm", justify="right")
            table.add_column("Shape", style="green")
            num_layers_badgrad = 0
            for layer in layers_info:
                if (
                    verbosity == 2
                    or layer["grad_status"] == "Grad None"
                    or layer["grad_status"] == "Zero Grad"
                ):
                    table.add_row(
                        layer["name"],
                        str(layer["requires_grad"]),
                        layer["grad_status"],
                        (
                            f"{layer['weight_norm']:.4f}"
                            if layer["weight_norm"] is not None
                            else "N/A"
                        ),
                        (
                            f"{layer['grad_norm']:.4f}"
                            if layer["grad_norm"] is not None
                            else "N/A"
                        ),
                        str(layer["shape"]),
                    )
                    num_layers_badgrad += 1
            if num_layers_badgrad > 0:
                console.print(table)

            # Summary table
            summary_table = Table(title="Gradient Check Summary", title_justify="left")
            summary_table.add_column("Status", style="cyan")
            summary_table.add_column("Count", style="magenta")

            summary_table.add_row(
                "[green]Layers requiring grad with valid gradients[/green]",
                str(len(reqgrad_ok)),
            )
            summary_table.add_row(
                "[cyan]Layers not requiring grads (Frozen)[/cyan]", str(len(frozen))
            )
            summary_table.add_row(
                "[orange3]Layers requiring grad with zero gradient[/orange3]",
                str(len(reqgrad_gzero)),
            )
            summary_table.add_row(
                "[red]Layers requiring grad with no gradient[/red]",
                str(len(reqgrad_gnone)),
            )
            summary_table.add_row(
                "[yellow]Pairs of layers with the same weights[/yellow]",
                str(len(same_weights)),
            )
            console.print(summary_table)

            # Duplicate weights table
            if len(duplicate_layers) > 0:
                dup_table = Table(
                    title="Duplicate Weights Layers", title_justify="left"
                )
                dup_table.add_column("Layer A Name", style="cyan")
                dup_table.add_column("Layer B Name", style="cyan")
                dup_table.add_column("Weight Norm A", justify="right")
                dup_table.add_column("Weight Norm B", justify="right")
                dup_table.add_column("Grad Norm A", justify="right")
                dup_table.add_column("Grad Norm B", justify="right")
                dup_table.add_column("Shape A", style="green")
                dup_table.add_column("Shape B", style="green")

                for layer_a, layer_b in duplicate_layers:
                    dup_table.add_row(
                        layer_a["name"],
                        layer_b["name"],
                        f"{layer_a['weight_norm']:.4f}",
                        f"{layer_b['weight_norm']:.4f}",
                        f"{layer_a['grad_norm']:.4f}",
                        f"{layer_b['grad_norm']:.4f}",
                        str(layer_a["shape"]),
                        str(layer_b["shape"]),
                    )
                console.print(dup_table)

        return gradsok, healthy_layers / total_num_layers

    def forwardpass_shape_check(
        self,
        framestack: torch.Tensor,
        Ts2t: torch.Tensor,
        depthmap_pred: torch.Tensor,
        warped: torch.Tensor,
        Ts2t_pred: tuple,
        verbosity=1,
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

        framestack, Ts2t, depthmap_pred, Ts2t_pred, warped = (
            framestack.shape,
            Ts2t.shape,
            depthmap_pred.shape,
            Ts2t_pred.shape,
            warped.shape,
        )
        passed = True
        if (
            not framestack[0]
            == Ts2t[0]
            == Ts2t_pred[0]
            == depthmap_pred[0]
            == warped[0]
        ):  # Batch Size
            passed = False
            if verbosity == 1:
                print("> Incompatible batch size")
        if not (
            framestack[-1] == depthmap_pred[-1] == warped[-1]
            and framestack[-2] == depthmap_pred[-2] == warped[-2]
        ):
            passed = False
            if verbosity == 1:
                print("> Incompatible spatial dimensions")
        if not (Ts2t == Ts2t_pred):
            passed = False
            if verbosity == 1:
                print("> Incompatible odometry shapes")
        return passed

    def device(self):
        return next(self.parameters()).device

    def fpass_memsize(self, dummy_input):
        def register_hook(layer):
            def hook(module, input, output):
                # print(output)
                if isinstance(output, tuple):
                    size_mb = sum(
                        o.element_size() * o.nelement() / (1024**2) for o in output
                    )
                elif isinstance(output, torch.Tensor):
                    size_mb = output.element_size() * output.nelement() / (1024**2)
                else:
                    size_mb = (
                        output[list(output.keys())[0]].element_size()
                        * output[list(output.keys())[0]].nelement()
                        / (1024**2)
                    )
                activation_sizes.append(size_mb)

            if isinstance(layer, nn.Module):
                hooks.append(layer.register_forward_hook(hook))

        activation_sizes = []
        hooks = []

        # Register hooks for each layer
        self.apply(register_hook)
        batch_size = dummy_input[0].shape[0]
        # Perform a forward pass
        with torch.no_grad():
            self(dummy_input[0].unsqueeze(0))

        # Remove hooks
        for hook in hooks:
            hook.remove()

        # Calculate the total size of activations
        total_size_mb = sum(activation_sizes)
        return total_size_mb * batch_size


# ! DEPRECATED
# ---------------------------------------------------------------------
class DINO_backbone(nn.Module):
    def __init__(
        self,
        config=None,
        size: str = "small",
        frozen: bool = True,
        output_hidden_states=False,
    ):
        """
        Initializes a DINO_backbone object.

        Args:
            pretrained (bool): Whether to load the pretrained DINO model or not. Defaults to True.
        """
        super(DINO_backbone, self).__init__()
        self.frozen = frozen
        self.size = size
        self.output_hidden_states = output_hidden_states
        self.processor = transformers.AutoImageProcessor.from_pretrained(
            f"facebook/dinov2-{size}"
        )
        if size == "small" or size == "base":
            self.dino = transformers.AutoModel.from_pretrained(
                f"facebook/dinov2-{size}",
                output_hidden_states=self.output_hidden_states,
            )
            if frozen:
                for param in self.dino.parameters():
                    param.requires_grad = False
        else:
            self.dino = transformers.Dinov2Model(config)
        self.processor.crop_size["height"] = 448
        self.processor.crop_size["width"] = 448
        self.processor.size["shortest_edge"] = 448
        self.processor.do_rescale = False

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the DINO_backbone.

        Args:
            source (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        sourcedevice = source.device
        dino_hidden = self.dino(
            self.processor(images=source, return_tensors="pt")["pixel_values"].to(
                sourcedevice
            )
        )
        if self.output_hidden_states:
            return dino_hidden.hidden_states
        return dino_hidden.last_hidden_state


class DINOINTEL_backbone(nn.Module):
    def __init__(self, size="large", intermediate_states=False):
        """
        Initializes a DPT_DepthEstimator object.

        Args:
            pretrained_backbone (bool): Whether to use a pretrained backbone. Defaults to False.
            pretrained_neck (bool): Whether to use a pretrained neck. Defaults to False.
            pretrained_head (bool): Whether to use a pretrained head. Defaults to False.
        """
        super(DINOINTEL_backbone, self).__init__()
        self.model = transformers.DPTForDepthEstimation.from_pretrained(
            f"Intel/dpt-{size}"
        )
        self.preprocess = transformers.AutoImageProcessor.from_pretrained(
            f"Intel/dpt-{size}", do_rescale=False
        )

        self.intermediate_states = intermediate_states
        if self.intermediate_states:
            self.model.config.output_hidden_states = True

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
        out = self.model.dpt(source)  # ["last_hidden_state"].unsqueeze(1)
        if self.intermediate_states:
            return out["hidden_states"]
        return out["last_hidden_state"]


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
            f"microsoft/swinv2-{self.size}-patch4-window8-256",
            do_rescale=False,
        )
        self.processor.do_rescale = False
        if self.size in ["tiny", "small", "base", "large"]:
            self.swin = transformers.AutoModel.from_pretrained(
                f"microsoft/swinv2-{self.size}-patch4-window8-256",
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
