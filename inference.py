"""Utility script for running UnReflectAnything inference on arbitrary image folders.

This module provides a command line entry point that loads
`UnReflect_Model_TokenInpainter` checkpoints and produces diffuse outputs for
every image inside a user-specified directory tree while preserving the input
directory structure.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import torch
import yaml
from dotmap import DotMap
from PIL import Image
from torch import Tensor
from torchvision.transforms import functional as TF
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from main import create_model_from_config, load_and_process_config
from models_old import UnReflect_Model_TokenInpainter
from utilities.run_resume import get_resume_info

# Optional imports for monitoring
try:
    from fvcore.nn import FlopCountAnalysis
    FVCORE_AVAILABLE = True
except ImportError:
    FVCORE_AVAILABLE = False
    FlopCountAnalysis = None

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False


@dataclass
class InferenceOptions:
    """Container for CLI/YAML driven inference options.

    Attributes:
        weights_path: Absolute path to the checkpoint with the serialized model
            state. The checkpoint must contain a ``model_state_dict`` entry.
        input_dir: Root directory that holds the RGB images to be processed. The
            script traverses the directory recursively and keeps the relative
            folder layout when exporting the diffuse predictions.
        output_dir: Destination directory that will mirror the input structure.
        run: Optional run identifier used to recover metadata from an existing
            experiment directory. This is the same identifier accepted in test
            mode by ``run_pipeline``.
        runs_dir: Optional base directory that stores experiment outputs. When
            provided together with ``run``, the script loads the saved config
            artifacts for model reconstruction.
        model_config_path: Optional configuration YAML that follows the training
            template (``config_train.yaml``). When specified, the configuration
            is parsed and used to build the architecture prior to loading the
            checkpoint weights.
        batch_size: Number of images processed per forward pass. The default is
            ``4``.
        device: Preferred device string (e.g. ``"cuda"`` or ``"cpu"``).
        image_extensions: Sequence of file suffixes that are treated as valid
            images. All comparisons are case-insensitive.
        resize_output: If ``True``, resize output images to match the original
            input image dimensions. Defaults to ``True``.
        brightness_threshold: Threshold value for computing highlight masks via
            intensity thresholding. Pixels with brightness (average of R, G, B)
            above this value are considered highlights. Defaults to ``0.7``.
        monitor_usage: If ``True``, track and report FLOPS and energy consumption
            metrics after inference completes. Defaults to ``False``.
    """

    weights_path: Path
    input_dir: Path
    output_dir: Path
    run: Optional[str] = None
    runs_dir: Optional[Path] = None
    model_config_path: Optional[Path] = None
    model_module: Optional[str] = None
    batch_size: int = 4
    device: str = "cuda"
    image_extensions: Sequence[str] = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
    resize_output: bool = True
    brightness_threshold: float = 0.7
    monitor_usage: bool = False


console = Console()


@contextmanager
def suppress_stdout_stderr():
    """Context manager to suppress stdout, stderr, and warnings output."""
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        old_warnings_showwarning = warnings.showwarning
        try:
            sys.stdout = devnull
            sys.stderr = devnull
            # Suppress warnings
            warnings.filterwarnings("ignore")
            warnings.showwarning = lambda *args, **kwargs: None
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            warnings.showwarning = old_warnings_showwarning
            warnings.resetwarnings()


class UsageMonitor:
    """Monitor FLOPS and energy consumption during inference."""

    def __init__(self, device: torch.device, model: torch.nn.Module):
        """Initialize the usage monitor.
        
        Args:
            device: The device being used for inference.
            model: The model to monitor.
        """
        self.device = device
        self.model = model
        self.is_cuda = device.type == "cuda"
        
        # FLOPS tracking
        self.flops_per_forward = None  # FLOPS for the batch used to compute it
        self.flops_per_image = None  # FLOPS per single image
        self.total_flops = 0
        self.forward_count = 0
        
        # Energy tracking
        self.energy_start = None
        self.energy_end = None
        self.energy_initialized = False
        
        # Time tracking
        self.total_forward_time = 0.0
        
        # Initialize GPU monitoring if available
        if self.is_cuda and PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self.energy_initialized = True
            except Exception as e:
                console.log(f"[yellow]Warning: Could not initialize GPU energy monitoring: {e}[/yellow]")
                self.energy_initialized = False
        else:
            self.energy_initialized = False

    def get_gpu_info(self) -> Optional[dict]:
        """Get GPU hardware information."""
        if not self.is_cuda or not PYNVML_AVAILABLE:
            return None
        
        try:
            name = pynvml.nvmlDeviceGetName(self.handle).decode("utf-8")
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            return {
                "name": name,
                "total_memory_gb": memory_info.total / (1024**3),
            }
        except Exception:
            return None

    def start_monitoring(self):
        """Start energy monitoring."""
        if self.energy_initialized:
            try:
                self.energy_start = pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle) / 1000.0  # Convert mJ to J
            except Exception:
                self.energy_start = None

    def stop_monitoring(self):
        """Stop energy monitoring."""
        if self.energy_initialized:
            try:
                self.energy_end = pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle) / 1000.0  # Convert mJ to J
            except Exception:
                self.energy_end = None

    def compute_flops(self, rgb_batch: Tensor, patch_mask_override: Tensor):
        """Compute FLOPS for a single forward pass.
        
        Args:
            rgb_batch: Input RGB batch tensor of shape [B,3,H,W].
            patch_mask_override: Patch mask tensor of shape [B,1,H,W].
        """
        if not FVCORE_AVAILABLE or FlopCountAnalysis is None:
            return
        
        if self.flops_per_forward is None:
            try:
                # Create a dummy input dict matching the model's expected input
                dummy_input = {"rgb": rgb_batch, "patch_mask_override": patch_mask_override}
                # Suppress fvcore warnings about unsupported operators
                with suppress_stdout_stderr():
                    # Use FlopCountAnalysis API
                    flop_counter = FlopCountAnalysis(self.model, dummy_input)
                    # Get total FLOPS for the batch
                    batch_flops = flop_counter.total()
                self.flops_per_forward = batch_flops
                # Compute FLOPS per image by dividing by batch size
                batch_size = rgb_batch.shape[0]
                self.flops_per_image = batch_flops / batch_size
            except Exception as e:
                console.log(f"[yellow]Warning: Could not compute FLOPS: {e}[/yellow]")
                self.flops_per_forward = None
                self.flops_per_image = None

    def record_forward(self, batch_size: int, forward_time: float):
        """Record a forward pass.
        
        Args:
            batch_size: Number of images in the batch.
            forward_time: Time taken for the forward pass in seconds.
        """
        self.forward_count += batch_size
        self.total_forward_time += forward_time
        
        if self.flops_per_image is not None:
            # Add FLOPS for this batch: FLOPS per image * batch size
            self.total_flops += self.flops_per_image * batch_size

    def get_energy_consumption_wh(self) -> Optional[float]:
        """Get total energy consumption in Watt-hours."""
        if self.energy_start is None or self.energy_end is None:
            return None
        # Energy is in Joules, convert to Wh: 1 Wh = 3600 J
        energy_joules = self.energy_end - self.energy_start
        return energy_joules / 3600.0

    def generate_report(self, total_images: int) -> Table:
        """Generate a tabular report of usage metrics.
        
        Args:
            total_images: Total number of images processed.
        
        Returns:
            A rich Table with the usage report.
        """
        table = Table(title="Energy and Compute Usage Report", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="green")
        table.add_column("Unit", style="yellow")

        # Hardware info
        if self.is_cuda:
            gpu_info = self.get_gpu_info()
            if gpu_info:
                table.add_row("Hardware", gpu_info["name"], "CUDA GPU")
                table.add_row("GPU Memory", f"{gpu_info['total_memory_gb']:.2f}", "GB")
            else:
                table.add_row("Hardware", "NVIDIA GPU", "CUDA")
        else:
            table.add_row("Hardware", "CPU", "CPU")

        table.add_row("", "", "")  # Separator

        # FLOPS metrics
        if self.flops_per_image is not None:
            # Per forward pass (for the batch size used during computation)
            if self.flops_per_forward is not None:
                flops_forward_g = self.flops_per_forward / 1e9
                table.add_row("FLOPS (Forward Pass)", f"{flops_forward_g:.2f}", "GFLOPs")
            
            # Per image
            flops_per_image_g = self.flops_per_image / 1e9
            table.add_row("FLOPS (Per Image)", f"{flops_per_image_g:.2f}", "GFLOPs")
            
            # Total dataset
            total_flops_t = self.total_flops / 1e12
            table.add_row("FLOPS (Total Dataset)", f"{total_flops_t:.4f}", "TFLOPs")
        else:
            table.add_row("FLOPS (Forward Pass)", "N/A", "(fvcore not available)")
            table.add_row("FLOPS (Per Image)", "N/A", "")
            table.add_row("FLOPS (Total Dataset)", "N/A", "")

        table.add_row("", "", "")  # Separator

        # Energy metrics
        energy_wh = self.get_energy_consumption_wh()
        if energy_wh is not None:
            # Per forward pass (average)
            if self.forward_count > 0:
                energy_per_image = energy_wh / self.forward_count
                table.add_row("Energy (Per Image)", f"{energy_per_image:.6f}", "Wh")
            
            # Total dataset
            table.add_row("Energy (Total Dataset)", f"{energy_wh:.4f}", "Wh")
            
            # Additional environmental metrics
            # CO2 equivalent (assuming average grid mix: ~0.5 kg CO2/kWh)
            co2_kg = energy_wh * 0.0005  # Convert Wh to kWh then multiply by 0.5
            table.add_row("CO2 Equivalent (Est.)", f"{co2_kg:.6f}", "kg CO2")
        else:
            table.add_row("Energy (Per Image)", "N/A", "(pynvml not available)")
            table.add_row("Energy (Total Dataset)", "N/A", "")
            table.add_row("CO2 Equivalent (Est.)", "N/A", "")

        table.add_row("", "", "")  # Separator

        # Performance metrics
        if self.total_forward_time > 0:
            avg_time_per_image = self.total_forward_time / self.forward_count if self.forward_count > 0 else 0
            table.add_row("Avg Time (Per Image)", f"{avg_time_per_image:.4f}", "seconds")
            table.add_row("Total Inference Time", f"{self.total_forward_time:.2f}", "seconds")
            
            if energy_wh is not None and self.total_forward_time > 0:
                avg_power = (energy_wh * 3600) / self.total_forward_time  # Convert Wh to J, then divide by time
                table.add_row("Average Power", f"{avg_power:.2f}", "Watts")

        table.add_row("", "", "")  # Separator
        table.add_row("Total Images Processed", str(total_images), "images")

        return table


def parse_cli() -> InferenceOptions:
    """Parse command line arguments and YAML file into inference options."""

    parser = argparse.ArgumentParser(description="Run UnReflectAnything diffuse inference")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="./config_inference.yaml",
        required=False,
        help="Absolute path to the inference YAML options file (default: ./config_inference.yaml)",
    )

    args = parser.parse_args()
    config_path = Path(args.config).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    console.log(f"Loading inference configuration from [bold]{config_path}[/bold]")

    with config_path.open("r", encoding="utf-8") as handle:
        raw_options = yaml.safe_load(handle)

    def _as_path(value: Optional[str]) -> Optional[Path]:
        return None if value is None else Path(value).expanduser().resolve()

    weights_path = _as_path(raw_options.get("weights_path"))
    input_dir = _as_path(raw_options.get("input_dir"))
    output_dir = _as_path(raw_options.get("output_dir"))

    if weights_path is None or not weights_path.exists():
        raise FileNotFoundError("weights_path must point to an existing checkpoint file")
    if input_dir is None or not input_dir.exists():
        raise FileNotFoundError("input_dir must point to an existing directory")
    if output_dir is None:
        raise ValueError("output_dir must be provided")

    output_dir.mkdir(parents=True, exist_ok=True)
    console.log(
        "✔️  Configuration loaded",
        # extra={
        #     "config": {
        #         # "weights_path": str(weights_path),
        #         "input_dir": str(input_dir),
        #         "output_dir": str(output_dir),
        #     }
        # },
    )

    batch_size = int(raw_options.get("batch_size", 4))
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    image_extensions = raw_options.get("image_extensions")
    if image_extensions is None:
        extensions: Sequence[str] = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
    else:
        extensions = tuple(ext.lower() for ext in image_extensions)

    resize_output = raw_options.get("resize_output", True)
    if not isinstance(resize_output, bool):
        raise ValueError("resize_output must be a boolean")

    brightness_threshold = float(raw_options.get("brightness_threshold", 0.7))
    if not (0.0 <= brightness_threshold <= 1.0):
        raise ValueError("brightness_threshold must be between 0.0 and 1.0")

    monitor_usage = raw_options.get("monitor_usage", False)
    if not isinstance(monitor_usage, bool):
        raise ValueError("monitor_usage must be a boolean")

    options = InferenceOptions(
        weights_path=weights_path,
        input_dir=input_dir,
        output_dir=output_dir,
        run=raw_options.get("run"),
        runs_dir=_as_path(raw_options.get("runs_dir")),
        model_config_path=_as_path(raw_options.get("model_config_path")),
        model_module=raw_options.get("model_module"),
        batch_size=batch_size,
        device=raw_options.get("device", "cuda"),
        image_extensions=extensions,
        resize_output=resize_output,
        brightness_threshold=brightness_threshold,
        monitor_usage=monitor_usage,
    )
    return options


def _load_config_from_checkpoint(checkpoint: dict) -> Optional[DotMap]:
    """Extract and normalize configuration information from a checkpoint."""

    raw_config = checkpoint.get("config")
    if raw_config is None:
        return None
    if isinstance(raw_config, DotMap):
        cfg = raw_config
    elif isinstance(raw_config, dict):
        cfg = DotMap(raw_config)
    else:
        # Some checkpoints may store JSON strings.
        cfg = DotMap(json.loads(raw_config)) if isinstance(raw_config, str) else None
    if cfg is not None:
        cfg.USE_TORCH_COMPILE = False
    return cfg


def _load_config_from_run(options: InferenceOptions) -> Optional[DotMap]:
    """Try to recover DotMap configuration from an existing run directory."""

    if options.run is None or options.runs_dir is None:
        return None
    resume_info = get_resume_info(options.run, str(options.runs_dir))
    if resume_info is None:
        return None
    run_config = resume_info.get("config")
    if run_config is None:
        return None
    if isinstance(run_config, DotMap):
        result = run_config
    else:
        result = DotMap(run_config)
    result.USE_TORCH_COMPILE = False
    return result


def _load_config_from_yaml(options: InferenceOptions) -> Optional[DotMap]:
    """Parse a training-style YAML configuration if provided."""

    if options.model_config_path is None:
        return None
    return load_and_process_config(config_path=str(options.model_config_path))


def load_model(options: InferenceOptions, device: torch.device) -> UnReflect_Model_TokenInpainter:
    """Build the model architecture and load checkpoint weights."""

    console.log(
        f"Loading checkpoint from [bold]{options.weights_path}[/bold] on device [bold]{device}[/bold]"
    )
    checkpoint = torch.load(options.weights_path, map_location="cpu", weights_only=False)

    # Try to reconstruct configuration, reporting chosen strategy for logging
    config = _load_config_from_checkpoint(checkpoint)
    config_source = None
    if config is not None:
        config_source = f"checkpoint [bold]{options.weights_path}[/bold]"
    else:
        config = _load_config_from_run(options)
        if config is not None:
            config_source = f"run directory [bold]{options.runs_dir}/{options.run}[/bold]"
        else:
            config = _load_config_from_yaml(options)
            if config is not None:
                config_source = f"YAML file [bold]{options.model_config_path}[/bold]"

    if config is None:
        raise RuntimeError(
            "Unable to reconstruct model configuration. Provide model_config_path"
            " or ensure the checkpoint/run stores a serialised config."
        )
    if config_source is not None:
        console.log(f"Model configuration loaded from {config_source}")

    if options.model_module is not None:
        config.MODEL.MODEL_MODULE = options.model_module
    config.USE_TORCH_COMPILE = False
    
    model = create_model_from_config(config, device)
    state_dict = checkpoint.get("model_state_dict")
    if state_dict is None:
        raise KeyError("Checkpoint does not contain model_state_dict")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Warning: missing keys when loading checkpoint: {missing}")
    if unexpected:
        print(f"Warning: unexpected keys when loading checkpoint: {unexpected}")

    model.eval()
    console.log("✔️  Model loaded and ready for inference")
    return model


def list_image_paths(root: Path, extensions: Sequence[str]) -> List[Path]:
    """Collect image files under ``root`` matching the provided extensions."""

    lower_exts = tuple(ext.lower() for ext in extensions)
    files = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in lower_exts
    ]
    if not files:
        raise RuntimeError(f"No images found under {root}")
    sorted_files = sorted(files)
    console.log(f"Discovered [bold]{len(sorted_files)}[/bold] images under {root}")
    return sorted_files


def load_image_batch(
    paths: Sequence[Path], target_size: Tuple[int, int], device: torch.device
) -> Tuple[Tensor, List[Tuple[int, int]]]:
    """Load a batch of images into a tensor of shape ``[B,3,H,W]``.
    
    Returns:
        A tuple containing:
        - The batch tensor of shape ``[B,3,H,W]``
        - A list of original image sizes ``[(H, W), ...]`` for each image
    """

    images = []
    original_sizes = []
    for path in paths:
        with Image.open(path) as img:
            rgb_img = img.convert("RGB")
            original_sizes.append(rgb_img.size[::-1])  # PIL size is (W, H), we need (H, W)
            tensor = TF.to_tensor(rgb_img)
            resized = TF.resize(tensor, target_size, antialias=True)
            images.append(resized)
    batch = torch.stack(images, dim=0)
    return batch.to(device=device, dtype=torch.float32), original_sizes


def compute_highlight_mask(rgb_batch: Tensor, threshold: float = 0.7) -> Tensor:
    """Compute binary highlight masks via intensity thresholding.

    The mask uses the simple brightness (average of R, G, B channels)
    and sets locations with brightness > threshold to one. The tensor shape is ``[B,1,H,W]``.
    
    Args:
        rgb_batch: Input RGB batch tensor of shape ``[B,3,H,W]``.
        threshold: Brightness threshold value. Pixels with brightness above this
            value are considered highlights. Defaults to ``0.7``.
    
    Returns:
        Binary mask tensor of shape ``[B,1,H,W]``.
    """

    # Compute brightness as the mean across the channel dimension (R,G,B)
    brightness = rgb_batch.mean(dim=1, keepdim=True)  # [B,1,H,W]
    mask = (brightness > threshold).to(rgb_batch.dtype)
    return mask


def save_diffuse_batch(
    diffuse_batch: Tensor,
    batch_paths: Sequence[Path],
    input_root: Path,
    output_root: Path,
    original_sizes: Optional[List[Tuple[int, int]]] = None,
    resize_output: bool = True,
) -> None:
    """Write diffuse predictions to disk preserving directory structure.
    
    Args:
        diffuse_batch: Tensor of shape ``[B,3,H,W]`` containing diffuse predictions.
        batch_paths: Sequence of input image paths.
        input_root: Root directory of input images.
        output_root: Root directory for output images.
        original_sizes: Optional list of original image sizes ``[(H, W), ...]``.
            Required if ``resize_output`` is ``True``.
        resize_output: If ``True``, resize output images to original dimensions.
    """

    diffuse_batch = diffuse_batch.clamp_(0.0, 1.0).cpu()
    for idx, (tensor, input_path) in enumerate(zip(diffuse_batch, batch_paths)):
        relative_path = input_path.relative_to(input_root)
        output_path = output_root / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if resize_output and original_sizes is not None:
            original_size = original_sizes[idx]
            # TF.resize expects (H, W) but PIL Image.size is (W, H), so we reverse
            tensor = TF.resize(tensor, original_size, antialias=True)
        
        image = TF.to_pil_image(tensor)
        image.save(output_path)


def run_inference(options: InferenceOptions) -> None:
    """Execute end-to-end inference on the dataset described by ``options``."""

    desired_device = torch.device(options.device if torch.cuda.is_available() else "cpu")
    model = load_model(options, desired_device)
    target_side = model.dinov3.config["image_size"]
    target_size = (target_side, target_side)

    image_paths = list_image_paths(options.input_dir, options.image_extensions)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    # Initialize usage monitor if requested
    monitor = None
    if options.monitor_usage:
        monitor = UsageMonitor(desired_device, model)
        if not FVCORE_AVAILABLE:
            console.log("[yellow]Warning: fvcore not available. FLOPS tracking will be disabled.[/yellow]")
        if not PYNVML_AVAILABLE and desired_device.type == "cuda":
            console.log("[yellow]Warning: pynvml not available. Energy tracking will be disabled.[/yellow]")
        monitor.start_monitoring()

    console.log(
        f"Starting inference over [bold]{len(image_paths)}[/bold] images with batch size {options.batch_size}"
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
        transient=False,
    ) as progress:
        task_id = progress.add_task("Processing", total=len(image_paths))

        for start in range(0, len(image_paths), options.batch_size):
            batch_paths = image_paths[start : start + options.batch_size]
            progress.update(task_id, description=f"Loading {batch_paths[0].name}")
            rgb_batch, original_sizes = load_image_batch(batch_paths, target_size, desired_device)
            patch_mask_override = compute_highlight_mask(rgb_batch, threshold=options.brightness_threshold)
            
            # Compute FLOPS on first batch if monitoring
            if monitor is not None and monitor.flops_per_image is None:
                monitor.compute_flops(rgb_batch, patch_mask_override)
            
            # Time the forward pass
            forward_start = time.time()
            with torch.no_grad():
                outputs = model({"rgb": rgb_batch, "patch_mask_override": patch_mask_override})
            forward_time = time.time() - forward_start
            
            # Record forward pass for monitoring
            if monitor is not None:
                monitor.record_forward(len(batch_paths), forward_time)
            
            diffuse = outputs.get("diffuse")
            if diffuse is None:
                raise KeyError("Model output does not contain 'diffuse'")
            save_diffuse_batch(
                diffuse,
                batch_paths,
                options.input_dir,
                options.output_dir,
                original_sizes=original_sizes if options.resize_output else None,
                resize_output=options.resize_output,
            )
            progress.advance(task_id, advance=len(batch_paths))

    # Stop monitoring and generate report
    if monitor is not None:
        monitor.stop_monitoring()
        console.log("")  # Empty line for spacing
        report_table = monitor.generate_report(len(image_paths))
        console.print(report_table)
        console.log("")  # Empty line for spacing

    console.log(f"✨ Inference complete. Results saved to [bold]{options.output_dir}[/bold]")


def main() -> None:
    """CLI entry point."""

    options = parse_cli()
    run_inference(options)


if __name__ == "__main__":
    main()


