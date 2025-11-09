import torch
import torch.nn as nn
import os
from typing import Optional
import torch.nn.functional as F


def get_model_parameter_summary(model):
    """
    Generate a comprehensive parameter summary for RGBPOLDecomposer or UnReflect_Model models.

    Args:
        model: RGBPOLDecomposer or UnReflect_Model instance

    Returns:
        dict: Detailed parameter summary with counts and breakdowns
    """
    allowed_types = ("RGBPOLDecomposer", "UnReflect_Model")
    if model.__class__.__name__ not in allowed_types:
        raise ValueError("Model must be RGBPOLDecomposer or UnReflect_Model")

    def count_parameters(module, trainable_only=False):
        """Count parameters in a module."""
        if trainable_only:
            return sum(p.numel() for p in module.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in module.parameters())

    def count_parameters_by_name(module, name_patterns):
        """Count parameters for modules matching name patterns."""
        total_params = 0
        trainable_params = 0

        for name, child in module.named_modules():
            if any(pattern in name for pattern in name_patterns):
                total_params += sum(p.numel() for p in child.parameters())
                trainable_params += sum(
                    p.numel() for p in child.parameters() if p.requires_grad
                )

        return total_params, trainable_params

    # Initialize summary
    summary = {
        "model_type": model.__class__.__name__,
        "total_parameters": 0,
        "trainable_parameters": 0,
        "frozen_parameters": 0,
        "components": {},
    }

    # RGB Encoder (DINOv3)
    dinov3_total, dinov3_trainable = (
        count_parameters(model.dinov3, trainable_only=False),
        count_parameters(model.dinov3, trainable_only=True),
    )
    summary["components"]["rgb_encoder"] = {
        "total": dinov3_total,
        "trainable": dinov3_trainable,
        "frozen": dinov3_total - dinov3_trainable,
        "description": "DINOv3 backbone for RGB feature extraction",
    }

    # POL components (only for RGBPOLDecomposer)
    if model.__class__.__name__ == "RGBPOLDecomposer":
        # POL Preprocessing
        pol_pre_total, pol_pre_trainable = (
            count_parameters(model.pol_pre, trainable_only=False),
            count_parameters(model.pol_pre, trainable_only=True),
        )
        summary["components"]["pol_preprocessing"] = {
            "total": pol_pre_total,
            "trainable": pol_pre_trainable,
            "frozen": pol_pre_total - pol_pre_trainable,
            "description": "Polarization preprocessing (AoLP, DoLP → [cos2θ, sin2θ, DoLP])",
        }

        # POL Encoder
        pol_enc_total, pol_enc_trainable = (
            count_parameters(model.pol_enc, trainable_only=False),
            count_parameters(model.pol_enc, trainable_only=True),
        )
        summary["components"]["pol_encoder"] = {
            "total": pol_enc_total,
            "trainable": pol_enc_trainable,
            "frozen": pol_enc_total - pol_enc_trainable,
            "description": "POLViT encoder for polarization feature extraction",
        }

        # Cross-attention modules
        cross_total, cross_trainable = (
            count_parameters(model.cross, trainable_only=False),
            count_parameters(model.cross, trainable_only=True),
        )
        summary["components"]["cross_attention"] = {
            "total": cross_total,
            "trainable": cross_trainable,
            "frozen": cross_total - cross_trainable,
            "description": "RGB-POL cross-attention fusion modules",
        }

    # Decoders
    decoders = {
        "specular_decoder": model.decS,
        "diffuse_decoder": model.decD,
        "highlight_decoder": model.decH,
    }

    for name, decoder in decoders.items():
        dec_total, dec_trainable = (
            count_parameters(decoder, trainable_only=False),
            count_parameters(decoder, trainable_only=True),
        )
        summary["components"][name] = {
            "total": dec_total,
            "trainable": dec_trainable,
            "frozen": dec_total - dec_trainable,
            "description": f"DPT decoder for {name.replace('_decoder', '')} component",
        }

    # Calculate totals
    summary["total_parameters"] = sum(
        comp["total"] for comp in summary["components"].values()
    )
    summary["trainable_parameters"] = sum(
        comp["trainable"] for comp in summary["components"].values()
    )
    summary["frozen_parameters"] = (
        summary["total_parameters"] - summary["trainable_parameters"]
    )

    return summary


def print_model_parameter_summary(model, detailed=True):
    """
    Print a formatted parameter summary for RGBPOLDecomposer or UnReflect_Model models.

    Args:
        model: RGBPOLDecomposer or UnReflect_Model instance
        detailed: Whether to print detailed breakdown by component
    """
    summary = get_model_parameter_summary(model)

    print(f"\n{'=' * 60}")
    print(f"MODEL PARAMETER SUMMARY: {summary['model_type']}")
    print(f"{'=' * 60}")

    # Overall statistics
    print("\n📊 OVERALL STATISTICS:")
    print(f"   Total Parameters:     {summary['total_parameters']:,}")
    print(f"   Trainable Parameters: {summary['trainable_parameters']:,}")
    print(f"   Frozen Parameters:    {summary['frozen_parameters']:,}")
    print(
        f"   Trainable Ratio:      {summary['trainable_parameters'] / summary['total_parameters'] * 100:.1f}%"
    )

    if detailed:
        print("\n🔍 DETAILED BREAKDOWN:")
        print(
            f"{'Component':<25} {'Total':<12} {'Trainable':<12} {'Frozen':<12} {'Ratio':<8}"
        )
        print(f"{'-' * 25} {'-' * 12} {'-' * 12} {'-' * 12} {'-' * 8}")

        for comp_name, comp_data in summary["components"].items():
            ratio = (
                comp_data["trainable"] / comp_data["total"] * 100
                if comp_data["total"] > 0
                else 0
            )
            print(
                f"{comp_name:<25} {comp_data['total']:<12,} {comp_data['trainable']:<12,} {comp_data['frozen']:<12,} {ratio:<7.1f}%"
            )

        print("\n📝 COMPONENT DESCRIPTIONS:")
        for comp_name, comp_data in summary["components"].items():
            print(f"   • {comp_name}: {comp_data['description']}")

    print(f"\n{'=' * 60}")


def get_model_size_mb(model):
    """
    Calculate model size in MB (approximate).

    Args:
        model: RGBPOLDecomposer or UnReflect_Model instance

    Returns:
        float: Model size in MB
    """
    summary = get_model_parameter_summary(model)
    # Assuming float32 (4 bytes per parameter)
    size_mb = summary["total_parameters"] * 4 / (1024 * 1024)
    return size_mb


def load_best_model_by_run(
    run_identifier: str,
    device: Optional[torch.device] = None,
    runs_dir: Optional[str] = None,
) -> nn.Module:
    """
    Load and return a model instantiated from a run's best checkpoint.

    Args:
        run_identifier: Run name or ID used to locate the run directory.
        device: Optional torch.device. Defaults to CUDA if available else CPU.
        runs_dir: Optional absolute path to the root runs directory. If None, uses the
                  same resolution logic as the training engine (RESUlTS_DIR env or default).

    Returns:
        nn.Module: The model put on the requested device, loaded with best weights, set to eval().

    Raises:
        FileNotFoundError: If the run or best checkpoint cannot be found.
        RuntimeError: If model instantiation or state loading fails.
    """
    # Resolve device
    load_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Resolve runs directory using same logic as engine initializers if not provided
    if runs_dir is None:
        try:
            from utilities import (
                engine_initializers as initialize,
            )  # local import to avoid cycles

            runs_dir = initialize.device_and_directories({})["runs_dir"]
        except Exception:
            # Fallback: environment variable or repo default
            runs_dir = os.path.expandvars(
                os.getenv(
                    "RESULTS_DIR",
                    os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs"),
                )
            )

    # Find the run and its models directory
    try:
        from utilities.run_resume import get_resume_info  # local import to avoid cycles

        resume_info = get_resume_info(run_identifier, runs_dir)
    except Exception as e:
        raise FileNotFoundError(
            f"Failed to resolve run '{run_identifier}' in runs_dir '{runs_dir}': {e}"
        )

    if resume_info is None:
        raise FileNotFoundError(f"Run not found or invalid: {run_identifier}")

    models_dir = resume_info.get("models_dir")
    run_dir = resume_info.get("run_dir")
    if models_dir is None or not os.path.isdir(models_dir):
        raise FileNotFoundError(f"Models directory not found for run: {run_identifier}")

    # Prefer weights_best.pt (EarlyStopping), fallback to best_model.pth (engine best)
    candidate_paths = [
        os.path.join(models_dir, "weights_best.pt"),
        os.path.join(models_dir, "best_model.pth"),
    ]
    best_ckpt = next((p for p in candidate_paths if os.path.exists(p)), None)
    if best_ckpt is None:
        raise FileNotFoundError(
            f"Best checkpoint not found. Looked for: {', '.join(candidate_paths)}"
        )

    # Load checkpoint and reconstruct model from saved config
    checkpoint = torch.load(best_ckpt, map_location=load_device, weights_only=False)

    # Prefer config packaged inside checkpoint; else use run's config from resume_info
    saved_config = checkpoint.get("config", resume_info.get("config"))
    if saved_config is None:
        # As a last resort, try to read config.json from the run directory
        import json

        config_path = os.path.join(run_dir, "config.json") if run_dir else None
        if config_path and os.path.exists(config_path):
            with open(config_path, "r") as f:
                saved_config = json.load(f)
        else:
            raise RuntimeError("No configuration found to rebuild model architecture")

    # Ensure DotMap for the model factory
    try:
        from dotmap import DotMap  # local import

        if not isinstance(saved_config, DotMap):
            saved_config = DotMap(saved_config)
    except Exception:
        # If DotMap import fails, attempt to use raw dict (factory expects DotMap but guard anyway)
        pass

    # Build model using the main factory to ensure identical architecture
    try:
        from main import (
            create_model_from_config,
        )  # local import to avoid top-level cycle

        model = create_model_from_config(saved_config, load_device)
    except Exception as e:
        raise RuntimeError(f"Failed to instantiate model from saved config: {e}")

    # Load weights (handle both full checkpoint and raw state_dict formats)
    state_dict = checkpoint.get("model_state_dict", None)
    if state_dict is None:
        # If the file is a plain state_dict
        state_dict = checkpoint

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if len(unexpected) > 0:
        # Not fatal; but surface to caller as warning via exception message for clarity
        # Users can inspect and decide to ignore
        pass

    model = model.to(load_device).eval()
    return model


def pixel_mask_to_patch_mask(
    mask_hw: torch.Tensor, patch_size: int, threshold: float = 0.0, invert: bool = False, soft: bool = False
) -> torch.Tensor:
    """
    Convert pixel mask to patch mask.
    
    Args:
        mask_hw: (B, 1, H, W) in [0,1]
        patch_size: int, spatial size of each patch
        threshold: float, threshold for determining masked pixels
        invert: bool, if True, invert the mask
        soft: bool, if True, output soft values in [0,1] representing proportion of pixels above threshold
        
    Returns:
        patch_mask: (B, N) where N=(H/P)*(W/P)
            - If soft=False: boolean tensor (patch is masked if ANY pixel is above threshold)
            - If soft=True: float tensor in [0,1] (proportion of pixels above threshold in each patch)
    """
    m = (mask_hw > threshold).float()
    if soft:
        # Use average pooling to get proportion of pixels above threshold in each patch
        m_small = F.avg_pool2d(m, kernel_size=patch_size, stride=patch_size)
        pm = m_small.flatten(1)  # (B, N) in [0,1]
    else:
        # Use max pooling to check if ANY pixel is above threshold
        m_small = F.max_pool2d(m, kernel_size=patch_size, stride=patch_size)
        pm = m_small.flatten(1).bool()  # (B, N) boolean
    if invert:
        if soft:
            pm = 1.0 - pm
        else:
            pm = torch.logical_not(pm)
    return pm

def patch_mask_to_pixel_mask(
    patch_mask: torch.Tensor, patch_size: int, soft: bool = False
) -> torch.Tensor:
    """
    Convert patch mask to pixel mask.

    Args:
        patch_mask: (B, N) boolean or float tensor, where N = (H/P)*(W/P)
        patch_size: int, spatial size of each patch
        soft: bool, if True, use bilinear interpolation for smooth transitions; 
              if False, use nearest neighbor interpolation

    Returns:
        mask_hw: (B, 1, H, W) float tensor in [0,1]
    """
    B, N = patch_mask.shape
    # Compute target H and W (assuming square arrangement, filled in row-major order)
    patch_grid_size = int(N ** 0.5)

    # Reshape (B, N) -> (B, 1, grid, grid)
    patch_mask_grid = patch_mask.reshape(B, 1, patch_grid_size, patch_grid_size).float()
    # Upsample to (B, 1, H, W)
    if soft:
        # Use bilinear interpolation for smooth transitions
        pixel_mask = torch.nn.functional.interpolate(
            patch_mask_grid,
            scale_factor=patch_size,
            mode="bilinear",
            align_corners=False
        )
    else:
        # Use nearest-neighbor interpolation
        pixel_mask = torch.nn.functional.interpolate(
            patch_mask_grid,
            scale_factor=patch_size,
            mode="nearest"
        )
    return pixel_mask