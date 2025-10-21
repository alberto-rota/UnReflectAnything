import math
import os
import random
from typing import Any, Optional, Tuple, Union

import cv2
import lovely_tensors as lt
import matplotlib.pyplot as plt
import numpy as np
import rerun as rr
import torch
from PIL import Image, ImageDraw, ImageFont
from rerun import RotationAxisAngle
from scipy.spatial.transform import Rotation
from sklearn.decomposition import PCA

import geometry
from utilities import *
from utilities.rotations import euler2axang, mat2axang

from .dev_utils import embedding2color


def rgb(
    t: torch.Tensor,
    as_tensor: Union[bool, str] = False,
    pca: Optional[PCA] = None,
    blackout: Optional[bool] = False,
    resize: Optional[Tuple[int, int]] = None,
    interpolation: str = "bilinear",
    colormap: Optional[str] = "magma",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    border: Optional[
        Union[
            dict,
            Tuple[Union[list, tuple, torch.Tensor, np.ndarray, str], int],
        ]
    ] = None,
    label: Optional[Union[Tuple[str, int, str], Tuple[str, int, int, str]]] = None,
    **kwargs: Any,
) -> Union[None, torch.Tensor, Image.Image]:
    r"""
    Display tensor as RGB image using lovely_tensors with robust input handling.

    Args:
        t (torch.Tensor):
            The tensor to display. Supports various shapes:
                - HxW (grayscale)
                - HxWx1 or HxWxC (grayscale or RGB)
                - 1xHxW or CxHxW (grayscale or RGB)
                - 1x1xHxW, 1xCxHxW, 1xHxWx1, 1xHxWxC (batched)
        as_tensor (Union[bool, str]):
            - False: Display image (default)
            - True: Return torch.Tensor
            - "pil": Return PIL Image
        pca (Optional[PCA]):
            Pre-fitted PCA object for high-dimensional tensors.
            If None and channels > 3, computes new PCA.
        resize (Optional[Tuple[int, int]]):
            Target size (height, width) for resizing. Defaults to None.
        interpolation (str):
            Interpolation method for resizing ("bilinear", "nearest", "bicubic"). Defaults to "bilinear".
        colormap (Optional[str]):
            Colormap name for single-channel images (e.g., "plasma", "viridis", "jet"). Defaults to None.
        vmin (Optional[float]):
            Minimum value for colormap normalization. Defaults to tensor min.
        vmax (Optional[float]):
            Maximum value for colormap normalization. Defaults to tensor max.
        border (Optional[Union[dict, Tuple[Union[list, tuple, torch.Tensor, np.ndarray, str], int]]]):
            Border specification. Prefer dict form: {"color": ..., "thickness": int}.
            Tuple form (color, thickness) is still accepted for backward compatibility.
            Color can be:
                - RGB list/tuple: [r, g, b] or (r, g, b) with values in [0, 1]
                - torch.Tensor: RGB tensor with values in [0, 1]
                - np.ndarray: RGB array with values in [0, 1]
                - str: Hex color code (e.g., "#FF0000")
            Thickness is the border width in pixels. Defaults to None.
        label (Optional[Union[dict, Tuple[str, int, str], Tuple[str, int, int, str]]]):
            Label specification. Prefer dict form:
                {"position": str, "height": int, "margin": int, "text": str, "style": str, "latex": bool}
            Tuple forms are still accepted: (position, height, text) or
                (position, height, padding, text). The padding value is treated as margin.
            Draws a white rectangular label with black text in Computer Modern font.
                - position: combinations of top/bottom with optional left/right and inside/outside,
                  e.g. "top", "bottom-right", "top-left-outside". If left/right omitted, centers.
                - height: rectangle height in pixels; text is scaled to fit height.
                - margin: pixels used as spacing from edges (and expansion for outside labels).
                  Inner padding around text is auto-scaled from height.
                - style: one of {"normal", "bold", "italic"}; defaults to "bold".
                - latex: if True, or if text is wrapped with $...$, render text as LaTeX.
                - text: the string to render inside the rectangle.
        **kwargs (Any):
            Additional keyword arguments passed to lt.rgb().

    Returns:
        Union[None, torch.Tensor, Image.Image]:
            Based on as_tensor parameter
    """

    # Normalize input tensor to standard format: CxHxW
    def normalize_tensor_shape(tensor):
        """Convert tensor to CxHxW format, handling various input shapes."""
        shape = tensor.shape
        ndim = len(shape)

        if ndim == 2:  # HxW -> 1xHxW
            return tensor.unsqueeze(0)  # Shape: (1, H, W)

        elif ndim == 3:  # CxHxW or HxWxC
            if shape[0] <= 4:  # Likely CxHxW
                return tensor  # Shape: (C, H, W)
            else:  # Likely HxWxC
                return tensor.permute(2, 0, 1)  # Shape: (C, H, W)

        elif ndim == 4:  # BxCxHxW or BxHxWxC
            if shape[0] == 1:  # Single batch
                if shape[1] <= 4:  # BxCxHxW
                    return tensor.squeeze(0)  # Shape: (C, H, W)
                else:  # BxHxWxC
                    return tensor.squeeze(0).permute(2, 0, 1)  # Shape: (C, H, W)
            else:
                raise ValueError(f"Batch size must be 1, got {shape[0]}")

        else:
            raise ValueError(f"Unsupported tensor shape: {shape}")

    # Convert tensor to standard format
    t = normalize_tensor_shape(t)

    # Handle high-dimensional tensors with PCA
    if t.shape[0] > 3:
        if pca is None:
            # Compute new PCA
            t, pca = embedding2color(
                t.permute(1, 2, 0).unsqueeze(0), pca=None
            )  # Add batch dimension
            t = t.squeeze(0)  # Remove batch dimension
            if blackout:
                t = blackout_pca(t)
        else:
            # Use provided PCA
            t = embedding2color(
                t.permute(1, 2, 0).unsqueeze(0), pca=pca
            )  # Add batch dimension
            t = t.squeeze(0)  # Remove batch dimension
            if blackout:
                t = blackout_pca(t)

    # Handle single-channel images with colormap
    if t.shape[0] == 1 and colormap is not None:
        # Get vmin and vmax values
        if vmin is None:
            vmin = t.min().item()
        if vmax is None:
            vmax = t.max().item()

        # Clamp values to vmin/vmax range
        t = torch.clamp(t, vmin, vmax)

        # Normalize to [0, 1] for colormap
        t_normalized = (t - vmin) / (vmax - vmin + 1e-8)

        # Apply colormap using matplotlib
        # t_normalized shape: (1, H, W)
        # Convert to numpy, apply colormap, convert back to torch
        t_np = t_normalized.squeeze(0).cpu().numpy()  # Shape: (H, W)
        if isinstance(colormap, str):
            import matplotlib.pyplot as plt

            cmap = plt.get_cmap(colormap)
        else:
            cmap = colormap
        t_colored = torch.from_numpy(cmap(t_np)[..., :3]).permute(
            2, 0, 1
        )  # Shape: (3, H, W)
        t = t_colored.to(t.device)

    # Handle grayscale to RGB conversion (if no colormap was applied)
    elif t.shape[0] == 1:
        t = t.repeat(3, 1, 1)  # Shape: (3, H, W)

    # Apply resize if requested
    if resize is not None:
        height, width = resize
        # Convert to interpolation mode
        if interpolation == "bilinear":
            mode = "bilinear"
        elif interpolation == "nearest":
            mode = "nearest"
        elif interpolation == "bicubic":
            mode = "bicubic"
        else:
            mode = "bilinear"

        # Resize tensor: (C, H, W) -> (C, new_H, new_W)
        t = torch.nn.functional.interpolate(
            t.unsqueeze(0),  # Add batch dimension for interpolate
            size=(height, width),
            mode=mode,
            align_corners=False if mode == "bilinear" else None,
        ).squeeze(0)  # Remove batch dimension

    # Normalize to [0, 1] range (only if not already normalized by colormap)
    if colormap is None or t.shape[0] != 1:
        t = (t - t.min()) / (t.max() - t.min() + 1e-8)

    # Apply border if specified
    if border is not None:
        # Support dict or tuple
        if isinstance(border, dict):
            color = border.get("color", [1.0, 1.0, 1.0])
            thickness = border.get("thickness", 1)
        else:
            color, thickness = border
        thickness = int(thickness) + 1
        # Convert color to RGB tensor with values in [0, 1]
        if isinstance(color, str):
            # Handle hex color
            if color.startswith("#"):
                color = color[1:]
            rgb_int = tuple(int(color[i : i + 2], 16) for i in (0, 2, 4))
            color_tensor = torch.tensor(
                [c / 255.0 for c in rgb_int], device=t.device, dtype=t.dtype
            )
        elif isinstance(color, (list, tuple)):
            # Handle list/tuple
            color_tensor = torch.tensor(color, device=t.device, dtype=t.dtype)
        elif isinstance(color, np.ndarray):
            # Handle numpy array
            color_tensor = torch.from_numpy(color).to(device=t.device, dtype=t.dtype)
        elif isinstance(color, torch.Tensor):
            # Handle torch tensor
            color_tensor = color.to(device=t.device, dtype=t.dtype)
        else:
            raise ValueError(f"Unsupported color type: {type(color)}")

        # Ensure color tensor is 1D with 3 values
        if color_tensor.dim() > 1:
            color_tensor = color_tensor.flatten()
        if color_tensor.numel() != 3:
            raise ValueError(
                f"Color must have exactly 3 RGB values, got {color_tensor.numel()}"
            )

        # Ensure color values are in [0, 1] range
        color_tensor = torch.clamp(color_tensor, 0, 1)

        # Create border by setting pixels at the edges
        # t shape: (C, H, W)
        C, H, W = t.shape

        # Create a copy of the tensor to avoid modifying the original
        t_with_border = t.clone()

        # Apply border using vectorized operations
        if thickness > 0:
            # Set top and bottom borders
            t_with_border[:, :thickness, :] = color_tensor.view(
                3, 1, 1
            )  # Shape: (3, thickness, W)
            t_with_border[:, -thickness + 1 :, :] = color_tensor.view(
                3, 1, 1
            )  # Shape: (3, thickness, W)

            # Set left and right borders
            t_with_border[:, :, :thickness] = color_tensor.view(
                3, 1, 1
            )  # Shape: (3, H, thickness)
            t_with_border[:, :, -thickness + 1 :] = color_tensor.view(
                3, 1, 1
            )  # Shape: (3, H, thickness)

        t = t_with_border

    # Apply label if specified
    if label is not None:
        # Accept dict or (pos, height, text) / (pos, height, padding, text)
        if isinstance(label, dict):
            pos_str = label.get("position", "top")
            rect_height = label.get("height", 24)
            margin = label.get("margin", None)
            label_text = label.get("text", "")
        elif isinstance(label, (list, tuple)) and len(label) == 3:
            pos_str, rect_height, label_text = label  # type: ignore
            margin = None
        elif isinstance(label, (list, tuple)) and len(label) == 4:
            pos_str, rect_height, margin, label_text = label  # type: ignore
        else:
            raise ValueError(
                "label must be a dict {position,height,margin,text} or a tuple (pos,height,text) or (pos,height,padding,text)"
            )

        rect_height = max(1, int(rect_height))
        C, H, W = t.shape
        np_img = (t.permute(1, 2, 0).detach().cpu().numpy() * 255.0).astype(np.uint8)
        np_img = np.ascontiguousarray(np_img)

        # Default inner padding around text (derived from height). External spacing uses 'margin'.
        pad = max(2, rect_height // 8)

        # Try to load Computer Modern (respect style: normal/bold/italic)
        pil_font = None
        try:
            import matplotlib as mpl

            ttf_dir = os.path.join(mpl.get_data_path(), "fonts", "ttf")
            bold_path = os.path.join(ttf_dir, "cmb10.ttf")
            italic_path = os.path.join(ttf_dir, "cmi10.ttf")
            regular_path = os.path.join(ttf_dir, "cmr10.ttf")
            # style variable may be undefined if tuple form used
            chosen_style = locals().get("style", "bold")
            chosen_path = regular_path
            if chosen_style == "bold" and os.path.exists(bold_path):
                chosen_path = bold_path
            elif chosen_style in ("italic", "italics") and os.path.exists(italic_path):
                chosen_path = italic_path
            elif os.path.exists(regular_path):
                chosen_path = regular_path
            elif os.path.exists(bold_path):
                chosen_path = bold_path
            elif os.path.exists(italic_path):
                chosen_path = italic_path
            else:
                chosen_path = regular_path
            # Binary search font size to fit height
            target_h = max(1, rect_height - 2 * pad)
            lo, hi = 4, 400
            for _ in range(10):
                mid = (lo + hi) // 2
                test_font = ImageFont.truetype(chosen_path, mid)
                # Measure using bbox
                bbox = test_font.getbbox(str(label_text))
                text_h = bbox[3] - bbox[1]
                if text_h <= target_h:
                    pil_font = test_font
                    lo = mid + 1
                else:
                    hi = mid - 1
            if pil_font is None:
                pil_font = ImageFont.truetype(chosen_path, max(4, target_h))
        except Exception:
            # Fallback to default PIL font
            pil_font = ImageFont.load_default()

        # Measure text size or LaTeX image size
        # If LaTeX support is requested, render it first, otherwise use PIL font
        is_latex = False
        if isinstance(label_text, str):
            s_txt = label_text.strip()
            is_latex = s_txt.startswith("$") and s_txt.endswith("$")
        if isinstance(label, dict):
            is_latex = bool(label.get("latex", is_latex))

        latex_img = None
        if is_latex:
            try:
                from matplotlib import rcParams

                rcParams.setdefault("mathtext.fontset", "cm")
                from matplotlib.mathtext import MathTextParser

                target_h = max(1, rect_height - 2 * pad)
                parser = MathTextParser("agg")
                rgba, _ = parser.to_rgba(str(label_text), dpi=200, color="black")
                latex_pil = Image.fromarray((rgba * 255).astype(np.uint8))
                if latex_pil.height > 0:
                    new_w = max(
                        1, int(round(latex_pil.width * (target_h / latex_pil.height)))
                    )
                    latex_pil = latex_pil.resize((new_w, int(target_h)), Image.LANCZOS)
                latex_img = latex_pil
                text_w, text_h = latex_img.size
            except Exception:
                latex_img = None

        if latex_img is None:
            dummy_img = Image.new("RGB", (10, 10), (255, 255, 255))
            dummy_draw = ImageDraw.Draw(dummy_img)
            bbox = dummy_draw.textbbox((0, 0), str(label_text), font=pil_font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
        rect_w = text_w + 2 * pad
        rect_h = rect_height

        # Parse position string
        tokens = (
            pos_str.lower().replace("_", "-").split("-")
            if isinstance(pos_str, str)
            else ["top"]
        )
        vpos = None
        if "top" in tokens:
            vpos = "top"
        if "bottom" in tokens:
            vpos = "bottom"
        hpos = "center"
        if "left" in tokens:
            hpos = "left"
        elif "right" in tokens:
            hpos = "right"
        inside = "outside" not in tokens

        def draw_label_on_pil(pil_img: Image.Image, x0: int, y0: int) -> None:
            draw = ImageDraw.Draw(pil_img)
            # Rectangle
            draw.rectangle([x0, y0, x0 + rect_w, y0 + rect_h], fill=(255, 255, 255))
            # Text position: left padding, vertically centered
            text_x = x0 + pad
            text_y = y0 + (rect_h - text_h) // 2
            if latex_img is not None:
                try:
                    pil_img.paste(latex_img, (int(text_x), int(text_y)), mask=latex_img)
                except Exception:
                    pil_img.paste(latex_img, (int(text_x), int(text_y)))
            else:
                # Adjust baseline using bbox top
                dummy_img2 = Image.new("RGB", (10, 10), (255, 255, 255))
                dummy_draw2 = ImageDraw.Draw(dummy_img2)
                bb2 = dummy_draw2.textbbox((0, 0), str(label_text), font=pil_font)
                base_adj = bb2[1]
                draw.text(
                    (int(text_x), int(text_y - base_adj)),
                    str(label_text),
                    font=pil_font,
                    fill=(0, 0, 0),
                )

        if inside:
            # Vertical placement
            if vpos is None:
                y0 = max(0, (H - rect_h) // 2)
            else:
                y0 = (0 + margin) if vpos == "top" else max(0, H - rect_h - margin)
            # Horizontal placement
            if hpos == "left":
                x0 = 0 + margin
            elif hpos == "right":
                x0 = max(0, W - rect_w - margin)
            else:
                x0 = max(0, (W - rect_w) // 2)

            x0 = min(x0, max(0, W - rect_w))
            y0 = min(y0, max(0, H - rect_h))

            pil_img = Image.fromarray(np_img)
            draw_label_on_pil(pil_img, int(x0), int(y0))
            np_img = np.array(pil_img)
            H, W = np_img.shape[:2]
        else:
            # Expand canvas by rectangle size plus margin on the exterior side
            expand_h = (rect_h + margin) if (vpos in ["top", "bottom"]) else 0
            expand_w = (rect_w + margin) if (hpos in ["left", "right"]) else 0
            new_H = H + expand_h
            new_W = W + expand_w
            canvas = Image.new("RGB", (new_W, new_H), (255, 255, 255))
            pil_img = Image.fromarray(np_img)

            # Paste original image
            offset_y = (rect_h + margin) if vpos == "top" else 0
            offset_x = (rect_w + margin) if hpos == "left" else 0
            canvas.paste(pil_img, (offset_x, offset_y))

            # Compute label rectangle position
            if vpos in ["top", "bottom"]:
                y0 = 0 if vpos == "top" else new_H - rect_h
                if hpos == "left":
                    x0 = 0
                elif hpos == "right":
                    x0 = new_W - rect_w
                else:
                    x0 = max(0, (new_W - rect_w) // 2)
            else:
                y0 = max(0, (new_H - rect_h) // 2)
                x0 = 0 if hpos == "left" else new_W - rect_w

            draw_label_on_pil(canvas, int(x0), int(y0))
            np_img = np.array(canvas)
            H, W = np_img.shape[:2]

        # Convert back to tensor
        t = (
            torch.from_numpy(np_img).to(device=t.device, dtype=t.dtype).permute(2, 0, 1)
            / 255.0
        )

    # Handle return types
    if as_tensor is True:
        return t
    elif as_tensor == "pil":
        # Convert tensor to PIL Image
        # Tensor is in CxHxW format, convert to HxWxC for PIL
        t_pil = t.permute(1, 2, 0).cpu().numpy()  # Shape: (H, W, C)
        t_pil = (t_pil * 255).astype(np.uint8)
        return Image.fromarray(t_pil)
    else:
        # Display the tensor
        display(lt.rgb(t, **kwargs))
        return None


def channels(t: torch.Tensor, as_tensor=False, **kwargs: Any) -> None:
    """
    Display tensor channels using lovely_tensors.

    Args:
        t (torch.Tensor): The tensor to display.
        **kwargs (Any): Additional keyword arguments passed to lt.chans().
    """
    # Normalize
    t = (t - t.mean()) / t.std()
    t = (t - t.min()) / (t.max() - t.min())
    # Display the tensor channels using lovely_tensors
    if not as_tensor:
        display(lt.chans(t, **kwargs))
    else:
        return lt.chans(t, **kwargs)


def plasma(depthmap: torch.Tensor) -> torch.Tensor:
    """
    Apply the plasma colormap to a depth map.

    Args:
    depthmap (torch.Tensor): A tensor representing the depth map.

    Returns:
    torch.Tensor: A tensor representing the depth map with the plasma colormap applied.
    """
    # Convert torch tensor to numpy
    if len(depthmap.shape) == 2:
        depthmap = depthmap.unsqueeze(0)
    if len(depthmap.shape) == 3:
        depthmap = (depthmap - depthmap.min()) / (depthmap.max() - depthmap.min())
        depthmap_np = depthmap.permute(2, 1, 0).cpu().detach().numpy()[..., 0]
        # Apply plasma colormap
        depthmap_np_plasma = plt.get_cmap("magma")(depthmap_np)[:, :, :3]
        # Convert back to torch tensor
        depthmap_rgb = torch.from_numpy(depthmap_np_plasma).permute(2, 1, 0)
    if len(depthmap.shape) == 4:
        depthmap = (depthmap - depthmap.min(dim=1).min(dim=2).min(dim=3)) / (
            depthmap.max(dim=1).min(dim=2).min(dim=3)
            - depthmap.min(dim=1).min(dim=2).min(dim=3)
        )
        depthmap_np = depthmap.permute(0, 2, 3, 1).cpu().detach().numpy()[..., 0]
        # Apply plasma colormap
        depthmap_np_plasma = plt.get_cmap("magma")(depthmap_np)[:, :, :, :3]
        # Convert back to torch tensor
        depthmap_rgb = torch.from_numpy(depthmap_np_plasma).permute(0, 3, 2, 1)
    return depthmap_rgb


def coloredbar(parts: list, colors: list, num_blocks: int) -> str:
    """
    Creates a colored bar using unicode block characters.


    Args:
        parts (list): A list of numbers representing the size of each part of the bar.
        colors (list): A list of colors for each part of the bar.
        num_blocks (int): The total number of blocks in the bar.

    Returns:
        str: A string representing the colored bar.
    """
    assert len(parts) == len(colors)
    total = sum(parts)
    block = "\u25a0"
    bar = ""
    for part, color in zip(parts, colors):
        num_part_blocks = round(part / total * num_blocks)
        bar += f"[{color}]{block * num_part_blocks}[/]"
    return bar


def viewPixelMatches(
    img1: Union[Image.Image, torch.Tensor],
    img2: Union[Image.Image, torch.Tensor],
    pts1: torch.Tensor,
    pts2: torch.Tensor,
    scores: torch.Tensor,
    topk: int = 20,
    use_actual_topk: bool = False,
) -> Image.Image:
    def to_pil(img):
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu()
            if img.dim() == 3:
                img = img.permute(1, 2, 0)
            return Image.fromarray(img.numpy().astype("uint8"))
        return img

    img1, img2 = to_pil(img1 * 255), to_pil(img2 * 255)

    # Select points based on mode
    n = pts2.shape[0]
    if use_actual_topk:
        # Get the actual top-k highest scoring points
        _, selected_indices = torch.topk(scores, min(topk, n))
    else:
        # Use evenly spaced points as before
        stride = max(1, n // topk)
        selected_indices = torch.arange(0, n, stride)[:topk]

    pts1, pts2 = pts1[selected_indices], pts2[selected_indices]
    scores = scores[selected_indices]

    # Calculate patch size based on image dimensions
    w1, h1 = img1.size
    w2, h2 = img2.size
    patch_size = int(
        min(w1, h1) * 2 / topk
    )  # Reasonable default relative to image size
    half_patch = patch_size // 2

    # Create canvas with space for patches
    h = max(h1, h2) + 2 * patch_size  # Add space for two rows of patches
    w = w1 + w2
    canvas = Image.new("RGB", (w, h))
    canvas.paste(img1, (0, 0))
    canvas.paste(img2, (w1, 0))

    # Draw matches
    draw = ImageDraw.Draw(canvas)

    # Handle score normalization for the case where all scores are equal
    score_min = scores.min()
    score_max = scores.max()
    if score_min == score_max:
        # If all scores are equal, set them all to 1.0 to indicate maximum confidence
        norm_scores = torch.ones_like(scores)
    else:
        norm_scores = (scores - score_min) / (score_max - score_min)

    # Extract and draw patches
    for i, ((x1, y1), (x2, y2), score) in enumerate(zip(pts1, pts2, scores)):
        # Draw match line
        color = (int(255 * (1 - score)), int(255 * score), 0)
        draw.line([x1, y1, x2 + w1, y2], fill=color, width=1)

        # Extract and paste patches
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        # First image patch
        left1 = max(0, x1 - half_patch)
        top1 = max(0, y1 - half_patch)
        # Check if patch is out of bounds
        if left1 >= w1 or top1 >= h1:
            continue
        patch1 = img1.crop(
            (left1, top1, min(w1, left1 + patch_size), min(h1, top1 + patch_size))
        )
        patch_x = i * patch_size
        canvas.paste(patch1, (patch_x, h - 2 * patch_size))

        # Second image patch
        left2 = max(0, x2 - half_patch)
        top2 = max(0, y2 - half_patch)
        if left2 >= w2 or top2 >= h2:
            continue
        patch2 = img2.crop((left2, top2, left2 + patch_size, top2 + patch_size))
        canvas.paste(patch2, (patch_x, h - patch_size))

        # Draw boxes around patches
        draw.rectangle(
            [patch_x, h - 2 * patch_size, patch_x + patch_size, h - patch_size],
            outline=color,
            width=1,
        )
        draw.rectangle(
            [patch_x, h - patch_size, patch_x + patch_size, h], outline=color, width=1
        )

    return canvas


def viewComparePixelMatches(
    img1: Union[Image.Image, torch.Tensor],
    img2: Union[Image.Image, torch.Tensor],
    pts1: torch.Tensor,
    pts2: torch.Tensor,
    pts2_true: torch.Tensor,
    scores: torch.Tensor,
    topk: int = 20,
    use_actual_topk: bool = False,
    as_tensor: bool = False,
) -> Image.Image:
    def to_pil(img):
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu()
            if img.dim() == 3:
                img = img.permute(1, 2, 0)
            img = img.numpy()
            if img.dtype != "uint8":
                img = (img * 255).astype("uint8")
            return Image.fromarray(img)
        return img

    img1, img2 = to_pil(img1), to_pil(img2)

    # Select points based on mode
    n = pts2.shape[0]
    if use_actual_topk:
        # Get the actual top-k highest scoring points
        _, selected_indices = torch.topk(scores, min(topk, n))
    else:
        # Use evenly spaced points as before
        stride = max(1, n // topk)
        selected_indices = torch.arange(0, n, stride)[:topk]

    pts1_selected, pts2_selected, pts2_true_selected, scores_selected = (
        pts1[selected_indices],
        pts2[selected_indices],
        pts2_true[selected_indices],
        scores[selected_indices],
    )

    # Calculate patch size based on image dimensions
    w1, h1 = img1.size
    w2, h2 = img2.size
    patch_size = int(
        min(w1, h1, w2, h2) * 2 / topk
    )  # Reasonable default relative to image size
    half_patch = patch_size // 2

    # Create canvas with space for patches and annotations
    canvas_height = max(h1, h2)  # Additional space for circles
    canvas_width = w1 + w2
    canvas = Image.new("RGB", (canvas_width, canvas_height), (255, 255, 255))
    canvas.paste(img1, (0, 0))
    canvas.paste(img2, (w1, 0))

    draw = ImageDraw.Draw(canvas)

    # Handle score normalization for coloring
    score_min = scores_selected.min()
    score_max = scores_selected.max()
    if score_min == score_max:
        norm_scores = torch.ones_like(scores_selected)
    else:
        norm_scores = (scores_selected - score_min) / (score_max - score_min)

    # Function to generate a random color
    def get_random_color():
        return tuple(random.randint(0, 255) for _ in range(3))

    # Extract and draw matches
    for i, ((x1, y1), (x2, y2), (x2_true, y2_true), score) in enumerate(
        zip(pts1_selected, pts2_selected, pts2_true_selected, scores)
    ):
        # Assign a unique random color for this match
        color = get_random_color()

        # Draw line from pts1 to pts2 with color based on score
        line_color = (int(255 * (1 - score)), int(255 * score), 0)
        draw.line([x1, y1, x2 + w1, y2], fill=line_color, width=1)

        # Draw line from pts2 to pts2_true with the same random color
        draw.line(
            [x2 + w1, y2, x2_true + w1, y2_true],
            fill=color,
            width=1,
        )

        # Draw small circles for pts1, pts2, and pts2_true
        radius = 2
        # pts1 circle
        draw.ellipse(
            [
                x1 - radius,
                y1 - radius,
                x1 + radius,
                y1 + radius,
            ],
            outline=color,
            width=2,
        )
        # pts2 circle
        draw.ellipse(
            [
                x2 + w1 - radius,
                y2 - radius,
                x2 + w1 + radius,
                y2 + radius,
            ],
            outline=color,
            width=2,
        )
        # pts2_true circle
        draw.ellipse(
            [
                x2_true + w1 - radius,
                y2_true - radius,
                x2_true + w1 + radius,
                y2_true + radius,
            ],
            outline=color,
            width=2,
        )
        # Return as tensor if requested
    if as_tensor:
        # Convert PIL image to tensor (3xHxW)
        canvas_np = np.array(canvas)
        canvas_tensor = torch.from_numpy(canvas_np).permute(2, 0, 1).float() / 255.0
        return canvas_tensor

    return canvas


def viewPatchMatches(
    img1: Union[Image.Image, torch.Tensor],
    img2: Union[Image.Image, torch.Tensor],
    similarity_matrix: torch.Tensor,
    patch_size: int = None,
    topk: int = 20,
    use_actual_topk: bool = False,
) -> Image.Image:
    def to_pil(img):
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu()
            if img.dim() == 3:
                img = img.permute(1, 2, 0)
            return Image.fromarray(img.numpy().astype("uint8"))
        return img

    img1, img2 = to_pil(img1 * 255), to_pil(img2 * 255)
    w1, h1 = img1.size
    w2, h2 = img2.size

    num_patches = similarity_matrix.shape[0]
    patch_size_h = h1 // int(math.sqrt(num_patches))
    patch_size_w = w1 // int(math.sqrt(num_patches))
    if patch_size is not None:
        patch_size_h = patch_size
        patch_size_w = patch_size
    # Get global min/max for normalization
    global_min = similarity_matrix.min().cpu()
    global_max = similarity_matrix.max().cpu()

    # Select points based on mode
    n = similarity_matrix.shape[0]
    if use_actual_topk:
        # Get the actual top-k highest similarity scores
        values, flat_indices = torch.topk(similarity_matrix.view(-1), topk)
        src_indices = flat_indices // similarity_matrix.shape[1]
        tgt_indices = flat_indices % similarity_matrix.shape[1]
    else:
        # Use evenly spaced points as before
        stride = max(1, n // topk)
        src_indices = torch.arange(0, n, stride)[:topk]
        values = torch.zeros(topk, dtype=torch.float32)
        tgt_indices = torch.zeros(topk, dtype=torch.int64)
        for i, src_idx in enumerate(src_indices):
            values[i], tgt_indices[i] = similarity_matrix[src_idx].max(dim=0)

    patches_per_row = w1 // patch_size_w

    def get_patch_center(idx):
        row = int(idx // patches_per_row)
        col = int(idx % patches_per_row)
        return (
            col * patch_size_w + patch_size_w // 2,
            row * patch_size_h + patch_size_h // 2,
        )

    w = w1 + w2
    display_patch_size = w // topk
    h = max(h1, h2) + 2 * display_patch_size
    canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    img1_rgba = img1.convert("RGBA")
    img2_rgba = img2.convert("RGBA")
    canvas.paste(img1_rgba, (0, 0))
    canvas.paste(img2_rgba, (w1, 0))

    draw = ImageDraw.Draw(canvas, "RGBA")
    # Normalize scores using global min/max
    norm_scores = (values - global_min) / (global_max - global_min)

    display_patch_size = w // topk
    h = max(h1, h2) + 2 * display_patch_size

    for i, (src_idx, tgt_idx, score) in enumerate(
        zip(src_indices, tgt_indices, norm_scores)
    ):
        src_x, src_y = get_patch_center(src_idx)
        tgt_x, tgt_y = get_patch_center(tgt_idx)

        # Opacity based on score (minimum opacity of 0.3)
        opacity = int(255 * (0.3 + 0.7 * score))
        color = (
            int(255 * (1 - score)),  # Red
            int(255 * score),  # Green
            0,  # Blue
            opacity,  # Alpha
        )

        # Draw line with transparency
        draw.line(
            [src_x, src_y, tgt_x + w1, tgt_y],
            fill=color,
            width=2,
        )

        # Draw rectangles with transparency
        draw.rectangle(
            [
                src_x - patch_size_w // 2,
                src_y - patch_size_h // 2,
                src_x + patch_size_w // 2,
                src_y + patch_size_h // 2,
            ],
            outline=color,
            width=2,
        )
        draw.rectangle(
            [
                tgt_x + w1 - patch_size_w // 2,
                tgt_y - patch_size_h // 2,
                tgt_x + w1 + patch_size_w // 2,
                tgt_y + patch_size_h // 2,
            ],
            outline=color,
            width=2,
        )

        # Extract and process patches
        src_patch = img1.crop(
            (
                src_x - patch_size_w // 2,
                src_y - patch_size_h // 2,
                src_x + patch_size_w // 2,
                src_y + patch_size_h // 2,
            )
        )
        tgt_patch = img2.crop(
            (
                tgt_x - patch_size_w // 2,
                tgt_y - patch_size_h // 2,
                tgt_x + patch_size_w // 2,
                tgt_y + patch_size_h // 2,
            )
        )

        # Resize patches
        src_patch = src_patch.resize(
            (display_patch_size, display_patch_size), resample=Image.NEAREST
        )
        tgt_patch = tgt_patch.resize(
            (display_patch_size, display_patch_size), resample=Image.NEAREST
        )

        patch_x = i * display_patch_size
        canvas.paste(src_patch, (patch_x, h - 2 * display_patch_size))
        canvas.paste(tgt_patch, (patch_x, h - display_patch_size))

        # Draw rectangles around patches with transparency
        draw.rectangle(
            [
                patch_x,
                h - 2 * display_patch_size,
                patch_x + display_patch_size,
                h - display_patch_size,
            ],
            outline=color,
            width=2,
        )
        draw.rectangle(
            [patch_x, h - display_patch_size, patch_x + display_patch_size, h],
            outline=color,
            width=2,
        )

    return canvas


def viewEpipolarGeometry(
    img1: Union[Image.Image, torch.Tensor],
    img2: Union[Image.Image, torch.Tensor],
    pts1: torch.Tensor,
    pts2: torch.Tensor,
    scores: torch.Tensor,
    F: torch.Tensor,
    pose_6d: Union[
        torch.Tensor, dict
    ] = None,  # [x, y, z, rx, ry, rz] or dict with 'gt' and 'pred' keys
    topk: int = 20,
    use_actual_topk: bool = False,
    as_tensor: bool = False,  # Return tensor instead of PIL image
) -> Union[Image.Image, torch.Tensor]:
    def to_pil(img):
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu()
            if img.dim() == 3:
                img = img.permute(1, 2, 0)
            return Image.fromarray(img.numpy().astype("uint8"))
        return img

    def compute_epipolar_line(point, F, in_img2=True):
        p = torch.tensor([point[0], point[1], 1.0], device=F.device)
        if in_img2:
            line = F @ p
        else:
            line = F.T @ p
        line = line / torch.sqrt(line[0] ** 2 + line[1] ** 2)
        return line.cpu().numpy()

    def get_line_endpoints(line, width, height):
        a, b, c = line
        points = []

        if abs(b) > 1e-8:
            y = -c / b
            if 0 <= y <= height:
                points.append((0, y))
            y = -(a * width + c) / b
            if 0 <= y <= height:
                points.append((width, y))

        if abs(a) > 1e-8:
            x = -c / a
            if 0 <= x <= width:
                points.append((x, 0))
            x = -(b * height + c) / a
            if 0 <= x <= width:
                points.append((x, height))

        return points[:2]

    def draw_arrow(draw, x, y, dx, dy, color, scale=50, width=2):
        """Draw an arrow starting at (x,y) in direction (dx,dy)"""
        # Normalize direction vector
        length = (dx * dx + dy * dy) ** 0.5
        if length < 1e-8:
            return

        dx, dy = dx / length, dy / length

        # Scale arrow length
        dx, dy = dx * scale, dy * scale

        # Arrow head parameters
        head_length = scale * 0.3
        head_width = scale * 0.2

        # Calculate arrow head points
        angle = math.atan2(dy, dx)
        angle1 = angle + math.pi * 3 / 4
        angle2 = angle - math.pi * 3 / 4

        x2, y2 = x + dx, y + dy
        x3 = x2 + head_length * math.cos(angle1)
        y3 = y2 + head_length * math.sin(angle1)
        x4 = x2 + head_length * math.cos(angle2)
        y4 = y2 + head_length * math.sin(angle2)

        # Draw arrow body and head
        draw.line([(x, y), (x2, y2)], fill=color, width=width)
        draw.line([(x2, y2), (x3, y3)], fill=color, width=width)
        draw.line([(x2, y2), (x4, y4)], fill=color, width=width)

    def draw_curved_arrow(
        draw, center, radius, start_angle, sweep_angle, color, width=2, num_points=50
    ):
        # Generate points along the arc
        points = []
        for i in range(num_points + 1):
            angle = math.radians(start_angle + (sweep_angle * i / num_points))
            x = center[0] + radius * math.cos(angle)
            y = center[1] + radius * math.sin(angle)
            points.append((x, y))
        # Draw the arc
        draw.line(points, fill=color, width=width)

        # Compute arrowhead based on the last segment direction
        if len(points) >= 2:
            x_last, y_last = points[-1]
            x_prev, y_prev = points[-2]
            dx, dy = x_last - x_prev, y_last - y_prev
            arrow_head_length = radius * 0.2  # Adjust relative to radius
            angle = math.atan2(dy, dx)
            # Two lines for the arrowhead at ±30° from the last segment
            angle1 = angle + math.pi / 6
            angle2 = angle - math.pi / 6
            x1 = x_last - arrow_head_length * math.cos(angle1)
            y1 = y_last - arrow_head_length * math.sin(angle1)
            x2 = x_last - arrow_head_length * math.cos(angle2)
            y2 = y_last - arrow_head_length * math.sin(angle2)
            draw.line([(x_last, y_last), (x1, y1)], fill=color, width=width)
            draw.line([(x_last, y_last), (x2, y2)], fill=color, width=width)

    def draw_z_motion(draw, x, y, z, color, radius=20):
        """Draw a circle with dot/cross to indicate z-motion"""
        # Draw circle
        bbox = [x - radius, y - radius, x + radius, y + radius]
        draw.ellipse(bbox, outline=color, width=2)

        # Draw indicator based on direction
        if z > 0:  # Forward motion: draw cross
            cross_size = radius * 0.6
            draw.line(
                [(x - cross_size, y - cross_size), (x + cross_size, y + cross_size)],
                fill=color,
                width=2,
            )
            draw.line(
                [(x - cross_size, y + cross_size), (x + cross_size, y - cross_size)],
                fill=color,
                width=2,
            )
        else:  # Backward motion: draw dot
            dot_radius = radius * 0.3
            draw.ellipse(
                [x - dot_radius, y - dot_radius, x + dot_radius, y + dot_radius],
                fill=color,
            )

    # Move tensors to CPU for processing
    scores = scores.detach().cpu()
    pts1 = pts1.detach().cpu()
    pts2 = pts2.detach().cpu()
    F = F.detach().cpu()

    img1, img2 = to_pil(img1 * 255), to_pil(img2 * 255)
    w1, h1 = img1.size
    w2, h2 = img2.size

    # Select points based on use_actual_topk parameter
    n = len(scores)

    if use_actual_topk:
        max_score = scores.max()
        max_score_indices = torch.where(scores == max_score)[0]

        if len(max_score_indices) > topk:
            perm = torch.randperm(len(max_score_indices))
            selected_indices = max_score_indices[perm[:topk]]
        else:
            remaining = topk - len(max_score_indices)
            if remaining > 0:
                _, other_indices = torch.topk(
                    scores[~torch.isin(torch.arange(len(scores)), max_score_indices)],
                    remaining,
                )
                selected_indices = torch.cat([max_score_indices, other_indices])
            else:
                selected_indices = max_score_indices[:topk]
    else:
        stride = max(1, n // topk)
        selected_indices = torch.arange(0, n, stride)[:topk]

    pts1, pts2 = pts1[selected_indices], pts2[selected_indices]
    scores = scores[selected_indices]

    # Create canvas
    margin = 1
    h_total = h1 + margin
    w_total = w1 + w2 + margin
    canvas = Image.new("RGB", (w_total, h_total))

    # Paste images
    canvas.paste(img1, (0, 0))
    canvas.paste(img2, (w1 + margin, 0))

    draw = ImageDraw.Draw(canvas)
    point_radius = 2

    # Generate random colors for each match
    colors = [(torch.rand(3) * 255).int().tolist() for _ in range(len(pts1))]

    # Draw matches and epipolar lines
    for i, ((x1, y1), (x2, y2), color) in enumerate(zip(pts1, pts2, colors)):
        x1, y1, x2, y2 = map(float, [x1, y1, x2, y2])

        # Compute epipolar lines
        line2 = compute_epipolar_line((x1, y1), F, in_img2=True)
        line1 = compute_epipolar_line((x2, y2), F, in_img2=False)

        # Get line endpoints
        endpoints2 = get_line_endpoints(line2, w2, h2)
        endpoints1 = get_line_endpoints(line1, w1, h1)

        # Draw points and epipolar lines
        # Left image (img1)
        draw.ellipse(
            [
                x1 - point_radius,
                y1 - point_radius,
                x1 + point_radius,
                y1 + point_radius,
            ],
            fill=tuple(color),
        )
        if len(endpoints1) == 2:
            draw.line(
                [
                    endpoints1[0][0],
                    endpoints1[0][1],
                    endpoints1[1][0],
                    endpoints1[1][1],
                ],
                fill=tuple(color),
                width=1,
            )

        # Right image (img2)
        x2_offset = w1 + margin
        draw.ellipse(
            [
                x2_offset + x2 - point_radius,
                y2 - point_radius,
                x2_offset + x2 + point_radius,
                y2 + point_radius,
            ],
            fill=tuple(color),
        )
        if len(endpoints2) == 2:
            draw.line(
                [
                    x2_offset + endpoints2[0][0],
                    endpoints2[0][1],
                    x2_offset + endpoints2[1][0],
                    endpoints2[1][1],
                ],
                fill=tuple(color),
                width=1,
            )

    if pose_6d is not None:
        # Define colors
        gt_color = (0, 255, 0)  # Green for ground truth
        pred_color = (255, 119, 2)  # Cyan for prediction

        if isinstance(pose_6d, dict):
            # Process both ground truth and prediction poses
            poses = {
                "gt": (pose_6d.get("gt"), gt_color),
                "pred": (pose_6d.get("pred"), pred_color),
            }

            # Calculate offset for side-by-side visualization
            # Position arrows at 1/3 and 2/3 of the width
            offsets = {"gt": w1 // 4, "pred": (w1 * 3) // 4}

            # Draw both pose visualizations
            for pose_type, (pose, color) in poses.items():
                if pose is not None:
                    pose = pose.detach().cpu()
                    center_x = offsets[pose_type]
                    center_y = h1 // 2
                    translation = pose[:3]  # Get translation vector [x, y, z]
                    rotation = pose[3:6]  # Get rotation vector [ex, ey, ez]

                    # Draw x-y motion arrow
                    draw_arrow(
                        draw,
                        center_x,
                        center_y,
                        float(translation[0]),
                        float(translation[1]),
                        color,
                    )

                    # Draw z-motion indicator
                    draw_z_motion(
                        draw, center_x, center_y, float(translation[2]), color
                    )

                    # Draw rotation visualization (new)
                    ex, ey, ez = rotation
                    center = (w1 // 2, h1 // 2)
                    radius = 100  # Adjust as needed
                    scale_factor = 1000
                    # Example parameters (you may need to adjust start angles and directions)
                    draw_curved_arrow(
                        draw,
                        center,
                        radius,
                        start_angle=0,
                        sweep_angle=ex * scale_factor,
                        width=4 if pose_type == "gt" else 2,
                        color=(255, 0, 0),
                    )
                    draw_curved_arrow(
                        draw,
                        center,
                        radius + 10,
                        start_angle=90,
                        sweep_angle=ey * scale_factor,
                        width=4 if pose_type == "gt" else 2,
                        color=(0, 255, 0),
                    )
                    draw_curved_arrow(
                        draw,
                        center,
                        radius + 20,
                        start_angle=180,
                        sweep_angle=ez * scale_factor,
                        width=4 if pose_type == "gt" else 2,
                        color=(0, 0, 255),
                    )

        else:
            # Original single pose visualization with green color
            pose_6d = pose_6d.detach().cpu()
            center_x = w1 // 2
            center_y = h1 // 2
            translation = pose_6d[:3]  # Get translation vector [x, y, z]
            rotation = pose_6d[3:6]  # Get rotation vector [ex, ey, ez]

            # Draw x-y motion arrow
            draw_arrow(
                draw,
                center_x,
                center_y,
                float(translation[0]),
                float(translation[1]),
                gt_color,
            )

            # Draw z-motion indicator
            draw_z_motion(draw, center_x, center_y, float(translation[2]), gt_color)

            # Draw rotation visualization (new)
            ex, ey, ez = rotation
            center = (w1 // 2, h1 // 2)
            radius = 100  # Adjust as needed
            scale_factor = 1000
            # Example parameters (you may need to adjust start angles and directions)
            draw_curved_arrow(
                draw,
                center,
                radius,
                start_angle=0,
                sweep_angle=ex * scale_factor,
                width=2,
                color=(255, 0, 0),
            )
            draw_curved_arrow(
                draw,
                center,
                radius + 10,
                start_angle=90,
                sweep_angle=ey * scale_factor,
                width=2,
                color=(0, 255, 0),
            )
            draw_curved_arrow(
                draw,
                center,
                radius + 20,
                start_angle=180,
                sweep_angle=ez * scale_factor,
                width=2,
                color=(0, 0, 255),
            )

    # Return as tensor if requested
    if as_tensor:
        # Convert PIL image to tensor (3xHxW)
        canvas_np = np.array(canvas)
        canvas_tensor = torch.from_numpy(canvas_np).permute(2, 0, 1).float() / 255.0
        return canvas_tensor

    return canvas


from PIL import Image


def viewTriplets(
    img1: Image.Image,
    img2: Image.Image,
    anchor_indices: list,
    positive_indices: list,
    negative_indices: list,
    patch_size: int = 8,
    num_triplets: int = 5,
    color_positive=(0, 255, 0, 180),  # Green for positives
    color_anchor=(255, 255, 0, 180),  # Green for positives
    color_negative=(255, 0, 0, 180),  # Red for negatives
) -> Image.Image:
    """
    Visualizes randomly selected triplets of patches on two images by drawing connections and
    displaying close-up patches of the positives, anchors, and negatives.

    Parameters:
        img1 (PIL.Image.Image): The first image (source).
        img2 (PIL.Image.Image): The second image (target).
        anchor_indices (list): List of flat indices for anchor patches in img1.
        positive_indices (list): List of flat indices for positive patches in img2.
        negative_indices (list): List of flat indices for negative patches in img2.
        patch_size (int): The size (width and height) of each patch.
        num_triplets (int): Number of triplets to randomly select and plot.
        color_positive (tuple): RGBA color for positive connections.
        color_negative (tuple): RGBA color for negative connections.

    Returns:
        PIL.Image.Image: An image showing the selected triplet visualizations and close-ups.
    """

    # Convert images to RGBA
    def to_pil(img):
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu()
            if img.dim() == 3:
                img = img.permute(1, 2, 0)
            return Image.fromarray(img.numpy().astype("uint8"))
        return img

    img1_rgba, img2_rgba = to_pil(img1 * 255), to_pil(img2 * 255)
    w1, h1 = img1_rgba.size
    w2, h2 = img2_rgba.size

    # Combine images side by side on a canvas
    canvas_width = w1 + w2
    canvas_height = max(h1, h2)
    canvas = Image.new("RGBA", (canvas_width, canvas_height), (255, 255, 255, 255))
    canvas.paste(img1_rgba, (0, 0))
    canvas.paste(img2_rgba, (w1, 0))

    draw = ImageDraw.Draw(canvas, "RGBA")

    # Determine grid layout for patches
    patches_per_row_img1 = w1 // patch_size
    patches_per_row_img2 = w2 // patch_size

    def get_patch_center(flat_idx, patches_per_row, patch_size):
        """Return the center (x, y) coordinates of a patch given its flat index."""
        row = flat_idx // patches_per_row
        col = flat_idx % patches_per_row
        center_x = col * patch_size + patch_size // 2
        center_y = row * patch_size + patch_size // 2
        return center_x, center_y

    # Randomly select triplet indices to plot
    total_triplets = len(anchor_indices)
    num_to_select = min(num_triplets, total_triplets)
    selected_indices = random.sample(range(total_triplets), num_to_select)

    # Draw connections on the canvas for the selected triplets
    for i, idx in enumerate(selected_indices):
        a_idx = anchor_indices[idx]
        p_idx = positive_indices[idx]
        n_idx = negative_indices[idx]

        # Centers for patches: anchor on img1, positives and negatives on img2.
        anchor_x, anchor_y = get_patch_center(a_idx, patches_per_row_img1, patch_size)
        pos_x, pos_y = get_patch_center(p_idx, patches_per_row_img2, patch_size)
        neg_x, neg_y = get_patch_center(n_idx, patches_per_row_img2, patch_size)

        # Adjust coordinates for img2 patches on the combined canvas.
        pos_x_canvas = pos_x + w1
        neg_x_canvas = neg_x + w1

        # Draw lines from anchor to positive and anchor to negative.
        draw.line(
            [(anchor_x, anchor_y), (pos_x_canvas, pos_y)], fill=color_positive, width=2
        )
        draw.line(
            [(anchor_x, anchor_y), (neg_x_canvas, neg_y)], fill=color_negative, width=2
        )

        # Optionally, draw rectangles around the patches.
        half_size = patch_size // 2
        draw.rectangle(
            [
                anchor_x - half_size,
                anchor_y - half_size,
                anchor_x + half_size,
                anchor_y + half_size,
            ],
            outline=color_anchor,
            width=2,
        )
        draw.rectangle(
            [
                pos_x_canvas - half_size,
                pos_y - half_size,
                pos_x_canvas + half_size,
                pos_y + half_size,
            ],
            outline=color_positive,
            width=2,
        )
        draw.rectangle(
            [
                neg_x_canvas - half_size,
                neg_y - half_size,
                neg_x_canvas + half_size,
                neg_y + half_size,
            ],
            outline=color_negative,
            width=2,
        )

        # Draw triplet index number next to the anchor patch
        number_x = anchor_x - patch_size  # Offset to the right of the anchor patch
        number_y = anchor_y  # Offset above the anchor patch
        draw.text(
            (number_x, number_y),
            str(i + 1),  # Triplet index (1-based)
            fill=color_anchor,  # Text color
            anchor="mm",  # Center alignment
            align="center",
        )

    # Create a new canvas area below to display close-ups of selected triplets
    display_patch_size = int(patch_size * 8)  # enlarge patches for close-up display
    num_selected = len(selected_indices)
    closeup_width = num_selected * display_patch_size
    closeup_height = 3 * display_patch_size  # three rows: positive, anchor, negative

    total_canvas_height = canvas_height + closeup_height
    new_canvas = Image.new(
        "RGBA", (canvas_width, total_canvas_height), (255, 255, 255, 255)
    )
    new_canvas.paste(canvas, (0, 0))  # paste original combined images at the top

    closeup_draw = ImageDraw.Draw(new_canvas, "RGBA")

    def get_patch_box(flat_idx, patches_per_row, patch_size):
        """Return the bounding box of a patch given its flat index."""
        row = flat_idx // patches_per_row
        col = flat_idx % patches_per_row
        x0 = col * patch_size
        y0 = row * patch_size
        return (int(x0), int(y0), int(x0 + patch_size), int(y0 + patch_size))

    # Paste close-up patches for each selected triplet
    for i, idx in enumerate(selected_indices):
        x_offset = i * display_patch_size

        # Retrieve flat indices for the current triplet
        a_idx = anchor_indices[idx]
        p_idx = positive_indices[idx]
        n_idx = negative_indices[idx]

        # Define bounding boxes for cropping patches
        pos_box = get_patch_box(p_idx, patches_per_row_img2, patch_size)
        anchor_box = get_patch_box(a_idx, patches_per_row_img1, patch_size)
        neg_box = get_patch_box(n_idx, patches_per_row_img2, patch_size)

        # Crop and resize patches for close-ups
        pos_patch = img2_rgba.crop(pos_box).resize(
            (display_patch_size, display_patch_size), Image.NEAREST
        )
        anchor_patch = img1_rgba.crop(anchor_box).resize(
            (display_patch_size, display_patch_size), Image.NEAREST
        )
        neg_patch = img2_rgba.crop(neg_box).resize(
            (display_patch_size, display_patch_size), Image.NEAREST
        )

        # Calculate vertical offsets for three rows
        base_y = canvas_height
        pos_y_offset = base_y
        anchor_y_offset = base_y + display_patch_size
        neg_y_offset = base_y + 2 * display_patch_size

        # Paste patches into their respective rows in the close-up area
        new_canvas.paste(pos_patch, (x_offset, pos_y_offset))
        new_canvas.paste(anchor_patch, (x_offset, anchor_y_offset))
        new_canvas.paste(neg_patch, (x_offset, neg_y_offset))

        # Optionally, draw rectangles around the pasted patches for clarity
        half_disp = display_patch_size // 2
        closeup_draw.rectangle(
            [
                x_offset,
                pos_y_offset,
                x_offset + display_patch_size,
                pos_y_offset + display_patch_size,
            ],
            outline=color_positive,
            width=2,
        )
        closeup_draw.rectangle(
            [
                x_offset,
                anchor_y_offset,
                x_offset + display_patch_size,
                anchor_y_offset + display_patch_size,
            ],
            outline=color_anchor,
            width=2,
        )
        closeup_draw.text(
            (x_offset + 10, anchor_y_offset + 10),
            str(i + 1),  # Triplet index (1-based)
            fill=color_anchor,  # Text color
            anchor="mm",  # Center alignment
            align="center",
        )
        closeup_draw.rectangle(
            [
                x_offset,
                neg_y_offset,
                x_offset + display_patch_size,
                neg_y_offset + display_patch_size,
            ],
            outline=color_negative,
            width=2,
        )

    return new_canvas


from typing import Union

import numpy as np
import torch
from PIL import Image


def viewCameraMotion(
    img1: Union[Image.Image, torch.Tensor],
    img2: Union[Image.Image, torch.Tensor],
    pose_6d: Union[
        torch.Tensor, dict
    ] = None,  # [x, y, z, rx, ry, rz] or dict with 'gt' and 'pred' keys
    as_tensor: bool = False,  # Return tensor instead of PIL image
) -> Union[Image.Image, torch.Tensor]:
    def to_pil(img):
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu()
            if img.dim() == 3:
                img = img.permute(1, 2, 0)
            return Image.fromarray(img.numpy().astype("uint8"))
        return img

    def draw_arrow(draw, x, y, dx, dy, color, scale=50, width=2):
        """Draw an arrow starting at (x,y) in direction (dx,dy)"""
        # Normalize direction vector
        length = (dx * dx + dy * dy) ** 0.5
        if length < 1e-8:
            return

        dx, dy = dx / length, dy / length

        # Scale arrow length
        dx, dy = dx * scale, dy * scale

        # Arrow head parameters
        head_length = scale * 0.3
        head_width = scale * 0.2

        # Calculate arrow head points
        angle = math.atan2(dy, dx)
        angle1 = angle + math.pi * 3 / 4
        angle2 = angle - math.pi * 3 / 4

        x2, y2 = x + dx, y + dy
        x3 = x2 + head_length * math.cos(angle1)
        y3 = y2 + head_length * math.sin(angle1)
        x4 = x2 + head_length * math.cos(angle2)
        y4 = y2 + head_length * math.sin(angle2)

        # Draw arrow body and head
        draw.line([(x, y), (x2, y2)], fill=color, width=width)
        draw.line([(x2, y2), (x3, y3)], fill=color, width=width)
        draw.line([(x2, y2), (x4, y4)], fill=color, width=width)

    def draw_curved_arrow(
        draw, center, radius, start_angle, sweep_angle, color, width=2, num_points=50
    ):
        # Generate points along the arc
        points = []
        for i in range(num_points + 1):
            angle = math.radians(start_angle + (sweep_angle * i / num_points))
            x = center[0] + radius * math.cos(angle)
            y = center[1] + radius * math.sin(angle)
            points.append((x, y))
        # Draw the arc
        draw.line(points, fill=color, width=width)

        # Compute arrowhead based on the last segment direction
        if len(points) >= 2:
            x_last, y_last = points[-1]
            x_prev, y_prev = points[-2]
            dx, dy = x_last - x_prev, y_last - y_prev
            arrow_head_length = radius * 0.2  # Adjust relative to radius
            angle = math.atan2(dy, dx)
            # Two lines for the arrowhead at ±30° from the last segment
            angle1 = angle + math.pi / 6
            angle2 = angle - math.pi / 6
            x1 = x_last - arrow_head_length * math.cos(angle1)
            y1 = y_last - arrow_head_length * math.sin(angle1)
            x2 = x_last - arrow_head_length * math.cos(angle2)
            y2 = y_last - arrow_head_length * math.sin(angle2)
            draw.line([(x_last, y_last), (x1, y1)], fill=color, width=width)
            draw.line([(x_last, y_last), (x2, y2)], fill=color, width=width)

    def draw_z_motion(draw, x, y, z, color, radius=20):
        """Draw a circle with dot/cross to indicate z-motion"""
        # Draw circle
        bbox = [x - radius, y - radius, x + radius, y + radius]
        draw.ellipse(bbox, outline=color, width=2)

        # Draw indicator based on direction
        if z > 0:  # Forward motion: draw cross
            cross_size = radius * 0.6
            draw.line(
                [(x - cross_size, y - cross_size), (x + cross_size, y + cross_size)],
                fill=color,
                width=2,
            )
            draw.line(
                [(x - cross_size, y + cross_size), (x + cross_size, y - cross_size)],
                fill=color,
                width=2,
            )
        else:  # Backward motion: draw dot
            dot_radius = radius * 0.3
            draw.ellipse(
                [x - dot_radius, y - dot_radius, x + dot_radius, y + dot_radius],
                fill=color,
            )

    # Convert images to PIL if they are tensors
    img1, img2 = to_pil(img1 * 255), to_pil(img2 * 255)
    w1, h1 = img1.size
    w2, h2 = img2.size

    # Create canvas
    margin = 1
    h_total = h1 + margin
    w_total = w1 + w2 + margin
    canvas = Image.new("RGB", (w_total, h_total))

    # Paste images
    canvas.paste(img1, (0, 0))
    canvas.paste(img2, (w1 + margin, 0))

    draw = ImageDraw.Draw(canvas)

    # Define colors
    gt_color = (0, 255, 0)  # Green for ground truth
    pred_color = (255, 119, 2)  # Cyan for prediction

    gt_color = (0, 255, 0)  # Green for ground truth
    pred_color = (255, 119, 2)  # Cyan for prediction

    if isinstance(pose_6d, dict):
        # Process both ground truth and prediction poses
        poses = {
            "gt": (pose_6d.get("gt"), gt_color),
            "pred": (pose_6d.get("pred"), pred_color),
        }

        # Calculate offset for side-by-side visualization
        # Position arrows at 1/3 and 2/3 of the width
        offsets = {"gt": w1 // 3, "pred": (w1 * 2) // 3}

        # Draw both pose visualizations
        for pose_type, (pose, color) in poses.items():
            if pose is not None:
                pose = pose.detach().cpu()
                center_x = offsets[pose_type]
                center_y = h1 // 2
                translation = pose[:3]  # Get translation vector [x, y, z]
                rotation = pose[3:6]  # Get rotation vector [ex, ey, ez]

                # Draw x-y motion arrow
                draw_arrow(
                    draw,
                    center_x,
                    center_y,
                    -float(translation[0]),
                    -float(translation[1]),
                    color,
                )

                # Draw z-motion indicator
                draw_z_motion(draw, center_x, center_y, -float(translation[2]), color)

                # Draw rotation visualization (new)
                ex, ey, ez = rotation
                center = (w1 // 2, h1 // 2)
                radius = 100  # Adjust as needed
                scale_factor = 100
                # Example parameters (you may need to adjust start angles and directions)
                draw_curved_arrow(
                    draw,
                    center,
                    radius,
                    start_angle=0,
                    sweep_angle=ex * scale_factor,
                    width=4 if pose_type == "gt" else 2,
                    color=(255, 0, 0),
                )
                draw_curved_arrow(
                    draw,
                    center,
                    radius + 10,
                    start_angle=90,
                    sweep_angle=ey * scale_factor,
                    width=4 if pose_type == "gt" else 2,
                    color=(0, 255, 0),
                )
                draw_curved_arrow(
                    draw,
                    center,
                    radius + 20,
                    start_angle=180,
                    sweep_angle=ez * scale_factor,
                    width=4 if pose_type == "gt" else 2,
                    color=(0, 0, 255),
                )

            else:
                # Original single pose visualization with green color
                pose_6d = pose_6d.detach().cpu()
                center_x = w1 // 2
                center_y = h1 // 2
                translation = pose_6d[:3]  # Get translation vector [x, y, z]
                rotation = pose_6d[3:6]  # Get rotation vector [ex, ey, ez]

                # Draw x-y motion arrow
                draw_arrow(
                    draw,
                    center_x,
                    center_y,
                    float(translation[0]),
                    float(translation[1]),
                    gt_color,
                )

                # Draw z-motion indicator
                draw_z_motion(draw, center_x, center_y, float(translation[2]), gt_color)

                # Draw rotation visualization (new)
                ex, ey, ez = rotation
                center = (w1 // 2, h1 // 2)
                radius = 100  # Adjust as needed
                scale_factor = 500
                # Example parameters (you may need to adjust start angles and directions)
                draw_curved_arrow(
                    draw,
                    center,
                    radius,
                    start_angle=0,
                    sweep_angle=ex * scale_factor,
                    width=2,
                    color=(255, 0, 0),
                )
                draw_curved_arrow(
                    draw,
                    center,
                    radius + 10,
                    start_angle=90,
                    sweep_angle=ey * scale_factor,
                    width=2,
                    color=(0, 255, 0),
                )
                draw_curved_arrow(
                    draw,
                    center,
                    radius + 20,
                    start_angle=180,
                    sweep_angle=ez * scale_factor,
                    width=2,
                    color=(0, 0, 255),
                )

    # Return as tensor if requested
    if as_tensor:
        # Convert PIL image to tensor (3xHxW)
        canvas_np = np.array(canvas)
        canvas_tensor = torch.from_numpy(canvas_np).permute(2, 0, 1).float() / 255.0
        return canvas_tensor

    return canvas


import numpy as np
import torch
from PIL import Image


def visualize_patch_matches_pil(
    spatches: torch.Tensor,
    tpatches: torch.Tensor,
    best_src_xy: torch.Tensor,
    best_tgt_xy: torch.Tensor,
    scores: torch.Tensor,
    num_examples: int = 5,
) -> Image.Image:
    """
        Visualizes matched pixels in source and target patches and returns a PIL image.
    S
        Args:
            spatches (torch.Tensor): Source patches of shape (N, 3, H, W).
            tpatches (torch.Tensor): Target patches of shape (N, 3, H, W).
            best_src_xy (torch.Tensor): Tensor of shape (N, 2) with matched (y, x) pixel coordinates in the source patch.
            best_tgt_xy (torch.Tensor): Tensor of shape (N, 2) with matched (y, x) pixel coordinates in the target patch.
            scores (torch.Tensor): Tensor of shape (N,) with scores for each match, used for coloring the dots.
            num_examples (int, optional): Number of random examples to visualize. Defaults to 5.

        Returns:
            Image.Image: The composed PIL image with source patches on top, target patches below, and colored dots.
    """
    import random

    N: int = spatches.shape[0]
    num_examples: int = min(num_examples, N)  # Ensure we don't exceed available samples
    indices: list[int] = random.sample(range(N), num_examples)

    # Normalize scores to [0, 1]
    scores: np.ndarray = scores.cpu().numpy()
    scores_norm: np.ndarray = (scores - scores.min()) / (
        scores.max() - scores.min() + 1e-8
    )

    # Create a red-to-green colormap using matplotlib
    cmap = plt.get_cmap("RdYlGn")
    colors = [cmap(s) for s in scores_norm]
    # Convert RGBA to RGB tuples scaled to 255
    colors_rgb = [(int(r * 255), int(g * 255), int(b * 255)) for r, g, b, _ in colors]

    # Assume all patches have the same size
    _, _, H, W = spatches.shape

    # Create blank image: width = num_examples * W, height = 2 * H
    total_width = num_examples * W
    total_height = 2 * H
    composed_image = Image.new("RGB", (total_width, total_height))

    draw = ImageDraw.Draw(composed_image)

    for idx, i in enumerate(indices):
        # Process source patch
        s_patch = spatches[i].cpu().numpy().transpose(1, 2, 0)  # (H, W, 3)
        s_patch = np.clip(s_patch * 255, 0, 255).astype(np.uint8)
        s_image = Image.fromarray(s_patch)
        composed_image.paste(s_image, (idx * W, 0))

        # Process target patch
        t_patch = tpatches[i].cpu().numpy().transpose(1, 2, 0)
        t_patch = np.clip(t_patch * 255, 0, 255).astype(np.uint8)
        t_image = Image.fromarray(t_patch)
        composed_image.paste(t_image, (idx * W, H))

        # Get matched pixel coordinates
        sx, sy = best_src_xy[i].cpu().numpy()
        tx, ty = best_tgt_xy[i].cpu().numpy()

        # Get color based on score
        color = colors_rgb[idx]

        # Draw dot on source patch
        radius = 1
        draw.ellipse(
            (idx * W + sx - radius, sy - radius, idx * W + sx + radius, sy + radius),
            fill=color,
            outline=None,
        )

        # Draw dot on target patch
        draw.ellipse(
            (
                idx * W + tx - radius,
                H + ty - radius,
                idx * W + tx + radius,
                H + ty + radius,
            ),
            fill=color,
            outline=None,
        )

    return composed_image


def overlay(image, heatmap, alpha=0.4, cmap="plasma"):
    """
    Overlay a heatmap on an image with a given 5opacity.

    :param image: Input image as a numpy array or PIL Image.
    :param heatmap: Heatmap as a numpy array.
    :param alpha: Opacity of the heatmap overlay (between 0 and 1).
    :param cmap: Color map to use for the heatmap.
    :return: The overlaid image as a numpy array.
    """
    # Ensure image is a numpy array
    if isinstance(image, Image.Image):
        image = np.array(image)

    # Normalize the heatmap
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

    # Apply colormap to the heatmap
    colored_heatmap = plt.get_cmap(cmap)(heatmap)[:, :, :3]  # Take RGB channels
    # Resize heatmap to match the image size if necessary
    if heatmap.shape[:2] != image.shape[:2]:
        colored_heatmap = np.array(
            Image.fromarray((colored_heatmap * 255).astype(np.uint8)).resize(
                (image.shape[1], image.shape[2])
            )
        )
        colored_heatmap = colored_heatmap / 255.0
    # Overlay the heatmap on the image
    overlayed_image = (1 - alpha) * image.permute(1, 2, 0) + alpha * colored_heatmap
    # overlayed_image = image.permute(1,2,0) * colored_heatmap.mean(axis=2, keepdims=True)
    overlayed_image = np.clip(overlayed_image * 255, 0, 255)
    return overlayed_image.permute(2, 0, 1)


def sampleinspect(sample: tuple):
    """
    Visualizes a sample from the dataset, including the source and target frames, and prints out the transformation.

    Args:
        sample (tuple): A tuple containing (framestack, Ts2t, paths) where:
            - framestack is a tensor of shape [2, C, H, W] containing source and target frames
            - Ts2t is a tensor of shape [6] or [4, 4] containing the transformation
            - paths is a tuple of (source_path, target_path) containing the paths to the frames
    """
    # Unpack the framestack and Ts2t from the sample

    if len(sample) == 2:
        framestack, Ts2t = sample[0], sample[1]
    else:
        framestack, Ts2t, paths = sample[0], sample[1], sample[2]

    if len(framestack.shape) == 5:
        framestack = framestack[0]
    if len(Ts2t.shape) == 2:
        Ts2t = Ts2t[0]

    fstack = torch.cat([framestack[0], framestack[1]], dim=-1)
    # Assuming source and target are the first and last in the framestack respectively
    if len(sample) == 3:
        source_path, target_path = paths
        print(f"Paths: \n-Source: {source_path}\n-Target: {target_path}")
    # print(f"Frameskip: {self.frameskip}")
    # Create a subplot for source and target
    rgb(fstack)

    # Print transformation information
    print("Transformation:")
    tprint(Ts2t.cpu().numpy())

    # Interpret and print the directional information based on the transformation vector
    direction = [
        "OBJ Right / CAM Left" if Ts2t[0] < 0 else "OBJ Left / CAM Right",
        "OBJ Down / CAM UP" if Ts2t[1] < 0 else "OBJ Up / CAM Down",
        "OBJ Farther / CAM Backward" if Ts2t[2] < 0 else "OBJ Closer / CAM Forward",
    ]
    print(" ".join(direction))
    translation, rax, rang = euler2axang(Ts2t)
    print(f"L\th: {translation.norm():.2f} mm")
    print(f"Rotation: {rang:.2f} deg")


def log_to_rerun(
    frame: int = 0,
    mode: str = "pred",
    is_keyframe: bool = False,
    color: Optional[list] = None,
    framestack: Optional[torch.Tensor] = None,
    depthmap: Optional[torch.Tensor] = None,
    warped: Optional[torch.Tensor] = None,
    K: Optional[torch.Tensor] = None,
    cloud: Optional[torch.Tensor] = None,
    rgb_vec: Optional[torch.Tensor] = None,
    points_match_3d: Optional[torch.Tensor] = None,
    camera_pose: Optional[torch.Tensor] = None,
    with_motion_trail: bool = True,
    base_pose: Optional[torch.Tensor] = None,
    images: Optional[dict] = None,
    metrics: Optional[dict] = None,
    cam_size: float = 0.1,
    **kwargs: Any,
) -> None:
    """
    Enhanced logging function for Rerun that visualizes data conditionally based on provided parameters.

    Args:
        frame (int, optional): Frame number for time sequence logging. Defaults to 0.
        mode (str, optional): Logging mode - "gt" for ground truth, "pred" for prediction, or custom. Defaults to "pred".
        is_keyframe (bool, optional): Whether to log data as a keyframe. Defaults to False.
        color (list, optional): RGB color to use for visualization elements. If None, uses mode-based defaults.
        framestack (Tensor, optional): [B,2,C,H,W] source and target frames.
        depthmap (Tensor, optional): [B,H,W] depth values.
        warped (Tensor, optional): [B,C,H,W] warped images.
        K (Tensor, optional): [B,3,3] camera intrinsics matrix.
        cloud (Tensor, optional): [B,4,N] point cloud coordinates.
        rgb_vec (Tensor, optional): [B,3,N] point cloud colors.
        points_match_3d (Tensor, optional): [B,N,3] matched 3D points.
        camera_pose (Tensor, optional): [B,6] camera pose parameters.
        base_pose (Tensor, optional): Base pose matrix for trajectory calculation.
        images (dict, optional): Dictionary of named images to log, where keys are image names and values are the images.
        metrics (dict, optional): Dictionary of named metrics to log, where keys are metric names and values are scalar values.
        **kwargs: Additional parameters for extensibility.
    """
    # Set the time sequence
    if frame is not None:
        rr.set_time_sequence("frame", frame)
    if base_pose is None and camera_pose is not None:
        base_pose = torch.eye(4)
        if camera_pose.shape[-1] == 6 and camera_pose.ndim > 1:
            base_pose = base_pose.unsqueeze(0)
        if camera_pose.shape[-1] == 4 and camera_pose.ndim > 2:
            base_pose = base_pose.unsqueeze(0)
    # Determine color based on mode
    if color is None:
        if mode == "gt":
            color = [0, 255, 0]  # Green for ground truth
        elif mode == "pred":
            color = [112, 219, 204]  # Teal for predictions
        else:
            color = [255, 0, 0]  # Default to red for custom modes

    # Set keyframe indicator
    kf_indicator = "_KF" if is_keyframe else "_F"

    # Image plane distance and axis length based on keyframe status
    ipd = 2 if is_keyframe else 1
    axl = 1 if is_keyframe else 0

    # Set global coordinate system

    # Log individual elements conditionally

    # Source and target frame logging
    if framestack is not None:
        # Source frame - apply keyframe marker if needed
        source_frame = mark_img(
            framestack[0].detach().cpu().permute(1, 2, 0).numpy(),
            "K" if is_keyframe else None,
        )

        # Target frame - apply keyframe marker if needed
        target_frame = mark_img(
            framestack[1].detach().cpu().permute(1, 2, 0).numpy(),
            "K" if is_keyframe else None,
        )

        rr.log(
            f"/cloud/camera_{mode}/{frame}_{kf_indicator}/camera/source",
            rr.Image(source_frame),
        )

        rr.log(
            f"/cloud/camera_{mode}/{frame}_{kf_indicator}/camera/target",
            rr.Image(target_frame),
        )

    # Depth visualization
    if depthmap is not None:
        depth_visualization = (
            plasma(depthmap).permute(1, 2, 0).cpu().detach().numpy() * 255
        )
        depth_visualization = mark_img(
            depth_visualization, "K" if is_keyframe else None
        )

        rr.log(
            f"/cloud/camera_{mode}/{frame}_{kf_indicator}/camera/depth",
            rr.Image(depth_visualization),
        )

    # Warped image logging
    if warped is not None:
        warped_visualization = warped.permute(1, 2, 0).cpu().detach().numpy() * 255
        warped_visualization = mark_img(
            warped_visualization, "K" if is_keyframe else None
        )

        rr.log(
            f"/cloud/camera_{mode}/{frame}_{kf_indicator}/camera/warped",
            rr.Image(warped_visualization),
        )

    # Camera intrinsics and pose visualization
    if camera_pose is not None:
        # Convert pose to transformation matrix if it's in Euler format
        if camera_pose.shape[-1] == 6:  # Euler format [tx, ty, tz, rx, ry, rz]
            if camera_pose.ndim == 1:
                transform = geometry.euler2mat(camera_pose.detach())
            else:
                transform = geometry.euler2mat(camera_pose[frame, :].detach())
        else:  # Assuming it's already a 4x4 transformation matrix
            transform = camera_pose[frame].detach()

        # Extract translation, axis and angle from transformation matrix
        tr, ax, ang = mat2axang(transform)
        # Log camera with transform
        rr.log(
            f"/cloud/camera_{mode}/{frame}_{kf_indicator}",
            rr.Transform3D(
                rotation=RotationAxisAngle(axis=ax, radians=ang),
                translation=tr,
                axis_length=axl,
            ),
        )
        if with_motion_trail:
            # Visualize motion trail between poses if previous pose is available
            if frame == 0:
                previous_pose = base_pose
            else:
                previous_pose = camera_pose[frame - 1].unsqueeze(0)

            if previous_pose.shape[-1] == 6:  # Euler format
                prev_transform = geometry.euler2mat(previous_pose.detach())
            else:  # Matrix format
                prev_transform = previous_pose.detach()
            # Extract previous translation and current translation for motion trail
            # Shape: (4, 4) - transformation matrices
            # Get the 4x4 transformation matrices for previous and current poses
            prev_transform = prev_transform.detach().cpu()  # Shape: (4, 4)
            curr_transform = transform.detach().cpu()  # Shape: (4, 4)

            # Compute the relative transformation from previous to current
            # T_rel = T_prev^(-1) * T_curr
            rel_transform = torch.matmul(
                torch.inverse(prev_transform), curr_transform
            )  # Shape: (4, 4)

            # Extract the translation component from the relative transformation
            motion_vector = -rel_transform[0, :3, 3]  # Shape: (3,)
            here = torch.tensor([0.0, 0.0, 0.0])
            # Log motion trail between previous and current position
            rr.log(
                f"/cloud/camera_{mode}/{frame}_{kf_indicator}/motion",
                rr.LineStrips3D(
                    torch.cat(
                        [
                            here.unsqueeze(0),
                            motion_vector.unsqueeze(0),
                        ],
                        dim=0,
                    )
                    .cpu()
                    .numpy(),
                    colors=color,
                ),
            )

        # Log camera intrinsics if available
    if K is not None:
        rr.log(
            f"/cloud/camera_{mode}/{frame}_{kf_indicator}/camera",
            rr.Pinhole(
                width=384,
                height=384,
                focal_length=(
                    K[0, 0].detach().cpu(),
                    K[1, 1].detach().cpu(),
                ),
                principal_point=(
                    K[0, 2].detach().cpu(),
                    K[1, 2].detach().cpu(),
                ),
                image_plane_distance=ipd * cam_size,
            ),
        )

    # Point cloud visualization
    if cloud is not None and rgb_vec is not None:
        rr.log(
            f"/cloud/points_{mode}{frame}",
            rr.Points3D(
                cloud[:3, :].permute(1, 0).detach().cpu(),
                colors=rgb_vec[:3, :].permute(1, 0).detach().cpu(),
            ),
        )

    # Matched 3D points visualization
    if points_match_3d is not None:
        # Sample points to avoid visual clutter
        sampled_points = points_match_3d.permute(1, 0)[::3]

        rr.log(
            f"/cloud/matches_{mode}{frame}",
            rr.Points3D(
                sampled_points.detach().cpu(),
                colors=torch.tensor(color),
                radii=0.5,
            ),
        )
        if K is not None:
            rr.log(
                f"/cloud/camlines_{mode}{frame}",
                rr.Arrows3D(vectors=sampled_points.cpu(), colors=color),
            )

    # Additional logging for metrics or errors
    if "metric" in kwargs and "value" in kwargs:
        rr.log(
            f"/{kwargs['metric']}_{mode}",
            rr.Scalar(kwargs["value"]),
        )

    # Log images from images dictionary
    if images is not None and isinstance(images, dict):
        for image_name, image_data in images.items():
            # Apply keyframe marker to the image if needed
            marked_image = mark_img(image_data, "K" if is_keyframe else None)

            rr.log(
                f"/image/{image_name}",
                rr.Image(marked_image),
            )

    # Handle metrics with colors
    if metrics is not None and isinstance(metrics, dict):
        for metric_name, metric_data in metrics.items():
            metric_value = metric_data
            metric_color = [0, 0, 255]  # Default blue color

            # Check if metric_data is a dict with 'value' and 'color' keys
            if isinstance(metric_data, dict) and "value" in metric_data:
                metric_value = metric_data["value"]

                # Extract color if provided
                if "color" in metric_data:
                    color_val = metric_data["color"]

                    # Parse hex color string
                    if isinstance(color_val, str):
                        if color_val.startswith("#"):
                            color_val = color_val[1:]
                        if color_val.startswith("0x"):
                            color_val = color_val[2:]

                        # Handle hex format
                        if len(color_val) == 6:
                            metric_color = [
                                int(color_val[0:2], 16),
                                int(color_val[2:4], 16),
                                int(color_val[4:6], 16),
                            ]

                        elif color_val.startswith("rgb(") and color_val.endswith(")"):
                            rgb_values = color_val[4:-1].split(",")
                            if len(rgb_values) == 3:
                                metric_color = [int(v.strip()) for v in rgb_values]
                    # Handle direct rgb list/tuple format
                    elif isinstance(color_val, (list, tuple)) and len(color_val) >= 3:
                        metric_color = [
                            int(color_val[0]),
                            int(color_val[1]),
                            int(color_val[2]),
                        ]

            # Log the metric with the specified color
            rr.log(
                f"metrics/{metric_name}",
                rr.SeriesLine(color=metric_color, name=metric_name, width=2),
                static=True,
            )
            rr.log(f"metrics/{metric_name}", rr.Scalar(metric_value))


import numpy as np
import torch


def log_rerun_line(
    source: Union[torch.Tensor, np.ndarray, list],
    target: Union[torch.Tensor, np.ndarray, list] = None,
    entity: str = "/line",
    **kwargs: Any,
) -> None:
    """
    Draws a 3D line from the origin to a pose, or between two poses, using Rerun's LineStrips3D.

    Parameters:
        source: The first pose, which can be:
            - 4x4 transformation matrix (numpy.ndarray or torch.Tensor)
            - 3-element translation vector (numpy.ndarray, torch.Tensor, or list)
            - Batched version with batch size 1
        target: (Optional) The second pose, same format as source.
                If None, the line is drawn from the origin to source.
        entity (str): The Rerun entity path to log the line strip.
    """

    def extract_position(pose):
        # Convert lists to numpy arrays
        if isinstance(pose, list):
            pose = np.array(pose)

        # Convert torch tensors to numpy arrays
        if isinstance(pose, torch.Tensor):
            pose = pose.detach().cpu().numpy()

        # Ensure the pose is a numpy array
        if not isinstance(pose, np.ndarray):
            raise TypeError(f"Unsupported type: {type(pose)}")

        # Handle batched inputs
        if pose.ndim == 3:
            if pose.shape[0] != 1:
                raise ValueError("Only batch size of 1 is supported.")
            pose = pose[0]
        elif pose.ndim == 2 and pose.shape[0] == 1:
            pose = pose[0]
        elif pose.ndim == 2 and pose.shape[1] == 1:
            pose = pose[:, 0]

        # Extract position from 4x4 matrix
        if pose.shape == (4, 4):
            return pose[:3, 3]
        # Handle 3-element vectors
        elif pose.shape == (3,):
            return pose
        elif pose.shape == (3, 1):
            return pose[:, 0]
        elif pose.shape == (1, 3):
            return pose[0]
        else:
            raise ValueError(f"Unsupported shape: {pose.shape}")

    # Extract positions
    pos_a = extract_position(source)
    if target is not None:
        pos_b = extract_position(target)
    else:
        pos_b = pos_a
        pos_a = np.zeros(3)

    # Stack positions to form the line
    line = np.stack([pos_a, pos_b], axis=0)

    # Log the line strip to Rerun
    rr.log(entity, rr.LineStrips3D([line], **kwargs))


def log_rerun_camera(
    K: Union[
        torch.Tensor, np.ndarray
    ],  # (3, 3) camera intrinsics matrix, torch.Tensor or np.ndarray
    pose: Union[
        torch.Tensor, np.ndarray
    ],  # (4, 4) camera-to-world pose, torch.Tensor or np.ndarray
    height: int = 384,  # int, image height
    width: int = 384,  # int, image width
    entity: str = "/camera",  # str, rerun entity path
    **kwargs: Any,
) -> None:
    """
    Logs a camera as a Rerun Pinhole object at a specific pose.

    Args:
        K: (3, 3) camera intrinsics matrix (fx, 0, cx; 0, fy, cy; 0, 0, 1)
        height: int, image height
        width: int, image width
        pose: (4, 4) camera-to-world pose matrix
        entity: str, rerun entity path
    """
    # Convert to numpy if torch
    if isinstance(K, torch.Tensor):
        K = K.detach().cpu().numpy()
    if isinstance(pose, torch.Tensor):
        pose = pose.detach().cpu().numpy()

    R = pose[:3, :3]
    t = pose[:3, 3]

    rot = Rotation.from_matrix(R)
    axis_angle = rot.as_rotvec()
    axis = axis_angle / (np.linalg.norm(axis_angle) + 1e-10)  # Normalize axis
    angle = np.linalg.norm(axis_angle)  # Extract angle

    rr.log(
        entity,
        rr.Transform3D(
            translation=t,
            rotation_axis_angle=rr.RotationAxisAngle(axis=axis, radians=angle),
            axis_length=0.1,
        ),
    )

    # Log the pinhole camera
    rr.log(
        entity,
        rr.Pinhole(
            image_from_camera=K,  # (3, 3)
            resolution=(width, height),  # (2,)
            camera_xyz=rr.components.ViewCoordinates.RDF,  # Default: X=Right, Y=Down, Z=Forward
            **kwargs,
        ),
    )


def bundle_to_rerun(
    source_points,
    target_points,
    batch_idx,
    K,
    depth_map_rgb1,
    rgb1,
    rgb2,
    transformations,
    inliers=None,
    sample_fraction=0.1,
    max_correspondences=25,
    point_size=0.05,
    camera_scale=1,
):
    """
    Create an integrated visualization for visual SLAM results using Rerun.
    """
    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)

    # Convert tensors to numpy arrays for processing
    if torch.is_tensor(source_points):
        source_points = source_points.cpu().numpy()
    if torch.is_tensor(target_points):
        target_points = target_points.cpu().numpy()
    if torch.is_tensor(batch_idx):
        batch_idx = batch_idx.cpu().numpy()
    if torch.is_tensor(K):
        K = K.cpu().numpy()
    if torch.is_tensor(depth_map_rgb1):
        depth_map_rgb1 = depth_map_rgb1.cpu().numpy()
    if torch.is_tensor(rgb1):
        rgb1 = rgb1.cpu().numpy()
    if torch.is_tensor(rgb2):
        rgb2 = rgb2.cpu().numpy()
    if torch.is_tensor(transformations):
        transformations = transformations.cpu().numpy()
    if torch.is_tensor(inliers) and inliers is not None:
        inliers = inliers.cpu().numpy()

    # Handle batch dimensions
    if len(depth_map_rgb1.shape) == 2:
        depth_map_rgb1 = depth_map_rgb1[np.newaxis, ...]
    if len(rgb1.shape) == 3:
        rgb1 = rgb1[np.newaxis, ...]
    if len(rgb2.shape) == 3:
        rgb2 = rgb2[np.newaxis, ...]
    if len(K.shape) == 2:
        K = K[np.newaxis, ...]
    if len(transformations.shape) == 2:
        transformations = transformations[np.newaxis, ...]

    # Get unique batch indices
    unique_batches = np.unique(batch_idx)
    b_idx = unique_batches[0]

    # Extract data for the first batch
    batch_mask = batch_idx == b_idx
    src_pts = source_points[batch_mask]
    tgt_pts = target_points[batch_mask]
    K_batch = K[0]
    depth_map = depth_map_rgb1[0]
    rgb1_batch = rgb1[0]
    rgb2_batch = rgb2[0]
    transformation = transformations[0]

    batch_inliers = None
    if inliers is not None:
        batch_inliers = inliers[batch_mask]

    # Get image dimensions
    h, w = depth_map.shape

    # Create point cloud from depth map
    # Create grid of pixels
    y, x = np.mgrid[0:h, 0:w]
    uv = np.stack([x.flatten(), y.flatten()], axis=1)

    # Get depths for all pixels
    depths = depth_map.flatten()

    # Filter valid depths
    valid_mask = depths > 0
    valid_uv = uv[valid_mask]
    valid_depths = depths[valid_mask]

    # Sample subset for visualization
    if sample_fraction < 1.0:
        n_points = valid_uv.shape[0]
        sample_size = int(n_points * sample_fraction)
        indices = np.random.choice(n_points, sample_size, replace=False)
        valid_uv = valid_uv[indices]
        valid_depths = valid_depths[indices]

    # Extract camera intrinsics
    fx, fy = K_batch[0, 0], K_batch[1, 1]
    cx, cy = K_batch[0, 2], K_batch[1, 2]

    # Back-project to 3D points
    points3d = np.zeros((valid_uv.shape[0], 3))
    points3d[:, 2] = valid_depths
    points3d[:, 0] = (valid_uv[:, 0] - cx) * valid_depths / fx
    points3d[:, 1] = (valid_uv[:, 1] - cy) * valid_depths / fy

    # Get colors for 3D points
    # Ensure rgb1_batch is in HWC format
    if len(rgb1_batch.shape) == 3 and rgb1_batch.shape[0] == 3:
        rgb1_batch = np.transpose(rgb1_batch, (1, 2, 0))

    colors = np.zeros((valid_uv.shape[0], 3))
    u = valid_uv[:, 0].astype(int)
    v = valid_uv[:, 1].astype(int)
    valid_idx = (v >= 0) & (v < h) & (u >= 0) & (u < w)
    colors[valid_idx] = rgb1_batch[v[valid_idx], u[valid_idx]]

    # Filter points with valid colors
    valid_color_mask = ~np.all(colors == 0, axis=1)
    points3d = points3d[valid_color_mask]
    colors = colors[valid_color_mask]

    # Convert colors to 0-1 range if they're in 0-255
    if np.max(colors) > 1.0:
        colors = colors / 255.0

    # Log 3D point cloud with increased point size
    rr.log(
        "slam/point_cloud",
        rr.Points3D(
            positions=points3d,
            colors=colors,
            radii=point_size,  # Increased point size
        ),
    )

    # Camera poses
    # Camera 1 (source)
    rr.log(
        "slam/camera1",
        rr.Transform3D(
            translation=[0, 0, 0],
            rotation_axis_angle=rr.RotationAxisAngle(axis=[0, 0, 1], radians=0),
            axis_length=camera_scale,
        ),
    )

    # Add pinhole projection for camera 1
    rr.log(
        "slam/camera1",
        rr.Pinhole(
            width=384,
            height=384,
            focal_length=(
                K_batch[0, 0],
                K_batch[1, 1],
            ),
            principal_point=(
                K_batch[0, 2],
                K_batch[1, 2],
            ),
            image_plane_distance=camera_scale,
        ),
    )

    # Camera 2 (target)
    # Extract rotation and translation from transformation matrix
    R = transformation[:3, :3]
    t = transformation[:3, 3]

    # Convert rotation matrix to axis-angle
    from scipy.spatial.transform import Rotation

    rot = Rotation.from_matrix(R)
    axis_angle = rot.as_rotvec()
    axis = axis_angle / (np.linalg.norm(axis_angle) + 1e-10)  # Normalize axis
    angle = np.linalg.norm(axis_angle)  # Extract angle

    rr.log(
        "slam/camera2",
        rr.Transform3D(
            translation=t,
            rotation_axis_angle=rr.RotationAxisAngle(axis=axis, radians=angle),
            axis_length=camera_scale,
        ),
    )

    # Add pinhole projection for camera 2
    rr.log(
        "slam/camera2",
        rr.Pinhole(
            width=384,
            height=384,
            focal_length=(
                K_batch[0, 0],
                K_batch[1, 1],
            ),
            principal_point=(
                K_batch[0, 2],
                K_batch[1, 2],
            ),
            image_plane_distance=camera_scale,
        ),
    )

    # Process source points to find correspondences
    src_pts_int = np.round(src_pts).astype(int)

    # Ensure points are within bounds
    valid_mask = (
        (src_pts_int[:, 0] >= 0)
        & (src_pts_int[:, 0] < w)
        & (src_pts_int[:, 1] >= 0)
        & (src_pts_int[:, 1] < h)
    )

    # Filter valid points
    valid_src_pts = src_pts[valid_mask]
    valid_tgt_pts = tgt_pts[valid_mask]
    valid_src_pts_int = src_pts_int[valid_mask]

    valid_inliers = None
    if batch_inliers is not None:
        valid_inliers = batch_inliers[valid_mask]

    # Get depths for valid source points
    valid_depths = np.zeros(len(valid_src_pts_int))
    for i, (x, y) in enumerate(valid_src_pts_int):
        valid_depths[i] = depth_map[y, x]

    # Filter points with valid depth
    non_zero_mask = valid_depths > 0
    final_src_pts = valid_src_pts[non_zero_mask]
    final_tgt_pts = valid_tgt_pts[non_zero_mask]
    final_depths = valid_depths[non_zero_mask]

    # Select points for visualization
    if valid_inliers is not None:
        final_inliers = valid_inliers[non_zero_mask]
        inlier_indices = np.where(final_inliers)[0]
        if len(inlier_indices) > max_correspondences:
            selected_indices = np.random.choice(
                inlier_indices, max_correspondences, replace=False
            )
        else:
            selected_indices = inlier_indices
    else:
        if len(final_src_pts) > max_correspondences:
            selected_indices = np.random.choice(
                len(final_src_pts), max_correspondences, replace=False
            )
        else:
            selected_indices = np.arange(len(final_src_pts))

    # Create 3D points for selected source points
    selected_src_pts = final_src_pts[selected_indices]
    selected_tgt_pts = final_tgt_pts[selected_indices]
    selected_depths = final_depths[selected_indices]

    # Compute 3D positions for source points
    src_points3d = np.zeros((len(selected_indices), 3))
    src_points3d[:, 2] = selected_depths
    src_points3d[:, 0] = (selected_src_pts[:, 0] - cx) * selected_depths / fx
    src_points3d[:, 1] = (selected_src_pts[:, 1] - cy) * selected_depths / fy

    # Log source points as 3D markers with increased size
    rr.log(
        "slam/source_points",
        rr.Points3D(
            positions=src_points3d,
            colors=[1.0, 0.0, 0.0],  # Red
            radii=0.15,  # Increased size for source points
        ),
    )

    # Project target points to 3D
    # Define z-plane in target camera space
    z_plane = camera_scale

    # Compute corners of image plane in target camera coordinates
    plane_corners_cam2 = np.array(
        [
            [(0 - cx) * z_plane / fx, (0 - cy) * z_plane / fy, z_plane],
            [(w - 1 - cx) * z_plane / fx, (0 - cy) * z_plane / fy, z_plane],
            [(w - 1 - cx) * z_plane / fx, (h - 1 - cy) * z_plane / fy, z_plane],
            [(0 - cx) * z_plane / fx, (h - 1 - cy) * z_plane / fy, z_plane],
        ]
    )

    # Transform to world coordinates
    plane_corners_world = np.zeros_like(plane_corners_cam2)
    for i, corner in enumerate(plane_corners_cam2):
        plane_corners_world[i] = R @ corner + t

    # Compute 3D positions for target points on the image plane
    tgt_points3d = np.ones((len(selected_indices), 3))
    for i, point in enumerate(selected_tgt_pts):
        # Normalize coordinates to [0,1]
        u_norm = point[0] / (w - 1)
        v_norm = point[1] / (h - 1)

        # Bilinear interpolation
        pos = (
            (1 - v_norm) * (1 - u_norm) * plane_corners_world[0]
            + (1 - v_norm) * u_norm * plane_corners_world[1]
            + v_norm * u_norm * plane_corners_world[2]
            + v_norm * (1 - u_norm) * plane_corners_world[3]
        )

        tgt_points3d[i] = pos

    # Log target points as 3D markers
    # rr.log(
    #     "slam/target_points",
    #     rr.Points3D(
    #         positions=tgt_points3d,
    #         colors=[0.0, 0.0, 1.0],  # Blue
    #         radii=0.15,  # Match source point size
    #     ),
    # )

    # Create line segments connecting source to target points
    for i in range(len(selected_indices)):
        start_point = src_points3d[i]
        end_point = tgt_points3d[i]

        # Log correspondence as line segment
        rr.log(
            f"slam/correspondence/{i}",
            rr.LineStrips3D(
                [np.stack([start_point, end_point])],
                colors=[1.0, 0.5, 0.0],  # Orange
                radii=0.02,  # Slightly thicker lines
            ),
        )

        # Draw lines from source points to camera center (origin)
        camera_center = np.zeros(3)  # Camera 1 is at origin
        # Draw lines from source points to camera2 center
        camera2_center = t  # Camera 2 center is at translation vector t
        rr.log(
            f"slam/source_to_target/{i}",
            rr.LineStrips3D(
                [np.stack([start_point, camera2_center])],  # Shape: (2, 3)
                colors=[0.0, 1.0, 0.5],  # Green-cyan
                radii=0.015,  # Slightly thinner than correspondence lines
            ),
        )

    # Ensure images are in correct format for visualization
    # For Camera 1
    if len(rgb1_batch.shape) == 3 and rgb1_batch.shape[0] == 3:
        rgb1_batch = np.transpose(rgb1_batch, (1, 2, 0))
    # For Camera 2
    if len(rgb2_batch.shape) == 3 and rgb2_batch.shape[0] == 3:
        rgb2_batch = np.transpose(rgb2_batch, (1, 2, 0))

    # Create copies of the images for drawing
    rgb1_with_points = rgb1_batch.copy()
    rgb2_with_points = rgb2_batch.copy()

    # Draw source points on image 1
    for pt in selected_src_pts:
        x, y = int(round(pt[0])), int(round(pt[1]))
        if 0 <= x < w and 0 <= y < h:
            # Draw a small red circle (5x5 pixels)
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    if dx * dx + dy * dy <= 4:  # Circle with radius 2
                        py, px = y + dy, x + dx
                        if 0 <= px < w and 0 <= py < h:
                            rgb1_with_points[py, px] = [1.0, 0.0, 0.0]  # Red

    # Draw target points on image 2
    for pt in selected_tgt_pts:
        x, y = int(round(pt[0])), int(round(pt[1]))
        if 0 <= x < w and 0 <= y < h:
            # Draw a small blue circle (5x5 pixels)
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    if dx * dx + dy * dy <= 4:  # Circle with radius 2
                        py, px = y + dy, x + dx
                        if 0 <= px < w and 0 <= py < h:
                            rgb2_with_points[py, px] = [0.0, 0.0, 1.0]  # Blue

    # Log images with points to separate entities
    rr.log("slam/camera1/image", rr.Image(rgb1_with_points))
    rr.log("slam/camera2/image", rr.Image(rgb2_with_points))


def mark_img(
    image: Union[torch.Tensor, np.ndarray, Image.Image], marker: Optional[str] = None
) -> Union[torch.Tensor, np.ndarray, Image.Image]:
    """
    Add a marker to the top-left corner o an image.

    Args:
        image (Union[torch.Tensor, np.ndarray, Image.Image]): Input image to mark
        marker (Optional[str]): Text to display on the image. If None, no marker is added.

    Returns:
        Union[torch.Tensor, np.ndarray, Image.Image]: Image with or without marker, same type as input
    """
    if marker is None:
        return image

    # Store original type for later conversion
    original_type: type = type(image)
    is_tensor: bool = isinstance(image, torch.Tensor)
    is_pil: bool = hasattr(image, "mode")  # Check if PIL Image
    original_shape: Optional[Tuple[int, ...]] = None

    # Convert PIL Image to numpy array if necessary
    if is_pil:
        image = np.array(image)

    # Handle torch tensor conversion to numpy
    if is_tensor:
        original_shape = image.shape
        # Convert torch tensor to numpy array
        if len(image.shape) == 3 and image.shape[0] in [1, 3]:  # CHW format
            image = image.permute(1, 2, 0).detach().cpu().numpy()
        else:  # Already in HWC format
            image = image.detach().cpu().numpy()

    # Handle case where image has values in 0-1 range vs 0-255 range
    if image.max() <= 1.0:
        image = image.copy() * 255
    else:
        image = image.copy()

    # Ensure image is in uint8 format for drawing
    image = image.astype(np.uint8)

    # Get dimensions for text positioning and sizing
    height: int
    width: int
    height, width = image.shape[:2]
    font_scale: float = max(0.5, min(width, height) / 500)  # Scale based on image size
    font_thickness: int = max(1, int(font_scale * 2))
    text_size: Tuple[int, int] = cv2.getTextSize(
        marker, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
    )[0]

    # Calculate position (top-left corner with small padding)
    x: int = 10
    y: int = text_size[1] + 10

    # Create a semi-transparent background for the text
    overlay: np.ndarray = image.copy()
    bg_padding: int = 5
    cv2.rectangle(
        overlay,
        (x - bg_padding, y - text_size[1] - bg_padding),
        (x + text_size[0] + bg_padding, y + bg_padding),
        (0, 0, 0),
        -1,
    )

    # Apply the rectangle with alpha blending
    alpha: float = 0.6
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    # Draw marker text in white
    cv2.putText(
        image,
        marker,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        font_thickness,
    )

    # Convert back to original format
    if is_tensor:
        if len(original_shape) == 3 and original_shape[0] in [1, 3]:  # Original was CHW
            image = torch.from_numpy(image).float() / 255.0
            image = image.permute(2, 0, 1)
        else:  # Original was HWC
            image = torch.from_numpy(image).float() / 255.0
    elif is_pil:
        from PIL import Image

        image = Image.fromarray(image)

    return image


def blackout_pca(tensor: torch.Tensor) -> torch.Tensor:
    """
    Replace most frequent color in tensor with black

    Args:
        tensor: Tensor of shape [3, H, W] containing RGB values

    Returns:
        Tensor of same shape with most frequent color replaced by black
    """
    # Reshape to [3, N] and transpose to get N RGB triplets
    colors = tensor.reshape(3, -1).T  # Shape: [N, 3]

    # Find most frequent color
    unique_colors, counts = torch.unique(colors, dim=0, return_counts=True)
    most_frequent_idx = counts.argmax()
    black_pcad = unique_colors[most_frequent_idx]

    # Create mask for pixels matching black_pcad color
    black_mask = (tensor == black_pcad.reshape(3, 1, 1)).all(dim=0)  # Shape: [H, W]

    # Replace black_pcad with pure black (0,0,0) using vectorized operations
    return torch.where(
        black_mask.unsqueeze(0).expand(3, -1, -1),  # Shape: [3, H, W]
        torch.zeros_like(tensor),
        tensor,
    )


def highlights_rerun_show(
    highlight_result,
    batch,
    resolution=(448, 448),
    width=800,
    height=800,
    show_normals=False,
    show_light_dir=False,
    show_view_dir=False,
):
    """
    Visualize highlight and geometry info with rerun, including highlight 3D points.

    Args:
        highlight_result (dict): Outputs from PolarHighlighter.
        batch (dict): Associated batch. Uses batch["diffuse"] for colors.
        resolution (tuple): Output resolution (height, width) for camera.
        width (int): Width of rerun viewer.
        height (int): Height of rerun viewer.
        show_normals (bool): Whether to plot surface normals as 3D arrows.
        show_light_dir (bool): Whether to plot light direction arrows.
        show_view_dir (bool): Whether to plot view direction arrows.
    """
    import rerun as rr
    import torch

    rr.init("polar_highlighter")
    rr.log("/", rr.ViewCoordinates.RDF)
    rr.log(
        "/cam",
        rr.Pinhole(
            image_from_camera=highlight_result["intrinsic"][0].cpu().numpy(),  # (3, 3)
            resolution=resolution,  # (2,)
            camera_xyz=rr.components.ViewCoordinates.RDF,  # Default: X=Right, Y=Down, Z=Forward
        ),
    )
    # Light position and color (white)
    light_camera_rf = highlight_result["light_pos"][0] * torch.tensor(
        [-1, -1, 1]
    ).to(highlight_result["light_pos"].device)
    rr.log(
        "/light",
        rr.Points3D(
            positions=light_camera_rf.cpu().numpy().reshape(1, 3),
            colors=torch.tensor([1.0, 1.0, 1.0]).cpu().numpy().reshape(1, 3),
            radii=0.005,
        ),
    )
    # Point cloud of (x, y, z) points colored by diffuse RGB
    rr.log(
        "/pcloud",
        rr.Points3D(
            positions=highlight_result["pcloud"]
            .view(1, 3, -1)[0]
            .permute(1, 0)
            .cpu()
            .numpy(),
            colors=batch["diffuse"].view(1, 3, -1)[0].permute(1, 0).cpu().numpy(),
        ),
    )
    # Constants for visualization clarity
    CLOUD_DOWNS = 1
    ARROW_LENGTH = 0.1
    ARROW_RADIUS = 0.005

    cloud = highlight_result["pcloud"][:, :, ::CLOUD_DOWNS, ::CLOUD_DOWNS]

    # ---- Add highlight as white 3D points ----
    # Assume highlight_result["highlight"] is [B, 1, H, W] or [B, H, W]
    highlight = highlight_result["highlight"]
    pcloud = highlight_result["pcloud"]

    # Get pcloud and highlight at full resolution, batch 0 only
    if highlight.dim() == 4:
        # [B, 1, H, W] --> [1, H, W]
        highlight_map = highlight[0, 0] if highlight.shape[1] == 1 else highlight[0]
    elif highlight.dim() == 3:
        highlight_map = highlight[
            0
        ]  # [H, W] (or [C, H, W], but not likely for highlight)
    else:
        raise ValueError("Unexpected highlight shape")

    # Downsample to match cloud (for visualization) if needed
    # (Optional, but if visualizing only downsampled pcloud, use same here)
    pcloud_vis = pcloud[0, :, ::CLOUD_DOWNS, ::CLOUD_DOWNS]
    highlight_vis = highlight_map[::CLOUD_DOWNS, ::CLOUD_DOWNS]

    # Find points where highlight is at least 0.5
    mask = highlight_vis >= 0.5
    if mask.sum() > 0:
        # Get 3D positions at those indices: [3, H, W] -> [mask.sum(), 3]
        pos = pcloud_vis[:, mask].T.cpu().numpy()
        # White color for all highlight points
        colors = torch.ones((pos.shape[0], 3), dtype=torch.float32).cpu().numpy()
        # Use intensity to set radius: 0.05 ... 0.10 (twice normal size)
        highlight_val = highlight_vis[mask].cpu().numpy()
        radii = 0.005 + 0.005 * ((highlight_val - 0.5) / 0.5).clip(
            0, 1
        )  # [0.05, 0.10] for [0.5, 1]

        rr.log(
            "/highlights_points",
            rr.Points3D(
                positions=pos,
                colors=colors,
                radii=radii,
            ),
        )

    if show_normals:
        # Normals as arrows, colored using HSV-like mapping for direction
        normals = highlight_result["normals"][:, :, ::CLOUD_DOWNS, ::CLOUD_DOWNS]
        rr.log(
            "/normals",
            rr.Arrows3D(
                origins=cloud.reshape(1, 3, -1)[0].permute(1, 0).cpu().numpy(),
                vectors=-normals.reshape(1, 3, -1)[0].permute(1, 0).cpu().numpy()
                * ARROW_LENGTH,
                colors=torch.nn.functional.normalize(
                    normals.reshape(1, 3, -1)[0].permute(1, 0), dim=1
                )
                .cpu()
                .numpy()
                * 0.5
                + 0.5,  # HSV-like coloring
                radii=ARROW_RADIUS,
            ),
        )
    if show_light_dir:
        # Light direction arrows (white)
        light_dir = highlight_result["light_dir"][:, :, ::CLOUD_DOWNS, ::CLOUD_DOWNS]
        rr.log(
            "/light_dir",
            rr.Arrows3D(
                origins=cloud.reshape(1, 3, -1)[0].permute(1, 0).cpu().numpy(),
                vectors=light_dir.reshape(1, 3, -1)[0].permute(1, 0).cpu().numpy()
                * ARROW_LENGTH,
                colors=torch.ones_like(light_dir.reshape(1, 3, -1)[0].permute(1, 0))
                .cpu()
                .numpy(),
                radii=ARROW_RADIUS,
            ),
        )
    if show_view_dir:
        # View direction arrows (red)
        view_dir = highlight_result["view_dir"][:, :, ::CLOUD_DOWNS, ::CLOUD_DOWNS]
        rr.log(
            "/view_dir",
            rr.Arrows3D(
                origins=cloud.reshape(1, 3, -1)[0].permute(1, 0).cpu().numpy(),
                vectors=-view_dir.reshape(1, 3, -1)[0].permute(1, 0).cpu().numpy()
                * ARROW_LENGTH,
                colors=torch.tensor([1.0, 0.0, 0.0])
                .expand_as(view_dir.reshape(1, 3, -1)[0].permute(1, 0))
                .cpu()
                .numpy(),
                radii=ARROW_RADIUS,
            ),
        )
    rr.notebook_show(width=width, height=height)
