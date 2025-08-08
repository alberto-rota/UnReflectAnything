# -------------------------------------------------------------------------------------------------#

"""Copyright (c) 2024 Asensus Surgical"""

""" Code Developed by: Alberto Rota """
""" Supervision: Uriya Levy, Gal Weizman, Stefano Pomati """

# -------------------------------------------------------------------------------------------------#

from typing import Any
import numpy as np
from rich import print as nativeprint
import torch
from PIL import Image
import re
import wandb


def gt2string(tensor: torch.Tensor) -> str:
    """
    Converts a tensor into a string representation.

    Parameters:
    tensor (torch.Tensor): The tensor to be converted.

    Returns:
    str: The string representation of the tensor.
    """
    append = ["X", "Y", "Z", "R", "P", "Y"]
    return " ".join([f"{a}:{x.item():+.4f}  " for x, a in zip(tensor, append)])


def pred2string(pos: torch.Tensor, rot: torch.Tensor) -> str:
    """
    Converts position and rotation tensors into a string representation.

    Parameters:
    pos (torch.Tensor): The position tensor to be converted.
    rot (torch.Tensor): The rotation tensor to be converted.

    Returns:
    str: The string representation of the position and rotation tensors.
    """
    tensor = torch.cat([pos, rot], dim=0)
    append = ["X", "Y", "Z", "R", "P", "Y"]
    return " ".join([f"{a}:{x.item():+.4f}  " for x, a in zip(tensor, append)])


def tprint(args, shape=True, dtype=False, device=False, grad_fn=False, **kwargs):

    sep = kwargs.get("sep", " ")
    end = kwargs.get("end", "\n")
    output = []
    np.set_printoptions(precision=4, suppress=True)

    def tensor_to_string(tensor):
        return str(tensor.cpu().detach().numpy())

    for arg in args:
        if isinstance(arg, torch.Tensor):

            infos = ""
            if shape:
                infos += f"Shape: {tuple(arg.shape)}"
            if dtype:
                infos += f"Dtype {str(arg.dtype).split('torch.')[1]}"
            if device:
                infos += f"Device: {arg.device}"
            if grad_fn:
                infos += (
                    f"Grad_fn: {arg.grad_fn}" if arg.grad_fn is not None else "NOGRAD"
                )
            if shape or dtype or device or grad_fn:
                infos += "\n"
            infos += tensor_to_string(arg)

            output.append(infos)
        elif (isinstance(arg, list) or isinstance(arg, tuple)) and all(
            isinstance(x, torch.Tensor) for x in arg
        ):
            print(f"{len(arg)} elements:", [x.shape for x in arg])
        else:
            output.append(str(arg))
    nativeprint(sep.join(output), end=end)


def print(*args: Any, **kwargs: Any) -> None:
    """
    Custom print function to handle both tensors and regular objects.

    If the argument is a torch.Tensor, use lovely_tensors to print.
    Otherwise, use the rich print function.

    Args:
        *args (Any): The arguments to print.
        **kwargs (Any): Additional keyword arguments passed to the print function.
    """
    # Check if any argument is a torch.Tensor
    if any(isinstance(arg, torch.Tensor) for arg in args):
        tprint(args, **kwargs)
    else:
        # Use the original print function for non-tensor arguments
        nativeprint(*args, **kwargs)


def align(input_str, max_length, alignment):
    if alignment == "left":
        # Trim the string from the right side if it exceeds the max_length
        input_str = input_str[:max_length]
        return input_str.ljust(max_length)
    elif alignment == "right":
        # Trim the string from the left side if it exceeds the max_length
        input_str = input_str[-max_length:]
        return input_str.rjust(max_length)
    elif alignment == "center":
        # For center alignment, take characters from the middle if trimming is needed
        if len(input_str) > max_length:
            start = (len(input_str) - max_length) // 2
            input_str = input_str[start : start + max_length]
        return input_str.center(max_length)
    else:
        raise ValueError("Alignment must be 'left', 'right', or 'center'.")


def RdGr(value):
    # Ensure the value is between 0 and 1
    value = max(0.0, min(1.0, value))

    # Map the value to a red (0) to green (1) gradient
    red = int(255 * (1 - value))
    green = int(255 * value)

    # Create a hex color code
    color = f"#{red:02x}{green:02x}00"

    # Create the text object with the specified color
    return f"[{color}]{(value*100):.2f}[/{color}]"


def metrics_for_wandb(metrics_dict, prefix_str, separator="/"):
    """
    Process dictionary keys to include a prefix if they don't already contain the separator.
    Skip any key-value pairs where the value is None.

    Args:
        metrics_dict (dict): The dictionary whose keys need to be processed
        prefix_str (str): The string to prepend to keys
        separator (str, optional): The separator character. Defaults to "/".

    Returns:
        dict: A new dictionary with the processed keys
    """
    result = {}

    for key, value in metrics_dict.items():
        # Skip None values
        if value is None:
            continue

        if separator in key:
            # Key already contains separator, keep it unchanged
            new_key = key
        else:
            # Add prefix to the key
            new_key = f"{prefix_str}{separator}{key}"

        if isinstance(value, Image.Image):
            # If the value is an Image, convert it to wandb.Image
            value = wandb.Image(value)
        result[new_key] = value

    return result


def strip_rich_markup(text):
    """
    Strip Rich markup tags from a string.

    Args:
        text (str): Text with Rich markup

    Returns:
        str: Text with Rich markup tags removed
    """
    # Remove [tag]...[/tag] pairs
    text = re.sub(r"\[([^\]]+)\](.*?)\[/\1\]", r"\2", text)

    # Remove remaining single tags like [tag]
    text = re.sub(r"\[([^\]]+)\]", "", text)

    return text
