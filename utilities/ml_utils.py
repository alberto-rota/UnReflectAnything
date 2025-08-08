# -------------------------------------------------------------------------------------------------#

""" Copyright (c) 2024 Asensus Surgical """

""" Code Developed by: Alberto Rota """
""" Supervision: Uriya Levy, Gal Weizman, Stefano Pomati """

# -------------------------------------------------------------------------------------------------#

import torch
import torchvision


def dinotransform(height, width) -> torchvision.transforms.Compose:
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
            torchvision.transforms.Resize((height, width)),
        ]
    )


def dinotransform_inv(inp="rgbvec") -> torchvision.transforms.Compose:
    """
    Define the postprocessing transformation for DINOv2

    Returns:
    torchvision.transforms.Compose: A composition of inverse torchvision transformations.
    """
    if inp == "image":
        return torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((512, 640)),
                torchvision.transforms.Normalize(
                    mean=(0.0, 0.0, 0.0), std=(1 / 58.395, 1 / 57.12, 1 / 57.375)
                ),
                torchvision.transforms.Normalize(
                    mean=(-123.675, -116.28, -103.53), std=(1.0, 1.0, 1.0)
                ),
                lambda x: 1 / 255.0 * x,
            ]
        )
    elif inp == "rbgvec":
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


def resizeTransform(height=384, width=384) -> torchvision.transforms.Compose:
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


# Compute the skew-symmetric matrix of t for each batch


def hessian_trace(model, loss_fn, data, target):
    """
    Estimate the trace of the Hessian matrix for a given model and loss function.

    This function estimates the trace of the Hessian matrix of the loss function
    with respect to the model's parameters. It uses Hutchinson's estimator, which
    is an efficient way to estimate the trace without computing the full Hessian.
    The method is useful for analyzing the curvature of the loss surface.

    Parameters:
    - model (torch.nn.Module): The model for which the Hessian trace is computed.
    - loss_fn (callable): The loss function used during the model's training.
    - data (torch.Tensor): The input data batch.
    - target (torch.Tensor): The target outputs for the given input data.

    Returns:
    - float: The estimated trace of the Hessian matrix.

    """

    # Reset gradients to zero to avoid accumulation from previous operations
    model.zero_grad()

    # Forward pass: compute the model's output given the input data
    output = model(data)

    # Compute the loss using the model's output and the target values
    loss = loss_fn(output, target)

    # Filter model parameters to only those that require gradients
    params_with_grad = [p for p in model.parameters() if p.requires_grad]

    # Compute gradients of the loss with respect to model parameters
    grads = torch.autograd.grad(loss, params_with_grad, create_graph=True)

    # Flatten the gradients to a single vector (grad_vector)
    grad_vector = torch.cat([grad.reshape(-1) for grad in grads if grad is not None])

    # Generate a random vector (v) with the same shape as grad_vector
    v = torch.randn(grad_vector.shape, device=grad_vector.device)

    # Compute the Hessian-vector product (Hv) for the gradient vector
    Hv = torch.autograd.grad(grad_vector @ v, params_with_grad)

    # Flatten the Hessian-vector product to a single vector (Hv_vector)
    Hv_vector = torch.cat([hv.reshape(-1) for hv in Hv]).detach()

    # Estimate the trace of the Hessian using Hutchinson's estimator
    trace_estimate = v @ Hv_vector

    # Return the estimated trace as a scalar value
    return trace_estimate.item()


def spatial_attention_maps(
    attention_output: torch.Tensor, reference_patch: int = 0, patch_size: int = 14
) -> torch.Tensor:
    """
    Generate spatial attention maps from the given attention output.

    Parameters:
    attention_output (torch.Tensor): The output tensor from the attention layer with shape [batch_size, num_heads, num_patches, num_patches].
    reference_patch (int): The reference patch index to use for extracting attention. Default is 0.
    patch_size (int): The size to which the attention map patches will be upsampled. Default is 14.

    Returns:
    torch.Tensor: The upsampled attention maps with shape [num_heads, patch_size * h_featmap, patch_size * w_featmap].
    """
    # Extract attention for the reference patch and reshape
    attentions = attention_output[reference_patch, 1:]

    # For example, if attention_output shape is [batch_size, num_heads, num_patches, num_patches]
    # and if num_patches is square of some integer then
    w_featmap = h_featmap = int((attention_output.shape[-1] - 1) ** 0.5)

    # Reshape attentions to spatial dimensions
    attentions = attentions.reshape(
        nh, w_featmap, h_featmap
    )  # --> [num_heads, h_featmap, w_featmap]

    # Upsample the attention maps to the desired patch size
    attentions = nn.functional.interpolate(
        attentions.unsqueeze(0), scale_factor=patch_size, mode="bicubic"
    )[0]
    # --> [num_heads, patch_size * h_featmap, patch_size * w_featmap]

    return attentions


def perc_req_grad(model):
    return (
        sum(param.requires_grad for param in model.parameters())
        / len(list(model.parameters()))
        * 100
    )


def perc_grad_finite(model):
    return (
        sum(
            param.grad is not None and torch.isfinite(param.grad).all()
            for param in model.parameters()
            if param.requires_grad
        )
        / sum(param.requires_grad for param in model.parameters())
        * 100
    )
