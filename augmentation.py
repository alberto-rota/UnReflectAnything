import torchvision.transforms as tvt
import torch
from utilities.rotations import (
    euler2mat,
    mat2euler,
)  # Assuming utilities.py contains these functions


def color_augmentation(
    framestack: torch.Tensor, camera_pose: list, p: float = 0.2, target_only=False
) -> tuple:
    """
    Applies a series of color augmentations to a given tensor of images, with a set probability.

    Parameters:
    - framestack (torch.Tensor): A tensor representing a stack of images.
    - tr (list): A list representing transformation parameters, not modified by this function.
    - p (float): Probability of applying sharpness adjustments. Default is 0.2.

    Returns:
    - tuple[torch.Tensor, list]: A tuple containing the augmented image tensor and the unchanged transformation list.
    """
    # Define a series of color augmentations

    def identity_transform(x):
        return x

    applyjitter = torch.rand(1) < p
    applyblur = torch.rand(1) < p
    applyequalize = torch.rand(1) < p
    augmentation_composition = tvt.Compose(
        [
            (
                tvt.ColorJitter(
                    brightness=(0.8, 1.2),
                    contrast=(0.5, 1.5),
                    saturation=(0.8, 1.5),
                    hue=(-0.05, 0.05),
                )
                if applyjitter
                else tvt.Lambda(identity_transform)
            ),
            tvt.RandomAdjustSharpness(sharpness_factor=0, p=p),
            tvt.RandomAdjustSharpness(sharpness_factor=2, p=p),
            (
                tvt.GaussianBlur(kernel_size=3, sigma=(0.01, 10))
                if applyblur
                else tvt.Lambda(identity_transform)
            ),
            # (
            #     tvt.RandomEqualize(p=applyequalize)
            #     if applyequalize
            #     else tvt.Lambda(identity_transform)
            # ),
        ]
    )

    if target_only:
        if framestack.ndim == 5:
            source, target = framestack[:, 0], framestack[:, 1]
        else:
            source, target = framestack[0], framestack[1]

        target_augmented = augmentation_composition(target)
    else:
        framestack = augmentation_composition(
            framestack
        )  # Apply augmentations to the image tensor
    return (
        (
            torch.stack(
                [source, target_augmented], dim=1 if framestack.ndim == 5 else 0
            ),
            camera_pose,
        )
        if target_only
        else framestack
    )


def geometric_augmentation(
    framestack: torch.Tensor,
    camera_pose: list,
    depthstack=None,
    p: float = 0.2,
) -> tuple:
    """
    Applies random geometric augmentations to a tensor of images, modifying the transformation vector accordingly, wiith
    a set probability.

    Parameters:
    - framestack (torch.Tensor): A tensor representing a stack of images.
    - tr (list): A list representing transformation parameters, modified based on applied augmentations.
    - p (float): Probability of applying each geometric augmentation. Default is 0.2.

    Returns:
    - tuple[torch.Tensor, list]: A tuple containing the augmented image tensor and the adjusted transformation list.
    """
    # Randomly decide which augmentations to apply based on probability p
    hflip = torch.rand(1) < p
    vflip = torch.rand(1) < p
    cwrotate = torch.rand(1) < p
    ccwrotate = torch.rand(1) < p
    unbatched = camera_pose.ndim == 1
    if depthstack is None:
        depthstack = torch.ones_like(framestack)
    if unbatched:
        camera_pose = camera_pose.unsqueeze(0)
        depthstack = depthstack.unsqueeze(0)
        source, target = framestack[0], framestack[1]
        depthsource, depthtarget = depthstack[0], depthstack[1]
    else:
        source, target = framestack[:, 0], framestack[:, 1]
        depthsource, depthtarget = (
            depthstack[:, 0],
            depthstack[:, 1],
        )  # Apply horizontal flip if selected
    if hflip:
        source = tvt.functional.hflip(source)
        target = tvt.functional.hflip(target)
        depthsource = tvt.functional.hflip(depthsource)
        depthtarget = tvt.functional.hflip(depthtarget)
        camera_pose[:, 0] = -camera_pose[:, 0]
    # Apply vertical flip if selected
    if vflip:
        source = tvt.functional.vflip(source)
        target = tvt.functional.vflip(target)
        depthsource = tvt.functional.vflip(depthsource)
        depthtarget = tvt.functional.vflip(depthtarget)
        camera_pose[:, 1] = -camera_pose[:, 1]

    # Apply clockwise rotation if selected
    if cwrotate:
        source = tvt.functional.rotate(source, -90, expand=True)
        target = tvt.functional.rotate(target, -90, expand=True)
        depthsource = tvt.functional.rotate(depthsource, -90, expand=True)
        depthtarget = tvt.functional.rotate(depthtarget, -90, expand=True)
        # Adjust transformation vector for rotation
        camera_pose[:, 0] = -camera_pose[:, 1]
        camera_pose[:, 1] = camera_pose[:, 0]

    # Apply counter-clockwise rotation if selected
    if ccwrotate:
        source = tvt.functional.rotate(source, 90, expand=True)
        target = tvt.functional.rotate(target, 90, expand=True)
        depthsource = tvt.functional.rotate(depthsource, 90, expand=True)
        depthtarget = tvt.functional.rotate(depthtarget, 90, expand=True)
        # Adjust transformation vector for rotation
        camera_pose[:, 0] = camera_pose[:, 1]
        camera_pose[:, 1] = -camera_pose[:, 0]

    framestack = torch.stack([source, target], dim=0 if unbatched else 1)
    depthstack = torch.stack([depthsource, depthtarget], dim=0 if unbatched else 1)
    return framestack, camera_pose.squeeze(0) if unbatched else camera_pose, depthstack


def reverse_augmentation(
    framestack: torch.Tensor, camera_pose: list, p: float = 0.2
) -> tuple:
    """
    Optionally applies reverse geometric augmentation to a tensor of images based on a probability and a transformation vector.

    Parameters:
    - framestack (torch.Tensor): A tensor representing a stack of images.
    - tr (list): A list representing transformation parameters, potentially inverted by this function.
    - p (float): Probability of reversing the augmentation. Default is 0.2.

    Returns:
    - tuple[torch.Tensor, list]: A tuple containing the potentially augmented image tensor and the adjusted (or inverted) transformation list.
    """
    # Randomly decide whether to apply reverse augmentation based on probability p
    if torch.rand(1) < p:
        source_to_target = euler2mat(
            camera_pose
        )  # Convert transformation vector to matrix
        target_to_source = torch.linalg.inv(
            source_to_target
        )  # Invert transformation matrix
        tr_inv = mat2euler(target_to_source)  # Convert inverted matrix back to vector
        return framestack, tr_inv

    return framestack, camera_pose


def standstill_augmentation(
    framestack: torch.Tensor, tr: list, p: float = 0.2
) -> tuple:
    """
    Optionally applies reverse geometric augmentation to a tensor of images based on a probability and a transformation vector.

    Parameters:
    - framestack (torch.Tensor): A tensor representing a stack of images.
    - tr (list): A list representing transformation parameters, potentially inverted by this function.
    - p (float): Probability of reversing the augmentation. Default is 0.2.

    Returns:
    - tuple[torch.Tensor, list]: A tuple containing the potentially augmented image tensor and the adjusted (or inverted) transformation list.
    """
    # Randomly decide whether to apply reverse augmentation based on probability p
    if torch.rand(1) < p:
        framestack[-1] = framestack[0]
        tr = torch.zeros_like(tr)
    return framestack, tr
