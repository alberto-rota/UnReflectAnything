#!/usr/bin/env python3
import os
import json
import shutil
from pathlib import Path
import subprocess
from PIL import Image
import io
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import tqdm
import time
import argparse

# Try to import GPU accelerated libraries
try:
    import cv2

    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import torch
    from torchvision import transforms

    TORCH_AVAILABLE = True
    if torch.cuda.is_available():
        TORCH_CUDA_AVAILABLE = True
        torch_device = torch.device("cuda")
    else:
        TORCH_CUDA_AVAILABLE = False
        torch_device = torch.device("cpu")
except ImportError:
    TORCH_AVAILABLE = False
    TORCH_CUDA_AVAILABLE = False


def create_dummy_json():
    """
    Create a dummy JSON with the same structure but with all numeric values replaced by -1.
    This function constructs a nested dictionary that mirrors the structure of the required JSON,
    but replaces all numeric values with -1 while preserving the structure.
    """
    return {
        "camera-calibration": {
            "DL": [[-1, -1, -1, -1, -1]],
            "DR": [[-1, -1, -1, -1, -1]],
            "KL": [[-1, -1, -1], [-1, -1, -1], [-1, -1, -1]],
            "KR": [[-1, -1, -1], [-1, -1, -1], [-1, -1, -1]],
            "R": [[-1, -1, -1], [-1, -1, -1], [-1, -1, -1]],
            "T": [[-1], [-1], [-1]],
        },
        "camera-pose": [
            [-1, -1, -1, -1],
            [-1, -1, -1, -1],
            [-1, -1, -1, -1],
            [-1, -1, -1, -1],
        ],
        "timestamp": -1,
    }


def compare_image_sizes(jpg_path, output_format="png"):
    """
    Compare the file sizes between a JPEG image and its PNG conversion.

    Args:
        jpg_path (str): Path to the JPEG image file
        output_format (str): Format to compare against (default: "png")

    Returns:
        tuple: (bool, str) - (True if the new format is smaller, preferred format)
    """
    # Load the original image
    if CV2_AVAILABLE:
        img = cv2.imread(jpg_path)
    else:
        img = Image.open(jpg_path)

    # Get the original file size
    original_size = os.path.getsize(jpg_path)

    # Create a BytesIO object to store the converted version temporarily
    buffer = io.BytesIO()

    # Save in the new format
    if CV2_AVAILABLE:
        # Convert to PIL Image for saving (CV2 doesn't directly support BytesIO)
        is_success, buffer_arr = cv2.imencode(f".{output_format}", img)
        buffer.write(buffer_arr.tobytes())
    else:
        img.save(buffer, format=output_format.upper())

    # Get the converted size
    new_size = buffer.tell()

    # Return True if converted is smaller, False otherwise, and the preferred format
    if new_size < original_size:
        return True, output_format
    else:
        return False, "jpg"


def process_image_pil(src_path, dst_path, preferred_format=None, resize_half=True):
    """
    Process an image using PIL by optionally resizing it to half its resolution and converting format.

    Args:
        src_path (str): Path to the source image
        dst_path (str): Base path for the destination image (without extension)
        preferred_format (str, optional): If provided, forces conversion to this format
                                         If None, compares sizes and chooses optimal
        resize_half (bool): If True, resize the image to half its resolution

    Returns:
        str: The full path to the saved image including extension
    """
    # Open the image
    img = Image.open(src_path)

    # Resize to half resolution if requested
    if resize_half:
        width, height = img.size
        img = img.resize((width // 2, height // 2), Image.Resampling.LANCZOS)

    # Determine the best format if not explicitly specified
    if preferred_format is None:
        # Compare sizes
        is_smaller, best_format = compare_image_sizes(src_path)
    else:
        best_format = preferred_format

    # Save in the determined format
    full_dst_path = f"{dst_path}.{best_format}"
    img.save(full_dst_path)

    return full_dst_path


def process_image_cv2(src_path, dst_path, preferred_format=None, resize_half=True):
    """
    Process an image using OpenCV by optionally resizing it to half its resolution and converting format.

    Args:
        src_path (str): Path to the source image
        dst_path (str): Base path for the destination image (without extension)
        preferred_format (str, optional): If provided, forces conversion to this format
                                         If None, compares sizes and chooses optimal
        resize_half (bool): If True, resize the image to half its resolution

    Returns:
        str: The full path to the saved image including extension
    """
    # Read the image with OpenCV
    img = cv2.imread(src_path)

    # Resize to half resolution if requested
    if resize_half:
        height, width = img.shape[:2]
        img = cv2.resize(
            img, (width // 2, height // 2), interpolation=cv2.INTER_LANCZOS4
        )

    # Determine the best format if not explicitly specified
    if preferred_format is None:
        # Compare sizes
        is_smaller, best_format = compare_image_sizes(src_path)
    else:
        best_format = preferred_format

    # Save in the determined format
    full_dst_path = f"{dst_path}.{best_format}"
    cv2.imwrite(full_dst_path, img)

    return full_dst_path


def process_image_torch(src_path, dst_path, preferred_format=None, resize_half=True):
    """
    Process an image using PyTorch with GPU acceleration for resizing.

    Args:
        src_path (str): Path to the source image
        dst_path (str): Base path for the destination image (without extension)
        preferred_format (str, optional): If provided, forces conversion to this format
                                         If None, compares sizes and chooses optimal
        resize_half (bool): If True, resize the image to half its resolution

    Returns:
        str: The full path to the saved image including extension
    """
    # Open the image with PIL first (PyTorch uses PIL internally)
    pil_img = Image.open(src_path)

    # Convert to PyTorch tensor
    img_tensor = transforms.ToTensor()(pil_img).unsqueeze(0)

    # Move to GPU if available
    if TORCH_CUDA_AVAILABLE:
        img_tensor = img_tensor.to(torch_device)

    # Resize if requested using PyTorch's GPU-accelerated interpolation
    if resize_half:
        width, height = pil_img.size
        resize_transform = transforms.Resize(
            (height // 2, width // 2),
            interpolation=transforms.InterpolationMode.BICUBIC,
        )
        img_tensor = resize_transform(img_tensor)

    # Convert back to PIL for saving
    if TORCH_CUDA_AVAILABLE:
        img_tensor = img_tensor.cpu()

    # Remove batch dimension and convert to PIL
    resized_img = transforms.ToPILImage()(img_tensor.squeeze(0))

    # Determine the best format if not explicitly specified
    if preferred_format is None:
        # Compare sizes
        is_smaller, best_format = compare_image_sizes(src_path)
    else:
        best_format = preferred_format

    # Save in the determined format
    full_dst_path = f"{dst_path}.{best_format}"
    resized_img.save(full_dst_path)

    return full_dst_path


def process_image(
    src_path, dst_path, preferred_format=None, resize_half=True, use_gpu=False
):
    """
    Process an image using the best available method.
    This function selects the optimal processing method based on available libraries.

    Args:
        src_path (str): Path to the source image
        dst_path (str): Base path for the destination image (without extension)
        preferred_format (str, optional): If provided, forces conversion to this format
        resize_half (bool): If True, resize the image to half its resolution
        use_gpu (bool): If True, try to use GPU acceleration

    Returns:
        str: The full path to the saved image including extension
    """
    # Use GPU-accelerated processing if requested and available
    if use_gpu and TORCH_AVAILABLE and TORCH_CUDA_AVAILABLE:
        return process_image_torch(src_path, dst_path, preferred_format, resize_half)
    elif CV2_AVAILABLE:
        return process_image_cv2(src_path, dst_path, preferred_format, resize_half)
    else:
        return process_image_pil(src_path, dst_path, preferred_format, resize_half)


def process_case_directory(
    case_index, case_dir, src_dir, dst_dir, preferred_format, resize_half, use_gpu
):
    """
    Process a single case directory, converting all images and creating JSON files.
    This function is designed to be run in parallel by multiple processes.

    Args:
        case_index (int): Index of the case (for versioning)
        case_dir (str): Name of the case directory
        src_dir (str): Source root directory
        dst_dir (str): Destination root directory
        preferred_format (str): Preferred image format (or None for auto)
        resize_half (bool): Whether to resize images to half resolution
        use_gpu (bool): Whether to use GPU acceleration

    Returns:
        tuple: (case_dir, num_images_processed)
    """
    try:
        # Define paths
        case_path = os.path.join(src_dir, "frames", case_dir)
        new_video_dir = f"v{case_index}"
        new_frame_dir = os.path.join(dst_dir, new_video_dir, "frame")
        new_poses_dir = os.path.join(dst_dir, new_video_dir, "poses_absolute")

        # Create destination directories
        os.makedirs(new_frame_dir, exist_ok=True)
        os.makedirs(new_poses_dir, exist_ok=True)

        # Get all image files and sort them
        image_files = sorted(
            [f for f in os.listdir(case_path) if f.endswith(".jpg")],
            key=lambda x: int(os.path.splitext(x)[0]),
        )

        # Create a dummy JSON structure once
        dummy_json = create_dummy_json()

        # Process each image in this case
        for j, img_file in enumerate(image_files):
            # Format the new filename with leading zeros (6 digits)
            new_img_base = f"{j:06d}"

            # Source image path
            src_img_path = os.path.join(case_path, img_file)

            # Base destination path (without extension)
            dst_img_base = os.path.join(new_frame_dir, new_img_base)

            # Process the image
            process_image(
                src_img_path, dst_img_base, preferred_format, resize_half, use_gpu
            )

            # Create the corresponding JSON file
            json_path = os.path.join(new_poses_dir, f"{new_img_base}.json")
            with open(json_path, "w") as f:
                json.dump(dummy_json, f, indent=4)

        return case_dir, len(image_files)
    except Exception as e:
        return case_dir, f"Error: {str(e)}"


def process_image_batch(batch_data):
    """
    Process a batch of images.
    This function is designed to be run by a worker in the process pool.

    Args:
        batch_data (tuple): (case_dir, image_files, src_case_path,
                            new_frame_dir, new_poses_dir, preferred_format,
                            resize_half, use_gpu)

    Returns:
        int: Number of images processed
    """
    (
        case_dir,
        image_files,
        src_case_path,
        new_frame_dir,
        new_poses_dir,
        preferred_format,
        resize_half,
        use_gpu,
    ) = batch_data

    dummy_json = create_dummy_json()
    processed_count = 0

    for j, img_file in enumerate(image_files):
        try:
            # Format the new filename with leading zeros (6 digits)
            new_img_base = f"{j:06d}"

            # Source image path
            src_img_path = os.path.join(src_case_path, img_file)

            # Base destination path (without extension)
            dst_img_base = os.path.join(new_frame_dir, new_img_base)

            # Process the image
            process_image(
                src_img_path, dst_img_base, preferred_format, resize_half, use_gpu
            )

            # Create the corresponding JSON file
            json_path = os.path.join(new_poses_dir, f"{new_img_base}.json")
            with open(json_path, "w") as f:
                json.dump(dummy_json, f, indent=4)

            processed_count += 1
        except Exception as e:
            print(f"Error processing {img_file} in {case_dir}: {e}")

    return processed_count


def adapt_directory_structure(
    src_dir,
    dst_dir,
    force_format=None,
    resize_half=True,
    parallel=True,
    num_processes=None,
    use_gpu=False,
    batch_size=50,
):
    """
    Adapt the directory structure from the original format to the new format.

    Args:
        src_dir (str): Path to the source directory (GraSP)
        dst_dir (str): Path to the destination directory (GRASP)
        force_format (str, optional): If provided, forces all images to this format
        resize_half (bool): If True, resize images to half resolution
        parallel (bool): If True, process directories in parallel
        num_processes (int, optional): Number of processes to use (default: CPU count)
        use_gpu (bool): If True, try to use GPU acceleration
        batch_size (int): Number of images to process in a batch
    """
    # Set default number of processes if not specified
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()

    # Convert provided string format to PIL-compatible format
    format_map = {"jpg": "JPEG", "png": "PNG"} if force_format else None
    preferred_format = format_map.get(force_format.lower()) if force_format else None

    # Ensure the destination directory exists
    os.makedirs(dst_dir, exist_ok=True)

    # Get all case directories
    frames_dir = os.path.join(src_dir, "frames")
    case_dirs = sorted(
        [
            d
            for d in os.listdir(frames_dir)
            if os.path.isdir(os.path.join(frames_dir, d))
        ]
    )

    # Report available acceleration methods
    print(f"Image processing options:")
    print(f" - OpenCV available: {CV2_AVAILABLE}")
    print(f" - PyTorch available: {TORCH_AVAILABLE}")
    print(f" - CUDA available: {TORCH_CUDA_AVAILABLE if TORCH_AVAILABLE else False}")
    print(
        f" - Using GPU acceleration: {use_gpu and TORCH_CUDA_AVAILABLE and TORCH_AVAILABLE}"
    )
    print(f" - Parallel processing: {parallel} (using {num_processes} processes)")
    print(f" - Resizing images: {resize_half}")
    print(f" - Image format: {force_format if force_format else 'auto'}")

    # Start timing
    start_time = time.time()

    if parallel:
        # Process case directories in parallel
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            # Submit jobs for each case directory
            futures = {}

            for i, case_dir in enumerate(case_dirs, 1):
                future = executor.submit(
                    process_case_directory,
                    i,
                    case_dir,
                    src_dir,
                    dst_dir,
                    preferred_format,
                    resize_half,
                    use_gpu,
                )
                futures[future] = case_dir

            # Create a progress bar
            with tqdm.tqdm(total=len(case_dirs), desc="Processing directories") as pbar:
                for future in as_completed(futures):
                    case_dir = futures[future]
                    try:
                        result = future.result()
                        pbar.update(1)
                        pbar.set_postfix_str(f"Completed {case_dir}")
                    except Exception as e:
                        print(f"Error processing {case_dir}: {e}")

    else:
        # Process directories sequentially but images within a directory in batches
        for i, case_dir in enumerate(case_dirs, 1):
            case_path = os.path.join(frames_dir, case_dir)

            # Create the new directories for this video
            new_video_dir = f"v{i}"
            new_frame_dir = os.path.join(dst_dir, new_video_dir, "frame")
            new_poses_dir = os.path.join(dst_dir, new_video_dir, "poses_absolute")

            os.makedirs(new_frame_dir, exist_ok=True)
            os.makedirs(new_poses_dir, exist_ok=True)

            # Get all image files and sort them
            image_files = sorted(
                [f for f in os.listdir(case_path) if f.endswith(".jpg")],
                key=lambda x: int(os.path.splitext(x)[0]),
            )

            # Process in batches using a ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=num_processes) as executor:
                # Create batches
                batches = []
                for k in range(0, len(image_files), batch_size):
                    batch = image_files[k : k + batch_size]
                    batches.append(
                        (
                            case_dir,
                            batch,
                            case_path,
                            new_frame_dir,
                            new_poses_dir,
                            preferred_format,
                            resize_half,
                            use_gpu,
                        )
                    )

                # Process batches in parallel
                total_images = len(image_files)
                completed_images = 0

                with tqdm.tqdm(
                    total=total_images, desc=f"Processing {case_dir}"
                ) as pbar:
                    for result in executor.map(process_image_batch, batches):
                        completed_images += result
                        pbar.update(result)

    # End timing
    elapsed_time = time.time() - start_time

    # Display performance information
    print(f"\nDirectory structure adaptation completed in {elapsed_time:.2f} seconds.")
    print(f"Processed {len(case_dirs)} case directories.")


def main():
    """Main function to parse arguments and run the adaptation process."""
    parser = argparse.ArgumentParser(
        description="Adapt GraSP directory structure to the new format."
    )
    parser.add_argument(
        "--src", type=str, default="GraSP", help="Source directory path"
    )
    parser.add_argument(
        "--dst", type=str, default="GRASP", help="Destination directory path"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["jpg", "png", "auto"],
        default="auto",
        help="Force output format (jpg, png) or auto-select smallest (auto)",
    )
    parser.add_argument(
        "--resize_half",
        action="store_true",
        default=False,
        help="Resize images to half their original resolution",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        default=True,
        help="Process directories in parallel (default: True)",
    )
    parser.add_argument(
        "--no-parallel",
        action="store_false",
        dest="parallel",
        help="Disable parallel processing",
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=None,
        help="Number of processes to use (default: number of CPU cores)",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        default=False,
        help="Use GPU acceleration if available",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Number of images to process in a batch (default: 50)",
    )

    args = parser.parse_args()

    # Convert 'auto' to None for the function
    force_format = None if args.format == "auto" else args.format

    # Run the adaptation
    adapt_directory_structure(
        args.src,
        args.dst,
        force_format,
        args.resize_half,
        args.parallel,
        args.processes,
        args.use_gpu,
        args.batch_size,
    )

    # Print information about the operation
    resize_info = "with" if args.resize_half else "without"
    print(
        f"Directory structure adapted successfully from {args.src} to {args.dst} {resize_info} image resizing"
    )


if __name__ == "__main__":
    multiprocessing.freeze_support()  # Required for Windows
    main()
