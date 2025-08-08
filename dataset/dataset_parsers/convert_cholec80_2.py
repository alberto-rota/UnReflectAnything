#!/usr/bin/env python3

import os
import subprocess
import tempfile
import shutil
from concurrent.futures import ProcessPoolExecutor
import math

# Define constants
SOURCE_BUCKET = "gs://alberto-bucket/CHOLEC80_vids/videos/"
DEST_BUCKET = "gs://alberto-bucket/CHOLEC80_frames/"
VIDEO_NUMBERS = ["78", "79", "80"]  # Specific videos to process
NUM_CORES = 8  # Number of cores to use for parallel processing
MAX_STORAGE_PERCENT = 70  # Maximum percentage of available storage to use
USE_GPU = True  # Set to True to use GPU-accelerated frame extraction


def check_gpu_availability():
    """Check if CUDA/GPU is available for FFmpeg."""
    try:
        cmd = "ffmpeg -hide_banner -hwaccels"
        result = subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE)
        output = result.stdout.decode("utf-8").lower()

        for accel in ["cuda", "nvenc", "vaapi", "vdpau"]:
            if accel in output:
                return accel
        return None
    except:
        return None


def process_video(args):
    """Process a single video by downloading it, extracting frames, and uploading frames to GCS."""
    video_num, gpu_accel = args
    video_path = f"{SOURCE_BUCKET}video{video_num}.mp4"
    dest_prefix = f"{DEST_BUCKET}v{video_num}/frame/"

    print(f"Processing video{video_num}...")

    # Create a temporary directory for processing
    with tempfile.TemporaryDirectory() as temp_dir:
        local_video_path = os.path.join(temp_dir, f"video{video_num}.mp4")
        frames_dir = os.path.join(temp_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)

        try:
            # Download the video
            print(f"Downloading video{video_num}.mp4...")
            download_cmd = f"gsutil cp {video_path} {local_video_path}"
            subprocess.run(download_cmd, shell=True, check=True)

            # Prepare FFmpeg command with or without GPU acceleration
            if gpu_accel == "cuda" or gpu_accel == "nvenc":
                # CUDA/NVENC acceleration
                ffmpeg_cmd = (
                    f"ffmpeg -hwaccel cuda -i {local_video_path} "
                    f"-vf fps=1 -q:v 1 {frames_dir}/%06d.png"
                )
            elif gpu_accel == "vaapi":
                # VAAPI acceleration
                ffmpeg_cmd = (
                    f"ffmpeg -hwaccel vaapi -hwaccel_output_format vaapi "
                    f"-i {local_video_path} -vf 'fps=1,hwdownload,format=nv12' "
                    f"-q:v 1 {frames_dir}/%06d.png"
                )
            else:
                # No GPU acceleration
                ffmpeg_cmd = f"ffmpeg -i {local_video_path} -vf fps=1 -q:v 1 {frames_dir}/%06d.png"

            # Extract frames
            print(
                f"Extracting frames from video{video_num}.mp4 {'with GPU' if gpu_accel else 'without GPU'}..."
            )
            subprocess.run(ffmpeg_cmd, shell=True, check=True)

            # Get frame count
            frames = [f for f in os.listdir(frames_dir) if f.endswith(".png")]
            frame_count = len(frames)

            if frame_count == 0:
                print(f"Warning: No frames were extracted from video{video_num}")
                return None

            print(f"Extracted {frame_count} frames from video{video_num}.mp4")

            # Upload frames to GCS
            print(f"Uploading frames for video{video_num} to {dest_prefix}...")

            # Create destination folder if it doesn't exist
            try:
                subprocess.run(
                    f"gsutil ls {dest_prefix}",
                    shell=True,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
            except subprocess.CalledProcessError:
                print(f"Creating destination folder {dest_prefix}...")
                # Note: In GCS, folders are created implicitly when objects are uploaded

            # Upload frames
            upload_cmd = f"gsutil -m cp {frames_dir}/*.png {dest_prefix}"
            subprocess.run(upload_cmd, shell=True, check=True)

            print(f"Completed processing video{video_num}.mp4")
            return video_num

        except subprocess.CalledProcessError as e:
            print(f"Error processing video{video_num}: {str(e)}")
            return None
        finally:
            # Delete the local video to free up space (if it exists)
            if os.path.exists(local_video_path):
                os.remove(local_video_path)


def get_free_space_gb():
    """Get free space in GB on the current filesystem."""
    stats = shutil.disk_usage("/")
    return stats.free / (1024**3)  # Convert bytes to GB


def main():
    # Check if required tools are installed
    try:
        subprocess.run(
            "gsutil --version", shell=True, check=True, stdout=subprocess.PIPE
        )
        subprocess.run(
            "ffmpeg -version", shell=True, check=True, stdout=subprocess.PIPE
        )
    except subprocess.CalledProcessError:
        print(
            "Error: gsutil or ffmpeg not found. Please install Google Cloud SDK and FFmpeg."
        )
        return

    # Check for GPU acceleration
    gpu_accel = None
    if USE_GPU:
        gpu_accel = check_gpu_availability()
        if gpu_accel:
            print(f"GPU acceleration available using {gpu_accel}")
        else:
            print("GPU acceleration not available, using CPU")

    # Check free space
    free_space_gb = get_free_space_gb()
    print(f"Free space: {free_space_gb:.2f} GB")

    if free_space_gb < 5:
        print("Warning: Less than 5GB free space available. Processing may fail.")
        return

    # Determine max concurrent processes based on available storage
    # Assume each video might take up to 5GB during processing (conservative estimate)
    estimated_space_per_video_gb = 5
    safe_space_gb = free_space_gb * (MAX_STORAGE_PERCENT / 100)
    max_concurrent = max(
        1, min(int(safe_space_gb / estimated_space_per_video_gb), NUM_CORES)
    )

    print(f"Processing with {max_concurrent} concurrent workers")

    # Prepare arguments for each video
    args = [(video_num, gpu_accel) for video_num in VIDEO_NUMBERS]

    # Process videos in parallel
    if max_concurrent > 1:
        with ProcessPoolExecutor(max_workers=max_concurrent) as executor:
            results = list(executor.map(process_video, args))

        # Count successful processes
        successful = [r for r in results if r is not None]
        print(
            f"Successfully processed {len(successful)} out of {len(VIDEO_NUMBERS)} videos."
        )
    else:
        # Process sequentially if only one worker is allowed
        print("Processing videos sequentially...")
        successful = 0
        for arg in args:
            if process_video(arg) is not None:
                successful += 1

        print(
            f"Successfully processed {successful} out of {len(VIDEO_NUMBERS)} videos."
        )


if __name__ == "__main__":
    main()
