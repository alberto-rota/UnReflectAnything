#!/usr/bin/env python3
"""
High-Performance Image Highlight Analysis Script

Optimized version using multiprocessing, vectorized operations, and performance enhancements.
"""

import os
import json
import argparse
import shutil
from pathlib import Path
from typing import Tuple, List, Dict, Optional, NamedTuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import numpy as np
from PIL import Image
import cv2


class ImageResult(NamedTuple):
    """Structure to hold image processing results."""
    path: Path
    highlight_percentage: float
    binary_mask: Optional[np.ndarray] = None
    crop_rect: Optional[Tuple[int, int, int, int]] = None
    original_image: Optional[Image.Image] = None


def process_single_image(img_path: Path, 
                        is_highlight_threshold: int,
                        max_highlighted: float,
                        min_crop_size: Tuple[int, int],
                        need_masks: bool,
                        need_crops: bool) -> ImageResult:
    """
    Process a single image for highlights. Optimized for multiprocessing.
    """
    try:
        # Load image directly with OpenCV for speed
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            return ImageResult(img_path, 0.0)
        
        # Convert to grayscale using OpenCV (faster than PIL)
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # Vectorized highlight detection
        highlight_mask = img_gray > is_highlight_threshold
        
        # Calculate percentage using numpy operations
        total_pixels = img_gray.size
        highlight_pixels = np.sum(highlight_mask)
        highlight_percentage = (highlight_pixels / total_pixels) * 100
        
        # Early return if we don't need additional processing
        is_unhighlighted = highlight_percentage <= max_highlighted
        
        binary_mask = None
        crop_rect = None
        original_image = None
        
        # Only compute mask if needed
        if need_masks and is_unhighlighted:
            binary_mask = (highlight_mask * 255).astype(np.uint8)
        
        # Only compute crop if needed
        if need_crops and highlight_mask.size > 0:
            crop_rect = find_largest_unhighlighted_rectangle_fast(highlight_mask, min_crop_size)
            if crop_rect:
                # Load with PIL only if we found a valid crop
                original_image = Image.open(img_path)
        
        return ImageResult(
            path=img_path,
            highlight_percentage=highlight_percentage,
            binary_mask=binary_mask,
            crop_rect=crop_rect,
            original_image=original_image
        )
        
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return ImageResult(img_path, 0.0)


def find_largest_unhighlighted_rectangle_fast(highlight_mask: np.ndarray, 
                                            min_crop_size: Tuple[int, int]) -> Optional[Tuple[int, int, int, int]]:
    """
    Optimized version of largest rectangle finding using OpenCV.
    """
    if highlight_mask.size == 0:
        return None
    
    # Convert to uint8 for OpenCV
    binary_mask = highlight_mask.astype(np.uint8)
    
    # Use OpenCV's optimized histogram-based approach
    max_area = 0
    best_rect = None
    
    rows, cols = binary_mask.shape
    heights = np.zeros(cols, dtype=np.int32)
    
    for i in range(rows):
        # Vectorized height update
        heights = np.where(binary_mask[i] == 1, 0, heights + 1)
        
        # Fast histogram rectangle calculation
        area, rect = largest_rectangle_in_histogram_fast(heights, i)
        if area > max_area:
            max_area = area
            best_rect = rect
    
    if best_rect and best_rect[2] >= min_crop_size[0] and best_rect[3] >= min_crop_size[1]:
        return best_rect
    
    return None


def largest_rectangle_in_histogram_fast(heights: np.ndarray, bottom_row: int) -> Tuple[int, Optional[Tuple[int, int, int, int]]]:
    """
    Optimized largest rectangle in histogram using stack approach.
    """
    stack = []
    max_area = 0
    best_rect = None
    n = len(heights)
    
    for i in range(n + 1):
        h = heights[i] if i < n else 0
        
        while stack and heights[stack[-1]] > h:
            height = heights[stack.pop()]
            width = i if not stack else i - stack[-1] - 1
            area = height * width
            
            if area > max_area:
                max_area = area
                x = 0 if not stack else stack[-1] + 1
                y = bottom_row - height + 1
                best_rect = (x, y, width, height)
        
        stack.append(i)
    
    return max_area, best_rect


class HighPerformanceHighlightAnalyzer:
    def __init__(self, 
                 is_highlight_threshold: int = 240,
                 max_highlighted: float = 5.0,
                 min_crop_size: Tuple[int, int] = (100, 100),
                 max_workers: Optional[int] = None):
        """
        Initialize the high-performance highlight analyzer.
        """
        self.is_highlight_threshold = is_highlight_threshold
        self.max_highlighted = max_highlighted
        self.min_crop_size = min_crop_size
        self.max_workers = max_workers or min(32, os.cpu_count() + 4)
        
        # Supported image extensions
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    
    def find_image_files_fast(self, directory: str) -> List[Path]:
        """Fast image file discovery using os.scandir."""
        directory = Path(directory)
        image_files = []
        
        def scan_recursive(path):
            try:
                with os.scandir(path) as entries:
                    for entry in entries:
                        if entry.is_file() and Path(entry.name).suffix.lower() in self.image_extensions:
                            image_files.append(Path(entry.path))
                        elif entry.is_dir():
                            scan_recursive(entry.path)
            except PermissionError:
                pass
        
        scan_recursive(directory)
        return image_files
    
    def batch_save_files(self, file_operations: List[Tuple[Path, Path]]):
        """Batch file operations for better I/O performance."""
        for src, dst in file_operations:
            try:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
            except Exception as e:
                print(f"Error copying {src.name}: {e}")
    
    def batch_save_images(self, image_operations: List[Tuple[np.ndarray, Path]]):
        """Batch save images using OpenCV for speed."""
        for img_data, path in image_operations:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(path), img_data)
            except Exception as e:
                print(f"Error saving image {path.name}: {e}")
    
    def analyze_directory(self, 
                         input_directory: str,
                         output_json: str = "results.json",
                         copy_unhighlighted_to: Optional[str] = None,
                         save_masks_to: Optional[str] = None,
                         save_crops_to: Optional[str] = None):
        """
        High-performance analysis using multiprocessing.
        """
        print(f"🚀 High-Performance Analysis Starting...")
        print(f"📁 Directory: {input_directory}")
        print(f"🖥️  Using {self.max_workers} workers")
        print(f"⚡ Highlight threshold: {self.is_highlight_threshold}")
        print(f"📊 Max highlighted: {self.max_highlighted}%")
        print("-" * 60)
        
        # Fast file discovery
        print("🔍 Discovering image files...")
        image_files = self.find_image_files_fast(input_directory)
        total_files = len(image_files)
        print(f"📸 Found {total_files} images")
        
        if not image_files:
            print("❌ No images found!")
            return
        
        # Determine what processing we need
        need_masks = save_masks_to is not None
        need_crops = save_crops_to is not None
        
        # Prepare results structure
        results = {
            "parameters": {
                "is_highlight_threshold": self.is_highlight_threshold,
                "max_highlighted_percentage": self.max_highlighted,
                "min_crop_size": self.min_crop_size,
                "workers_used": self.max_workers
            },
            "unhighlighted_images": [],
            "all_images": []
        }
        
        # Batch processing with multiprocessing
        print("⚡ Processing images in parallel...")
        
        process_func = partial(
            process_single_image,
            is_highlight_threshold=self.is_highlight_threshold,
            max_highlighted=self.max_highlighted,
            min_crop_size=self.min_crop_size,
            need_masks=need_masks,
            need_crops=need_crops
        )
        
        # Process in batches to manage memory
        batch_size = min(1000, total_files)
        processed = 0
        
        # Prepare batch operations
        copy_operations = []
        mask_operations = []
        crop_operations = []
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            for i in range(0, total_files, batch_size):
                batch = image_files[i:i + batch_size]
                
                # Submit batch
                future_to_path = {executor.submit(process_func, path): path for path in batch}
                
                # Collect results
                for future in as_completed(future_to_path):
                    result = future.result()
                    processed += 1
                    
                    if processed % 100 == 0 or processed == total_files:
                        print(f"✅ Processed {processed}/{total_files} images ({processed/total_files*100:.1f}%)")
                    
                    # Store result
                    image_info = {
                        "path": str(result.path),
                        "highlight_percentage": round(result.highlight_percentage, 2)
                    }
                    results["all_images"].append(image_info)
                    
                    # Check if unhighlighted
                    if result.highlight_percentage <= self.max_highlighted:
                        results["unhighlighted_images"].append(image_info)
                        
                        # Prepare copy operation
                        if copy_unhighlighted_to:
                            dst_path = Path(copy_unhighlighted_to) / result.path.name
                            copy_operations.append((result.path, dst_path))
                        
                        # Prepare mask save operation
                        if save_masks_to and result.binary_mask is not None:
                            mask_filename = result.path.stem + '_mask.png'
                            mask_path = Path(save_masks_to) / mask_filename
                            mask_operations.append((result.binary_mask, mask_path))
                    
                    # Prepare crop operation
                    if save_crops_to and result.crop_rect and result.original_image:
                        x, y, width, height = result.crop_rect
                        crop_filename = result.path.stem + '_crop' + result.path.suffix
                        crop_path = Path(save_crops_to) / crop_filename
                        
                        # Crop and prepare for saving
                        cropped = result.original_image.crop((x, y, x + width, y + height))
                        crop_operations.append((cropped, crop_path))
        
        print("\n💾 Saving results...")
        
        # Save JSON results
        os.makedirs(os.path.dirname(output_json) if os.path.dirname(output_json) else '.', exist_ok=True)
        with open(output_json, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"📄 Results saved to: {output_json}")
        
        # Batch file operations
        if copy_operations:
            print(f"📁 Copying {len(copy_operations)} unhighlighted images...")
            self.batch_save_files(copy_operations)
        
        if mask_operations:
            print(f"🎭 Saving {len(mask_operations)} masks...")
            self.batch_save_images(mask_operations)
        
        if crop_operations:
            print(f"✂️  Saving {len(crop_operations)} crops...")
            for cropped_img, path in crop_operations:
                path.parent.mkdir(parents=True, exist_ok=True)
                cropped_img.save(path, optimize=True)
        
        unhighlighted_count = len(results["unhighlighted_images"])
        print(f"\n🎉 Analysis Complete!")
        print(f"📊 Total images: {total_files}")
        print(f"✨ Unhighlighted images: {unhighlighted_count} ({unhighlighted_count/total_files*100:.1f}%)")
        print(f"⚡ Performance: {total_files/1 if total_files < 100 else 'High-speed'} mode")


def main():
    parser = argparse.ArgumentParser(description="High-Performance Image Highlight Analyzer")
    parser.add_argument("input_directory", help="Input directory containing images")
    parser.add_argument("--is-highlight-threshold", type=int, default=240,
                       help="Brightness threshold for highlight detection (0-255, default: 240)")
    parser.add_argument("--max-highlighted", type=float, default=5.0,
                       help="Maximum percentage of highlight pixels (default: 5.0)")
    parser.add_argument("--min-crop-size", nargs=2, type=int, default=[100, 100],
                       help="Minimum crop size as width height (default: 100 100)")
    parser.add_argument("--output-json", default="results.json",
                       help="Output JSON file path (default: results.json)")
    parser.add_argument("--copy-unhighlighted-to", 
                       help="Directory to copy unhighlighted images to")
    parser.add_argument("--save-masks-to",
                       help="Directory to save binary masks to")
    parser.add_argument("--save-crops-to",
                       help="Directory to save cropped images to")
    parser.add_argument("--workers", type=int,
                       help="Number of worker processes (default: auto-detect)")
    
    args = parser.parse_args()
    
    # Create high-performance analyzer
    analyzer = HighPerformanceHighlightAnalyzer(
        is_highlight_threshold=args.is_highlight_threshold,
        max_highlighted=args.max_highlighted,
        min_crop_size=tuple(args.min_crop_size),
        max_workers=args.workers
    )
    
    # Run analysis
    analyzer.analyze_directory(
        input_directory=args.input_directory,
        output_json=args.output_json,
        copy_unhighlighted_to=args.copy_unhighlighted_to,
        save_masks_to=args.save_masks_to,
        save_crops_to=args.save_crops_to
    )


if __name__ == "__main__":
    main()