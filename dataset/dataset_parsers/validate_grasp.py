#!/usr/bin/env python3
import os
import argparse
import json
import sys
from PIL import Image
from collections import defaultdict


def count_files_with_extension(directory, extension):
    """
    Count files with a specific extension in a directory.

    Args:
        directory (str): Path to the directory to scan
        extension (str): File extension to count (e.g., '.jpg', '.json')

    Returns:
        int: Number of files with the specified extension
    """
    if not os.path.exists(directory):
        return 0
    return len(
        [f for f in os.listdir(directory) if f.lower().endswith(extension.lower())]
    )


def verify_image_resolution(image_path, expected_width, expected_height):
    """
    Verify if an image has the expected resolution.

    Args:
        image_path (str): Path to the image file
        expected_width (int): Expected width of the image
        expected_height (int): Expected height of the image

    Returns:
        bool: True if the image has the expected resolution, False otherwise
    """
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            return width == expected_width and height == expected_height
    except Exception as e:
        print(f"Error checking image resolution for {image_path}: {e}")
        return False


def verify_json_structure(json_path):
    """
    Verify if a JSON file has the expected structure with all numeric values as -1.

    Args:
        json_path (str): Path to the JSON file

    Returns:
        bool: True if the JSON has the expected structure, False otherwise
    """
    try:
        with open(json_path, "r") as f:
            data = json.load(f)

        # Check for expected keys
        expected_keys = {"camera-calibration", "camera-pose", "timestamp"}
        if set(data.keys()) != expected_keys:
            return False

        # Check timestamp value
        if data.get("timestamp") != -1:
            return False

        # Recursive function to check all numeric values are -1
        def check_values(obj):
            if isinstance(obj, dict):
                return all(check_values(v) for v in obj.values())
            elif isinstance(obj, list):
                return all(check_values(v) for v in obj)
            elif isinstance(obj, (int, float)):
                return obj == -1
            else:
                return True  # Non-numeric values are ignored

        return check_values(data)

    except Exception as e:
        print(f"Error validating JSON structure for {json_path}: {e}")
        return False


def scan_directory_for_issues(
    directory, extension, validator_func=None, expected_name_format=None, **kwargs
):
    """
    Scan a directory for files with a specific extension and validate them.

    Args:
        directory (str): Path to the directory to scan
        extension (str): File extension to validate
        validator_func (callable, optional): Function to validate each file
        expected_name_format (str, optional): Expected format for filenames (e.g., '{:06d}')
        **kwargs: Additional arguments to pass to the validator function

    Returns:
        tuple: (valid_count, invalid_count, invalid_files)
    """
    if not os.path.exists(directory):
        return 0, 0, []

    files = [f for f in os.listdir(directory) if f.lower().endswith(extension.lower())]
    valid_count = 0
    invalid_count = 0
    invalid_files = []

    for filename in files:
        file_path = os.path.join(directory, filename)

        # Check filename format if expected_name_format is provided
        if expected_name_format:
            basename = os.path.splitext(filename)[0]
            try:
                index = int(basename)
                expected_filename = f"{expected_name_format.format(index)}{extension}"
                if filename != expected_filename:
                    invalid_files.append(
                        (
                            file_path,
                            f"Incorrect filename format. Expected: {expected_filename}",
                        )
                    )
                    invalid_count += 1
                    continue
            except ValueError:
                invalid_files.append((file_path, "Filename is not numeric"))
                invalid_count += 1
                continue

        # Validate file with provided function
        if validator_func:
            if validator_func(file_path, **kwargs):
                valid_count += 1
            else:
                invalid_files.append((file_path, "Failed validation"))
                invalid_count += 1
        else:
            valid_count += 1

    return valid_count, invalid_count, invalid_files


def main():
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(
        description="Validate GraSP directory structure conversion"
    )
    parser.add_argument(
        "--src", default="GraSP", help="Source directory (default: GraSP)"
    )
    parser.add_argument(
        "--dest", default="GRASP", help="Destination directory (default: GRASP)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show detailed information"
    )
    parser.add_argument(
        "--check-resolution", action="store_true", help="Verify image resolutions"
    )
    parser.add_argument(
        "--check-json", action="store_true", help="Verify JSON structure"
    )
    parser.add_argument(
        "--original-width",
        type=int,
        default=1280,
        help="Original image width (default: 1280)",
    )
    parser.add_argument(
        "--original-height",
        type=int,
        default=800,
        help="Original image height (default: 800)",
    )
    args = parser.parse_args()

    # Set paths
    src_root = args.src
    dest_root = args.dest
    verbose = args.verbose

    # Get source frame directories
    src_frames_dir = os.path.join(src_root, "frames")

    # Check if the source directory exists
    if not os.path.exists(src_frames_dir):
        print(f"Error: Source directory '{src_frames_dir}' not found.")
        return 1

    # Get all case directories in source and sort them naturally
    src_case_dirs = sorted(
        [
            d
            for d in os.listdir(src_frames_dir)
            if os.path.isdir(os.path.join(src_frames_dir, d))
        ],
        key=lambda x: (
            int(x.replace("CASE", ""))
            if x.startswith("CASE") and x[4:].isdigit()
            else float("inf")
        ),
    )

    # Count expected version directories
    num_versions = len(src_case_dirs)

    # Verify destination directory structure
    all_valid = True
    missing_dest_dirs = []

    # Check all destination version directories exist
    for i in range(1, num_versions + 1):
        version_dir = f"v{i}"
        frame_dir = os.path.join(dest_root, version_dir, "frame")
        poses_dir = os.path.join(dest_root, version_dir, "poses_absolute")

        if not os.path.exists(frame_dir):
            missing_dest_dirs.append(frame_dir)
            all_valid = False

        if not os.path.exists(poses_dir):
            missing_dest_dirs.append(poses_dir)
            all_valid = False

    if missing_dest_dirs:
        print("Error: The following destination directories are missing:")
        for dir_path in missing_dest_dirs:
            print(f"  - {dir_path}")
        if not args.verbose:
            print("Stopping validation. Use --verbose to continue despite errors.")
            return 1

    # Count files in each source directory
    src_file_counts = {}
    for case_dir in src_case_dirs:
        src_dir_path = os.path.join(src_frames_dir, case_dir)
        jpg_count = count_files_with_extension(src_dir_path, ".jpg")
        src_file_counts[case_dir] = jpg_count

    # Prepare data structures for validation
    version_validation = {}
    total_src_jpg = 0
    total_dest_jpg = 0
    total_dest_json = 0
    total_valid_jpg = 0
    total_valid_json = 0

    # Print header for the validation report
    print("\nValidation Report:")
    print("-" * 100)
    header = f"{'Source Case':<12} {'Dest Version':<12} {'Source JPGs':<12} {'Dest JPGs':<12} {'Dest JSONs':<12}"
    if args.check_resolution:
        header += f" {'Valid Res.':<12}"
    if args.check_json:
        header += f" {'Valid JSON':<12}"
    header += f" {'Status':<10}"
    print(header)
    print("-" * 100)

    # Validate each case directory mapping
    for i, case_dir in enumerate(src_case_dirs, 1):
        version_dir = f"v{i}"
        src_dir_path = os.path.join(src_frames_dir, case_dir)
        dest_frame_dir = os.path.join(dest_root, version_dir, "frame")
        dest_poses_dir = os.path.join(dest_root, version_dir, "poses_absolute")

        # Count files
        src_jpg_count = src_file_counts[case_dir]
        dest_jpg_count = count_files_with_extension(
            dest_frame_dir, ".jpg"
        ) + count_files_with_extension(dest_frame_dir, ".png")
        dest_json_count = count_files_with_extension(dest_poses_dir, ".json")

        total_src_jpg += src_jpg_count
        total_dest_jpg += dest_jpg_count
        total_dest_json += dest_json_count

        # Check file count match
        counts_match = src_jpg_count == dest_jpg_count == dest_json_count

        # Initialize validation result
        validation_result = {
            "counts_match": counts_match,
            "src_jpg_count": src_jpg_count,
            "dest_jpg_count": dest_jpg_count,
            "dest_json_count": dest_json_count,
            "resolution_check": None,
            "json_check": None,
            "invalid_files": [],
        }

        # Check image resolutions if requested
        resolution_valid = None
        if args.check_resolution and counts_match:
            valid_count, invalid_count, invalid_files = scan_directory_for_issues(
                dest_frame_dir,
                ".jpg",  # Check JPG files
                verify_image_resolution,
                expected_name_format="{:06d}",
                expected_width=args.original_width // 2,
                expected_height=args.original_height // 2,
            )

            # Also check PNG files if present
            png_valid_count, png_invalid_count, png_invalid_files = (
                scan_directory_for_issues(
                    dest_frame_dir,
                    ".png",  # Check PNG files
                    verify_image_resolution,
                    expected_name_format="{:06d}",
                    expected_width=args.original_width // 2,
                    expected_height=args.original_height // 2,
                )
            )

            valid_count += png_valid_count
            invalid_count += png_invalid_count
            invalid_files.extend(png_invalid_files)

            resolution_valid = invalid_count == 0
            total_valid_jpg += valid_count

            validation_result["resolution_check"] = resolution_valid
            validation_result["invalid_files"].extend(invalid_files)

        # Check JSON structure if requested
        json_valid = None
        if args.check_json and counts_match:
            valid_count, invalid_count, invalid_files = scan_directory_for_issues(
                dest_poses_dir,
                ".json",
                verify_json_structure,
                expected_name_format="{:06d}",
            )

            json_valid = invalid_count == 0
            total_valid_json += valid_count

            validation_result["json_check"] = json_valid
            validation_result["invalid_files"].extend(invalid_files)

        # Determine overall status
        status = "VALID"
        if not counts_match:
            status = "INVALID"
            all_valid = False
        elif args.check_resolution and resolution_valid is False:
            status = "INVALID"
            all_valid = False
        elif args.check_json and json_valid is False:
            status = "INVALID"
            all_valid = False

        # Store validation result
        version_validation[version_dir] = validation_result

        # Print validation row
        row = f"{case_dir:<12} {version_dir:<12} {src_jpg_count:<12} {dest_jpg_count:<12} {dest_json_count:<12}"
        if args.check_resolution:
            res_status = (
                "VALID"
                if resolution_valid
                else "INVALID" if resolution_valid is False else "N/A"
            )
            row += f" {res_status:<12}"
        if args.check_json:
            json_status = (
                "VALID" if json_valid else "INVALID" if json_valid is False else "N/A"
            )
            row += f" {json_status:<12}"
        row += f" {status:<10}"
        print(row)

        # If verbose and there are invalid files, print them
        if verbose and validation_result["invalid_files"]:
            print("  Invalid files:")
            for file_path, reason in validation_result["invalid_files"][
                :5
            ]:  # Show only first 5 for brevity
                print(f"    - {os.path.basename(file_path)}: {reason}")
            if len(validation_result["invalid_files"]) > 5:
                print(
                    f"    ... and {len(validation_result['invalid_files']) - 5} more."
                )

    # Print summary
    print("-" * 100)
    summary_row = f"{'TOTAL':<12} {'N/A':<12} {total_src_jpg:<12} {total_dest_jpg:<12} {total_dest_json:<12}"
    if args.check_resolution:
        summary_row += f" {total_valid_jpg:<12}"
    if args.check_json:
        summary_row += f" {total_valid_json:<12}"
    summary_row += f" {'VALID' if all_valid else 'INVALID':<10}"
    print(summary_row)
    print("-" * 100)

    # Final validation result
    if all_valid:
        print("\nValidation PASSED: All checks successful!")
        if total_src_jpg == 0:
            print(
                "Warning: No source files found. Did you specify the correct directories?"
            )
    else:
        print("\nValidation FAILED: Some checks did not pass.")

        # Show invalid versions
        invalid_versions = [
            v
            for v, data in version_validation.items()
            if not data["counts_match"]
            or (data["resolution_check"] is False)
            or (data["json_check"] is False)
        ]

        if invalid_versions:
            print("Invalid versions:", ", ".join(invalid_versions))

        # Provide troubleshooting advice
        print("\nPossible issues:")
        print("1. Missing or mismatched file counts")
        print("2. Images not resized to half the original resolution")
        print("3. JSON files do not have the expected structure with -1 values")
        print("4. Files were renamed incorrectly")
        print("5. The conversion didn't complete successfully")

        print("\nTroubleshooting steps:")
        print("1. Run with --verbose to see detailed information about invalid files")
        print("2. Check the conversion script logs for errors")
        print("3. Make sure the original image dimensions provided are correct")

    return 0 if all_valid else 1


if __name__ == "__main__":
    sys.exit(main())
