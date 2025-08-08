import os
import shutil
import json
import argparse
import math


def create_dir_if_not_exists(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)


def nullify(obj):
    """Recursively replace all values with null"""
    if isinstance(obj, dict):
        return {k: nullify(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [nullify(item) for item in obj]
    else:
        return -1


def main():
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(
        description="Convert CHOLEC80 directory structure to adapted format"
    )
    parser.add_argument(
        "--src", default="CHOLEC80", help="Source directory (default: CHOLEC80)"
    )
    parser.add_argument(
        "--dest",
        default="CHOLEC80_adapted",
        help="Destination directory (default: CHOLEC80_adapted)",
    )
    parser.add_argument(
        "--max-versions",
        type=int,
        default=80,
        help="Maximum number of version directories to create (default: 80)",
    )
    parser.add_argument(
        "--version-prefix",
        default="v",
        help='Prefix for version directories (default: "v")',
    )
    args = parser.parse_args()

    # Define paths
    src_root = args.src
    dest_root = args.dest
    max_versions = args.max_versions
    version_prefix = args.version_prefix

    # Get subdirectories in numerical order
    src_frame_dir = os.path.join(src_root, "frame")

    # Check if the source directory exists
    if not os.path.exists(src_frame_dir):
        print(f"Error: Source directory '{src_frame_dir}' not found.")
        return

    # Get all subdirectories and sort them numerically
    src_subdirs = sorted(
        [
            d
            for d in os.listdir(src_frame_dir)
            if os.path.isdir(os.path.join(src_frame_dir, d))
        ],
        key=lambda x: int(x) if x.isdigit() else float("inf"),
    )

    # JSON template structure
    json_template = {
        "camera-calibration": {
            "DL": [
                [
                    -0.0005951574421487749,
                    -0.0005466293077915907,
                    0.0,
                    0.0,
                    0.0018295900663360953,
                ]
            ],
            "DR": [
                [
                    -0.00023428065469488502,
                    -0.0007689339690841734,
                    0.0,
                    0.0,
                    0.0007763953180983663,
                ]
            ],
            "KL": [
                [1035.30810546875, 0.0, 596.9550170898438],
                [0.0, 1035.087646484375, 520.4100341796875],
                [0.0, 0.0, 1.0],
            ],
            "KR": [
                [1035.1741943359375, 0.0, 688.3618774414062],
                [0.0, 1034.97900390625, 521.07080078125],
                [0.0, 0.0, 1.0],
            ],
            "R": [
                [1.0, 1.9485649318085052e-05, -0.00015232479199767113],
                [-1.950531623151619e-05, 1.0, -0.00012911413796246052],
                [0.00015232227451633662, 0.0001291171065531671, 1.0],
            ],
            "T": [
                [-4.14339017868042],
                [-0.023819703608751297],
                [-0.0019068525871261954],
            ],
        },
        "camera-pose": [
            [
                0.9999965796048702,
                -0.0025485094056294315,
                -0.0002287824528534348,
                -0.02928418649968023,
            ],
            [
                0.002548294731588833,
                0.9999962957559811,
                -0.0008537852186089054,
                -0.10912194375544004,
            ],
            [
                0.00023095502338880993,
                0.0008533662118120291,
                0.9999997628384426,
                0.3536461362789396,
            ],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "timestamp": 1558660318896583,
    }

    # Nullify the template (replace all values with null)
    nullified_template = nullify(json_template)

    # Determine the number of source directories
    num_src_dirs = len(src_subdirs)
    print(f"Found {num_src_dirs} source directories")

    # Calculate the number of version directories to create
    num_versions = min(num_src_dirs, max_versions)
    print(f"Will create {num_versions} version directories")

    # Create all version directories
    for i in range(1, num_versions + 1):
        version_dir = f"{version_prefix}{i}"
        create_dir_if_not_exists(os.path.join(dest_root, version_dir, "frame"))
        create_dir_if_not_exists(os.path.join(dest_root, version_dir, "poses_absolute"))

    # Map source directories to version directories
    version_mapping = {}

    # Simple case: one source directory to one version
    if num_src_dirs <= max_versions:
        for i, subdir in enumerate(src_subdirs, 1):
            version_mapping[subdir] = f"{version_prefix}{i}"
    else:
        # Complex case: distribute multiple source directories across versions
        dirs_per_version = math.ceil(num_src_dirs / max_versions)
        for i, subdir in enumerate(src_subdirs):
            version_num = (i // dirs_per_version) + 1
            if version_num <= max_versions:
                version_mapping[subdir] = f"{version_prefix}{version_num}"
            else:
                version_mapping[subdir] = f"{version_prefix}{max_versions}"

    # Counters for each version
    version_counters = {f"{version_prefix}{i}": 0 for i in range(1, num_versions + 1)}

    # Process each directory
    for subdir in src_subdirs:
        try:
            # Get the mapped version
            dest_version = version_mapping.get(subdir)
            if not dest_version:
                print(f"Warning: No version mapping for directory {subdir}. Skipping.")
                continue

            # Get all image files in the subdirectory
            src_img_dir = os.path.join(src_frame_dir, subdir)
            img_files = sorted(
                [
                    f
                    for f in os.listdir(src_img_dir)
                    if f.endswith(".jpg") and f[:-4].isdigit()
                ],
                key=lambda x: int(x[:-4]),
            )

            print(
                f"Processing directory {subdir} -> {dest_version} ({len(img_files)} images)"
            )

            for img_file in img_files:
                # Create new filenames with 6-digit numbering
                counter = version_counters[dest_version]
                new_img_name = f"{counter:06d}.jpg"
                new_json_name = f"{counter:06d}.json"
                version_counters[dest_version] += 1

                # Copy the image file
                src_img_path = os.path.join(src_img_dir, img_file)
                dest_img_path = os.path.join(
                    dest_root, dest_version, "frame", new_img_name
                )
                shutil.copy2(src_img_path, dest_img_path)

                # Create the JSON file
                dest_json_path = os.path.join(
                    dest_root, dest_version, "poses_absolute", new_json_name
                )
                with open(dest_json_path, "w") as f:
                    json.dump(nullified_template, f, indent=4)

        except ValueError:
            print(f"Warning: Could not process directory '{subdir}'. Skipping.")
            continue

    # Print summary statistics
    print("\nConversion complete. Files created per version:")
    total_files = 0
    for version, count in sorted(version_counters.items()):
        print(f"- {version}: {count} files")
        total_files += count
    print(f"Total: {total_files} files created")


if __name__ == "__main__":
    main()
