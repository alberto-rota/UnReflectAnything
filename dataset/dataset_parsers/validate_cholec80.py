import os
import argparse
import json
import math
from collections import defaultdict
import sys

def count_files_with_extension(directory, extension):
    """Count files with a specific extension in a directory"""
    if not os.path.exists(directory):
        return 0
    return len([f for f in os.listdir(directory) if f.lower().endswith(extension.lower())])

def main():
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description='Validate CHOLEC80 directory structure conversion')
    parser.add_argument('--src', default='CHOLEC80', help='Source directory (default: CHOLEC80)')
    parser.add_argument('--dest', default='CHOLEC80_adapted', help='Destination directory (default: CHOLEC80_adapted)')
    parser.add_argument('--max-versions', type=int, default=80, help='Maximum number of version directories (default: 80)')
    parser.add_argument('--version-prefix', default='v', help='Prefix for version directories (default: "v")')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed information')
    args = parser.parse_args()
    
    # Define paths
    src_root = args.src
    dest_root = args.dest
    max_versions = args.max_versions
    version_prefix = args.version_prefix
    verbose = args.verbose
    
    # Get all source subdirectories
    src_frame_dir = os.path.join(src_root, "frame")
    
    # Check if the source directory exists
    if not os.path.exists(src_frame_dir):
        print(f"Error: Source directory '{src_frame_dir}' not found.")
        return 1
    
    # Get all subdirectories and sort them numerically
    src_subdirs = sorted([d for d in os.listdir(src_frame_dir) 
                          if os.path.isdir(os.path.join(src_frame_dir, d))],
                         key=lambda x: int(x) if x.isdigit() else float('inf'))
    
    # Calculate the number of version directories
    num_src_dirs = len(src_subdirs)
    num_versions = min(num_src_dirs, max_versions)
    
    # Create the same mapping logic as in the conversion script
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
    
    # Check all destination version directories exist
    missing_dest_dirs = []
    for i in range(1, num_versions + 1):
        version_dir = f"{version_prefix}{i}"
        frame_dir = os.path.join(dest_root, version_dir, "frame")
        poses_dir = os.path.join(dest_root, version_dir, "poses_absolute")
        
        if not os.path.exists(frame_dir):
            missing_dest_dirs.append(frame_dir)
        if not os.path.exists(poses_dir):
            missing_dest_dirs.append(poses_dir)
    
    if missing_dest_dirs:
        print("Error: The following destination directories are missing:")
        for dir_path in missing_dest_dirs:
            print(f"  - {dir_path}")
        return 1
    
    # Count files in each source directory
    src_file_counts = {}
    for subdir in src_subdirs:
        src_dir_path = os.path.join(src_frame_dir, subdir)
        jpg_count = count_files_with_extension(src_dir_path, '.jpg')
        src_file_counts[subdir] = jpg_count
    
    # Count files in each destination directory
    dest_jpg_counts = {}
    dest_json_counts = {}
    for i in range(1, num_versions + 1):
        version_dir = f"{version_prefix}{i}"
        frame_dir = os.path.join(dest_root, version_dir, "frame")
        poses_dir = os.path.join(dest_root, version_dir, "poses_absolute")
        
        jpg_count = count_files_with_extension(frame_dir, '.jpg')
        json_count = count_files_with_extension(poses_dir, '.json')
        
        dest_jpg_counts[version_dir] = jpg_count
        dest_json_counts[version_dir] = json_count
    
    # Group source directories by destination version
    src_by_version = defaultdict(list)
    for subdir, version in version_mapping.items():
        src_by_version[version].append(subdir)
    
    # Compare file counts
    all_valid = True
    total_src_jpg = 0
    total_dest_jpg = 0
    total_dest_json = 0
    
    # Print header
    print("\nValidation Report:")
    print("-" * 80)
    print(f"{'Source Dir':<12} {'Dest Version':<12} {'Source JPGs':<12} {'Dest JPGs':<12} {'Dest JSONs':<12} {'Status':<10}")
    print("-" * 80)
    
    # First check each source directory individually
    for subdir in src_subdirs:
        version = version_mapping[subdir]
        src_jpg = src_file_counts[subdir]
        total_src_jpg += src_jpg
        
        # In verbose mode, show each directory mapping
        if verbose:
            status = "OK" if src_jpg > 0 else "EMPTY"
            print(f"{subdir:<12} {version:<12} {src_jpg:<12} {'N/A':<12} {'N/A':<12} {status:<10}")
    
    # Now check each destination version directory
    version_valid = {}
    for version, subdirs in src_by_version.items():
        # Calculate expected file count for this version
        expected_jpg_count = sum(src_file_counts[subdir] for subdir in subdirs)
        actual_jpg_count = dest_jpg_counts[version]
        actual_json_count = dest_json_counts[version]
        
        total_dest_jpg += actual_jpg_count
        total_dest_json += actual_json_count
        
        # Compare counts
        counts_match = (expected_jpg_count == actual_jpg_count == actual_json_count)
        if counts_match:
            status = "VALID"
        else:
            status = "INVALID"
            all_valid = False
        
        version_valid[version] = counts_match
        
        # List source directories that map to this version
        src_dirs_str = ", ".join(subdirs) if len(subdirs) <= 3 else f"{len(subdirs)} directories"
        
        print(f"{'N/A':<12} {version:<12} {expected_jpg_count:<12} {actual_jpg_count:<12} {actual_json_count:<12} {status:<10}")
        if verbose and not counts_match:
            print(f"  Source directories: {src_dirs_str}")
    
    # Print summary
    print("-" * 80)
    print(f"{'TOTAL':<12} {'N/A':<12} {total_src_jpg:<12} {total_dest_jpg:<12} {total_dest_json:<12} {'VALID' if all_valid else 'INVALID':<10}")
    print("-" * 80)
    
    # Final validation result
    if all_valid:
        print("\nValidation PASSED: All file counts match!")
        if total_src_jpg == 0:
            print("Warning: No source files found. Did you specify the correct directories?")
    else:
        print("\nValidation FAILED: Some file counts don't match.")
        
        # Show invalid versions
        invalid_versions = [v for v, valid in version_valid.items() if not valid]
        print("Invalid versions:", ", ".join(invalid_versions))
        
        # Provide troubleshooting advice
        print("\nPossible issues:")
        print("1. The conversion script didn't complete successfully")
        print("2. Files were modified after conversion")
        print("3. Version mapping logic has changed between conversion and validation")
    
    return 0 if all_valid else 1

if __name__ == "__main__":
    sys.exit(main())