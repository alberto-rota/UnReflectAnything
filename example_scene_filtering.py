#!/usr/bin/env python3
"""
Example demonstrating the improved scene filtering functionality in RGBP_Dataset.
"""

from dataset.rgbp import RGBP_Dataset

def example_string_filtering():
    """Examples using single string filtering."""
    
    print("=== String-based Scene Filtering ===")
    
    # Include all scenes containing "kitchen"
    dataset1 = RGBP_Dataset(
        root_dir="/path/to/dataset",
        include="kitchen",  # Single string - matches any scene with "kitchen" in name
        few_images=True
    )
    print(f"Scenes with 'kitchen': {dataset1.get_loaded_scenes()}")
    
    # Exclude all scenes containing "outdoor" 
    dataset2 = RGBP_Dataset(
        root_dir="/path/to/dataset",
        exclude="outdoor",  # Single string - excludes any scene with "outdoor" in name
        few_images=True
    )
    print(f"Scenes without 'outdoor': {dataset2.get_loaded_scenes()}")
    
    # Include scenes with "room" but exclude those with "dark"
    dataset3 = RGBP_Dataset(
        root_dir="/path/to/dataset", 
        include="room",
        exclude="dark",
        few_images=True
    )
    print(f"Scenes with 'room' but not 'dark': {dataset3.get_loaded_scenes()}")


def example_list_filtering():
    """Examples using list-based filtering."""
    
    print("\n=== List-based Scene Filtering ===")
    
    # Include specific scenes (exact match or substring)
    dataset1 = RGBP_Dataset(
        root_dir="/path/to/dataset",
        include=["scene_001", "scene_002", "kitchen"],  # Exact matches + substring
        few_images=True
    )
    print(f"Included scenes: {dataset1.get_loaded_scenes()}")
    
    # Exclude multiple patterns
    dataset2 = RGBP_Dataset(
        root_dir="/path/to/dataset",
        exclude=["test", "debug", "broken"],  # Multiple exclusion patterns
        few_images=True
    )
    print(f"Scenes excluding test/debug/broken: {dataset2.get_loaded_scenes()}")
    
    # Complex filtering: include training scenes but exclude validation ones
    dataset3 = RGBP_Dataset(
        root_dir="/path/to/dataset",
        include=["train", "scene"],  # Include scenes with "train" or "scene"
        exclude=["val", "validation"],  # But exclude validation scenes
        few_images=True
    )
    print(f"Training scenes only: {dataset3.get_loaded_scenes()}")


def example_backward_compatibility():
    """Examples showing backward compatibility with deprecated parameters."""
    
    print("\n=== Backward Compatibility (with deprecation warnings) ===")
    
    # This will work but show deprecation warnings
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        dataset_old = RGBP_Dataset(
            root_dir="/path/to/dataset",
            scene_names=["scene_001", "scene_002"],  # Deprecated
            ignore_scenes=["test_scene"],  # Deprecated
            few_images=True
        )
        
        # Print warnings
        for warning in w:
            print(f"Warning: {warning.message}")
    
    print(f"Old parameter usage still works: {dataset_old.get_loaded_scenes()}")


def example_mixed_usage():
    """Examples showing different combinations."""
    
    print("\n=== Mixed Usage Examples ===")
    
    # Use with different polarization formats
    dataset_single = RGBP_Dataset(
        root_dir="/path/to/dataset",
        polarization_format="single_file_clock",
        include="indoor",
        exclude=["broken", "corrupt"],
        few_images=True
    )
    
    dataset_separate = RGBP_Dataset(
        root_dir="/path/to/dataset",
        polarization_format="separate_files", 
        include=["scene_", "room_"],  # Include scenes starting with these patterns
        exclude="test",
        few_images=True
    )
    
    print(f"Single file format scenes: {dataset_single.get_loaded_scenes()}")
    print(f"Separate files format scenes: {dataset_separate.get_loaded_scenes()}")


if __name__ == "__main__":
    print("RGBP Dataset Scene Filtering Examples")
    print("=====================================")
    
    print("\nNew parameter names:")
    print("- 'include' (replaces 'scene_names'): str or List[str]")
    print("- 'exclude' (replaces 'ignore_scenes'): str or List[str]")
    print("\nFiltering behavior:")
    print("- String: Substring matching (e.g., 'kitchen' matches 'kitchen_001', 'my_kitchen')")
    print("- List: Each item can be exact match or substring")
    print("- Both exact matches and substring matches are supported")
    print("- exclude filter is applied first, then include filter")
    
    # Uncomment to test with actual data
    # example_string_filtering()
    # example_list_filtering() 
    # example_backward_compatibility()
    # example_mixed_usage()
