#!/usr/bin/env python3
"""
Pytest-based unit tests for RGBP_Dataset class and dataset creation from configuration.

This test suite covers:
1. RGBP_Dataset creation from config_train.yaml
2. Individual dataset class instantiation (SCRREAM, HOUSECAT6D, POLARGB)
3. Dataset concatenation functionality
4. Data loading from concatenated datasets
5. Polarization format support
6. Scene filtering functionality
7. Configuration parsing and validation

Author: Generated for UnReflectAnything project
"""

import pytest
import tempfile
import shutil
import os
import yaml
import torch
import numpy as np
from PIL import Image
from unittest.mock import patch
import warnings

# Import the classes we want to test
from dataset.rgbp import (
    RGBP_Dataset,
    SCRREAM_Dataset,
    HOUSECAT6D_Dataset,
    POLARGB_Dataset,
    create_datasets_from_config,
    load_config_and_create_datasets,
)
from torch.utils.data import ConcatDataset


# ===============================================
# FIXTURES
# ===============================================


@pytest.fixture(scope="session")
def temp_dir():
    """Create a temporary directory for test data."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture(scope="session")
def test_config(temp_dir):
    """Load actual config_train.yaml and modify paths for testing."""
    import yaml

    # Load the actual config file
    with open("config_train.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Update dataset paths to use temp directory for testing
    for dataset_name in config["parameters"]["DATASETS"]["value"]:
        dataset_config = config["parameters"]["DATASETS"]["value"][dataset_name]
        # Update root directory to use temp directory
        dataset_config["ROOT_DIR"] = os.path.join(temp_dir, dataset_name)
        # Enable few_images for faster testing
        dataset_config["FEW_IMAGES"] = True

    return config


@pytest.fixture(scope="session")
def config_file(test_config, temp_dir):
    """Create a config file for testing."""
    config_path = "../config_train.yaml"
    with open(config_path, "w") as f:
        yaml.dump(test_config, f)
    return config_path


@pytest.fixture(scope="session")
def mock_datasets(temp_dir):
    """Create mock dataset directory structures and files."""
    datasets_info = [
        (
            "SCRREAM",
            "single_file_clock",
            ["scene01", "scene09_full_00", "scene09_reduced_00"],
        ),
        ("HOUSECAT6D", "single_file_clock", ["scene01", "scene02"]),
        ("POLARGB", "separate_files", ["train_hard_scene_31"]),
    ]

    def create_mock_rgb_image(path: str):
        """Create a mock RGB image file."""
        img = Image.new("RGB", (224, 224), color=(128, 128, 128))
        img.save(path)

    def create_mock_polarization_image_single(path: str):
        """Create a mock polarization image for single_file_clock format."""
        img = Image.new("RGB", (448, 448), color=(100, 100, 100))
        img.save(path)

    def create_mock_intrinsics(path: str):
        """Create a mock intrinsics file."""
        K = np.eye(3, dtype=np.float32)
        K[0, 0] = 500.0  # fx
        K[1, 1] = 500.0  # fy
        K[0, 2] = 112.0  # cx
        K[1, 2] = 112.0  # cy
        np.savetxt(path, K.flatten())

    for dataset_name, pol_format, scenes in datasets_info:
        dataset_dir = os.path.join(temp_dir, dataset_name)

        for scene in scenes:
            scene_dir = os.path.join(dataset_dir, scene)
            rgb_dir = os.path.join(scene_dir, "rgb")
            pol_dir = os.path.join(scene_dir, "pol")

            os.makedirs(rgb_dir, exist_ok=True)
            os.makedirs(pol_dir, exist_ok=True)

            # Create mock RGB image
            rgb_path = os.path.join(rgb_dir, "000001.png")
            create_mock_rgb_image(rgb_path)

            # Create mock polarization images based on format
            if pol_format == "single_file_clock":
                pol_path = os.path.join(pol_dir, "000001.png")
                create_mock_polarization_image_single(pol_path)
            elif pol_format == "separate_files":
                for angle in ["000", "045", "090", "135"]:
                    pol_path = os.path.join(pol_dir, f"000001_{angle}.png")
                    create_mock_rgb_image(pol_path)

            # Create mock intrinsics file
            intrinsics_path = os.path.join(scene_dir, "intrinsics.txt")
            create_mock_intrinsics(intrinsics_path)

    return temp_dir


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Set up test environment before each test."""
    torch.manual_seed(42)
    np.random.seed(42)
    warnings.filterwarnings("ignore", category=UserWarning)


# ===============================================
# CONFIGURATION AND PARSING TESTS
# ===============================================


class TestConfigurationParsing:
    """Test configuration loading and parsing."""

    def test_config_loading_and_parsing(self, config_file):
        """Test that configuration is loaded and parsed correctly."""
        with open(config_file, "r") as f:
            loaded_config = yaml.safe_load(f)

        assert "parameters" in loaded_config
        assert "DATASETS" in loaded_config["parameters"]
        assert "value" in loaded_config["parameters"]["DATASETS"]

        datasets_config = loaded_config["parameters"]["DATASETS"]["value"]
        assert "SCRREAM" in datasets_config
        assert "HOUSECAT6D" in datasets_config
        assert "POLARGB" in datasets_config

    def test_invalid_config_handling(self, temp_dir):
        """Test handling of invalid configurations."""
        # Test with missing DATASETS section
        invalid_config = {"parameters": {}}

        with patch("builtins.print"):
            with pytest.raises(ValueError, match="No datasets found"):
                create_datasets_from_config(invalid_config)

        # Test with invalid config file path
        with pytest.raises(ValueError, match="Error loading configuration"):
            load_config_and_create_datasets("/nonexistent/path.yaml")


# ===============================================
# DATASET CREATION TESTS
# ===============================================


class TestDatasetCreation:
    """Test dataset creation from configuration."""

    def test_create_datasets_from_config(self, test_config, mock_datasets):
        """Test create_datasets_from_config function."""
        with patch("builtins.print"):
            datasets = create_datasets_from_config(test_config)

        # Check that all expected keys are present
        assert "training" in datasets
        assert "validation" in datasets
        assert "test" in datasets

        # Check that training and validation datasets are created
        assert datasets["training"] is not None
        assert datasets["validation"] is not None

        # Check that datasets are ConcatDataset instances
        assert isinstance(datasets["training"], ConcatDataset)
        assert isinstance(datasets["validation"], ConcatDataset)

    def test_load_config_and_create_datasets(self, config_file, mock_datasets):
        """Test load_config_and_create_datasets function."""
        with patch("builtins.print"):
            datasets = load_config_and_create_datasets(config_file)

        assert "training" in datasets
        assert "validation" in datasets
        assert isinstance(datasets["training"], ConcatDataset)
        assert isinstance(datasets["validation"], ConcatDataset)

    @pytest.mark.parametrize("dataset_name", ["SCRREAM", "HOUSECAT6D", "POLARGB"])
    def test_specific_dataset_creation(self, test_config, mock_datasets, dataset_name):
        """Test creation of specific dataset types."""
        with patch("builtins.print"):
            datasets = create_datasets_from_config(
                test_config, dataset_names=[dataset_name]
            )

        assert datasets["training"] is not None
        assert isinstance(datasets["training"], ConcatDataset)

        # Check that individual datasets in concat are correct type
        expected_class = {
            "SCRREAM": SCRREAM_Dataset,
            "HOUSECAT6D": HOUSECAT6D_Dataset,
            "POLARGB": POLARGB_Dataset,
        }[dataset_name]

        for dataset in datasets["training"].datasets:
            assert isinstance(dataset, expected_class)


# ===============================================
# INDIVIDUAL DATASET CLASS TESTS
# ===============================================


class TestDatasetClasses:
    """Test individual RGBP_Dataset classes."""

    def test_scrream_dataset_initialization(self, test_config, mock_datasets):
        """Test SCRREAM_Dataset initialization."""
        dataset_config = test_config["parameters"]["DATASETS"]["value"]["SCRREAM"]

        dataset = SCRREAM_Dataset(
            root_dir=dataset_config["ROOT_DIR"],
            polarization_format=dataset_config["POLARIZATION_FORMAT"],
            target_size=tuple(dataset_config["TARGET_SIZE"]),
            resize_mode=dataset_config["RESIZE_MODE"],
            rho_s=dataset_config["RHO_S"],
            eps=dataset_config["EPS"],
            use_cache=dataset_config["USE_CACHE"],
            simplify_upsampling=dataset_config["SIMPLIFY_UPSAMPLING"],
            few_images=dataset_config["FEW_IMAGES"],
        )

        assert isinstance(dataset, SCRREAM_Dataset)
        assert isinstance(dataset, RGBP_Dataset)
        assert dataset.polarization_format == "single_file_clock"
        assert dataset.target_size == (224, 224)
        assert dataset.resize_mode == "crop"

    def test_housecat6d_dataset_initialization(self, test_config, mock_datasets):
        """Test HOUSECAT6D_Dataset initialization."""
        dataset_config = test_config["parameters"]["DATASETS"]["value"]["HOUSECAT6D"]

        dataset = HOUSECAT6D_Dataset(
            root_dir=dataset_config["ROOT_DIR"],
            polarization_format=dataset_config["POLARIZATION_FORMAT"],
            target_size=tuple(dataset_config["TARGET_SIZE"]),
            few_images=dataset_config["FEW_IMAGES"],
        )

        assert isinstance(dataset, HOUSECAT6D_Dataset)
        assert isinstance(dataset, RGBP_Dataset)

    def test_polargb_dataset_initialization(self, test_config, mock_datasets):
        """Test POLARGB_Dataset initialization."""
        dataset_config = test_config["parameters"]["DATASETS"]["value"]["POLARGB"]

        dataset = POLARGB_Dataset(
            root_dir=dataset_config["ROOT_DIR"],
            polarization_format=dataset_config["POLARIZATION_FORMAT"],
            target_size=tuple(dataset_config["TARGET_SIZE"]),
            few_images=dataset_config["FEW_IMAGES"],
        )

        assert isinstance(dataset, POLARGB_Dataset)
        assert isinstance(dataset, RGBP_Dataset)
        assert dataset.polarization_format == "separate_files"


# ===============================================
# DATASET FUNCTIONALITY TESTS
# ===============================================


class TestDatasetFunctionality:
    """Test RGBP_Dataset functionality and data loading."""

    @pytest.mark.parametrize(
        "pol_format", ["single_file_clock", "separate_files", "mosaic"]
    )
    def test_polarization_format_validation(self, temp_dir, mock_datasets, pol_format):
        """Test polarization format validation."""
        dataset_dir = os.path.join(temp_dir, "SCRREAM")

        dataset = RGBP_Dataset(
            root_dir=dataset_dir, polarization_format=pol_format, few_images=True
        )
        assert dataset.polarization_format == pol_format

    def test_invalid_polarization_format(self, temp_dir, mock_datasets):
        """Test invalid polarization format raises error."""
        dataset_dir = os.path.join(temp_dir, "SCRREAM")

        with pytest.raises(ValueError, match="polarization_format must be one of"):
            RGBP_Dataset(
                root_dir=dataset_dir,
                polarization_format="invalid_format",
                few_images=True,
            )

    @pytest.mark.parametrize("resize_mode", ["crop", "resize", "pad"])
    def test_resize_mode_validation(self, temp_dir, mock_datasets, resize_mode):
        """Test resize mode validation."""
        dataset_dir = os.path.join(temp_dir, "SCRREAM")

        dataset = RGBP_Dataset(
            root_dir=dataset_dir, resize_mode=resize_mode, few_images=True
        )
        assert dataset.resize_mode == resize_mode

    def test_invalid_resize_mode(self, temp_dir, mock_datasets):
        """Test invalid resize mode raises error."""
        dataset_dir = os.path.join(temp_dir, "SCRREAM")

        with pytest.raises(ValueError, match="resize_mode must be one of"):
            RGBP_Dataset(
                root_dir=dataset_dir, resize_mode="invalid_mode", few_images=True
            )

    def test_scene_filtering_include(self, temp_dir, mock_datasets):
        """Test scene filtering with include filter."""
        dataset_dir = os.path.join(temp_dir, "SCRREAM")

        dataset = RGBP_Dataset(
            root_dir=dataset_dir, include=["scene01"], few_images=True
        )

        loaded_scenes = dataset.get_loaded_scenes()
        assert "scene01" in loaded_scenes
        assert "scene09_full_00" not in loaded_scenes

    def test_scene_filtering_exclude(self, temp_dir, mock_datasets):
        """Test scene filtering with exclude filter."""
        dataset_dir = os.path.join(temp_dir, "SCRREAM")

        dataset = RGBP_Dataset(
            root_dir=dataset_dir,
            exclude=["scene09"],  # Should exclude scenes containing 'scene09'
            few_images=True,
        )

        loaded_scenes = dataset.get_loaded_scenes()
        assert "scene01" in loaded_scenes
        # Should exclude both scene09_full_00 and scene09_reduced_00
        assert "scene09_full_00" not in loaded_scenes
        assert "scene09_reduced_00" not in loaded_scenes

    def test_dataset_length_and_indexing(self, temp_dir, mock_datasets):
        """Test dataset length and item access."""
        dataset_dir = os.path.join(temp_dir, "SCRREAM")

        dataset = RGBP_Dataset(root_dir=dataset_dir, few_images=True)

        # Check that dataset has expected length
        assert len(dataset) > 0

        # Test that we can access items
        if len(dataset) > 0:
            sample = dataset[0]
            assert isinstance(sample, dict)

            # Check expected keys in sample
            expected_keys = [
                "rgb",
                "specular",
                "diffuse",
                "intrinsics",
                "DoLP",
                "AoP",
                "f_spec",
            ]
            for key in expected_keys:
                assert key in sample, f"Missing key: {key}"

            # Check tensor shapes
            assert sample["rgb"].shape[0] == 3  # RGB channels
            assert sample["rgb"].shape[1] == 224  # Height
            assert sample["rgb"].shape[2] == 224  # Width


# ===============================================
# DATA LOADING TESTS
# ===============================================


class TestDataLoading:
    """Test data loading functionality."""

    def test_data_loading_single_file_clock(self, temp_dir, mock_datasets):
        """Test data loading with single_file_clock polarization format."""
        dataset_dir = os.path.join(temp_dir, "SCRREAM")

        dataset = RGBP_Dataset(
            root_dir=dataset_dir,
            polarization_format="single_file_clock",
            target_size=(224, 224),
            few_images=True,
        )

        if len(dataset) > 0:
            sample = dataset[0]

            # Check polarization data keys
            pol_keys = ["I0", "I45", "I90", "I135", "S0", "S1", "S2", "DoLP", "AoP"]
            for key in pol_keys:
                assert key in sample
                assert isinstance(sample[key], torch.Tensor)

    @pytest.mark.parametrize(
        "resize_mode,target_size",
        [("crop", (224, 224)), ("resize", (224, 224)), ("pad", (224, 224))],
    )
    def test_tensor_resizing(self, temp_dir, mock_datasets, resize_mode, target_size):
        """Test tensor resizing functionality."""
        dataset_dir = os.path.join(temp_dir, "SCRREAM")

        dataset = RGBP_Dataset(
            root_dir=dataset_dir,
            target_size=target_size,
            resize_mode=resize_mode,
            few_images=True,
        )

        if len(dataset) > 0:
            sample = dataset[0]

            # Check that tensors are resized correctly
            assert sample["rgb"].shape[1] == target_size[0]
            assert sample["rgb"].shape[2] == target_size[1]


# ===============================================
# CONCATENATION TESTS
# ===============================================


class TestDatasetConcatenation:
    """Test dataset concatenation functionality."""

    def test_concatenated_dataset_creation(self, test_config, mock_datasets):
        """Test that concatenated datasets are created correctly."""
        with patch("builtins.print"):
            datasets = create_datasets_from_config(test_config)

        train_dataset = datasets["training"]
        val_dataset = datasets["validation"]

        # Check that datasets are ConcatDataset instances
        assert isinstance(train_dataset, ConcatDataset)
        assert isinstance(val_dataset, ConcatDataset)

        # Check that individual datasets are correct types
        for dataset in train_dataset.datasets:
            assert isinstance(dataset, RGBP_Dataset)

        for dataset in val_dataset.datasets:
            assert isinstance(dataset, RGBP_Dataset)

    def test_concatenated_dataset_data_loading(self, test_config, mock_datasets):
        """Test data loading from concatenated datasets."""
        with patch("builtins.print"):
            datasets = create_datasets_from_config(test_config)

        train_dataset = datasets["training"]

        if len(train_dataset) > 0:
            # Test loading samples from concatenated dataset
            sample = train_dataset[0]
            assert isinstance(sample, dict)

            # Test that we can iterate through the dataset
            sample_count = 0
            for sample in train_dataset:
                sample_count += 1
                assert isinstance(sample, dict)
                if sample_count >= 5:  # Test first 5 samples
                    break

            assert sample_count > 0

    def test_validation_scene_splitting(self, test_config, mock_datasets):
        """Test that validation scenes are split correctly."""
        with patch("builtins.print"):
            datasets = create_datasets_from_config(test_config)

        train_dataset = datasets["training"]
        val_dataset = datasets["validation"]

        # Get scenes from each dataset
        train_scenes = set()
        val_scenes = set()

        for dataset in train_dataset.datasets:
            if hasattr(dataset, "get_loaded_scenes"):
                train_scenes.update(dataset.get_loaded_scenes())

        for dataset in val_dataset.datasets:
            if hasattr(dataset, "get_loaded_scenes"):
                val_scenes.update(dataset.get_loaded_scenes())

        # Check that validation scenes are not in training
        val_scene_patterns = ["scene09_full_00", "scene09_reduced_00"]
        for pattern in val_scene_patterns:
            if any(pattern in scene for scene in val_scenes):
                # If validation scene is found, it shouldn't be in training
                assert not any(pattern in scene for scene in train_scenes)


# ===============================================
# ERROR HANDLING TESTS
# ===============================================


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_missing_intrinsics_file(self, temp_dir):
        """Test handling of missing intrinsics files."""
        # Create a dataset directory without intrinsics files
        test_dir = os.path.join(temp_dir, "test_no_intrinsics")
        scene_dir = os.path.join(test_dir, "scene01")
        rgb_dir = os.path.join(scene_dir, "rgb")
        pol_dir = os.path.join(scene_dir, "pol")

        os.makedirs(rgb_dir, exist_ok=True)
        os.makedirs(pol_dir, exist_ok=True)

        # Create mock images but no intrinsics
        rgb_img = Image.new("RGB", (224, 224), color=(128, 128, 128))
        rgb_img.save(os.path.join(rgb_dir, "000001.png"))

        pol_img = Image.new("RGB", (448, 448), color=(100, 100, 100))
        pol_img.save(os.path.join(pol_dir, "000001.png"))

        dataset = RGBP_Dataset(root_dir=test_dir, few_images=True)

        if len(dataset) > 0:
            # Should load successfully with identity intrinsics
            sample = dataset[0]
            assert "intrinsics" in sample
            assert isinstance(sample["intrinsics"], torch.Tensor)

    def test_few_images_mode(self, temp_dir, mock_datasets):
        """Test few_images mode functionality."""
        dataset_dir = os.path.join(temp_dir, "SCRREAM")

        # Create dataset with few_images=True
        dataset = RGBP_Dataset(root_dir=dataset_dir, few_images=True)

        # Should limit to 100 samples maximum
        assert len(dataset) <= 100

        # Create dataset with few_images=False
        dataset_full = RGBP_Dataset(root_dir=dataset_dir, few_images=False)

        # Should have same or more samples
        assert len(dataset_full) >= len(dataset)


# ===============================================
# INTEGRATION TESTS
# ===============================================


class TestIntegration:
    """Integration tests for the complete workflow."""

    def test_end_to_end_workflow(self, config_file, mock_datasets):
        """Test complete end-to-end workflow from config to data loading."""
        # Load config and create datasets
        with patch("builtins.print"):
            datasets = load_config_and_create_datasets(config_file)

        # Verify datasets are created
        assert datasets["training"] is not None
        assert datasets["validation"] is not None

        # Test data loading from training dataset
        train_dataset = datasets["training"]
        if len(train_dataset) > 0:
            sample = train_dataset[0]

            # Verify sample structure
            assert isinstance(sample, dict)
            assert "rgb" in sample
            assert "specular" in sample
            assert "diffuse" in sample
            assert "intrinsics" in sample

            # Verify tensor properties
            assert isinstance(sample["rgb"], torch.Tensor)
            assert sample["rgb"].dtype == torch.float32
            assert len(sample["rgb"].shape) == 3  # [C, H, W]
            assert sample["rgb"].shape[0] == 3  # RGB channels

    def test_dataloader_compatibility(self, test_config, mock_datasets):
        """Test compatibility with PyTorch DataLoader."""
        from torch.utils.data import DataLoader

        with patch("builtins.print"):
            datasets = create_datasets_from_config(test_config)

        train_dataset = datasets["training"]

        if len(train_dataset) > 0:
            # Create DataLoader
            dataloader = DataLoader(
                train_dataset,
                batch_size=2,
                shuffle=False,
                num_workers=0,  # Use 0 for testing to avoid multiprocessing issues
            )

            # Test loading a batch
            batch = next(iter(dataloader))

            # Verify batch structure
            assert isinstance(batch, dict)
            assert "rgb" in batch

            # Verify batch dimensions
            rgb_batch = batch["rgb"]
            assert rgb_batch.shape[0] <= 2  # Batch size
            assert rgb_batch.shape[1] == 3  # RGB channels
            assert rgb_batch.shape[2] == 224  # Height
            assert rgb_batch.shape[3] == 224  # Width


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
