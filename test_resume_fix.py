#!/usr/bin/env python3
"""
Test script to verify the resume functionality fix.
This script tests the config file loading functionality.
"""

import os
import json
import tempfile
from utilities.run_resume import validate_run_for_resume

def test_config_loading():
    """Test that the resume functionality can load config from both config.json and hyperparams.json"""
    
    # Create a temporary directory structure
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create models directory
        models_dir = os.path.join(temp_dir, "models")
        os.makedirs(models_dir)
        
        # Create a dummy checkpoint
        checkpoint_path = os.path.join(models_dir, "checkpoint_epoch_1.pth")
        with open(checkpoint_path, 'w') as f:
            f.write("dummy checkpoint")
        
        # Test 1: config.json exists
        config_data = {"EPOCHS": 100, "LEARNING_RATE": 0.001}
        config_path = os.path.join(temp_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config_data, f)
        
        is_valid, error_msg, run_info = validate_run_for_resume(temp_dir)
        print(f"Test 1 (config.json): Valid={is_valid}, Error='{error_msg}'")
        if is_valid:
            print(f"  Config loaded: {run_info['config']['EPOCHS']} epochs")
        
        # Test 2: Only hyperparams.json exists (legacy format)
        os.remove(config_path)
        hyperparams_data = {"training": config_data}
        hyperparams_path = os.path.join(temp_dir, "hyperparams.json")
        with open(hyperparams_path, 'w') as f:
            json.dump(hyperparams_data, f)
        
        is_valid, error_msg, run_info = validate_run_for_resume(temp_dir)
        print(f"Test 2 (hyperparams.json): Valid={is_valid}, Error='{error_msg}'")
        if is_valid:
            print(f"  Config loaded: {run_info['config']['EPOCHS']} epochs")
        
        # Test 3: Neither file exists
        os.remove(hyperparams_path)
        is_valid, error_msg, run_info = validate_run_for_resume(temp_dir)
        print(f"Test 3 (no config files): Valid={is_valid}, Error='{error_msg}'")
        
        print("\nAll tests completed!")

if __name__ == "__main__":
    test_config_loading()
