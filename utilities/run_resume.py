"""
Utilities for resuming training runs.
Contains functions to discover, validate, and load existing training runs.
"""

import os
import json
import glob
from typing import Optional, Dict, Any, Tuple
import torch
from logger import get_logger

logger = get_logger(__name__)


def find_run_by_name_or_id(runs_dir: str, run_identifier: str) -> Optional[str]:
    """
    Find a run directory by name or ID.
    
    Args:
        runs_dir (str): Directory containing all runs
        run_identifier (str): Run name or run ID to search for
        
    Returns:
        Optional[str]: Path to the run directory if found, None otherwise
    """
    if not os.path.exists(runs_dir):
        logger.error(f"Runs directory does not exist: {runs_dir}")
        return None
    
    # First, try exact match with run name
    run_path = os.path.join(runs_dir, run_identifier)
    if os.path.exists(run_path):
        return run_path
    
    # If not found, search for runs that contain the identifier
    for run_dir in os.listdir(runs_dir):
        if run_identifier in run_dir:
            run_path = os.path.join(runs_dir, run_dir)
            if os.path.isdir(run_path):
                return run_path
    
    logger.warning(f"Run not found: {run_identifier}")
    return None


def validate_run_for_resume(run_dir: str) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """
    Validate that a run directory contains the necessary files for resuming.
    
    Args:
        run_dir (str): Path to the run directory
        
    Returns:
        Tuple[bool, str, Optional[Dict]]: (is_valid, error_message, run_info)
    """
    if not os.path.exists(run_dir):
        return False, f"Run directory does not exist: {run_dir}", None
    
    # Check for models directory
    models_dir = os.path.join(run_dir, "models")
    if not os.path.exists(models_dir):
        return False, "No models directory found in run", None
    
    # Check for config file (try config.json first, then hyperparams.json as fallback)
    config_path = os.path.join(run_dir, "config.json")
    hyperparams_path = os.path.join(run_dir, "hyperparams.json")
    
    config = None
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
        except Exception as e:
            return False, f"Failed to load config.json: {e}", None
    elif os.path.exists(hyperparams_path):
        try:
            with open(hyperparams_path, 'r') as f:
                hyperparams_data = json.load(f)
                # Extract config from hyperparams structure
                config = hyperparams_data.get("training", hyperparams_data)
        except Exception as e:
            return False, f"Failed to load hyperparams.json: {e}", None
    else:
        return False, "No config.json or hyperparams.json found in run directory", None
    
    # Check for checkpoints
    checkpoint_files = glob.glob(os.path.join(models_dir, "checkpoint_epoch_*.pth"))
    best_model_path = os.path.join(models_dir, "weights_best.pt")
    
    if not checkpoint_files and not os.path.exists(best_model_path):
        return False, "No checkpoints found in models directory", None
    
    # Find the latest checkpoint
    latest_checkpoint = None
    latest_epoch = -1
    
    for checkpoint_file in checkpoint_files:
        try:
            # Extract epoch number from filename
            filename = os.path.basename(checkpoint_file)
            epoch_str = filename.replace("checkpoint_epoch_", "").replace(".pth", "")
            epoch_num = int(epoch_str)
            
            if epoch_num > latest_epoch:
                latest_epoch = epoch_num
                latest_checkpoint = checkpoint_file
        except ValueError:
            continue
    
    # If no regular checkpoints found, use best model
    if latest_checkpoint is None and os.path.exists(best_model_path):
        latest_checkpoint = best_model_path
        # Try to get epoch from best model
        try:
            checkpoint = torch.load(best_model_path, map_location='cpu', weights_only=False)
            latest_epoch = checkpoint.get('epoch', 0)
        except Exception:
            latest_epoch = 0
    
    if latest_checkpoint is None:
        return False, "No valid checkpoints found", None
    
    run_info = {
        'run_dir': run_dir,
        'config': config,
        'latest_checkpoint': latest_checkpoint,
        'latest_epoch': latest_epoch,
        'models_dir': models_dir
    }
    
    return True, "", run_info


def load_checkpoint_for_resume(checkpoint_path: str, device: torch.device) -> Optional[Dict[str, Any]]:
    """
    Load checkpoint data for resuming training.
    
    Args:
        checkpoint_path (str): Path to the checkpoint file
        device (torch.device): Device to load the checkpoint on
        
    Returns:
        Optional[Dict]: Checkpoint data if successful, None otherwise
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # Validate checkpoint structure
        required_keys = ['model_state_dict', 'optimizer_state_dict', 'epoch', 'config']
        missing_keys = [key for key in required_keys if key not in checkpoint]
        
        if missing_keys:
            logger.error(f"Checkpoint missing required keys: {missing_keys}")
            return None
        
        return checkpoint
        
    except Exception as e:
        logger.error(f"Failed to load checkpoint {checkpoint_path}: {e}")
        return None


def get_resume_info(run_identifier: str, runs_dir: str) -> Optional[Dict[str, Any]]:
    """
    Get all information needed to resume a training run.
    
    Args:
        run_identifier (str): Run name or ID to resume
        runs_dir (str): Directory containing all runs
        
    Returns:
        Optional[Dict]: Resume information if successful, None otherwise
    """
    # Find the run directory
    run_dir = find_run_by_name_or_id(runs_dir, run_identifier)
    if run_dir is None:
        return None
    
    # Validate the run
    is_valid, error_msg, run_info = validate_run_for_resume(run_dir)
    if not is_valid:
        logger.error(f"Run validation failed: {error_msg}")
        return None
    
    logger.info(f"Found valid run to resume: {run_dir}")
    logger.info(f"Latest checkpoint: {run_info['latest_checkpoint']}")
    logger.info(f"Latest epoch: {run_info['latest_epoch']}")
    
    return run_info
