
#  MODULES AND DATASET LOADING
import torch
from dotmap import DotMap
import os, yaml, socket, time
import argparse
import debugpy
import ast
from rich.traceback import install
from engine import Engine
from dataset.rgbp import load_config_and_create_datasets
from models import RGBPOLDecomposer
from dotenv import load_dotenv
from logger import get_logger
logger = get_logger(__name__).set_context("IMPORT")
load_dotenv()

# Optional utilities (if available)
try:
    from utilities import *
except ImportError:
    logger.warning("Some utilities not available", context="WARNING")


def create_model_from_config(config, device):
    """
    Create the RGBPOLDecomposer model from configuration.
    
    Args:
        config: Configuration dictionary containing model parameters
        device: Device to place the model on
        
    Returns:
        RGBPOLDecomposer: The initialized model
    """
    # Access model configuration from the nested structure
    model_config = config.get("MODEL", {})#.get("value", {})
    
    # Get image dimensions from config (check multiple possible locations)
    target_size = None
    for dataset_name in ["SCRREAM", "HOUSECAT6D", "POLARGB"]:
        dataset_config = config.get("DATASETS", {}).get("value", {}).get(dataset_name, {})
        if "TARGET_SIZE" in dataset_config:
            target_size = dataset_config["TARGET_SIZE"]
            break
    
    if target_size is None:
        target_size = (224, 224)  # Default fallback
    
    if isinstance(target_size, list):
        target_size = tuple(target_size)
    
    # RGB Encoder configuration
    rgb_encoder_config = model_config.get("RGB_ENCODER", {})
    dinov3_cfg = {
        "model_name": rgb_encoder_config.get("ENCODER", "facebook/dinov3-vits16-pretrain-lvd1689m"),
        "image_size": rgb_encoder_config.get("IMAGE_SIZE", min(target_size)),
        "freeze_backbone": rgb_encoder_config.get("FREEZE_BACKBONE", True),
        "return_selected_layers": rgb_encoder_config.get("RETURN_SELECTED_LAYERS", [3, 6, 9, 12]),
        "return_last_hidden_state": rgb_encoder_config.get("RETURN_LAST_HIDDEN_STATE", False),
        "return_as_feature_maps": False,
        "return_cls_token": False,
    }

    # POL Encoder configuration
    pol_encoder_config = model_config.get("POL_ENCODER", {})
    pol_enc_cfg = {
        "in_ch": 3,
        "embed_dim": pol_encoder_config.get("EMBED_DIM", 384),
        "depth": pol_encoder_config.get("DEPTH", 4),
        "n_heads": pol_encoder_config.get("N_HEADS", 12),
        "patch_size": pol_encoder_config.get("PATCH_SIZE", 16),
    }
    # Cross-attention configuration
    cross_attn_config = model_config.get("CROSS_ATTN", {})
    cross_attn_cfg = {
        "embed_dim": cross_attn_config.get("EMBED_DIM", 384),
        "n_heads": cross_attn_config.get("N_HEADS", 12),
        "dropout": cross_attn_config.get("DROPOUT", 0.1),
        "bi_directional": cross_attn_config.get("BI_DIRECTIONAL", False)
    }
    
    # Decoder configuration
    decoder_config = model_config.get("DECODER", {})
    decoder_cfg = {
        "use_bn": decoder_config.get("USE_BN", True),
        "readout_type": decoder_config.get("READOUT_TYPE", "ignore"),
        "feature_dim": decoder_config.get("FEATURE_DIM", 384),
        "output_image_size": decoder_config.get("OUTPUT_IMAGE_SIZE", min(target_size)),
        "output_channels": decoder_config.get("OUTPUT_CHANNELS", 3),
    }


    # Create the main model
    model = RGBPOLDecomposer(
        dinov3=dinov3_cfg,
        pol_encoder=pol_enc_cfg,
        pol_cross_attn=cross_attn_cfg,
        spec_decoder=decoder_cfg,
        diffuse_decoder=decoder_cfg,
        highlight_decoder=decoder_cfg,
    ).to(device)
    
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters", context="MODEL")
    return model


def create_datasets_from_config(config, config_path):
    """
    Create training and validation datasets from configuration using the new system.
    
    This function now uses the improved dataset creation system that:
    - Reads from YAML config files with DATASETS section
    - Supports multiple datasets (SCRREAM, HOUSECAT6D, etc.)
    - Creates dataset-specific classes
    - Returns ConcatDatasets for multi-dataset training
    
    Args:
        config: Configuration dictionary (DotMap) containing dataset parameters
        config_path: Path to the YAML config file for direct loading
        
    Returns:
        dict: Dictionary containing datasets with keys expected by Engine
    """
    try:
        # Use the new dataset creation system
        datasets = load_config_and_create_datasets(config_path)
        
        # Convert keys to match what Engine expects (capitalize first letter)
        result = {
            "Training": datasets.get('training'),
            "Validation": datasets.get('validation'), 
            "Test": datasets.get('test'),
            "workers": config.get("WORKERS", 4)
        }
                
        return result
        
    except Exception as e:
        logger.error(f"Failed to create datasets using new system: {e}")
        logger.warning("This may be due to:")
        logger.warning("1. Missing or incorrect DATASETS section in config file")
        logger.warning("2. Invalid dataset root directories")
        logger.warning("3. Missing dataset classes")
        logger.warning("Please check your config_train.yaml file structure")
        raise


def load_and_process_config(config_path, config=None, unknown_args=None, boot_mode=False):
    """
    Load and process configuration from file or direct input.
    
    Args:
        config_path (str): Path to the YAML configuration file
        config (dict, optional): Direct configuration dictionary (overrides file loading)
        unknown_args (list, optional): List of unknown command-line arguments to process
        boot_mode (bool): Whether to enable boot mode with minimal parameters
        
    Returns:
        DotMap: Processed configuration object
    """
    if config is None:
        # Load the configuration file
        with open(config_path, "r") as f:
            config_yaml = yaml.safe_load(f)
        config_parameters = config_yaml["parameters"]
        config_dict = {
            k: v.get("value") for k, v in config_parameters.items() if v is not None
        }
    else:
        config_dict = config

    # Process the unknown command-line arguments
    if unknown_args:
        additional_args = {}
        for arg in unknown_args:
            if arg.startswith("--"):
                key_value = arg.lstrip("--").split("=", 1)
                if len(key_value) == 2:
                    key, value = key_value
                    additional_args[key.upper()] = value
                else:
                    key = key_value[0]
                    additional_args[key.upper()] = None

        # Override parameters in the configuration with command-line arguments
        for key, value in additional_args.items():
            if key in config_dict:
                orig_value = config_dict[key]
                orig_type = type(orig_value)
                try:
                    if orig_type == bool:
                        if value.lower() in ("true", "1", "yes"):
                            new_value = True
                        elif value.lower() in ("false", "0", "no"):
                            new_value = False
                        else:
                            raise ValueError(f"Cannot parse boolean value: {value}")
                    elif orig_type == list:
                        new_value = ast.literal_eval(value)
                    else:
                        new_value = orig_type(value)
                    config_dict[key] = new_value
                except Exception as e:
                    print(f"Could not convert value for {key}: {value}, error: {e}")
            else:
                print(f"Warning: Unknown parameter {key}")

    # Convert the configuration dictionary to a DotMap for easy access
    config = DotMap(config_dict)

    # Override parameters if boot mode is enabled
    if boot_mode:
        config.BATCH_SIZE = 1
        config.EPOCHS = 1
        config.NO_WANDB = True
        # Set FEW_IMAGES to True for all datasets in boot mode
        if hasattr(config, 'DATASETS') and config.DATASETS is not None:
            for dataset_name, dataset_config in config.DATASETS.items():
                if isinstance(dataset_config, dict):
                    dataset_config['FEW_IMAGES'] = True
        logger.info("Boot mode enabled - using minimal parameters for quick testing")
    
    return config


def run_pipeline(mode="train", config=None):
    """
    Common pipeline for train and test modes.

    Args:
        mode (str): Operation mode ('train' or 'test')
    """
    install(show_locals=False)

    # Argparse
    parser = argparse.ArgumentParser(description=f"{mode.capitalize()} the network")
    parser.add_argument(
        "--nodebug", "-nd", action="store_false", help="Disable debug mode"
    )
    parser.add_argument(
        "--record",
        "-r",
        action="store_true",
        help="Save the session for comparison",
    )
    parser.add_argument("--stop", "-s", action="store_true", help="Stop VM when done")
    parser.add_argument(
        "--boot",
        "-b",
        action="store_true",
        help="Run in boot mode with minimal parameters",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=f"config_{mode}.yaml",
        help="Path to the config file",
    )

    # Parse known and unknown arguments
    args, unknown = parser.parse_known_args()

    if args.nodebug:
        debugpy.listen(("localhost", 5678))
    
    # Show title screen if available
    try:
        titlescreen()
    except:
        logger.info("=" * 50, context="INFO")
        logger.info("UnReflectAnything - Reflection Removal Training", context="INFO")
        logger.info("=" * 50, context="INFO")


    logger.info(f"Torch Version: {torch.__version__}")
    logger.info(f"Python Version: {os.sys.version.split()[0]}")
    logger.info(f"CUDA version: {torch.version.cuda}")
    logger.info(f"CUDNN version: {torch.backends.cudnn.version()}")
    
    # Get CPU info
    try:
        cpu_affinity = os.sched_getaffinity(os.getpid())
        NUM_WORKERS = len(list(cpu_affinity))
        logger.info(f"Cores available: {NUM_WORKERS} {sorted(list(cpu_affinity))}")
    except Exception as e:
        NUM_WORKERS = 4
        logger.info(f"Using default workers: {NUM_WORKERS}", context="INFO")
    
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ACCELERATED = torch.cuda.is_available()

    # Load and process configuration
    CONFIG_PATH = args.config
    config = load_and_process_config(
        config_path=CONFIG_PATH,
        config=config,
        unknown_args=unknown,
        boot_mode=args.boot
    )

    def get_unique_note():
        notes_past_file = os.path.join("assets", "notes_past.txt")
        existing_notes = set()
        if os.path.exists(notes_past_file):
            with open(notes_past_file, "r") as f:
                existing_notes = set(line.strip() for line in f if line.strip())

        while True:
            print()
            note = input("Describe this run: ").strip()
            if note == " ":
                return note
            if not note:
                continue
            if note in existing_notes:
                print("Already in Use")
            else:
                with open(notes_past_file, "a") as f:
                    f.write(note + "\n")
                return note

    # Check if 'NOTES' is empty or not
    config.RECORD = args.record
    if args.record:
        if not config.get("NOTES"):
            config.NOTES = get_unique_note()
        else:
            if not config.NOTES.strip():
                config.NOTES = get_unique_note()
            else:
                notes_past_file = os.path.join("assets", "notes_past.txt")
                existing_notes = set()
                if os.path.exists(notes_past_file):
                    with open(notes_past_file, "r") as f:
                        existing_notes = set(line.strip() for line in f if line.strip())
                if config.NOTES in existing_notes:
                    config.NOTES = get_unique_note()
                else:
                    with open(notes_past_file, "a") as f:
                        f.write(config.NOTES + "\n")

    # Run the appropriate function based on mode
    try:
        if mode == "train":
            # Create model
            model = create_model_from_config(config, DEVICE)
            
            # Create datasets for training
            dataset = create_datasets_from_config(config, CONFIG_PATH)
            
            # Initialize engine
            engine = Engine(
                model=model,  # Pass the created model
                dataset=dataset,
                config=config,
                no_wandb=config.get("NO_WANDB", False),
                notes=config.get("NOTES", ""),
            )
            
            # Train the model
            engine.trainloop()
            
            # Load best model and test
            engine.reinstantiate_model_from_checkpoint()
            engine.test()
            
        elif mode == "test":
            # Create model
            model = create_model_from_config(config, DEVICE)
            
            # Create datasets for testing
            dataset = create_datasets_from_config(config, CONFIG_PATH)
            
            # Initialize engine
            engine = Engine(
                model=model,  # Pass the created model
                dataset=dataset,
                config=config,
                no_wandb=config.get("NO_WANDB", False),
                notes=config.get("NOTES", ""),
            )
            
            # Load best model and test
            engine.reinstantiate_model_from_checkpoint()
            engine.test()
            
        else:
            raise ValueError(f"Unknown mode: {mode}")
    except Exception as e:
        raise e
    finally:
        if args.stop and socket.gethostname() == "alberto-vm-03":
            print("\n[red]!!! Stopping VM in 60 seconds !!![/red]")
            try:
                time.sleep(60)
            except KeyboardInterrupt:
                print("\nVM stop aborted.")
            else:
                os.system(
                    "gcloud compute instances stop alberto-vm-03 --zone=us-central1-a"
                )
