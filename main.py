#  MODULES AND DATASET LOADING
from dotenv import load_dotenv
load_dotenv()
import argparse
import ast
import os
import socket
import time
from typing import Dict, Any, Optional, List

import debugpy
import torch
import yaml
from dotmap import DotMap
from rich.traceback import install

from dataset.rgbp import from_config
from engine import Engine
from logger import get_logger
import models
from utilities import *
import utilities.engine_initializers as initialize

logger = get_logger(__name__).set_context("IMPORT")


def create_model_from_config(config: DotMap, device: torch.device):
    """
    Create the RGBPOLDecomposer model from configuration.
    
    This function initializes a RGBPOLDecomposer model by extracting configuration
    parameters for different components (RGB encoder, POL encoder, cross-attention,
    and decoders) and creates the model with the specified architecture.

    Args:
        config (DotMap): Configuration dictionary containing model parameters including:
            - MODEL: Model architecture configuration
            - DATASETS: Dataset configurations for extracting target image size
        device (torch.device): PyTorch device to place the model on (e.g., 'cuda' or 'cpu')

    Returns:
        RGBPOLDecomposer: The initialized model ready for training or inference
        
    Note:
        The model expects input tensors of shape [B×3×H×W] for RGB images and 
        [B×3×H×W] for polarization images, where B is batch size, H and W are 
        height and width respectively.
    """
    # Access model configuration from the nested structure
    model_config = config.get("MODEL", {})  # .get("value", {})

    # Get image dimensions from config (check multiple possible locations)
    target_size = None
    for dataset_name in ["SCRREAM", "HOUSECAT6D", "POLARGB"]:
        dataset_config = (
            config.get("DATASETS", {}).get("value", {}).get(dataset_name, {})
        )
        if "TARGET_SIZE" in dataset_config:
            target_size = dataset_config["TARGET_SIZE"]
            break

    if target_size is None:
        target_size = (224, 224)  # Default fallback

    if isinstance(target_size, list):
        target_size = tuple(target_size)

    # # RGB Encoder configuration
    rgb_encoder_config = model_config.get("RGB_ENCODER", {})
    dinov3_cfg = {
        "model_name": rgb_encoder_config.get(
            "ENCODER", "facebook/dinov3-vits16-pretrain-lvd1689m"
        ),
        "image_size": rgb_encoder_config.get("IMAGE_SIZE", min(target_size)),
        "freeze_backbone": rgb_encoder_config.get("FREEZE_BACKBONE", True),
        "return_selected_layers": rgb_encoder_config.get(
            "RETURN_SELECTED_LAYERS", [3, 6, 9, 12]
        ),
        "return_last_hidden_state": rgb_encoder_config.get(
            "RETURN_LAST_HIDDEN_STATE", False
        ),
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
        "bi_directional": cross_attn_config.get("BI_DIRECTIONAL", False),
    }

    # Decoder configuration - support both flexible and legacy formats
    decoders_config = model_config.get("DECODERS", None)
    
    if decoders_config is not None:
        # New flexible decoder format
        decoders = {}
        for decoder_name, decoder_params in decoders_config.items():
            # Build decoder config with defaults
            decoder_cfg = {
                "feature_dim": decoder_params.get("FEATURE_DIM", 384),
                "reassemble_out_channels": decoder_params.get("REASSEMBLE_OUT_CHANNELS", [12,24,48,92]),
                "reassemble_factors": decoder_params.get("REASSEMBLE_FACTORS", [4.0, 2.0, 1.0, 0.5]),
                "readout_type": decoder_params.get("READOUT_TYPE", "ignore"),
                "use_bn": decoder_params.get("USE_BN", True),
                "dropout": decoder_params.get("DROPOUT", 0.0),
                "output_image_size": decoder_params.get("OUTPUT_IMAGE_SIZE", [min(target_size), min(target_size)]),
                "output_channels": decoder_params.get("OUTPUT_CHANNELS", 3),
            }
            decoders[decoder_name] = decoder_cfg
        
        decoder_kwargs = {"decoders": decoders}
    else:
        # Legacy decoder format - single DECODER config applied to all three decoders
        decoder_config = model_config.get("DECODER", {})
        decoder_cfg = {
            "feature_dim": decoder_config.get("FEATURE_DIM", 384),
            "reassemble_out_channels": decoder_config.get("REASSEMBLE_OUT_CHANNELS", [12,24,48,92]),
            "reassemble_factors": decoder_config.get("REASSEMBLE_FACTORS", [4.0, 2.0, 1.0, 0.5]),
            "readout_type": decoder_config.get("READOUT_TYPE", "ignore"),
            "use_bn": decoder_config.get("USE_BN", True),
            "output_image_size": decoder_config.get("OUTPUT_IMAGE_SIZE", [min(target_size), min(target_size)]),
            "output_channels": decoder_config.get("OUTPUT_CHANNELS", 3),
        }
        
        decoder_kwargs = {
            "spec_decoder": decoder_cfg,
            "diffuse_decoder": decoder_cfg,
            "highlight_decoder": decoder_cfg,
        }

    shared_dinov3 = models.DINOv3(dinov3_cfg).to(device)
    # Create the main model
    model_class_str = model_config.get("MODEL_CLASS", "RGBPOLDecomposer")
    # Get the model class from the string name
    model_class = getattr(models, model_class_str)
    
    # Build model kwargs based on model type
    model_kwargs = {
        "dinov3": shared_dinov3,
        **decoder_kwargs,
    }
    
    # Add POL-specific configs only for RGBPOLDecomposer
    if model_class_str == "RGBPOLDecomposer":
        model_kwargs.update({
            "pol_encoder": pol_enc_cfg,
            "pol_cross_attn": cross_attn_cfg,
        })
    
    model = model_class(**model_kwargs).to(device)
    torch.cuda.empty_cache()
    # from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
    # from models import DINOv3
    # from sd_decomposer import StableDiffusionDecomposer

    # # 1) DINO encoder (use your config to return selected_hidden_states)
    # dino = DINOv3({
    #     "model_name": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    #     "image_size": 224,
    #     "freeze_backbone": True,
    #     "return_selected_layers": [3, 6, 9, 12],   # example
    #     "return_as_feature_maps": False,
    # }).cuda()

    # # 2) SD parts (load checkpoints compatible with each other; SD 1.5 shown as example)
    # sd_vae  = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae").cuda()
    # sd_unet = UNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="unet").cuda()
    # sched   = DDPMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler")

    # # 3) Model
    # model = StableDiffusionDecomposer(
    #     dinov3=dino,
    #     sd_vae=sd_vae,
    #     sd_unet=sd_unet,
    #     scheduler=sched,
    #     adapter_cfg=dict(ctx_dim=768, ctx_len=77, n_layers_proj=2, n_heads=8,
    #                      reduce_mode="learned_pool", layer_fusion="weighted_sum"),
    #     freeze_vae=True,
    #     freeze_unet=True,               # start frozen
    #     unfreeze_unet_attn_qkv=False,   # optionally True later
    # ).to(device)

    
    logger.info(
        f"Model with class {model.__class__.__name__} created with {sum(p.numel() for p in model.parameters()):,} parameters",
        context="MODEL",
    )

    return model


def create_datasets_from_config(config: DotMap) -> Dict[str, Any]:
    """
    Create training and validation datasets from configuration using the new system.

    This function uses the improved dataset creation system that:
    - Reads from YAML config files with DATASETS section
    - Supports multiple datasets (SCRREAM, HOUSECAT6D, POLARGB, etc.)
    - Creates dataset-specific classes with proper data loading
    - Returns ConcatDatasets for multi-dataset training scenarios
    - Handles data preprocessing and augmentation pipelines

    Args:
        config (DotMap): Configuration dictionary containing dataset parameters including:
            - DATASETS: Dataset configurations for different data sources
            - WORKERS: Number of data loading workers

    Returns:
        Dict[str, Any]: Dictionary containing datasets with keys:
            - 'Training': Training dataset (torch.utils.data.Dataset)
            - 'Validation': Validation dataset (torch.utils.data.Dataset) 
            - 'Test': Test dataset (torch.utils.data.Dataset)
            - 'workers': Number of workers for data loading (int)
            
    Raises:
        Exception: If dataset creation fails due to missing config sections,
                  invalid paths, or missing dataset classes
                  
    Note:
        Datasets return samples as dictionaries with tensors of shape:
        - 'rgb': [B×3×H×W] - RGB input images
        - 'pol': [B×3×H×W] - Polarization input images  
        - 'target': [B×3×H×W] - Target reflection-free images
    """
    try:
        # Use the new dataset creation system
        datasets = from_config(config)

        # Convert keys to match what Engine expects (capitalize first letter)
        result = {
            "Training": datasets.get("training"),
            "Validation": datasets.get("validation"),
            "Test": datasets.get("test"),
            "workers": config.get("WORKERS", 4),
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


def load_and_process_config(
    config_path: str, 
    config: Optional[Dict[str, Any]] = None, 
    unknown_args: Optional[List[str]] = None, 
    boot_mode: bool = False
) -> DotMap:
    """
    Load and process configuration from file or direct input.
    
    This function handles configuration loading with support for:
    - YAML file parsing with parameter extraction
    - Command-line argument override processing
    - Type-safe parameter conversion
    - Boot mode for quick testing with minimal parameters
    - Automatic dataset configuration updates

    Args:
        config_path (str): Absolute path to the YAML configuration file
        config (Optional[Dict[str, Any]]): Direct configuration dictionary that 
                                          overrides file loading if provided
        unknown_args (Optional[List[str]]): List of unknown command-line arguments 
                                           to process as parameter overrides
        boot_mode (bool): Whether to enable boot mode with minimal parameters:
                         - BATCH_SIZE=1, EPOCHS=1, NO_WANDB=True
                         - FEW_IMAGES=True for all datasets

    Returns:
        DotMap: Processed configuration object with dot-notation access to parameters
        
    Raises:
        FileNotFoundError: If config_path does not exist
        yaml.YAMLError: If YAML file parsing fails
        ValueError: If parameter type conversion fails
        
    Note:
        Command-line arguments should follow the format --PARAMETER=value.
        Boolean parameters accept: true/1/yes for True, false/0/no for False.
        List parameters should be valid Python literals (e.g., "[1,2,3]").
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
                    if orig_type is bool:
                        if value.lower() in ("true", "1", "yes"):
                            new_value = True
                        elif value.lower() in ("false", "0", "no"):
                            new_value = False
                        else:
                            raise ValueError(f"Cannot parse boolean value: {value}")
                    elif orig_type is list:
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
        for dataset_name, dataset_config in config.DATASETS.items():
            if isinstance(dataset_config, dict):
                dataset_config["FEW_IMAGES"] = True
        logger.info("Boot mode enabled - using minimal parameters for quick testing")

    return config


def run_pipeline(mode: str = "train", config: Optional[Dict[str, Any]] = None) -> None:
    """
    Common pipeline for train and test modes.
    
    This is the main entry point that orchestrates the entire machine learning pipeline:
    - Environment setup and device detection
    - Configuration loading and processing
    - Model and dataset instantiation
    - Training or testing execution via Engine
    - Checkpoint management and evaluation
    
    The pipeline supports both training and testing modes with comprehensive
    logging, debugging capabilities, and optional experiment tracking.

    Args:
        mode (str): Operation mode, either 'train' or 'test'. Defaults to 'train'.
                   - 'train': Runs full training loop with validation and final testing
                   - 'test': Loads best checkpoint and runs evaluation only
        config (Optional[Dict[str, Any]]): Optional configuration dictionary to use
                                          instead of loading from file

    Returns:
        None
        
    Raises:
        ValueError: If mode is not 'train' or 'test'
        FileNotFoundError: If configuration file is not found
        RuntimeError: If model creation or dataset loading fails
        
    Note:
        This function handles:
        - CUDA device detection and setup
        - Multi-core CPU detection for data loading
        - Debug server setup if enabled
        - Experiment note management for reproducibility
        - VM auto-shutdown for cloud environments
    """
    install(show_locals=False)

    # Argparse
    parser = argparse.ArgumentParser(description=f"{mode.capitalize()} the network")
    parser.add_argument(
        "--debug", "-d", action="store_true", help="Enable debug mode"
    )
    parser.add_argument(
        "--wait-debugger-attach", "-wd", action="store_true", help="Wait for debugger to attach"
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
    parser.add_argument(
        "--resume-run",
        type=str,
        help="Resume training from an existing run. Provide the run name or run ID.",
    )
    parser.add_argument(
        "--distill-from",
        type=str,
        help="Train using knowledge distillation from a teacher model. Provide the run name or run ID of the teacher model.",
    )

    # Parse known and unknown arguments
    args, unknown = parser.parse_known_args()

    if args.debug:
        debug_port = int(os.getenv("DEBUGPY_PORT"))
        # Get machine hostname/IP for remote debugging
        import socket
        hostname = socket.gethostname()
        try:
            # Try to get the actual IP address
            ip_address = socket.gethostbyname(hostname)
        except:
            ip_address = "localhost"
        
        logger.info(f"Debug mode enabled on port {debug_port}")
        logger.info(f"Connect VSCode debugger to: {hostname} ({ip_address}:{debug_port})")
        debugpy.listen(("0.0.0.0", debug_port))  # Listen on all interfaces for remote connections
        if args.wait_debugger_attach:
            logger.info("Waiting for debugger to attach...")
            debugpy.wait_for_client()

    # Show title screen if available
    try:
        titlescreen()  
    except Exception:
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
        CPU_AFFINITY = len(list(cpu_affinity))
        logger.info(f"Cores available: {CPU_AFFINITY} {sorted(list(cpu_affinity))}")
    except Exception:
        logger.info(f"Couldn't get CPU affinity", context="INFO")

    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and process configuration
    CONFIG_PATH = args.config
    config = load_and_process_config(
        config_path=CONFIG_PATH,
        config=config,
        unknown_args=unknown,
        boot_mode=args.boot,
    )

    def get_unique_note() -> str:
        """
        Get a unique experiment note from user input.
        
        This function manages experiment note collection to ensure reproducibility
        and avoid duplicate experiment descriptions. It maintains a history of
        past notes and prompts the user until a unique description is provided.

        Returns:
            str: A unique experiment note describing the current run
            
        Note:
            - Notes are stored in 'assets/notes_past.txt' for history tracking
            - Empty input is rejected and user is re-prompted
            - Single space (' ') is accepted as a valid note
            - Duplicate notes are rejected with "Already in Use" message
        """
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
            # Check if we need to resume from an existing run
            if hasattr(args, 'resume_run') and args.resume_run:
                logger.info(f"Resuming training from run: {args.resume_run}", context="RESUME")
                
                # Create model
                model = create_model_from_config(config, DEVICE)

                # Create datasets for training
                dataset = create_datasets_from_config(config)

                # Initialize engine
                engine = Engine(
                    model=model,  # Pass the created model
                    dataset=dataset,
                    config=config,
                    no_wandb=config.get("NO_WANDB", False),
                    notes=config.get("NOTES", ""),
                )
                
                # Mark that we will resume to skip directory setup during init
                engine._will_resume = True
                
                # Resume from the specified run
                if not engine.resume_from_run(args.resume_run):
                    logger.error("Failed to resume training. Exiting.", context="RESUME")
                    return
                
                # Train the model (will start from the correct epoch)
                engine.trainloop()

                # Load best model and test
                engine.reinstantiate_model_from_checkpoint()
                engine.test()

            else:
                # Normal training (new run)
                # Create model
                model = create_model_from_config(config, DEVICE)

                # Create datasets for training
                dataset = create_datasets_from_config(config)

                # Initialize engine
                engine = Engine(
                    model=model,  # Pass the created model
                    dataset=dataset,
                    config=config,
                    no_wandb=config.get("NO_WANDB", False),
                    resume_run_id=args.resume_run,
                    notes=config.get("NOTES", ""),
                )

                # Train the model
                engine.trainloop()

                # Load best model and test
                engine.reinstantiate_model_from_checkpoint()
                engine.test()

        elif mode == "test":
            # Resolve run identifier from config for test resume
            run_identifier = config.get("RUN", None)
            if run_identifier is None or (isinstance(run_identifier, str) and len(run_identifier.strip()) == 0):
                logger.error("RUN name must be provided for test mode (via config or --RUN)")
                return

            # Discover existing run directories without creating anything new
            from utilities.run_resume import get_resume_info
            resume_info = get_resume_info(run_identifier, initialize.device_and_directories(config)["runs_dir"])
            if resume_info is None:
                logger.error(f"No existing run matched RUN name: {run_identifier}")
                return

            # Create model and datasets using the current config
            model = create_model_from_config(config, DEVICE)
            dataset = create_datasets_from_config(config)

            # Initialize engine in resume mode, resuming the existing WandB run
            engine = Engine(
                model=model,
                dataset=dataset,
                config=config,
                notes=config.get("NOTES", ""),
                no_wandb=False,
                resume_run_id=None,  # let initializer resolve RUN by display name
                will_resume=True,
                resume_info=resume_info,
            )

            # Load best model from existing run and test
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
