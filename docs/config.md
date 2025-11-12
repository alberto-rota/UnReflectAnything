# Configuration Files

Configuration files contain all parameters used to load datasets, initialize the model architecture, run the training loop, and save results.

There are three default configuration files in the project directory:

- `config_train.yaml` is used for training the pipeline
- `config_test.yaml` is used to run tests on a pipeline that has already been trained with saved results
- `config_sweep.yaml` is used to run a WandB sweep

!!! info
    The syntax and structure of the files are designed so that each file can be easily adapted for other purposes with minimal changes. For example, it's straightforward to take the training configuration from `config_train.yaml` and use it for a WandB sweep by adding extra hyperparameters to the file.

Each file contains a "parameters" section, which includes the actual hyperparameters that can be configured.

The `config_sweep.yaml` file additionally contains the following sections:

```yaml
method: grid
metric:
  goal: minimize
  name: Validation/epoch/Loss

program: sweep_agent.py

project: UnReflectAnything
```

These sections control how the WandB sweep is performed.

For hyperparameter configuration:
- All top-level hyperparameters use a "value" key in training configs
- In sweep configs, use "values" (plural) with a list or distribution following [WandB sweep conventions](https://docs.wandb.ai/guides/sweeps/define-sweep-configuration/)

!!! example "Adapting the training config to a sweep config"
    If a training experiment was satisfactory and you want to run a sweep around the successful configuration:
    
    1. Copy the "parameters" section from `config_train.yaml` to `config_sweep.yaml`
    2. Change hyperparameters to be swept: replace the "value" key with "values" and specify the list or distribution

## Configuration Parameters

This section explains each parameter in the `config_train.yaml` file:

### Model Architecture

#### `MODEL.RGB_ENCODER`
- **`ENCODER`**: Pre-trained model identifier (e.g., "facebook/dinov3-vitb16-pretrain-lvd1689m")
- **`IMAGE_SIZE`**: Input image size for the RGB encoder (224×224 pixels)
- **`RETURN_SELECTED_LAYERS`**: List of transformer layers to extract features from [3, 6, 9, 12]

#### `MODEL.POL_ENCODER`
- **`EMBED_DIM`**: Embedding dimension for polarization encoder (768)
- **`DEPTH`**: Number of transformer layers (4)
- **`N_HEADS`**: Number of attention heads (12)
- **`PATCH_SIZE`**: Size of image patches for tokenization (16×16)

#### `MODEL.CROSS_ATTN`
- **`EMBED_DIM`**: Embedding dimension for cross-attention module (768)
- **`N_HEADS`**: Number of attention heads (12)
- **`DROPOUT`**: Dropout rate for regularization (0.1)
- **`BI_DIRECTIONAL`**: Whether to use bidirectional attention (False)

#### `MODEL.DECODER`
- **`FEATURE_DIM`**: Feature dimension for decoder (768)
- **`REASSEMBLE_OUT_CHANNELS`**: Output channels for feature reassembly [96, 192, 384, 768]
- **`REASSEMBLE_FACTORS`**: Scaling factors for multi-scale features [4.0, 2.0, 1.0, 0.5]
- **`READOUT_TYPE`**: Type of feature readout ("ignore")
- **`USE_BN`**: Whether to use batch normalization (False)
- **`OUTPUT_IMAGE_SIZE`**: Final output image dimensions [224, 224]
- **`OUTPUT_CHANNELS`**: Number of output channels (4)

### Dataset Configuration

#### `DATASETS.SCRREAM`
- **`ROOT_DIR`**: Path to SCRREAM dataset directory
- **`TRAIN_SCENES`**: List of training scenes (empty = all scenes)
- **`VAL_SCENES`**: List of validation scenes
- **`POLARIZATION_FORMAT`**: Format of polarization data ("single_file_clock")
- **`RHO_S`**: Surface reflectance parameter (0.6)
- **`EPS`**: Small epsilon value for numerical stability (1e-8)
- **`TARGET_SIZE`**: Target image size after preprocessing [224, 224]
- **`RESIZE_MODE`**: Image resizing method ("resize")
- **`USE_CACHE`**: Whether to cache preprocessed data (True)
- **`SIMPLIFY_UPSAMPLING`**: Simplified upsampling strategy (True)
- **`FEW_IMAGES`**: Use subset of images for debugging (False)

#### `DATASETS.HOUSECAT6D` & `DATASETS.POLARGB`
Similar parameters as SCRREAM with dataset-specific paths and scene configurations.

### Data Loading

- **`BATCH_SIZE`**: Number of samples per batch (16)
- **`WORKERS`**: Number of data loading workers (8)
- **`SHUFFLE`**: Whether to shuffle training data (True)
- **`PIN_MEMORY`**: Pin memory for faster GPU transfer (True)
- **`PREFETCH_FACTOR`**: Number of batches to prefetch (1)

### Optimization

- **`LEARNING_RATE`**: Initial learning rate (1e-4)
- **`WEIGHT_DECAY`**: L2 regularization strength (1e-7)
- **`EPOCHS`**: Maximum number of training epochs (30)
- **`GRADIENT_ACCUMULATION_STEPS`**: Steps to accumulate gradients before update (4)
- **`WARMUP`**: Learning rate warmup steps (50)
- **`GRADIENT_CLIPPING_MAX_NORM`**: Maximum gradient norm for clipping (1.0)

#### `LR_SCHEDULER`
Learning rate scheduling options:
- **`ONPLATEAU`**: Reduce LR when validation loss plateaus
  - `PATIENCE`: Epochs to wait before reducing (10)
  - `FACTOR`: Reduction factor (0.1)
- **`STEPWISE`**: Reduce LR at fixed intervals
  - `N_STEPS`: Number of reduction steps (3)
  - `GAMMA`: Reduction factor (0.5)

#### Optimizer Settings
- **`SWITCH_OPTIMIZER_EPOCH`**: Epoch to switch optimizers (null = no switch)
- **`OPTIMIZER_BOOTSTRAP_NAME`**: Initial optimizer ("Adam")
- **`OPTIMIZER_REFINING_NAME`**: Secondary optimizer ("Adam")
- **`EARLY_STOPPING_PATIENCE`**: Epochs without improvement before stopping (20)
- **`SAVE_INTERVAL`**: Steps between model checkpoints (1000)

### Logging and Monitoring

- **`LOG_INTERVAL`**: Steps between console logs (1)
- **`WANDB_LOG_INTERVAL`**: Steps between WandB metric logs (4)
- **`IMAGE_LOG_INTERVAL`**: Steps between image logging (100)
- **`NO_WANDB`**: Disable WandB logging (False)
- **`MODEL_WATCHER_FREQ_WANDB`**: Frequency for model parameter monitoring (100)
- **`WANDB_ENTITY`**: WandB organization name ("unreflect-anything")
- **`WANDB_PROJECT`**: WandB project name ("UnReflectAnything")
- **`NOTES`**: Experiment description/notes
- **`RESULTS_DIR`**: Directory to save training results

