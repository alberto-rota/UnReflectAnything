# Dataset Structure

The UnReflectAnything dataset module supports three main polarization datasets for reflection removal and 6D pose estimation:

- **HOUSECAT6D**: 6D object pose estimation with polarization data
- **SCRREAM**: Specular reflection removal dataset
- **PolaRGB**: Polarization-guided RGB processing dataset

## Directory Structure

Each dataset follows a standardized directory structure with scene-based organization. The module supports two different polarization data formats, which are automatically detected based on the file naming convention.

### Format 1: Single Polarization Files

For datasets where each polarization image is stored as a single file (typically containing 4 quadrants):

```
DATASET_ROOT/
├── scene_001/
│   ├── rgb/
│   │   ├── 000000.png
│   │   ├── 000001.png
│   │   └── ...
│   ├── pol/
│   │   ├── 000000.png        # Single file with 4 polarization quadrants
│   │   ├── 000001.png
│   │   └── ...
│   ├── specular/             # Optional: ground truth specular components
│   ├── diffuse/              # Optional: ground truth diffuse components  
│   ├── normals/              # Optional: surface normal maps
│   └── intrinsics.txt        # Camera intrinsics matrix
├── scene_002/
│   └── ... (same structure)
└── ...
```

### Format 2: Separate Polarization Files

For datasets where each polarization angle is stored in separate files:

```
DATASET_ROOT/
├── scene_001/
│   ├── rgb/
│   │   ├── 000000.png
│   │   ├── 000001.png
│   │   └── ...
│   ├── pol/
│   │   ├── 000000_000.png    # 0° polarization
│   │   ├── 000000_045.png    # 45° polarization
│   │   ├── 000000_090.png    # 90° polarization
│   │   ├── 000000_135.png    # 135° polarization
│   │   ├── 000001_000.png
│   │   ├── 000001_045.png
│   │   ├── 000001_090.png
│   │   ├── 000001_135.png
│   │   └── ...
│   ├── specular/             # Optional: ground truth specular components
│   ├── diffuse/              # Optional: ground truth diffuse components
│   ├── normals/              # Optional: surface normal maps
│   └── intrinsics.txt        # Camera intrinsics matrix
├── scene_002/
│   └── ... (same structure)
└── ...
```

## Configuration

The polarization data format is automatically detected, but you can explicitly specify it in your `config.yaml` file:

```yaml
parameters:
  DATASETS:
    value:
      YOUR_DATASET:
        ROOT_DIR: "/path/to/dataset"
        POLARIZATION_FORMAT: "single_file_clock"  # or "separate_files"
        # ... other parameters
```

### Polarization Format Options

| Format | Description | File Pattern |
|--------|-------------|--------------|
| `single_file_clock` | Single file with 4 quadrants arranged clockwise | `000000.png` |
| `separate_files` | Four separate files for each angle | `000000_000.png`, `000000_045.png`, etc. |

## File Specifications

### Required Files

- **RGB images** (`rgb/`): Standard RGB images in PNG format
- **Polarization data** (`pol/`): Polarization measurements (format depends on dataset)
- **Camera intrinsics** (`intrinsics.txt`): 3×3 camera intrinsics matrix

### Optional Files

- **Specular components** (`specular/`): Ground truth specular reflection maps
- **Diffuse components** (`diffuse/`): Ground truth diffuse reflection maps  
- **Surface normals** (`normals/`): Surface normal maps for geometric analysis

### Camera Intrinsics Format

The `intrinsics.txt` file should contain a 3×3 camera intrinsics matrix:

```
fx  0   cx
0   fy  cy
0   0   1
```

Where:
- `fx`, `fy`: Focal lengths in pixels
- `cx`, `cy`: Principal point coordinates
- Values should be space or tab separated

## Usage Examples

### Loading a Dataset

```python
from dataset import SCRREAM_Dataset

# Load dataset with automatic format detection
dataset = SCRREAM_Dataset(
    root_dir="/path/to/scrream/data",
    target_size=(512, 640),
    resize_mode="crop"
)

# Access a sample
sample = dataset[0]
# Returns: RGB, polarization data, intrinsics, and derived components
```

### Configuration-Based Loading

```python
from dataset import load_config_and_create_datasets

# Load datasets from configuration file
datasets = load_config_and_create_datasets("config_train.yaml")
train_loader = DataLoader(datasets['training'], batch_size=16)
```