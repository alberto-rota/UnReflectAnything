
## Configuration

The polarization data format (clockwise arrangement or in separate files) **must** be explicited in the `config.yaml` section:


```yaml
parameters:
  DATASETS:
    value:
      YOUR_DATASET:
        ROOT_DIR: "/path/to/dataset"
        POLARIZATION_FORMAT: "single_file_clock"  # or "separate_files"
        # ... other parameters
```

Specifically, please specify the way in which the polarization data is saved

| Format | Description | File Pattern |
|--------|-------------|--------------|
| `single_file_clock` | Single file with 4 quadrants arranged clockwise | `000000.png` |
| `separate_files` | Four separate files for each angle | `000000_000.png`, `000000_045.png`, etc. |


The `intrinsics.txt` file should contain a 3×3 camera intrinsics matrix, with the following 3x3 format
```
7.054699707031250000e+02 0.000000000000000000e+00 5.541227923106198432e+02
0.000000000000000000e+00 7.030218505859375000e+02 4.324074733765810379e+02
0.000000000000000000e+00 0.000000000000000000e+00 1.000000000000000000e+00
```

!!!question "Unknown intrinsics"
    If the intrinsic file is not provided, the K matrix will not be returned by the dataset class

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

::: dataset.rgbp