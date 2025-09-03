# Adding a New Dataset

To add support for a new dataset (e.g., "NEWDATASET"), you need to:

1. Create a dataset class that inherits from the base RGBP_Dataset class
2. Register the dataset in the dataset factory
3. Add the dataset configuration to your config file

## Step 1: Create the Dataset Class

Create a new dataset class that inherits from the `RGBP_Dataset` base class:

```python
class NEWDATASET_Dataset(RGBP_Dataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Add any NEWDATASET-specific initialization here
        
    def _load_data(self):
        # Override this method if you need custom data loading logic
        pass
        
    def _process_sample(self, sample):
        # Override this method for custom preprocessing
        return super()._process_sample(sample)
```

!!! tip "Dataset Inheritance"
    The `RGBP_Dataset` class provides the base functionality for polarization datasets. Most datasets can inherit directly from it with minimal customization.

## Step 2: Register the Dataset

Add your new dataset class to the `dataset_classes` dictionary in the `create_datasets_from_config()` function:

```python
dataset_classes = {
    'SCRREAM': SCRREAM_Dataset,
    'HOUSECAT6D': HOUSECAT6D_Dataset,
    'POLARGB': POLARGB_Dataset,
    'NEWDATASET': NEWDATASET_Dataset,  # Add this line
}
```

## Step 3: Configure the Dataset

Add the dataset configuration to your `config.yaml` file under the `DATASETS` section:

```yaml
parameters:
  DATASETS:
    value:
      SCRREAM:
        ROOT_DIR: "/path/to/scrream"
        TRAIN_SCENES: []
        VAL_SCENES: ["scene09_full_00"]
        # ... other config
        
      NEWDATASET:
        ROOT_DIR: "/path/to/newdataset"
        TRAIN_SCENES: []
        VAL_SCENES: ["scene1", "scene2"]
        POLARIZATION_FORMAT: "single_file_clock"
        RHO_S: 0.6
        EPS: 1.0e-8
        TARGET_SIZE: [224, 224]
        RESIZE_MODE: "resize"
        USE_CACHE: true
        SIMPLIFY_UPSAMPLING: true
        FEW_IMAGES: false
```

## Dataset Configuration Parameters

For detailed information about each configuration parameter, see the [Configuration Parameters](config.md#dataset-configuration) section.

## Automatic Discovery

Once you've completed these steps, the dataset loading system will automatically discover and use your new dataset during training and validation.
