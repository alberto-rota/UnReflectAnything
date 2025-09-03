# Training Pipeline

The training pipeline is executed by executing 
```bash
python train.py
```

which wraps `main.py` and reads the `config_train.yaml` file containing all configuration parameters.

## Command Line Arguments

- `--config=/path/to/your/config.yaml`: Reads from the specified config file rather than the default one

- `--boot`: Launches training with minimal settings for quick debugging: `batch_size = 1`,`epochs = 1`, only loads 10 frames per dataset, disables wandb logging. Useful for quickly debugging training and validation loops

- `--nodebug`: Disables the default debugger instantiation

!!! tip Important
    All parameters in `config_train.yaml` can be overridden via command line arguments.
    - **Config file**: Parameters use `UPPERCASE_AND_UNDERSCORED` format (e.g., `LEARNING_RATE`, `BATCH_SIZE`)
    - **CLI override**: Use `lowercase_and_underscored` format (e.g., `learning_rate`, `batch_size`)

!!! example Example
    ```bash
    python train.py --epochs=300
    ```
    This command will override the `EPOCHS` parameter in `config_train.yaml` with the value `300`.


## Modifying the Pipeline

The `main.py` script performs three key operations after reading the config file:

1. **Instantiates the dataset class** - Loads and prepares the training/validation data
2. **Instantiates the model** - Creates the neural network architecture
3. **Passes everything to the Engine** - The Engine object handles training, validation, and testing

### Key Files to Modify

To customize the pipeline for your experiments, modify the following files:

- **`models.py`** - Change model architecture, encoders, decoders, and neural network components
- **`engine.py`** - Modify supervision strategy and training/validation logic  
- **`losses.py`** - Implement custom loss functions
- **`optimization.py`** - Configure optimizers, schedulers, and early stopping


---
:::main

