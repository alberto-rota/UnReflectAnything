# Training Pipeline

The training pipeline is executed by running `python train.py`, which calls `main.py` and reads the `config_train.yaml` file containing all configuration parameters.

## Command Line Arguments

### Debug Mode
- `--boot`: Launches training with minimal settings for quick debugging: `batch_size = 1`,`epochs = 1`, only loads 10 frames per dataset, disables wandb logging. Useful for quickly debugging training and validation loops

### Debugger Control
- `--nodebug`: Disables the default debugger instantiation

## Configuration Override

!!! tip Important
    All parameters in `config_train.yaml` can be overridden via command line arguments.

### Parameter Naming Convention
- **Config file**: Parameters use `UPPERCASE_AND_UNDERSCORED` format (e.g., `LEARNING_RATE`, `BATCH_SIZE`)
- **CLI override**: Use `lowercase_and_underscored` format (e.g., `learning_rate`, `batch_size`)

### Example
```bash
python train.py --epochs=300
```
This command will override the `EPOCHS` parameter in `config_train.yaml` with the value `300`.

## Architecture

`train.py` serves as a wrapper for `main.py`, which contains the core training logic.

:::main

