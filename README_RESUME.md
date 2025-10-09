# Resume Training Functionality

This document explains how to use the resume functionality to split training across multiple script launches.

## Overview

The resume functionality allows you to continue training from where you left off after an interruption or cancellation. This is particularly useful for:

- Long training runs that might be interrupted
- Splitting training across multiple sessions
- Resuming after system crashes or time limits
- Continuing training with different hardware

## How It Works

When you start training, the system automatically saves checkpoints that include:
- Model weights and optimizer state
- Current epoch number
- Learning rate scheduler state
- Early stopping state
- Training and validation metrics history
- Wandb run information

When you resume, the system:
- Loads the latest checkpoint
- Restores all training state
- Continues from the exact epoch where training stopped
- Maintains wandb logging continuity

## Usage Examples

### Basic Resume

1. **Start initial training:**
   ```bash
   python train.py --epochs=100
   ```
   
   This will create a run with a unique name (e.g., `my_experiment_20241201_143022`)

2. **Training gets interrupted at epoch 50**

3. **Resume training:**
   ```bash
   python train.py --resume-run=my_experiment_20241201_143022
   ```
   
   Training will continue from epoch 51 to 100

### Finding Run Names

You can find available runs in the `runs/` directory:

```bash
ls runs/
# Output example:
# my_experiment_20241201_143022
# another_experiment_20241201_150000
```

### Partial Run Name Matching

You can use partial run names for convenience:

```bash
# If you have runs: my_experiment_20241201_143022, my_experiment_20241201_150000
python train.py --resume-run=my_experiment_20241201_143022
# or
python train.py --resume-run=143022  # Will match the first run containing this string
```

### Resume with Different Configuration

You can resume with modified hyperparameters:

```bash
python train.py --resume-run=my_experiment_20241201_143022 --learning_rate=0.001
```

**Note:** Some parameters like model architecture cannot be changed when resuming.

## What Gets Preserved

When resuming, the following state is restored:

- ✅ **Model weights** - Exact model state
- ✅ **Optimizer state** - Momentum, learning rate schedules, etc.
- ✅ **Scheduler state** - Learning rate scheduler state
- ✅ **Early stopping state** - Patience counter and best validation loss
- ✅ **Training metrics** - Complete history of training metrics
- ✅ **Validation metrics** - Complete history of validation metrics
- ✅ **Wandb continuity** - Same wandb run, preserving all logging
- ✅ **Epoch number** - Continues from the exact epoch where it stopped

## What Can Be Modified

When resuming, you can override these parameters:

- `--epochs` - Change total number of epochs
- `--learning_rate` - Modify learning rate
- `--batch_size` - Change batch size (if compatible)
- `--save_interval` - Modify checkpoint saving frequency
- Most other training hyperparameters

## Error Handling

The system provides clear error messages for common issues:

### Run Not Found
```
ERROR: Run not found: my_nonexistent_run
```
**Solution:** Check available runs with `ls runs/`

### No Checkpoints Found
```
ERROR: No checkpoints found in models directory
```
**Solution:** Ensure the run has completed at least one epoch

### Checkpoint Loading Failed
```
ERROR: Failed to load checkpoint data
```
**Solution:** Check if checkpoint files are corrupted or incomplete

## Best Practices

1. **Use descriptive run names** - This makes it easier to identify runs later
2. **Check available runs** - Use `ls runs/` to see what's available
3. **Verify checkpoint integrity** - The system will validate checkpoints before resuming
4. **Monitor wandb continuity** - Ensure the same wandb run is being resumed
5. **Save frequently** - Use appropriate `SAVE_INTERVAL` in your config

## Technical Details

### Checkpoint Structure

Checkpoints are saved with the following structure:
```
runs/
└── my_experiment_20241201_143022/
    ├── models/
    │   ├── checkpoint_epoch_10.pth
    │   ├── checkpoint_epoch_20.pth
    │   ├── checkpoint_epoch_30.pth
    │   └── best_model.pth
    ├── config.json          # Configuration file (new runs)
    ├── hyperparams.json     # Hyperparameters file (legacy compatibility)
    ├── training_metrics.csv
    └── validation_metrics.csv
```

**Note:** The system saves both `config.json` and `hyperparams.json` for compatibility. The resume functionality will use `config.json` if available, or fall back to `hyperparams.json` for older runs.

### Wandb Integration

- Resume functionality automatically detects and resumes the same wandb run
- All metrics and logs are preserved in the same wandb project
- No duplicate runs are created

### Memory Considerations

- Resume functionality loads the complete checkpoint into memory
- For very large models, ensure sufficient RAM/VRAM
- The system will provide clear error messages if memory is insufficient

## Troubleshooting

### Common Issues

1. **"Run not found" error**
   - Check the exact run name in the `runs/` directory
   - Use partial matching if the full name is long

2. **"No config.json or hyperparams.json found" error**
   - This happens with very old runs that don't have config files
   - The system now automatically saves both files for new runs
   - For old runs, you may need to manually create a config.json file

3. **"Failed to load checkpoint" error**
   - Verify checkpoint files exist and are not corrupted
   - Check file permissions

4. **"Wandb resume failed" error**
   - Ensure wandb is properly configured
   - Check internet connection for wandb sync

5. **Training starts from epoch 0 instead of resuming**
   - Verify the checkpoint contains epoch information
   - Check that the resume process completed successfully

### Getting Help

If you encounter issues:

1. Check the log output for detailed error messages
2. Verify your run directory structure
3. Ensure all required files are present
4. Check file permissions and disk space

## Example Workflow

Here's a complete example of splitting training across two sessions:

### Session 1 (Initial Training)
```bash
# Start training for 50 epochs
python train.py --epochs=50 --notes="Initial training run"

# Output shows run name: my_experiment_20241201_143022
# Training completes or gets interrupted at epoch 30
```

### Session 2 (Resume Training)
```bash
# Resume training for additional 50 epochs (total 100)
python train.py --resume-run=my_experiment_20241201_143022 --epochs=100

# Training continues from epoch 31 to 100
# All metrics and wandb logging are preserved
```

This allows you to effectively split long training runs across multiple sessions while maintaining complete state continuity.
