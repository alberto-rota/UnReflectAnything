# Logger

The logger module provides comprehensive logging capabilities for the UnReflectAnything framework.

## Overview

The logger supports:
- Multiple logging levels (DEBUG, INFO, WARNING, ERROR)
- File and console output
- TensorBoard integration
- WandB integration
- Custom loggers

## Features

### Logging Levels
- **DEBUG**: Detailed debugging information
- **INFO**: General information about program execution
- **WARNING**: Warning messages for potential issues
- **ERROR**: Error messages for serious problems

### Output Formats
- **Console**: Real-time logging to terminal
- **File**: Persistent logging to files
- **TensorBoard**: Visualization of training metrics
- **WandB**: Experiment tracking and visualization

### Metrics Logging
- Training loss curves
- Validation metrics
- Model parameters
- System resources

## Usage

```python
from logger import Logger

logger = Logger(name="unreflectanything")
logger.info("Starting training...")
logger.log_metrics({"loss": 0.5, "accuracy": 0.95})
```
