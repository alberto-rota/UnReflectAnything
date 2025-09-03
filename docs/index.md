# UnReflectAnything

UnReflectAnything is a deep learning method for removing specular reflections and highlights from RGB images. At inference time, the model takes an RGB image containing specular reflections as input and outputs a modified image with the reflections removed.

## Getting Started

### Clone the Repository

```bash
git clone https://github.com/mertkiray/unreflect-anything
cd unreflect-anything
```

### Development Environment Setup

1. **Install the uv package manager** from [https://docs.astral.sh/uv/getting-started/installation/](https://docs.astral.sh/uv/getting-started/installation/)

2. **Create a virtual environment** in the current directory:
   ```bash
   uv venv -p 3.10.9
   ```

3. **Activate the virtual environment**:
   ```bash
   source .venv/bin/activate
   ```

!!! note "Python Version"
    Python 3.10.9 is the version used for testing and experiments. Other versions may be supported

4. **Install dependencies**:
   ```bash
   uv pip install -r requirements.txt
   ```

At this point, you're ready to start developing or running experiments.

## Project Structure

The key components of the project are:

### Core Components

- **Configuration Files** (`config.yaml`): Contain all parameters for the end-to-end pipeline, controlling dataset initialization, model architecture, optimization parameters, and housekeeping tasks (logging, wandb, result saving).

- **Main Pipeline** (`main.py`): The backbone of the pipeline that reads config files and instantiates pipeline components (datasets, models, optimizations). This file is not called directly; instead, use `train.py` and `test.py`.

- **Dataset Module** (`dataset/`): Creates dataset objects used during training and inference.

- **Model Architecture** (`models.py`): Contains PyTorch `nn.Module` classes for the model architecture.

- **Training Engine** (`engine.py`): Contains the training/validation/test loop logic and the overall optimization pipeline.

### Auxiliary Modules

Other files and modules (`utilities/`, `geometry.py`, `optimization.py`, `losses.py`, etc.) are auxiliary modules that provide supporting functionality.

!!! warning "Legacy Code"
    Several files have been imported from Alberto's older project (e.g., the pipelines and networks modules). These may be sparsely used or completely unused in the current codebase and will be removed in due time.
