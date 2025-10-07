#!/usr/bin/env python3
"""
Example script showing how to use the enhanced checkpoint saving/loading functionality.

This demonstrates how to:
1. Save checkpoints with model class information
2. Load models from checkpoints with model class information
3. Create model instances dynamically from checkpoint data
"""

import torch
from engine import Engine

def example_usage():
    """
    Example of how to use the enhanced checkpoint functionality.
    """
    
    # Example 1: Loading a model from checkpoint with class information
    checkpoint_path = "path/to/your/checkpoint.pth"
    
    # Method 1: Using the static method to create model from checkpoint
    checkpoint_data = Engine.create_model_from_checkpoint(
        checkpoint_path=checkpoint_path,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    
    if checkpoint_data:
        model = checkpoint_data["model"]
        config = checkpoint_data["config"]
        epoch = checkpoint_data["epoch"]
        model_class_name = checkpoint_data["model_class_name"]
        
        print(f"Loaded model of class: {model_class_name}")
        print(f"From epoch: {epoch}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # You can now use the model for inference
        # model.eval()
        # with torch.no_grad():
        #     output = model(your_input_data)
    
    # Method 2: Using the engine's reinstantiate method (existing functionality)
    # This still works as before, but now the checkpoint contains model class info
    # engine = Engine(model=your_model, dataset=your_dataset, config=your_config)
    # engine.reinstantiate_model_from_checkpoint(checkpoint_path)
    
    print("Checkpoint loading completed successfully!")

if __name__ == "__main__":
    example_usage()
