import wandb
import numpy as np
from dotenv import load_dotenv
import os

# Load API key from .env
load_dotenv()
wandb.login(key=os.getenv("WANDB_API_KEY"))

# Initialize API
api = wandb.Api()

# Get all runs from the project
runs = api.runs("unreflect-anything/UnReflectAnything")

print(f"Total runs: {len(runs)}")
print("\nChecking first few runs for available metrics...")

for i, run in enumerate(runs[:3]):  # Check first 3 runs
    print(f"\n{'='*60}")
    print(f"Run {i+1}: {run.name} (ID: {run.id})")
    print(f"State: {run.state}")
    
    # Get system metrics (final values)
    print("\nSummary metrics:")
    for key in list(run.system.keys())[:20]:  # Show first 20 keys
        print(f"  {key}: {run.system[key]}")
    
    # Try to get history keys
    history = run.history(pandas=False)
    if len(history) > 0:
        print(f"\nHistory available: {len(history)} steps")
        print("Available keys in history:")
        print(f"  {list(history[0].keys())[:20]}")  # Show first 20 keys
    else:
        print("\nNo history data available")
    
    # Check if system metrics exist
    system_keys = [k for k in run.system.keys() if 'gpu' in k.lower() or 'power' in k.lower() or 'system' in k.lower()]
    if system_keys:
        print(f"\nSystem/GPU/Power related keys found:")
        for key in system_keys[:10]:
            print(f"  {key}")

print(f"\n{'='*60}")
print("Please check the output above and let me know the correct metric name.")