
#  MODULES AND DATASET LOADING
import torch

from dotmap import DotMap
import main

# FIxing the issue
from utilities import *
import os, yaml
import argparse
import debugpy

from rich.traceback import install
import wandb

install(show_locals=False)


def sweep_agent():
    with wandb.init():
        config = wandb.config
        config_dict = dict(config)
        main.run_pipeline(mode="train", config=config_dict)


sweep_agent()
