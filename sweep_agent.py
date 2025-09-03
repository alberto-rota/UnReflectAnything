#  MODULES AND DATASET LOADING

import main

# FIxing the issue
from utilities import *

from rich.traceback import install
import wandb

install(show_locals=False)


def sweep_agent():
    with wandb.init():
        config = wandb.config
        config_dict = dict(config)
        main.run_pipeline(mode="train", config=config_dict)


sweep_agent()
