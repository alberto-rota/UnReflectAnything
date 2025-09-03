#  MODULES AND DATASET LOADING

# FIxing the issue
from rich.traceback import install

import main
import wandb

install(show_locals=False)


def sweep_agent(config=None):
    with wandb.init(config=config):
        config = wandb.config
        main.test(config)


sweep_agent()
