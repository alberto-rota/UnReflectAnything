"""Copyright (c) 2024 Asensus Surgical"""

""" Code Developed by: Alberto Rota """
""" Supervision: Uriya Levy, Gal Weizman, Stefano Pomati """

# -------------------------------------------------------------------------------------------------#
#  MODULES AND DATASET LOADING

import main

# FIxing the issue

from rich.traceback import install
import wandb

install(show_locals=False)


def sweep_agent(config=None):
    with wandb.init(config=config):
        config = wandb.config
        main.test(config)


sweep_agent()
