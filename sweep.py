# -------------------------------------------------------------------------------------------------#

"""Copyright (c) 2024 Asensus Surgical"""

""" Code Developed by: Alberto Rota """
""" Supervision: Uriya Levy, Gal Weizman, Stefano Pomati """

# -------------------------------------------------------------------------------------------------#

# %% MODULES AND DATASET LOADING
import torch
import cv2 as cv
import matplotlib

matplotlib.rcParams.update(matplotlib.rcParamsDefault)

import main


from utilities import *
import wandb
import os
import sys
import yaml
from datetime import datetime


def trainsweep(config=None):
    with wandb.init(config=config):
        config = wandb.config
        main.train(config)

    if __name__ == "__main__":
        titlescreen()
    print("> Modules loaded")
    print("Torch Version: ", torch.__version__)
    print("OpenCV Version: ", cv.__version__)
    print("Python Version: ", os.sys.version.split()[0], "\n")
    print("> CUDA available: ", torch.cuda.is_available())

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ACCELERATED = torch.cuda.is_available()
    if len(sys.argv) == 1:
        print("> No YAML file specified, using config_sweep.yaml as default")
        SWEEP_PATH = "config_sweep.yaml"
    if len(sys.argv) > 1 and not sys.argv[1].endswith(".yaml"):
        print("> Input specified is not a YAML file")
        exit()
    if len(sys.argv) > 1:
        SWEEP_PATH = sys.argv[1]
    SWEEP_CONFIG = yaml.safe_load(open(SWEEP_PATH, "r"))
    SWEEP_CONFIG["name"] = (
        f"{SWEEP_CONFIG['name']}_{datetime.now().strftime('%d-%m-%Y_%H:%M')}"
    )
    # -------------------------------------------------------------------------------------------------#

    ORIGINAL_HEIGHT, ORIGINAL_WIDTH = 1024, 1280
    DOWNSAMPLE = 0.5
    calibration = cvcalib_fromyaml()
    INTRINSICS = intrinsics(calibration["M2"], DOWNSAMPLE).to(DEVICE)
    DISTORTION = calibration["D1"].to(DEVICE)

    WORKERS = 8 if ACCELERATED else 0

    sweep_id = wandb.sweep(SWEEP_CONFIG, project="DVO")
    os.environ["WANDB_AGENT_MAX_INITIAL_FAILURES"] = "50"
    wandb.agent(sweep_id, trainsweep, count=1000)
