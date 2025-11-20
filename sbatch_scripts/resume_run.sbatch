#!/bin/bash -l
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --time=1-00:00:00
#SBATCH --job-name=resume
#SBATCH --mail-user=alberto1.rota@polimi.it
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --output=/home/hpc/v120bb/v120bb18/job_outputs/job_%j.out
#SBATCH --begin=now

unset SLURM_EXPORT_ENV             

# Enabling internet access
export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80

export WS_DIR=/anvme/workspace/v120bb18-unreflectanything
export DATASET_DIR="$WS_DIR/datasets"
export RESULTS_DIR="/anvme/workspace/v120bb18-unreflectanything/results"

export RUN_NAME="crisp-bee-609"

# Python modules
module load python
cd $WORK/UnReflectAnything
source $WORK/UnReflectAnything/.venv/bin/activate

# # Running training
python train.py -nd --resume-run $RUN_NAME
