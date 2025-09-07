#!/bin/bash -l
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --time=1-00:00:00
#SBATCH --job-name=train_a100
#SBATCH --mail-user=alberto1.rota@polimi.it
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=job_%j.out
#SBATCH --begin=now

unset SLURM_EXPORT_ENV             

# Enabling internet access
export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80

export WS_DIR=/anvme/workspace/v120bb18-unreflectanything
export DATASET_DIR="$WS_DIR/datasets"
export RESULTS_DIR="$WS_DIR/results"

# Python modules
module load python
cd $HOME/UnReflectAnything
source $HOME/UnReflectAnything/.venv/bin/activate

# # Running training
# python train.py
python train.py

# # Moving results on permanent storage
# echo "$(date): Copying results to $SLOW_DATASET_DIR"
# cp -r "$RESULTS_DIR" "$SLOW_DATASET_DIR"
# echo "$(date): Results copied to $SLOW_DATASET_DIR"