#!/bin/bash -l
#SBATCH --gres=gpu:a40:1
#SBATCH --partition=a40
#SBATCH --time=5:00:00
#SBATCH --job-name=debug
#SBATCH --mail-user=alberto1.rota@polimi.it
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
unset SLURM_EXPORT_ENV             # enable export of environment from this script to srun

# Python modules
module load python
source /home/hpc/v120bb/v120bb18/UnReflectAnything/.venv/bin/activate

# Enabling internet access
export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80

export NOTEBOOK_PORT=23476
jupyter notebook --no-browser --port=$NOTEBOOK_PORT --ip=0.0.0.0 --NotebookApp.token='' --NotebookApp.password=''