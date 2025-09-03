#!/bin/bash -l
#SBATCH --gres=gpu:a40:1
#SBATCH --partition=a40
#SBATCH --time=10:00:00
#SBATCH --job-name=debug_batchjob
#SBATCH --mail-user=alberto1.rota@polimi.it
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
unset SLURM_EXPORT_ENV             # enable export of environment from this script to srun

module load python
     # Setup job environment (load modules, stage data, ...)
PORT=23476
jupyter notebook --no-browser --port=$PORT --ip=0.0.0.0 --NotebookApp.token='' --NotebookApp.password=''
