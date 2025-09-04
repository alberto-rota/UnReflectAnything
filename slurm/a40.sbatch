#!/bin/bash -l
#SBATCH --gres=gpu:a40:1
#SBATCH --partition=a40
#SBATCH --time=1-00:00:00
#SBATCH --job-name=train_a40
#SBATCH --mail-user=alberto1.rota@polimi.it
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16

unset SLURM_EXPORT_ENV             


# Enabling internet access
export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80

# Moving dataset on fast storage
export SLOW_DATASET_DIR="$WORK/datasets"
export SLOW_RESULTS_DIR="$WORK/results"

echo "$(date): Copying datasets to $TMPDIR"
cp -r "$SLOW_DATASET_DIR" "$TMPDIR"
echo "$(date): Datasets copied to $TMPDIR"

export DATASET_DIR="$TMPDIR/$SLURM_JOB_ID.alex/datasets"
export RESULTS_DIR="$TMPDIR/$SLURM_JOB_ID.alex/results"

# Python modules
module load python
source $HOME/UnReflectAnything/.venv/bin/activate

# Running training
python train.py

# Moving results on permanent storage
echo "$(date): Copying results to $SLOW_DATASET_DIR"
cp -r "$RESULTS_DIR" "$SLOW_DATASET_DIR"
echo "$(date): Results copied to $SLOW_DATASET_DIR"