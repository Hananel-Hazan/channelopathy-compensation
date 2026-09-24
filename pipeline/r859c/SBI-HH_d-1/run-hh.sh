#!/usr/bin/env bash
#
#SBATCH --job-name=sbi-HH
#SBATCH --time=00-70:00:00
#SBATCH --mem=16g
#SBATCH --output=./res_%j.txt
#SBATCH -e ./res_%j.err 
#SBATCH -n 8
#SBATCH --constraint=rhel7



source "$(conda info --base)/etc/profile.d/conda.sh"
eval "$(conda shell.bash hook)"

conda activate sbi
unset DISPLAY

cd "$SLURM_SUBMIT_DIR"   # submit this script from its own directory
python neuron-sbi-hh.py

exit 
