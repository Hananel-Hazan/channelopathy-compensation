#!/usr/bin/env bash
#
#SBATCH --job-name=GA-6_rl7
#SBATCH --time=1-00:00:00
#SBATCH -p rh7batch
#SBATCH --mem=16g
#SBATCH --output=./res_%j.%N.txt
#SBATCH -e ./res_%j.%N.err 
#SBATCH -N 1
#SBATCH -n 16
##SBATCH --constraint=rhel7




source "$(conda info --base)/etc/profile.d/conda.sh"
eval "$(conda shell.bash hook)"

conda activate deap
unset DISPLAY

cd "$SLURM_SUBMIT_DIR"   # submit this script from its own directory
python -m scoop neuron-ga.py

exit 
