#!/usr/bin/env bash
#
#SBATCH --job-name=GA-4_search
#SBATCH --time=00-52:00:00
#SBATCH --mem=16g
#SBATCH --output=./res_%j.txt
#SBATCH -e ./res_%j.err 
#SBATCH -n 64



source "$(conda info --base)/etc/profile.d/conda.sh"
eval "$(conda shell.bash hook)"

conda activate deap
unset DISPLAY

cd "$SLURM_SUBMIT_DIR"   # submit this script from its own directory
python -m scoop neuron-ga.py

exit 
