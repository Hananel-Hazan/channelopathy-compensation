#!/usr/bin/env bash
#
#SBATCH --job-name=GA-6_rl6
#SBATCH --time=00-02:00:00
#SBATCH --mem=16g
#SBATCH --output=./res_%j.txt
#SBATCH -e ./res_%j.err 
#SBATCH -n 64
##SBARCH -N 2
#SBATCH --constraint=rhel6



source "$(conda info --base)/etc/profile.d/conda.sh"
eval "$(conda shell.bash hook)"

conda activate deap
unset DISPLAY

cd "$SLURM_SUBMIT_DIR"   # submit this script from its own directory

hosts=$(bash -c hostname)
python -m scoop --host $hosts neuron-ga.py

exit 
