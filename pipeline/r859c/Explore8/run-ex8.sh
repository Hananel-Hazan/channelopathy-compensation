#!/usr/bin/env bash
#
#SBATCH --job-name=Explor8

#SBATCH --output=./logs/res_%j_%N.txt
#SBATCH --error=./logs/res_%j_%N.err 

#SBATCH --constraint=rhel7


#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=12G         # memory per cpu-core (4G is default)
#SBATCH --time=04-00:00:00       # total run time limit (DD-HH:MM:SS)
#SBATCH --array=0-600             # job array with index values 0, 1, 2, 3, 4
# (the original job ran under a site-specific reservation, omitted here)
#SBATCH -p preempt


source "$(conda info --base)/etc/profile.d/conda.sh"
eval $(conda shell.bash hook)

conda activate deap
unset DISPLAY

cd "$SLURM_SUBMIT_DIR"   # submit this script from its own directory
python exploration.v3.run.py --parts 1

exit 
