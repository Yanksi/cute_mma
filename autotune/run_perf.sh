#!/bin/bash
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --mem-per-cpu=2G
#SBATCH --output="run_autotune.out"
#SBATCH --nodelist=ault25

module load cuda/12.1.1 cmake/3.21.1
cd ~/cute_mma/
python autotune/runner.py "$@"
