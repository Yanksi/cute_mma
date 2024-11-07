#!/bin/bash
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --mem-per-cpu=2G
#SBATCH --output="run_autotune.out"
#SBATCH --nodelist=ault25

target_dtype="all"
target_layout="all"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --dtype=*) target_dtype="${1#*=}" ;;
        --layout=*) target_layout="${1#*=}" ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

module load cuda/12.1.1 cmake/3.21.1
cd ~/cute_mma/
python autotune/runner.py --dtype=$target_dtype --layout=$target_layout