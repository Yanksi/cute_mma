#!/bin/bash
#SBATCH -n 1
#SBATCH --cpus-per-task=32
#SBATCH --time=4:00:00
#SBATCH --mem-per-cpu=2G
#SBATCH --output="run_make.out"
RECONFIGURE=false
TARGET="all"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --reconfigure) RECONFIGURE=true ;;
        --target=*) TARGET="${1#*=}" ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

module load cuda/12.1.1 cmake/3.21.1
cd ~/cute_mma/sgemm

if $RECONFIGURE; then
    rm -rf autotune_configs
    python autotune/gen_autotune_config.py
    rm -rf build_autotune
    mkdir build_autotune
    cd build_autotune
    cmake -DAUTOTUNE=1 -DCMAKE_BUILD_TYPE=Release ..
else
    cd build_autotune
fi

make $TARGET -k -j 32 2>/dev/null