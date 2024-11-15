$RECONFIGURE=false
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --reconfigure) RECONFIGURE=true ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

module load cuda/12.1.1 cmake/3.21.1
cd ~/cute_mma/

if $RECONFIGURE; then
    echo "Regenerating autotune_configs"
    rm -rf autotune_configs
    python autotune/gen_autotune_config.py
    echo "Reconfiguring build_autotune"
    rm -rf build_autotune
    mkdir build_autotune
    cd build_autotune
    cmake -DAUTOTUNE=1 -DCMAKE_BUILD_TYPE=Release ..
else
    cd build_autotune
fi

cd ..

sbatch --output="build_hh_NT.out" autotune/run_build.sh --target=half_half_NT
sbatch --output="build_hh_TN.out" autotune/run_build.sh --target=half_half_TN
sbatch --output="build_ff_NT.out" autotune/run_build.sh --target=float_float_NT
sbatch --output="build_ff_TN.out" autotune/run_build.sh --target=float_float_TN