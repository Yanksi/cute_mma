# CUTE MMA
Matrix matrix multiplication with tensorcores on Ampere architecture implemented with CuTe, which is part of the [cutlass](https://github.com/NVIDIA/cutlass/tree/main) library.

## Code structure
The overall logic of the code closely resembles the [`sgemm_sm80`](https://github.com/NVIDIA/cutlass/blob/main/examples/cute/tutorial/sgemm_sm80.cu) example provided by CuTe. The major modification is that the code in this repository works with both `half` and `float` datatype, while the computations are done with tensorcores. The new `gemm_device` function in `gemm_tc.hpp` generalize the size of the `smem->rmem` pipeline to work together with tensorcores.

The creation of those `TiledCopy`s and `TiledMMA`s are done in `gemm_sm80_tc`. `gemm_tn` and `gemm_nt` are the two versions of the `gemm` function for different layouts. The hyperparameters for these two functions would are stored in `gemm_config.hpp`. The default (currently the best) values for these hyperparameters can be found in folder `default_configs`.

## Compile
If the [cutlass]((https://github.com/NVIDIA/cutlass/tree/main)) and [argparse](https://github.com/p-ranav/argparse) library has already been cloned in the system, set environment variables `CUTLASS_DIR` and `ARGPARSE_DIR` to the root directory of those two libraries to avoid cmake cloning the two repositories again during configuration process.

The compilation of the code can be done by following

```bash
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make default
```

Target `default` contains two targets, `gemm_float` and `gemm_half`. Once finished compiling, both version of the `gemm` can be runned to checkout the performance. The `-h` flag can be provided to checkout all the available arguments and flags of the program.

## Autotuning
Autotuning of hyperparameters have also been (kinda) implemented. It works by generating all the possible configurations and compile a version of executable for each of them. The related files are all inside folder `autotune`. `run_build.sh` is responsible for generating and compiling all the autotune targets. `run_perf.sh` is responsible for benchmarking all the compiled executables and output the result to `results.pkl`. `analyse.py` is used to analyse the generated pickle file to findout the best configuration among all.