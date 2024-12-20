import os
import subprocess
import re
import pathlib
from itertools import product
import numpy as np
from tqdm import tqdm
import pickle
import asyncio
import argparse


async def run_command_and_postprocessing(command, postprocess=lambda out, err: out, semaphore=None, avai_gpus=[0]):
    async with semaphore:
        gpu = avai_gpus.pop()
        command = f"CUDA_VISIBLE_DEVICES={gpu} {command}"
        process = await asyncio.create_subprocess_shell(command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        out, err = await process.communicate()
        avai_gpus.append(gpu)
        return postprocess(out, err)

def get_flops(out, err):
    perf_string = re.compile(r'\[\d+\.\d\]')
    result = perf_string.search(out.decode('utf-8'))
    if result is None:
        return (np.nan, out, err)
    result = result.group(0)
    result = result[1:-1]
    return (float(result), out, err)

def get_executables(root_path):
    executables = []
    for path in root_path.iterdir():
        if path.is_file() and path.name.startswith("gemm"):
            executables.append(path)
    return executables

async def main(target_dtype, target_layout, save_raw=False):
    results = {}
    if os.path.exists("results.pkl"):
        with open("results.pkl", "rb") as f:
            results = pickle.load(f)

    n_gpus = 4
    gpu_ids = list(range(n_gpus))
    semaphore = asyncio.Semaphore(n_gpus)
    tasks = []
    dtypes = [target_dtype]
    if target_dtype == "all":
        dtypes = ["float_float", "half_half"]
    layouts = [target_layout]
    if target_layout == "all":
        # layouts = ["TN", "NT"]
        layouts = ["tn", "nt"]
    for dtype, layout in product(dtypes, layouts):
        executable_root = pathlib.Path(f"build_autotune/{dtype}/{layout.upper()}")
        list_executables = get_executables(executable_root)
        curr_result = results.setdefault(dtype, {})
        for i, executable in enumerate(list_executables):
            curr_name = executable.name
            if curr_name in curr_result and layout in curr_result[curr_name]:
                continue
            command = " ".join([str(executable), "-m 8192", "-n 8192", "-k 4096", f"--transA {layout[0]}", f"--transB {layout[1]}"])
            task = asyncio.create_task(run_command_and_postprocessing(command, postprocess=get_flops, semaphore=semaphore, avai_gpus=gpu_ids))
            tasks.append((task, curr_result, curr_name, layout))
    
    with tqdm(total=len(tasks)) as pbar:
        for i, (task, curr_result, curr_name, layout) in enumerate(tasks):
            result = await task
            curr_result.setdefault(curr_name, {})[layout] = result
            pbar.update(1)
            if (i % 10) == 0:
                with open("results.pkl", "wb") as f:
                    pickle.dump(results, f)
    
    with open("results.pkl", "wb") as f:
        pickle.dump(results, f)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--dtype", type=str, default="half_half")
    argparser.add_argument("--layout", type=str, default="TN")
    args = argparser.parse_args()
    target_dtype = args.dtype
    target_layout = args.layout
    asyncio.run(main(target_dtype, target_layout))