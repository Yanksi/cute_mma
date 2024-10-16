import os
import subprocess
import re
import pathlib
from itertools import product
import numpy as np
from tqdm import tqdm
import pickle
import asyncio


async def run_command_and_postprocessing(command, postprocess=lambda out, err: None, semaphore=None, avai_gpus=[0]):
    async with semaphore:
        gpu = avai_gpus.pop()
        command = f"CUDA_VISIBLE_DEVICES={gpu} {command}"
        process = await asyncio.create_subprocess_shell(command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        out, err = await process.communicate()
        avai_gpus.append(gpu)
        return postprocess(out, err)

def postprocess(out, err):
    perf_string = re.compile(r'\[\d+\.\d\]')
    result = perf_string.search(out.decode('utf-8'))
    result = result.group(0)
    result = result[1:-1]
    return float(result)

def get_executables(root_path):
    executables = []
    for path in root_path.iterdir():
        if path.is_file() and path.name.startswith("gemm"):
            executables.append(path)
    return executables

async def main():
    results = {}
    if os.path.exists("results.pkl"):
        with open("results.pkl", "rb") as f:
            results = pickle.load(f)

    n_gpus = 4
    gpu_ids = list(range(n_gpus))
    semaphore = asyncio.Semaphore(n_gpus)
    tasks = []
    for dtype, layout in product(["float_float", "half_half"], zip("TN", "NT")):
        executable_root = pathlib.Path(f"build_autotune/{dtype}/{''.join(layout)}")
        list_executables = get_executables(executable_root)
        curr_result = results.setdefault(dtype, {})
        for i, executable in enumerate(list_executables):
            curr_name = executable.name
            if curr_name in curr_result and layout in results[curr_name]:
                continue
            command = " ".join([str(executable), "-m 8192", "-n 8192", "-k 4096", f"--transA {layout[0]}", f"--transB {layout[1]}"])
            task = asyncio.create_task(run_command_and_postprocessing(command, postprocess, semaphore, gpu_ids))
            tasks.append((task, curr_result, curr_name, layout))
    
    with tqdm(total=len(tasks)) as pbar:
        best_flops = 0
        for i, (task, curr_result, curr_name, layout) in enumerate(tasks):
            flops = await task
            curr_result.setdefault(curr_name, {})[layout] = flops
            best_flops = max(best_flops, flops)
            pbar.set_postfix(best_flops=best_flops)
            pbar.update(1)
            if (i % 10) == 0:
                with open("results.pkl", "wb") as f:
                    pickle.dump(results, f)
    
    with open("results.pkl", "wb") as f:
        pickle.dump(results, f)

if __name__ == "__main__":
    asyncio.run(main())