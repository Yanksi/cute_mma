import textwrap
import inspect
import itertools
import pathlib

class ConfigParams:
    config_format = textwrap.dedent("""
    namespace {name} {{
        const static int bM = {bm};
        const static int bN = {bn};
        const static int bK = {bk};
        const static int bP = {bp};
        const static bool block_tiling_copy = {block_tiling};
        using warp_layout = Layout<Shape<Int<{warp_layout[0]}>, Int<{warp_layout[1]}>>>;
        using mma_atom = {mma_atom};
    }}
    """)
    def __init__(self, name, *,
                 bm=128, bn=128,
                 bk=16, bp=4,
                 block_tiling="true",
                 warp_layout=(2,4),
                 mma_atom="SM80_16x8x8_F16F16F16F16_TN"):
        self.name = name
        self.bm = bm
        self.bn = bn
        self.bk = bk
        self.bp = bp
        self.block_tiling = block_tiling
        self.warp_layout = warp_layout
        self.mma_atom = mma_atom
    
    def __str__(self):
        return self.config_format.format(**self.__dict__)
    
    def tocsv(self):
        return f"{self.bm},{self.bn},{self.bk},{self.bp},{self.block_tiling},{self.warp_layout[0]},{self.warp_layout[1]},{self.mma_atom}"

class ConfigFile:
    file_header = inspect.cleandoc("""
    #pragma once
    #include <cute/layout.hpp>
    using namespace cute;
    #define DTYPE {dtype}
    """)
    def __init__(self, f_name, dtype, config_params):
        self.f_name = f_name
        self.dtype = dtype
        self.config_params = config_params
    
    def write(self):
        with open(self.f_name, "w") as f:
            f.write(self.file_header.format(dtype=self.dtype))
            for config in self.config_params:
                f.write(str(config))

class MMMConfigSpace:
    def __init__(self, name, dtype):
        self.name = name
        self.bm = [64, 128, 256]
        self.bn = [64, 128, 256]
        self.bk = [8, 16, 32]
        self.bp = [2, 3, 4]
        self.block_tiling = ["true", "false"]
        self.warp_layouts = [
            (1,2), # 2 warps per block
            (2,1),
            (1,4), # 4 warps per block
            (4,1),
            (2,2),
            (1,8), # 8 warps per block
            (8,1),
            (4,2),
            (2,4),
        ]
        if dtype == "float":
            self.mma_atoms = [
                "SM80_16x8x4_F32TF32TF32F32_TN",
                "SM80_16x8x8_F32TF32TF32F32_TN"
            ]
        elif dtype == "half":
            self.mma_atoms = [
                "SM80_16x8x8_F16F16F16F16_TN",
                "SM80_16x8x16_F16F16F16F16_TN"
            ]
        else:
            raise ValueError(f"Unsupported dtype {dtype}")
    
    def csvheader(self):
        return "bm,bn,bk,bp,block_tiling,warp_layout_x,warp_layout_y,mma_atom"
    
    def configs(self):
        for bm, bn, bk, bp, block_tiling, warp_layout, mma_atom in itertools.product(
            self.bm, self.bn, self.bk, self.bp, self.block_tiling, self.warp_layouts, self.mma_atoms):
            yield ConfigParams(f"{self.name}", bm=bm, bn=bn, bk=bk, bp=bp, block_tiling=block_tiling, warp_layout=warp_layout, mma_atom=mma_atom)

for dtype in ["float", "half"]:
    params_tn = MMMConfigSpace("ParamTN", dtype)
    params_nt = MMMConfigSpace("ParamNT", dtype)
    with open(f"{dtype}_param_tn.csv", "w") as f_tn, open(f"{dtype}_param_nt.csv", "w") as f_nt:
        f_tn.write(f"id,{params_tn.csvheader()}\n")
        f_nt.write(f"id,{params_nt.csvheader()}\n")
        for i, (ptn, pnt) in enumerate(zip(params_tn.configs(), params_nt.configs())):
            cf_path = pathlib.Path(f"{dtype}/autotune_configs/config{i}/gemm_config.hpp")
            cf_path.parent.mkdir(parents=True, exist_ok=True)
            config_file = ConfigFile(cf_path, dtype, [ptn, pnt])
            config_file.write()
            f_tn.write(f"config{i},{ptn.tocsv()}\n")
            f_nt.write(f"config{i},{pnt.tocsv()}\n")

# class PerformanceTuner:
#     def __init__(self, build_dir, compilation_target):
#         self.build_dir = build_dir
#         self.compilation_target = compilation_target
        
#     def generate_config(self):
         