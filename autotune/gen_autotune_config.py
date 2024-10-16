import textwrap
import inspect
import itertools
import pathlib

class ConfigParams:
    config_format = textwrap.dedent("""
    template<CUTE_MMA_Layout ALayout, CUTE_MMA_Layout BLayout>
    struct Params <{dtypeo}, {dtyper}, ALayout, BLayout> {{
        const static int bM = {bm};
        const static int bN = {bn};
        const static int bK = {bk};
        const static int bP = {bp};
        const static bool block_tiling_copy = {block_tiling};
        using warp_layout = Layout<Shape<Int<{warp_layout[0]}>, Int<{warp_layout[1]}>>>;
        using mma_atom = {mma_atom};
    }};
    """)
    def __init__(self, *,
                 dtypeo="float", dtyper="float",
                 bm=128, bn=128,
                 bk=16, bp=4,
                 block_tiling="true",
                 warp_layout=(2,4),
                 mma_atom="SM80_16x8x8_F16F16F16F16_TN"):
        self.dtypeo = dtypeo
        self.dtyper = dtyper
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
    #include <common.hpp>
    #include <cute/layout.hpp>
    #include <cute/atom/mma_atom.hpp>
    namespace cute {{
    {configs}
    }}
    """)
    def __init__(self, f_name, config_params):
        self.f_name = f_name
        self.config_params = config_params
    
    def write(self):
        configs = "\n".join(str(config) for config in self.config_params)
        all_configs = textwrap.indent(configs, "    ")
        with open(self.f_name, "w") as f:
            f.write(self.file_header.format(configs=all_configs))

class MMMConfigSpace:
    def __init__(self, dtypeo, dtyper):
        self.dtypeo = dtypeo
        self.dtyper = dtyper
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
        if dtypeo == dtyper == "float":
            self.mma_atoms = [
                "SM80_16x8x4_F32TF32TF32F32_TN",
                "SM80_16x8x8_F32TF32TF32F32_TN"
            ]
        elif dtypeo == dtyper == "half":
            self.mma_atoms = [
                "SM80_16x8x8_F16F16F16F16_TN",
                "SM80_16x8x16_F16F16F16F16_TN"
            ]
        else:
            raise ValueError(f"Unsupported dtype combination {dtypeo} and {dtyper}")
    
    def csvheader(self):
        return "bm,bn,bk,bp,block_tiling,warp_layout_x,warp_layout_y,mma_atom"
    
    def configs(self):
        for bm, bn, bk, bp, block_tiling, warp_layout, mma_atom in itertools.product(
            self.bm, self.bn, self.bk, self.bp, self.block_tiling, self.warp_layouts, self.mma_atoms):
            yield ConfigParams(
                dtypeo=self.dtypeo,
                dtyper=self.dtyper,
                bm=bm, bn=bn, bk=bk,
                bp=bp,
                block_tiling=block_tiling,
                warp_layout=warp_layout,
                mma_atom=mma_atom)

for dtypeo, dtyper in [("float", "float"), ("half", "half")]:
    params = MMMConfigSpace(dtypeo, dtyper)
    csv_folder = pathlib.Path("autotune_configs/config_csvs")
    csv_folder.mkdir(parents=True, exist_ok=True)
    with open(f"autotune_configs/config_csvs/{dtypeo}_{dtyper}_param.csv", "w") as f:
        f.write(f"id,{params.csvheader()}\n")
        for i, p in enumerate(params.configs()):
            cf_path = pathlib.Path(f"autotune_configs/{dtypeo}_{dtyper}/config{i}/gemm_config.hpp")
            cf_path.parent.mkdir(parents=True, exist_ok=True)
            config_file = ConfigFile(cf_path, [p])
            config_file.write()
            f.write(f"{i},{p.tocsv()}\n")

# class PerformanceTuner:
#     def __init__(self, build_dir, compilation_target):
#         self.build_dir = build_dir
#         self.compilation_target = compilation_target
        
#     def generate_config(self):
         