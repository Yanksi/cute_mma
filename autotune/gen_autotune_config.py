import textwrap
import inspect
import itertools
import pathlib

class ConfigParams:
    config_format = textwrap.dedent("""
    template<>
    struct Params <{dtypeo}, {dtyper}, CUTE_MMA_{Alayout}, CUTE_MMA_{Blayout}> {{
        const static int bM = {bm};
        const static int bN = {bn};
        const static int bK = {bk};
        const static int bP = {bp};
        const static bool block_tiling_copy = {block_tiling};
        using warp_layout = Layout<Shape<Int<{warp_layout[0]}>, Int<{warp_layout[1]}>>>;
        using mma_atom = {mma_atom};
        using s2r_atom = {s2r_atom};
    }};
    """)
    def __init__(self, *,
                 dtypeo="half", dtyper="half",
                 Alayout="T", Blayout="N",
                 bm=128, bn=128,
                 bk=16, bp=4,
                 block_tiling="true",
                 warp_layout=(2,4),
                 mma_atom="SM80_16x8x8_F16F16F16F16_TN",
                 s2r_atom="Copy_Atom<AutoVectorizingCopy, half>"):
        self.dtypeo = dtypeo
        self.dtyper = dtyper
        self.Alayout = Alayout
        self.Blayout = Blayout
        self.bm = bm
        self.bn = bn
        self.bk = bk
        self.bp = bp
        self.block_tiling = block_tiling
        self.warp_layout = warp_layout
        self.mma_atom = mma_atom
        self.s2r_atom = s2r_atom
    
    def __str__(self):
        return self.config_format.format(**self.__dict__)

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

def dict_product(options):
    """
    >>> list(dict_product({'number': [1, 2], 'character': 'ab'}))
    [{'character': 'a', 'number': 1},
     {'character': 'a', 'number': 2},
     {'character': 'b', 'number': 1},
     {'character': 'b', 'number': 2}]
    """
    return (dict(zip(options.keys(), x)) for x in itertools.product(*options.values()))

MMA_ATOMS = {
    ("float", "float"): [
        "SM80_16x8x4_F32TF32TF32F32_TN",
        "SM80_16x8x8_F32TF32TF32F32_TN"
    ],
    ("half", "half"): [
        "SM80_16x8x8_F16F16F16F16_TN",
        "SM80_16x8x16_F16F16F16F16_TN"
    ]
}

S2R_ATOMS = {
    ("float", "float") : {
        ("T", "N"): [
            "Copy_Atom<SM75_U32x4_LDSM_N, half_t>",
            "Copy_Atom<SM75_U32x2_LDSM_N, half_t>",
            "Copy_Atom<SM75_U32x1_LDSM_N, half_t>",
            "Copy_Atom<AutoVectorizingCopy, float>"
        ],
        ("N", "T"): [
            "Copy_Atom<AutoVectorizingCopy, float>"
        ],
    },
    ("half", "half") : {
        ("T", "N"): [
            "Copy_Atom<SM75_U32x4_LDSM_N, half_t>",
            "Copy_Atom<SM75_U32x2_LDSM_N, half_t>",
            "Copy_Atom<SM75_U32x1_LDSM_N, half_t>",
            "Copy_Atom<AutoVectorizingCopy, half>"
        ],
        ("N", "T"): [
            "Copy_Atom<SM75_U16x8_LDSM_T, half_t>",
            "Copy_Atom<SM75_U16x4_LDSM_T, half_t>",
            "Copy_Atom<SM75_U16x2_LDSM_T, half_t>",
            "Copy_Atom<AutoVectorizingCopy, half>"
        ],
    },
}

class MMMConfigSpace:
    def __init__(self, dtypeo, dtyper, Alayout, Blayout):
        self.dtypeo = dtypeo
        self.dtyper = dtyper
        self.Alayout = Alayout
        self.Blayout = Blayout
        self.generated_configs = []
        self.params = {
            "bm": [128, 256],
            "bn": [128, 256],
            "bk": [16, 32],
            "bp": [2, 3],
            "block_tiling": ["true"],    #["true", "false"],
            "warp_layout": [
                # (1,2), # 2 warps per block
                # (2,1),
                (1,4), # 4 warps per block
                (4,1),
                (2,2),
                (1,8), # 8 warps per block
                (8,1),
                (4,2),
                (2,4),
            ],
            "mma_atom": MMA_ATOMS[(dtypeo, dtyper)],
            "s2r_atom": S2R_ATOMS[(dtypeo, dtyper)][(Alayout, Blayout)]
        }
    
    def getcsv(self):
        keys = list(self.params.keys())
        header = "id," + ",".join(keys)
        body_lines = [",".join(str(params[k]) for k in keys) for params in self.generated_configs]
        body_lines = [f"{i}, {line}" for i, line in enumerate(body_lines)]
        body = "\n".join(body_lines)
        return f"{header}\n{body}"
    
    def configs(self):
        self.generated_configs = []
        for params in dict_product(self.params):
            self.generated_configs.append(params)
            yield ConfigParams(
                dtypeo=self.dtypeo,
                dtyper=self.dtyper,
                Alayout=self.Alayout,
                Blayout=self.Blayout,
                **params)

for dtypeo, dtyper in [("float", "float"), ("half", "half")]:
    for Alayout, Blayout in [("T", "N"), ("N", "T")]:
        params = MMMConfigSpace(dtypeo, dtyper, Alayout, Blayout)
        csv_folder = pathlib.Path("autotune_configs/config_csvs")
        csv_folder.mkdir(parents=True, exist_ok=True)
        
        for i, p in enumerate(params.configs()):
            cf_path = pathlib.Path(f"autotune_configs/{dtypeo}_{dtyper}/{Alayout}{Blayout}/config{i}/gemm_config.hpp")
            cf_path.parent.mkdir(parents=True, exist_ok=True)
            config_file = ConfigFile(cf_path, [p])
            config_file.write()
        
        with open(f"autotune_configs/config_csvs/{dtypeo}_{dtyper}_{Alayout}{Blayout}_param.csv", "w") as f:
            f.write(params.getcsv())

         