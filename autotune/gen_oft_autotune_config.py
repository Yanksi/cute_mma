import textwrap
import inspect
import itertools
import pathlib

class ConfigParams:
    config_format = textwrap.dedent("""
    struct CurrCompParams {{
        static const unsigned int bM = {bm};
        static const unsigned int bN = {bn};
        static const unsigned int bK = {bk};
        static const unsigned int bP1 = {bp1};
        static const unsigned int bP2 = {bp2};
        using warp_layout1 = cute::Layout<cute::Shape<cute::Int<{warp_size1}>>>;
        using warp_layout2 = cute::Layout<cute::Shape<cute::Int<{warp_layout[0]}>, cute::Int<{warp_layout[0]}>>>;
    }};
    """)
    def __init__(self, *,
                 bm=128, bn=256,
                 bk=16, bp1=3, bp2=2,
                 warp_size1=2,
                 warp_layout=(2,4)):
        self.bm = bm
        self.bn = bn
        self.bk = bk
        self.bp1 = bp1
        self.bp2 = bp2
        self.warp_size1 = warp_size1
        self.warp_layout = warp_layout
    
    def __str__(self):
        return self.config_format.format(**self.__dict__)

class ConfigFile:
    file_header = inspect.cleandoc("""
    #pragma once
    #include <cute/tensor.hpp>
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

class OFTConfigSpace:
    def __init__(self):
        self.generated_configs = []
        self.params = {
            "bm": [128, 256],
            "bn": [128, 256],
            "bk": [16, 32],
            "bp1": [2, 3],
            "bp2": [2, 3],
            "warp_size1": [2, 4],
            "warp_layout": [
                (2,2),
                (4,2),
                (2,4),
            ]
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
            yield ConfigParams(**params)

params = OFTConfigSpace()
csv_folder = pathlib.Path("autotune_configs/config_csvs")
csv_folder.mkdir(parents=True, exist_ok=True)

for i, p in enumerate(params.configs()):
    cf_path = pathlib.Path(f"autotune_configs/config{i}/oft_config.hpp")
    cf_path.parent.mkdir(parents=True, exist_ok=True)
    config_file = ConfigFile(cf_path, [p])
    config_file.write()

with open(f"autotune_configs/config_csvs/param.csv", "w") as f:
    f.write(params.getcsv())

         