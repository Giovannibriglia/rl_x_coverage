from typing import Dict

from src.algos.ippo import IPPO
from src.algos.mappo import MAPPO


def get_marl_algo(config: Dict):
    name = config["algo_name"]
    algos = {
        "ippo": IPPO,
        "mappo": MAPPO,
    }
    try:
        algo_cls = algos[name.lower()]
        algo = algo_cls()
        algo.setup(config)
        return algo
    except KeyError:
        raise ValueError(f"Unknown MARL algorithm: {name}")
