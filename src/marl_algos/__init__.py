from src import IPPO_KEYWORD, MAPPO_KEYWORD

from src.marl_algos.ippo import MarlIPPO

from src.marl_algos.mappo import MarlMAPPO


MARL_ALGORITHMS = {IPPO_KEYWORD: MarlIPPO, MAPPO_KEYWORD: MarlMAPPO}
