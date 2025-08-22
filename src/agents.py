from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict

import torch


class MarlBase(ABC):
    def __init__(self, env, configs: Dict):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.env = env
        self.max_steps = None
        self.n_agents = None

        self._setup(configs)

    @abstractmethod
    def _setup(self, configs: Dict):
        raise NotImplementedError

    @abstractmethod
    def train_and_evaluate(
        self,
        env_train,
        envs_test: dict[str, Any],
        main_dir: Path,
        seed: int,
        n_checkpoints_train: int = 50,
        n_checkpoints_eval: int = 50,
    ):
        raise NotImplementedError
