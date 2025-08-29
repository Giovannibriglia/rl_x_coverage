from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch

from src import (
    IPPO_KEYWORD,
    MAPPO_KEYWORD,
    POLICIES_FOLDER,
    SCALARS_FOLDER,
    TEST_FOLDER,
    TRAIN_FOLDER,
    VORONOI_BASED_KEYWORD,
)
from src.algos import get_marl_algo

from src.plotting import plot_exp

from src.utils import (
    get_algo_dict,
    get_env_dict,
    get_torch_rl_env,
    rollout_eval,
    setup_folders,
)
from torchrl.envs import TransformedEnv
from tqdm import tqdm

from vmas.scenarios.coverage import VoronoiBasedActor

MAPPO_CONFIG_PATH = "./config/algos/mappo.yaml"
IPPO_CONFIG_PATH = "./config/algos/ippo.yaml"


def train_marl(
    algos_config: List[Dict],
    root_dir: Path,
    checkpoints_train: List,
    train_env_dict: Dict | None = None,
) -> List[int]:
    if train_env_dict is not None:
        dict_path = root_dir / TRAIN_FOLDER / "env_config.json"
        # make sure parent folders exist
        dict_path.parent.mkdir(parents=True, exist_ok=True)

        with open(dict_path, "w") as f:
            json.dump(train_env_dict, f, indent=4)

    for algo_config in algos_config:
        # algo_name = algo_config["algo_name"]
        algo = get_marl_algo(algo_config)

        train_csv_dir = root_dir / TRAIN_FOLDER / SCALARS_FOLDER
        policy_dir = root_dir / TRAIN_FOLDER / POLICIES_FOLDER

        algo.train_algo(
            checkpoints=checkpoints_train,
            train_csv_dir=train_csv_dir,
            policy_dir=policy_dir,
        )

    return checkpoints_train


def evaluation(
    algos_config: List[Dict],
    all_test_envs: Dict[str, TransformedEnv],
    checkpoints_train: List[int],
    root_dir: Path,
    n_checkpoints_eval: int = 25,
    with_voronoi: bool = True,
    seed: int = 0,
):

    for algo_config in algos_config:
        algo_name = algo_config["algo_name"]
        algo = get_marl_algo(algo_config)

        pbar = tqdm(
            all_test_envs.items(),
            total=len(list(all_test_envs.keys())),
            desc=f"evaluating {algo_name}...",
        )
        for test_env_name, test_env in pbar:

            for chkpt_n in checkpoints_train:
                pbar.set_postfix(
                    env=test_env_name, policy_chkpt=f"{algo_name}_chkpt_{chkpt_n}"
                )

                path_chkpt_n = (
                    root_dir
                    / TRAIN_FOLDER
                    / POLICIES_FOLDER
                    / f"{algo_name}_chkpt_{chkpt_n}.pt"
                )

                if algo_name == IPPO_KEYWORD:
                    new_policy = algo.transfer_policy(
                        target_env=test_env,
                        checkpoint_path=path_chkpt_n,
                        new_share_params_actor=False,
                        old_share_params_actor=False,
                    )
                elif algo_name == MAPPO_KEYWORD:
                    new_policy = algo.transfer_policy(
                        target_env=test_env,
                        checkpoint_path=path_chkpt_n,
                        new_share_params_actor=True,
                        old_share_params_actor=True,
                    )
                else:
                    raise NotImplementedError

                rollout_eval(
                    policy=new_policy,
                    env=test_env,
                    main_dir=root_dir / TEST_FOLDER / test_env_name,
                    filename=f"{algo_name}_chkpt_{chkpt_n}",
                    seed=seed,
                    with_video=True,
                    n_checkpoints_eval=n_checkpoints_eval,
                )

    if with_voronoi:
        pbar = tqdm(
            all_test_envs.items(),
            total=len(list(all_test_envs.keys())),
            desc=f"evaluating {VORONOI_BASED_KEYWORD}...",
        )

        for test_env_name, test_env in pbar:
            pbar.set_postfix(env=test_env_name)

            voronoi_based_policy = VoronoiBasedActor(test_env)

            rollout_eval(
                policy=voronoi_based_policy,
                env=test_env,
                main_dir=root_dir / TEST_FOLDER / test_env_name,
                filename=VORONOI_BASED_KEYWORD,
                seed=seed,
                with_video=True,
                n_checkpoints_eval=n_checkpoints_eval,
            )


def main(
    experiment_folder: str,
    batch_experiments: str,
    max_steps_train: int,
    n_envs: int,
    max_steps_eval: int,
    n_checkpoints_train: int,
    n_checkpoints_eval: int,
    seed: int = 0,
):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    n_iters = max_steps_train

    #  Define all your batches and their (kind, agents, gauss) specs:
    all_batches = {
        "batch1": {
            "train": ("base", 3, 1),
            "test": [
                ("base", 3, 1),
                ("base", 3, 3),
                ("base", 7, 1),
                ("base", 7, 7),
                ("base", 13, 1),
                ("base", 13, 13),
            ],
        },
        "batch2": {
            "train": ("base", 3, 3),
            "test": [
                ("base", 3, 1),
                ("base", 3, 3),
                ("base", 7, 1),
                ("base", 7, 7),
                ("base", 13, 1),
                ("base", 13, 13),
            ],
        },
        "batch3": {
            "train": ("base", 3, 3),
            "test": [
                ("dynamic", 3, 3),
                ("dynamic", 7, 7),
                ("dynamic", 13, 13),
            ],
        },
        "batch4": {
            "train": ("base", 3, 3),
            "test": [
                ("non_convex1", 3, 3),
                ("non_convex1", 7, 7),
                ("non_convex1", 13, 13),
            ],
        },
        "batch5": {
            "train": ("base", 3, 3),
            "test": [
                ("non_convex2", 3, 3),
                ("non_convex1", 7, 7),
                ("non_convex1", 13, 13),
            ],
        },
        "batch6": {
            "train": ("pfov1", 3, 3),
            "test": [
                ("pfov1", 3, 1),
                ("pfov1", 3, 3),
                ("pfov1", 7, 1),
                ("pfov1", 7, 7),
                ("pfov1", 13, 1),
                ("pfov1", 13, 13),
            ],
        },
    }

    if batch_experiments == "all":
        batch_specs = all_batches
    elif batch_experiments == "test":
        batch_specs = {
            "test": {
                "train": ("base", 3, 1),
                "test": [
                    ("base", 3, 1),
                    # ("dynamic", 3, 1),
                    # ("non_convex1", 3, 1),
                    ("non_convex2", 3, 1),
                ],
            }
        }
    else:
        batch_specs = {batch_experiments: all_batches[batch_experiments]}

    for batch_name, specs in batch_specs.items():
        exp_dir = setup_folders(experiment_folder, batch_name)

        train_spec = specs["train"]
        test_specs = specs["test"]

        kind, n_agents, n_gauss = train_spec
        train_env_dict, train_env_name = get_env_dict(
            kind, n_agents, n_gauss, max_steps_train, n_envs, n_iters
        )

        train_env = get_torch_rl_env(
            env_config=train_env_dict, device=device, fix_seed=False
        )

        max_steps = train_env_dict["max_steps"]

        assert n_checkpoints_train >= 2, "checkpoints must be at least 2"

        checkpoints_train = [
            int(round(i * (max_steps - 1) / (n_checkpoints_train - 1)))
            for i in range(n_checkpoints_train)
        ]

        algos_config = [
            get_algo_dict(IPPO_CONFIG_PATH, train_env_dict, train_env),
            get_algo_dict(MAPPO_CONFIG_PATH, train_env_dict, train_env),
        ]

        train_marl(algos_config, exp_dir, checkpoints_train, train_env_dict)

        all_test_envs = {}
        for kind, n_agents, n_gauss in test_specs:
            (
                test_env_dict,
                test_env_name,
            ) = get_env_dict(
                kind,
                n_agents,
                n_gauss,
                max_steps_eval,
                n_envs,
                n_iters,
            )
            all_test_envs[test_env_name] = get_torch_rl_env(
                env_config=test_env_dict, device=device, fix_seed=True
            )

        evaluation(
            algos_config,
            all_test_envs,
            checkpoints_train,
            exp_dir,
            n_checkpoints_eval,
            True,
            seed,
        )

        plot_exp(exp_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RL experiments with batches")
    parser.add_argument(
        "--experiment_folder",
        type=str,
        default="coverage",
        help="Folder of the experiment",
    )
    parser.add_argument(
        "--batch_experiments",
        type=str,
        default="all",
        choices=[
            "all",
            "test",
            "batch1",
            "batch2",
            "batch3",
            "batch4",
            "batch5",
            "batch6",
        ],
        help="Which batch to run",
    )
    parser.add_argument(
        "--max_steps_train",
        type=int,
        default=512,
        help="Maximum training steps per episode",
    )
    parser.add_argument(
        "--n_envs",
        type=int,
        default=32,
        help="Number of environments to consider in parallel",
    )
    parser.add_argument(
        "--max_steps_eval",
        type=int,
        default=1024,
        help="Maximum evaluation steps per episode",
    )
    parser.add_argument(
        "--n_checkpoints_train",
        type=int,
        default=25,
        help="Number of checkpoints during training",
    )
    parser.add_argument(
        "--n_checkpoints_eval",
        type=int,
        default=25,
        help="Number of checkpoints during evaluation",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")

    args = parser.parse_args()

    main(
        experiment_folder=args.experiment_folder,
        batch_experiments=args.batch_experiments,
        max_steps_train=args.max_steps_train,
        n_envs=args.n_envs,
        max_steps_eval=args.max_steps_eval,
        n_checkpoints_train=args.n_checkpoints_train,
        n_checkpoints_eval=args.n_checkpoints_eval,
        seed=args.seed,
    )
