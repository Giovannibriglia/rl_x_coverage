import argparse
from typing import Dict, Tuple

import torch

from src import TEST_KEYWORD, TRAIN_KEYWORD
from src.simulation import Simulation


def get_env_dict(
    kind: str,
    n_agents: int,
    n_gauss: int,
    max_steps_env: int = 500,
    frames_per_batch_env: int = 5000,
    n_iters_env: int = 500,
    seed: int = 42,
) -> Tuple[Dict, str]:

    env_dict = {
        "max_steps": max_steps_env,
        "n_agents": n_agents,
        "seed": seed,
        "frames_per_batch": frames_per_batch_env,
        "scenario_name": "voronoi",
        "env_kwargs": {
            "n_gaussians": n_gauss,
            "grid_spacing": 0.05,
            "centralized": False,
            "shared_rew": False,
            "n_iters": n_iters_env,
            "n_rays": 360,
            "lidar_range": 0.5,
        },
    }

    if kind == "basic":
        pass
    elif kind == "pofv1":
        env_dict["env_kwargs"]["angle_end"] = torch.pi + 0.05
        env_dict["env_kwargs"]["n_rays"] = 180
    elif kind == "non_convex1":
        env_dict["env_kwargs"]["n_obstacles"] = 8
    elif kind == "non_convex2":
        env_dict["env_kwargs"]["L_env"] = True
    elif kind == "dynamic":
        env_dict["env_kwargs"]["dynamic"] = True
    else:
        raise ValueError(f"unknown env kind: {kind}")

    env_name = f"{kind}_{n_agents}agents_{n_gauss}gauss"

    return env_dict, env_name


def main(
    experiment_name: str = "test",
    batch_experiments: str = "all",
    max_steps: int = 500,
    frames_per_batch: int = 5000,
    max_steps_evaluation: int = 1000,
    n_checkpoints: int = 50,
):

    n_iters = max_steps

    #  Define all your batches and their (kind, agents, gauss) specs:
    all_batches = {
        "batch0": {
            "train": [
                ("basic", 3, 1),
            ],
            "test": [
                ("basic", 3, 3),
            ],
        },
        "batch1": {
            "train": [
                ("basic", 3, 1),
            ],
            "test": [
                ("basic", 3, 1),
                ("basic", 3, 3),
                ("basic", 5, 1),
                ("basic", 5, 5),
                ("basic", 7, 1),
                ("basic", 7, 7),
            ],
        },
        "batch2": {
            "train": [
                ("basic", 3, 3),
            ],
            "test": [
                ("basic", 3, 3),
                ("basic", 5, 5),
                ("basic", 7, 7),
            ],
        },
        "batch3": {
            "train": [
                ("basic", 3, 3),
            ],
            "test": [
                ("dynamic", 3, 1),
                ("dynamic", 3, 3),
                ("dynamic", 5, 1),
                ("dynamic", 5, 5),
                ("dynamic", 7, 1),
                ("dynamic", 7, 7),
            ],
        },
        "batch4": {
            "train": [
                ("basic", 3, 3),
            ],
            "test": [
                ("pofv1", 3, 1),
                ("pofv1", 3, 3),
                ("pofv1", 5, 1),
                ("pofv1", 5, 5),
                ("pofv1", 7, 1),
                ("pofv1", 7, 7),
            ],
        },
        "batch5": {
            "train": [
                ("basic", 3, 3),
            ],
            "test": [
                ("non_convex1", 3, 1),
                ("non_convex1", 3, 3),
                ("non_convex1", 5, 1),
                ("non_convex1", 5, 5),
                ("non_convex1", 7, 1),
                ("non_convex1", 7, 7),
            ],
        },
        "batch6": {
            "train": [
                ("basic", 3, 3),
            ],
            "test": [
                ("non_convex2", 3, 1),
                ("non_convex2", 3, 3),
                ("non_convex2", 5, 1),
                ("non_convex2", 5, 5),
                ("non_convex2", 7, 1),
                ("non_convex2", 7, 7),
            ],
        },
    }

    if batch_experiments == "all":
        batch_specs = all_batches
    else:
        batch_specs = {batch_experiments: all_batches[batch_experiments]}

    # Build env_configs by looping through each batch:
    env_configs = {}
    for batch_name, specs in batch_specs.items():
        # unpack specs
        train_specs = specs["train"]
        test_specs = specs["test"]

        # build name→dict mappings
        train_envs = {
            env_name: env_dict
            for env_dict, env_name in (
                get_env_dict(
                    kind, n_agents, n_gauss, max_steps, frames_per_batch, n_iters
                )
                for kind, n_agents, n_gauss in train_specs
            )
        }
        test_envs = {
            env_name: env_dict
            for env_dict, env_name in (
                get_env_dict(
                    kind,
                    n_agents,
                    n_gauss,
                    max_steps_evaluation,
                    int(max_steps_evaluation * (frames_per_batch / max_steps)),
                    n_iters,
                )
                for kind, n_agents, n_gauss in test_specs
            )
        }

        env_configs[batch_name] = {
            TRAIN_KEYWORD: train_envs,
            TEST_KEYWORD: test_envs,
        }

    algo_configs = {
        "ippo": {
            "algo_name": "IPPO",
            "max_steps": max_steps,
            "n_agents": 3,
            "frames_per_batch": frames_per_batch,
            "n_iters": n_iters,
            "num_epochs": 50,
            "minibatch_size": 256,
            "max_grad_norm": 0.5,
            "clip_epsilon": 0.2,
            "entropy_eps": 0.01,
            "gamma": 0.99,
            "lambda": 0.95,
            "lr": 3e-4,
        }
    }

    sim = Simulation()
    sim.run(
        env_configs=env_configs,
        algo_configs=algo_configs,
        experiment_name=experiment_name,
        n_checkpoints=n_checkpoints,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RL experiments with batches")
    parser.add_argument(
        "--experiment_name", type=str, default="test", help="Name of the experiment"
    )
    parser.add_argument(
        "--batch_experiments",
        type=str,
        default="all",
        choices=[
            "all",
            "batch0",
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
        "--max_steps", type=int, default=500, help="Maximum training steps per episode"
    )
    parser.add_argument(
        "--frames_per_batch", type=int, default=5000, help="Frames per batch"
    )
    parser.add_argument(
        "--max_steps_evaluation",
        type=int,
        default=1000,
        help="Maximum evaluation steps per episode",
    )
    parser.add_argument(
        "--n_checkpoints", type=int, default=50, help="Number of checkpoints"
    )

    args = parser.parse_args()
    main(
        experiment_name=args.experiment_name,
        batch_experiments="batch0",  # args.batch_experiments,
        max_steps=10,  # args.max_steps,
        frames_per_batch=40,  # args.frames_per_batch,
        max_steps_evaluation=8,  # args.max_steps_evaluation,
        n_checkpoints=4,  # args.n_checkpoints,
    )

# TODO: plots all
# TODO: QMIX - implement and save
