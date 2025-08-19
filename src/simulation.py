from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt

from tensordict.nn import set_composite_lp_aggregate
from torchrl.envs import check_env_specs, RewardSum, TransformedEnv, VmasEnv
from tqdm import tqdm
from vmas import make_env
from vmas.scenarios.voronoi import VoronoiPolicy
from vmas.simulator.utils import save_video

from src import (
    PLOTS_DIR_KEYWORD,
    POLICIES_FOLDER_NAME,
    SCALARS_FOLDER_NAME,
    TEST_KEYWORD,
    TRAIN_KEYWORD,
    TRAIN_SCALARS_FOLDER_NAME,
    VIDEOS_FOLDER_NAME,
)
from src.marl_algos.ippo import MarlIPPO
from src.marl_algos.mappo import MarlMAPPO

from src.utils import (
    get_files_in_folder,
    get_first_layer_folders,
    group_by_checkpoints,
    read_csv_strict,
    save_csv,
)


class Simulation:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def _setup_folders(self, experiment_name: str = ""):
        if experiment_name != "":
            self.root_dir = f"runs/{experiment_name}"
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.root_dir = f"runs/test_{timestamp}"

        self.root_dir = Path(self.root_dir)
        os.makedirs(self.root_dir)

        print("Experiment root:", self.root_dir)

    def _setup_torch_rl_env(self, env_config: Dict):

        frames_per_batch = env_config["frames_per_batch"]
        max_steps = env_config["max_steps"]
        scenario = env_config["scenario_name"]
        n_agents = env_config["n_agents"]
        seed = env_config.get("seed", 42)
        env_kwargs = env_config.get("env_kwargs", {})

        set_composite_lp_aggregate(False)
        num_envs = frames_per_batch // max_steps
        torch_rl_raw_env = VmasEnv(
            scenario=scenario,
            num_envs=num_envs,
            continuous_actions=True,
            max_steps=max_steps,
            device=self.device,
            n_agents=n_agents,
            seed=seed,
            **env_kwargs,
        )

        torch_rl_transformed_env = TransformedEnv(
            torch_rl_raw_env,
            RewardSum(
                in_keys=[torch_rl_raw_env.reward_key],
                out_keys=[("agents", "episode_reward")],
            ),
        )
        check_env_specs(torch_rl_transformed_env)

        return torch_rl_transformed_env

    def _get_marl_algo(self, env, algo_configs: Dict):
        if algo_configs["algo_name"] == "IPPO":
            agent_cls = MarlIPPO
        elif algo_configs["algo_name"] == "MAPPO":
            agent_cls = MarlMAPPO
        else:
            raise NotImplementedError

        return agent_cls(env, algo_configs)

    def run(
        self,
        env_configs,
        algo_configs,
        experiment_name: str = "",
        n_checkpoints: int = 10,
    ):

        self._setup_folders(experiment_name)

        for batch_n, train_test in env_configs.items():
            folder_batch = self.root_dir / f"{batch_n}"
            folder_batch.mkdir(exist_ok=True)

            train_envs = train_test[TRAIN_KEYWORD]
            test_envs = train_test[TEST_KEYWORD]

            for train_env_name, train_env_config in train_envs.items():
                folder_exp = folder_batch / train_env_name
                folder_exp.mkdir(exist_ok=True)

                env_train_torch_rl = self._setup_torch_rl_env(train_env_config)

                test_envs_torch_rl = {
                    env_test_name: self._setup_torch_rl_env(test_env_config)
                    for env_test_name, test_env_config in test_envs.items()
                }

                for algo_name, algo_config in algo_configs.items():
                    marl_agent = self._get_marl_algo(env_train_torch_rl, algo_config)

                    marl_agent.train_and_evaluate(
                        env_train=env_train_torch_rl,
                        envs_test=test_envs_torch_rl,
                        main_dir=folder_exp,
                        n_checkpoints_train=n_checkpoints,
                        n_checkpoints_eval=n_checkpoints,
                    )

                for test_env_name, test_env_config in test_envs.items():

                    save_dir = folder_exp / test_env_name

                    self.use_voronoi_based_heuristic(
                        test_env_config,
                        main_dir=save_dir,
                        n_checkpoints=n_checkpoints,
                    )

        return self.root_dir

    def use_voronoi_based_heuristic(
        self, env_config, main_dir, n_checkpoints: int = 100, with_video: bool = False
    ):
        frames_per_batch = env_config["frames_per_batch"]
        max_steps = env_config["max_steps"]
        scenario = env_config["scenario_name"]
        n_agents = env_config["n_agents"]
        seed = env_config.get("seed", 42)
        env_kwargs = env_config.get("env_kwargs", {})

        env_kwargs["n_agents"] = n_agents
        num_envs = frames_per_batch // max_steps

        env = make_env(
            scenario=scenario,
            num_envs=num_envs,
            device=self.device,
            continuous_actions=True,
            wrapper=None,
            seed=seed,
            **env_kwargs,
        )

        policy = VoronoiPolicy(env=env, continuous_action=True)
        obs = torch.stack(env.reset(), dim=0)

        assert n_checkpoints >= 2, "Need at least 2 checkpoints (first and last)."

        if n_checkpoints >= max_steps:
            checkpoints = range(max_steps)
        else:
            checkpoints = [
                int(round(i * (max_steps - 1) / (n_checkpoints - 1)))
                for i in range(n_checkpoints)
            ]

        if with_video:
            frame_list = []

        metrics = {
            "reward": np.zeros((num_envs, n_checkpoints, n_agents)),
            "eta": np.zeros((num_envs, n_checkpoints, n_agents)),
            "beta": np.zeros((num_envs, n_checkpoints, n_agents)),
            "n_collisions": np.zeros((num_envs, n_checkpoints, n_agents)),
        }

        for t in tqdm(range(max_steps), desc="Voronoi-based heuristic evaluation..."):
            # compute actions for each agent
            actions = [
                policy.compute_action(obs[i], u_range=env.agents[i].u_range)
                for i in range(n_agents)
            ]

            # step the environment
            obs, rews, dones, info_list = env.step(actions)

            # Check if this iteration is a checkpoint
            if t in checkpoints:
                for ag_id in range(n_agents):
                    metrics["reward"][:, checkpoints.index(t), ag_id] = (
                        rews[:][ag_id].cpu().numpy()
                    )

                    metrics["eta"][:, checkpoints.index(t), ag_id] = (
                        info_list[ag_id]["eta"].cpu().numpy()
                    )

                    metrics["beta"][:, checkpoints.index(t), ag_id] = (
                        info_list[ag_id]["beta"].cpu().numpy()
                    )

                    metrics["n_collisions"][:, checkpoints.index(t), ag_id] = (
                        info_list[ag_id]["n_collisions"].cpu().numpy()
                    )

            if with_video:
                # render frames if needed
                frame_list.append(
                    env.render(
                        mode="rgb_array",
                        agent_index_focus=None,
                        visualize_when_rgb=True,
                    )
                )

        if with_video:
            # save video
            video_path = main_dir / VIDEOS_FOLDER_NAME
            os.makedirs(video_path, exist_ok=True)
            video_name = f"{video_path}/voronoi_based"
            save_video(video_name, frame_list, 1 / env.scenario.world.dt)

        # write records to CSV using csv module
        csv_path = main_dir / SCALARS_FOLDER_NAME / "voronoi_based.csv"

        save_csv(
            csv_path,
            n_agents,
            checkpoints,
            metrics["reward"],
            metrics["eta"],
            metrics["beta"],
            metrics["n_collisions"],
        )

    def make_plots(self, main_dir: Path | str):

        experiments = get_first_layer_folders(main_dir)
        # print("Experiments: ", experiments)

        for exp_dir in experiments:
            train_folders = get_first_layer_folders(exp_dir)
            # print("Train folders: ", train_folders)

            pbar = tqdm(train_folders)

            for train_dir in pbar:
                test_folders = get_first_layer_folders(train_dir)
                # print("Test folders: ", test_folders)
                pbar.set_description(desc=f"Plotting {str(exp_dir).split("/")[-1]}...")

                for test_dir in test_folders:
                    pbar.set_postfix(folder=str(test_dir).split("/")[:-2])

                    if POLICIES_FOLDER_NAME in str(test_dir):
                        continue

                    elif TRAIN_SCALARS_FOLDER_NAME in str(test_dir):
                        list_csv_train = get_files_in_folder(test_dir, "csv")
                        train_on = str(test_dir).split("/")[-2].replace("_", " ")
                        title = "Train on " + train_on
                        dir_save = test_dir / PLOTS_DIR_KEYWORD
                        data, metrics = self._sort_list_of_csv(list_csv_train)
                        self._plot_results(metrics, data, title, dir_save)

                    else:
                        test_res = test_dir / SCALARS_FOLDER_NAME
                        list_all_csv_test = get_files_in_folder(test_res, "csv")
                        groups_by_chkpt = group_by_checkpoints(list_all_csv_test)

                        df_mean_test = {}

                        for chkpt, list_csv_test in groups_by_chkpt.items():
                            test_on = str(test_res).split("/")[-2].replace("_", " ")
                            title = "Test on " + test_on + " chkpt: " + str(chkpt)
                            dir_save = (
                                test_res / PLOTS_DIR_KEYWORD / f"checkpoint_{chkpt}"
                            )
                            data, metrics = self._sort_list_of_csv(list_csv_test)
                            self._plot_results(metrics, data, title, dir_save)

                            for algo_name, df in data.items():
                                df_mean = df.mean(axis=0)

                                if algo_name not in df_mean_test:
                                    # first entry: start a new DataFrame
                                    df_mean_test[algo_name] = df_mean.to_frame().T
                                else:
                                    # append to existing DataFrame
                                    df_mean_test[algo_name] = pd.concat(
                                        [df_mean_test[algo_name], df_mean.to_frame().T],
                                        ignore_index=True,
                                    )

                        chkpts = list(groups_by_chkpt.keys())
                        for algo_name in df_mean_test.keys():
                            df_mean_test[algo_name]["step"] = chkpts

                        for algo, df in df_mean_test.items():
                            # ensure step is numeric
                            df["step"] = pd.to_numeric(df["step"], errors="coerce")

                            # sort whole rows by step, reset index
                            df_mean_test[algo] = df.sort_values(by="step").reset_index(
                                drop=True
                            )

                        test_on = str(test_res).split("/")[-2].replace("_", " ")
                        title = "Test on " + test_on + " all"
                        dir_save = test_res / PLOTS_DIR_KEYWORD / "all"
                        self._plot_results(metrics, df_mean_test, title, dir_save)

    @staticmethod
    def _sort_list_of_csv(list_of_csv_path: List):
        """
        Plot mean ± std curves for each metric across multiple CSVs.
        'step' is always used as the x-axis, not plotted as a metric.

        Args:
            list_of_csv_path: list of CSV paths (str or Path).

        Returns:
            Dict mapping <metric_name> -> saved image Path.
        """
        csv_paths = [Path(p) for p in list_of_csv_path]

        # Load CSVs
        data = {}
        for csv_path in csv_paths:
            df = read_csv_strict(csv_path)
            df.columns = [c.strip() for c in df.columns]
            if "step" not in df.columns:
                raise ValueError(f"'step' column not found in {csv_path}")

            name = csv_path.stem.lower()
            if "mappo" in name:
                algo_name = "mappo"
            elif "ippo" in name:
                algo_name = "ippo"
            elif "voronoi" in name:
                algo_name = "voronoi"
            else:
                algo_name = csv_path.stem  # fallback

            data[algo_name] = df

        # Find all metrics (exclude "step")
        metrics = set()
        for df in data.values():
            for col in df.columns:
                if col.endswith("_iqm"):
                    base = col[:-4]
                    if f"{base}_iqrstd" in df.columns and base != "step":
                        metrics.add(base)

        if not metrics:
            raise ValueError("No valid metrics with *_iqm and *_iqrstd found.")

        return data, list(metrics)

    @staticmethod
    def _plot_results(
        metrics: List,
        data: Dict[str, pd.DataFrame],
        title: str,
        dir_save: Path,
        img_format: str = "png",
    ):
        algo_colors = {
            "mappo": "#d62728",  # red
            "ippo": "#2ca02c",  # green
            "voronoi": "#1f77b4",  # blue
        }

        os.makedirs(dir_save, exist_ok=True)

        for metric in metrics:
            plt.figure(dpi=500, figsize=(6, 4))
            plt.title(title)
            for algo_name, df in data.items():
                mean_col, std_col = f"{metric}_iqm", f"{metric}_iqrstd"
                if mean_col not in df.columns or std_col not in df.columns:
                    continue

                x = df["step"].values
                y = df[mean_col].values
                s = df[std_col].values

                color = algo_colors.get(algo_name, None)

                plt.plot(x, y, label=algo_name, color=color)
                plt.fill_between(x, y - s, y + s, alpha=0.1, color=color)

            plt.xlabel("Step")
            plt.ylabel(metric.replace("_", " "))
            plt.legend(loc="best")
            plt.tight_layout()
            img_save = dir_save / f"{metric}.{img_format}"
            plt.savefig(img_save)
            # plt.show()
            plt.close()
