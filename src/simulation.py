from __future__ import annotations

import json
import os
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torchrl.envs import TransformedEnv
from tqdm import tqdm
from vmas.scenarios.voronoi import VoronoiBasedActor

from src import (
    IPPO_KEYWORD,
    MAPPO_KEYWORD,
    PLOTS_FOLDER,
    POLICIES_FOLDER,
    SCALARS_FOLDER,
    TEST_FOLDER,
    TRAIN_FOLDER,
    VORONOI_BASED_KEYWORD,
)
from src.algos import MARL_ALGORITHMS
from src.base_algo import MarlAlgo
from src.utils import (
    evaluate_and_record,
    get_files_in_folder,
    get_first_layer_folders,
    group_by_checkpoints,
    read_csv_strict,
    seed_everything,
)


class Simulation:
    def __init__(
        self,
        device: str = None,
        seed: int = 42,
        experiment_folder: str = "test",
        experiment_name: str = "",
    ):
        self.device = (
            device
            if device is not None
            else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        )
        self.seed = seed
        seed_everything(seed)

        self._setup_folders(experiment_folder, experiment_name)

    def _setup_folders(
        self, experiment_folder: str = "test", experiment_name: str = ""
    ):

        self.root_dir = Path(f"runs/{experiment_folder}")
        os.makedirs(self.root_dir, exist_ok=True)

        if experiment_name != "":
            self.root_dir = self.root_dir / experiment_name
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.root_dir = self.root_dir / f"{experiment_folder}test_{timestamp}"

        self.root_dir = Path(self.root_dir)
        os.makedirs(self.root_dir)

        print("Experiment root:", self.root_dir)

    @staticmethod
    def get_algo(config: Dict) -> MarlAlgo:
        algo_cls = MARL_ALGORITHMS[config["algo_name"]]

        algo = algo_cls()
        algo.setup(config)

        return algo

    def train_and_evaluate(
        self,
        algos_config: List[Dict],
        all_test_envs: Dict[str, TransformedEnv] | None = None,
        n_checkpoints_train: int = 10,
        n_checkpoints_eval: int = 25,
        with_voronoi: bool = True,
        train_env_dict: Dict | None = None,
    ):
        if train_env_dict is not None:
            dict_path = self.root_dir / TRAIN_FOLDER / "env_dict.json"
            with open(dict_path, "w") as f:
                json.dump(train_env_dict, f, indent=4)

        for algo_config in algos_config:
            algo = self.get_algo(algo_config)

            train_csv_dir = self.root_dir / TRAIN_FOLDER / SCALARS_FOLDER
            policy_dir = self.root_dir / TRAIN_FOLDER / POLICIES_FOLDER

            checkpoints = algo.train_algo(
                n_checkpoints=n_checkpoints_train,
                train_csv_dir=train_csv_dir,
                policy_dir=policy_dir,
            )

        for algo_config in algos_config:
            algo = self.get_algo(algo_config)
            algo_name = algo.algo_name

            pbar = tqdm(
                all_test_envs.items(),
                total=len(list(all_test_envs.keys())),
                desc=f"evaluating {algo_name}...",
            )
            for test_env_name, test_env in pbar:

                for chkpt_n in checkpoints:

                    path_chkpt_n = (
                        self.root_dir
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

                    evaluate_and_record(
                        policy=new_policy,
                        env=test_env,
                        main_dir=self.root_dir / TEST_FOLDER / test_env_name,
                        filename=f"{algo_name}_chkpt_{chkpt_n}",
                        seed=self.seed,
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

                voronoi_based_policy = VoronoiBasedActor(test_env)

                evaluate_and_record(
                    policy=voronoi_based_policy,
                    env=test_env,
                    main_dir=self.root_dir / TEST_FOLDER / test_env_name,
                    filename=VORONOI_BASED_KEYWORD,
                    seed=self.seed,
                    with_video=True,
                    n_checkpoints_eval=n_checkpoints_eval,
                )

        return self.root_dir

    def plot_exp(self, exp_dir: Path):
        train_dir = exp_dir / TRAIN_FOLDER
        test_dir = exp_dir / TEST_FOLDER

        METRIC_FOR_BEST = "team_reward_iqm"

        # TRAIN
        list_csv_train = get_files_in_folder(train_dir / SCALARS_FOLDER, "csv")
        dir_save = train_dir / SCALARS_FOLDER / PLOTS_FOLDER
        data, metrics = self._sort_list_of_csv(list_csv_train)
        self._plot_results(metrics, data, "train", dir_save)

        # TEST
        test_folders = get_first_layer_folders(test_dir)
        # print("Test folders: ", test_folders)

        per_algo_rows = defaultdict(list)

        pbar = tqdm(test_folders)
        for test_dir in test_folders:
            pbar.set_description(desc=f"Plotting {str(test_dir)}...")
            # print(test_dir)

            test_res = test_dir / SCALARS_FOLDER
            list_all_csv_test = get_files_in_folder(test_res, "csv")
            groups_by_chkpt = group_by_checkpoints(list_all_csv_test)

            # For each algo we'll collect one 1-row DF per checkpoint
            _rows_per_algo = defaultdict(list)

            for chkpt, list_csv_test in groups_by_chkpt.items():
                test_on = str(test_res).split("/")[-2].replace("_", " ")
                title = "Test on " + test_on + " chkpt: " + str(chkpt)
                dir_save = test_res / PLOTS_FOLDER / f"checkpoint_{chkpt}"

                data, metrics = self._sort_list_of_csv(list_csv_test)
                self._plot_results(metrics, data, title, dir_save)

                for algo_name, df in data.items():
                    if algo_name == VORONOI_BASED_KEYWORD:
                        continue

                    mean_row = (
                        df.mean(axis=0, numeric_only=True).to_frame().T
                    )  # 1-row DF

                    # drop "step" if it sneaks in
                    if "step" in mean_row.columns:
                        mean_row = mean_row.drop(columns=["step"])

                    mean_row.insert(
                        0, "checkpoint", chkpt
                    )  # keep the checkpoint as a column
                    _rows_per_algo[algo_name].append(mean_row)

            # Build the final dict: {algo_name: DataFrame with n_rows = n_checkpoints}
            df_mean_test = {
                algo: pd.concat(rows, ignore_index=True)
                .sort_values("checkpoint")
                .reset_index(drop=True)
                for algo, rows in _rows_per_algo.items()
            }

            # Replace the above with your actual loop. Below is the aggregation step:
            for algo, df in df_mean_test.items():
                # Keep only checkpoint + the metric (drop NaNs)
                sub = df[["checkpoint", METRIC_FOR_BEST]].copy()
                sub = sub.replace([np.inf, -np.inf], np.nan).dropna(
                    subset=[METRIC_FOR_BEST]
                )

                per_algo_rows[algo].append(sub)

        # Now compute average metric per checkpoint across all test envs, pick best
        best_overall = {}  # algo -> dict with the best checkpoint info

        for algo, parts in per_algo_rows.items():
            if not parts:
                continue

            combined = pd.concat(parts, ignore_index=True)

            # Must have the metric column
            if METRIC_FOR_BEST not in combined.columns:
                print(f"[warn] {algo}: metric '{METRIC_FOR_BEST}' missing, skipping.")
                continue

            # Average over tests for each checkpoint
            avg_per_ckpt = (
                combined.groupby("checkpoint", as_index=False)[METRIC_FOR_BEST]
                .mean()
                .rename(columns={METRIC_FOR_BEST: f"avg_{METRIC_FOR_BEST}"})
            )

            if avg_per_ckpt.empty:
                print(f"[warn] {algo}: no checkpoints to average, skipping.")
                continue

            # Pick the best checkpoint (tie-breaker: smallest checkpoint id)
            best_row = avg_per_ckpt.sort_values(
                [f"avg_{METRIC_FOR_BEST}", "checkpoint"], ascending=[False, True]
            ).iloc[0]
            best_chkpt = int(best_row["checkpoint"])
            best_avg = float(best_row[f"avg_{METRIC_FOR_BEST}"])

            # Build paths (NO trailing comma!)
            src = train_dir / POLICIES_FOLDER / f"{algo}_chkpt_{best_chkpt}.pt"
            dst = train_dir / POLICIES_FOLDER / f"{algo}_best.pt"

            # Ensure folder exists
            dst.parent.mkdir(parents=True, exist_ok=True)

            if not src.exists():
                print(f"[warn] {algo}: source policy not found: {src}")
            else:
                shutil.copy(src, dst)

            best_overall[algo] = {
                "best_checkpoint": best_chkpt,
                "avg_metric": best_avg,
                "best_policy_path": dst,
            }

            print(
                f"{algo} - {METRIC_FOR_BEST}: {best_avg:.6f} - "
                f"Checkpoint: {best_chkpt} - Policy path: {dst}"
            )

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
                algo_name = MAPPO_KEYWORD
            elif "ippo" in name:
                algo_name = IPPO_KEYWORD
            elif "voronoi" in name:
                algo_name = VORONOI_BASED_KEYWORD
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
            "mappo": "orange",
            "ippo": "green",
            "voronoi": "blue",
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
