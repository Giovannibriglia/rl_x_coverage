import csv
import os
from datetime import datetime
from pathlib import Path
from typing import Dict

import torch

from build.lib.src import SCALARS_FOLDER_NAME
from tensordict.nn import set_composite_lp_aggregate
from torchrl.envs import check_env_specs, RewardSum, TransformedEnv, VmasEnv
from tqdm import tqdm
from vmas import make_env
from vmas.scenarios.voronoi import VoronoiPolicy
from vmas.simulator.utils import save_video

from src import TEST_KEYWORD, TRAIN_KEYWORD, VIDEOS_FOLDER_NAME
from src.ippo import MarlIPPO


class Simulation:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def _setup_folders(self, experiment_name: str = ""):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.root_dir = Path.cwd() / "runs" / f"{experiment_name}_{timestamp}"
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
            """elif algo_configs["algo_name"] == "QMIX":
            agent_cls = MarlQmix"""
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

                for algo_name, algo_config in algo_configs.items():
                    marl_agent = self._get_marl_algo(env_train_torch_rl, algo_config)

                    test_envs_torch_rl = {
                        env_test_name: self._setup_torch_rl_env(test_env_config)
                        for env_test_name, test_env_config in test_envs.items()
                    }

                    marl_agent.train_and_evaluate(
                        env_train=env_train_torch_rl,
                        envs_test=test_envs_torch_rl,
                        main_dir=folder_exp,
                        n_checkpoints=n_checkpoints,
                        n_checkpoints_metrics=n_checkpoints,
                    )

                for test_env_name, test_env_config in test_envs.items():

                    save_dir = folder_exp / test_env_name

                    self.use_voronoi_based_heuristic(
                        test_env_config,
                        main_dir=save_dir,
                        n_checkpoints=n_checkpoints,
                    )

        self.make_plots(self.root_dir)

    def use_voronoi_based_heuristic(
        self, env_config, main_dir, n_checkpoints: int = 100
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

        # Determine checkpoint intervals
        n_checkpoints = max(1, n_checkpoints)
        interval = max(max_steps // n_checkpoints, 1)

        # Prepare storage for logging
        records = []

        frame_list = []
        for t in tqdm(range(max_steps), desc="Voronoi-based heuristic evaluation..."):
            # compute actions for each agent
            actions = [
                policy.compute_action(obs[i], u_range=env.agents[i].u_range)
                for i in range(n_agents)
            ]

            # step the environment
            obs, rews, dones, info_list = env.step(actions)

            # Check if this iteration is a checkpoint
            if (t % interval == 0) or (t == max_steps - 1):
                # compute per-agent mean reward across parallel envs
                agent_means = [rew.mean().item() for rew in rews]
                # compute sum of all rewards across agents and envs
                all_rewards = torch.cat([rew.flatten() for rew in rews], dim=0)
                overall_mean = all_rewards.mean().item()
                # compute team mean (same as flatten mean but equivalent)
                team_mean = torch.stack(rews, dim=1).mean().item()

                # aggregate info across envs and agents
                eta_vals = torch.stack(
                    [info["eta"].mean(dim=0) for info in info_list]
                ).cpu()
                beta_vals = torch.stack(
                    [info["beta"].mean(dim=0) for info in info_list]
                ).cpu()
                coll_vals = torch.stack(
                    [info["n_collisions"].mean(dim=0) for info in info_list]
                ).cpu()

                # compute team means
                eta_mean = eta_vals.mean().item()
                beta_mean = beta_vals.mean().item()
                collisions_mean = coll_vals.mean().item()

                # log record for this checkpoint
                rec = {"iter": t}

                # per-agent rewards
                for i, m in enumerate(agent_means):
                    rec[f"agent_{i}_reward"] = m
                # team reward metrics
                rec["team_mean_reward"] = team_mean
                rec["overall_mean_reward"] = overall_mean

                # per-agent eta, beta, collisions
                for j, v in enumerate(eta_vals.tolist()):
                    rec[f"eta_agent_{j}"] = v
                for j, v in enumerate(beta_vals.tolist()):
                    rec[f"beta_agent_{j}"] = v
                for j, v in enumerate(coll_vals.tolist()):
                    rec[f"collisions_agent_{j}"] = v

                # team mean metrics
                rec["eta_mean"] = eta_mean
                rec["beta_mean"] = beta_mean
                rec["n_collisions_mean"] = collisions_mean

                records.append(rec)

            # render frames if needed
            frame_list.append(
                env.render(
                    mode="rgb_array",
                    agent_index_focus=None,
                    visualize_when_rgb=True,
                )
            )

        # save video
        video_path = main_dir / VIDEOS_FOLDER_NAME
        os.makedirs(video_path, exist_ok=True)
        video_name = f"{video_path}/voronoi_tesselation"
        save_video(video_name, frame_list, 1 / env.scenario.world.dt)

        # write records to CSV using csv module
        csv_path = main_dir / SCALARS_FOLDER_NAME / "voronoi_based.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        if records:
            with open(csv_path, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                header = list(records[0].keys())
                writer.writerow(header)
                for rec in records:
                    writer.writerow([rec[k] for k in header])

    def make_plots(self, directory: Path):
        """

        :param directory:
        :return: TRAIN: tables: rewards, eta, beta, #n_collisions over episodes only RL agents
                 TEST: episodic rewards, cumulative rewards, eta, beta, #n_collisions of RL agents and Voronoi based Heuristic on each evaluation checkpoint
        """

        pass
