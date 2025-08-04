import os
from datetime import datetime
from pathlib import Path
from typing import Dict

import torch
from tensordict.nn import set_composite_lp_aggregate
from torchrl.envs import check_env_specs, RewardSum, TransformedEnv, VmasEnv
from tqdm import tqdm

from vmas import make_env
from vmas.scenarios.voronoi import VoronoiPolicy
from vmas.simulator.utils import save_video

from src import VIDEOS_FOLDER_NAME
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
    ):

        self._setup_folders(experiment_name)

        for envs_setup_name, envs_config in env_configs.items():

            envs_train = envs_config["envs_train"]
            envs_test = envs_config["envs_test"]

            for env_train_name, env_train_config in envs_train:
                for env_test_name, env_test_config in envs_test:

                    folder_exp = (
                        self.root_dir / envs_setup_name / env_train_name / env_test_name
                    )

                    env_train_torch_rl = self._setup_torch_rl_env(env_train_config)
                    env_test_torch_rl = self._setup_torch_rl_env(env_test_config)

                    for algo_name, algo_config in algo_configs.items():
                        marl_agent = self._get_marl_algo(
                            env_train_torch_rl, algo_config
                        )

                        marl_agent.train_and_evaluate(
                            env_train=env_train_torch_rl,
                            env_test=env_test_torch_rl,
                            main_dir=folder_exp,
                        )

                    self.use_voronoi_based_heuristic(
                        env_test_config,
                        main_dir=folder_exp,
                    )

        self.make_plots(self.root_dir)

    def use_voronoi_based_heuristic(self, env_config, main_dir):

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
            # Environment specific variables
            **env_kwargs,
        )

        policy = VoronoiPolicy(env=env, continuous_action=True)
        obs = torch.stack(env.reset(), dim=0)

        frame_list = []

        for t in tqdm(range(max_steps)):
            actions = [None] * n_agents

            for i in range(n_agents):
                actions[i] = policy.compute_action(
                    obs[i], u_range=env.agents[i].u_range
                )
            obs, rews, dones, info = env.step(actions)
            # rewards = torch.stack(rews, dim=1)
            # global_reward = rewards.mean(dim=1)
            # sum_reward = torch.sum(rewards)
            # mean_global_reward = global_reward.mean(dim=0)
            # print("Mean reward: ", mean_global_reward)
            frame_list.append(
                env.render(
                    mode="rgb_array",
                    agent_index_focus=None,
                    visualize_when_rgb=True,
                )
            )

        video_name = main_dir / VIDEOS_FOLDER_NAME / "voronoi_tesselation.mp4"
        os.makedirs(video_name.parent, exist_ok=True)
        save_video(video_name, frame_list, 1 / env.scenario.world.dt)

    def make_plots(self, directory: Path):
        """

        :param directory:
        :return: TRAIN: tables: rewards, eta, beta, #n_collisions over episodes only RL agents
                 TEST: episodic rewards, cumulative rewards, eta, beta, #n_collisions of RL agents and Voronoi based Heuristic on each evaluation checkpoint
        """

        pass
