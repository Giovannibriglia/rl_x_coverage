#  Copyright (c) 2022-2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
import time

from pathlib import Path
from typing import Type

import torch
from matplotlib import pyplot as plt

from tqdm import tqdm

from vmas import make_env
from vmas.simulator.heuristic_policy import BaseHeuristicPolicy, RandomPolicy
from vmas.simulator.utils import save_video

vmas_dir = Path(__file__).parent

import os
import sys

sys.path.append(os.path.dirname(vmas_dir))

# from algorithms.VoronoiCoverage import VoronoiCoverage

mydev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: ", mydev)


def run_heuristic(
    scenario_name: str,
    heuristic: Type[BaseHeuristicPolicy] = RandomPolicy,
    n_steps: int = 200,
    n_envs: int = 32,
    render: bool = False,
    save_render: bool = False,
    device: str = mydev,
    seed: int = 42,
    **env_kwargs,
):
    assert not (save_render and not render), "To save the video you have to render it"
    if env_kwargs is None:
        env_kwargs = {}

    # Scenario specific variables
    env = make_env(
        scenario=scenario_name,
        num_envs=n_envs,
        device=device,
        continuous_actions=True,
        wrapper=None,
        seed=seed,
        # Environment specific variables
        **env_kwargs,
    )
    if heuristic == VoronoiPolicy:
        policy = heuristic(env=env, continuous_action=True)
    else:
        policy = heuristic(continuous_action=True)

    frame_list = []  # For creating a gif
    init_time = time.time()
    step = 0
    total_reward = 0
    n_agents = len(env.agents)
    obs = torch.stack(env.reset(), dim=0)

    rewards_for_plot = torch.zeros((n_steps, n_envs, n_agents))
    global_rewards = []
    sum_rewards = []
    for timestep in tqdm(range(n_steps)):
        step += 1
        actions = [None] * n_agents
        # robots = [a.state.pos for a in env.agents]
        # robots = torch.stack(robots).to(device)
        # voro = VoronoiCoverage(robots, env.scenario.pdf, env.scenario.grid_spacing, env.scenario.xdim, env.scenario.ydim, env.scenario.world.device, centralized=True)
        # voro.partitioning()
        # t_act = time.time()
        for i in range(n_agents):
            actions[i] = policy.compute_action(obs[i], u_range=env.agents[i].u_range)
            # print(f"{i}", actions[i])
        # print("t action: ", time.time()-t_act)
        obs, rews, dones, info = env.step(actions)
        # print(rews)
        rewards = torch.stack(rews, dim=1)
        global_reward = rewards.mean(dim=1)
        sum_reward = torch.sum(rewards)
        mean_global_reward = global_reward.mean(dim=0)
        # print("Mean reward: ", mean_global_reward)
        global_rewards.append(mean_global_reward.cpu().numpy())
        sum_rewards.append(sum_reward.cpu().numpy())
        rewards_for_plot[timestep] = rewards

        total_reward += mean_global_reward
        if render:
            frame_list.append(
                env.render(
                    mode="rgb_array",
                    agent_index_focus=None,
                    visualize_when_rgb=True,
                )
            )

    env_n = 0
    plt.figure(dpi=500)
    for n in range(n_agents):
        plt.plot(
            rewards_for_plot[:, env_n, n].cpu().numpy(),
            label=f"ag{n}: {torch.mean(rewards_for_plot[:, env_n, n]):.3f}",
        )
    plt.plot(global_rewards, label="global")
    # plt.plot(sum_rewards, label="sum")
    plt.legend(loc="best")
    plt.show()

    total_time = time.time() - init_time
    if render and save_render:
        save_video(scenario_name, frame_list, 1 / env.scenario.world.dt)

    print(
        f"It took: {total_time}s for {n_steps} steps of {n_envs} parallel environments on device {device}\n"
        f"The average total reward was {total_reward}"
    )


if __name__ == "__main__":
    from vmas.scenarios.voronoi import VoronoiPolicy

    run_heuristic(
        scenario_name="voronoi",
        heuristic=VoronoiPolicy,
        n_envs=4,
        n_steps=100,
        render=True,
        save_render=True,
        centralized=False,  # mdp or pomdp in terms of pdf; but robots are seen only if within the agent's lidar range. #TODO; error in compute coverage function
        shared_rew=False,
        n_gaussians=1,
        n_rays=50,
        grid_spacing=0.2,
        lidar_range=0.5,
        n_agents=3,
        seed=2,
    )

# ok: [True, True], [


# 1) centralized: True, shared_rew: True
# 2) centralized: True, shared_rew: False
