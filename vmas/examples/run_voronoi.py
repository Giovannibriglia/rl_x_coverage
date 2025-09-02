import time

from pathlib import Path
from typing import Type

import numpy as np
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


def run_heuristic(
    scenario_name: str,
    heuristic: Type[BaseHeuristicPolicy] = RandomPolicy,
    n_steps: int = 200,
    n_envs: int = 32,
    render: bool = False,
    save_render: bool = False,
    device: str = "cpu",
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

    env.seed(seed)

    if heuristic == VoronoiPolicy:
        policy = heuristic(env=env, continuous_action=True)
    else:
        policy = heuristic(continuous_action=True)

    frame_list = []  # For creating a gif
    init_time = time.time()
    step = 0
    n_agents = len(env.agents)
    obs = torch.stack(env.reset(), dim=0)

    rewards_for_plot = torch.zeros((n_steps, n_envs, n_agents))

    for timestep in tqdm(range(n_steps)):
        step += 1
        actions = [None] * n_agents
        for i in range(n_agents):
            actions[i] = policy.compute_action(obs[i], u_range=env.agents[i].u_range)
            # print(f"{i}", actions[i])
        # print("t action: ", time.time()-t_act)
        obs, rews, dones, info = env.step(actions)
        # print(rews)
        for n in range(n_agents):
            if not torch.equal(
                info[n]["n_collisions"], torch.zeros(n_envs, device=device)
            ):
                print(f"Agent {n}:", info[n]["n_collisions"])
        rewards = torch.stack(rews, dim=1)
        rewards_for_plot[timestep] = rewards

        if render:
            frame_list.append(
                env.render(
                    mode="rgb_array",
                    agent_index_focus=None,
                    visualize_when_rgb=True,
                )
            )

    r = rewards_for_plot.cpu().numpy()  # shape [T, E, A] = [steps, envs, agents]
    T = r.shape[0]
    xs = np.arange(T)

    plt.figure(dpi=500)

    # Per-agent mean ± std across envs
    for n in range(n_agents):
        m = r[:, :, n].mean(axis=1)  # [T]
        s = r[:, :, n].std(axis=1, ddof=0)  # [T]
        plt.plot(xs, m, label=f"ag{n}: {m.mean():.3f} ± {s.mean():.3f}", alpha=0.8)
        plt.fill_between(xs, m - s, m + s, alpha=0.15)

    # Global mean ± std across envs and agents
    m_g = r.mean(axis=(1, 2))  # [T]
    s_g = r.std(axis=(1, 2), ddof=0)  # [T]
    plt.plot(
        xs,
        m_g,
        linestyle="--",
        linewidth=2,
        label=f"global: {m_g.mean():.3f} ± {s_g.mean():.3f}",
        alpha=0.8,
    )
    plt.fill_between(xs, m_g - s_g, m_g + s_g, alpha=0.15)

    plt.xlabel("timestep")
    plt.ylabel("reward")
    plt.title("Per-agent and global reward (mean ± std across envs)")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig("voronoi_example_plot.png")
    plt.show()

    total_time = time.time() - init_time
    if render and save_render:
        save_video(scenario_name, frame_list, 1 / env.scenario.world.dt)

    print(
        f"It took: {total_time}s for {n_steps} steps of {n_envs} parallel environments on device {device}\n"
    )


if __name__ == "__main__":
    from vmas.scenarios.coverage import VoronoiPolicy

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device: ", device)

    run_heuristic(
        scenario_name="coverage",
        heuristic=VoronoiPolicy,
        n_envs=8,
        device=device,
        n_steps=100,
        render=True,
        save_render=True,
        centralized=False,  # mdp or pomdp in terms of pdf; but robots are seen only if within the agent's lidar range.
        n_gaussians=3,
        n_rays=50,
        grid_spacing=0.05,
        lidar_range=0.5,
        n_agents=3,
        seed=1,
    )
