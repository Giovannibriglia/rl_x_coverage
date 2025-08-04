# evaluate_voronoi.py
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations

from typing import List, Tuple, Type

import matplotlib.pyplot as plt
import numpy as np

import torch

from tensordict import TensorDict
from tensordict.nn import NormalParamExtractor, TensorDictModule
from torchrl.envs import check_env_specs, RewardSum, TransformedEnv
from torchrl.envs.libs.vmas import VmasEnv  # torchrl ≥ 0.4
from torchrl.modules import MultiAgentMLP, ProbabilisticActor, TanhNormal
from tqdm import tqdm

from vmas import make_env as make_native_env
from vmas.simulator.heuristic_policy import BaseHeuristicPolicy
from vmas.simulator.utils import save_video

# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────
SCENARIO = "voronoi"
N_AGENTS = 3
N_GAUSSIANS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ENV_KWARGS = {
    "n_agents": N_AGENTS,
    "n_gaussians": N_GAUSSIANS,
    "n_rays": 50,
    "grid_spacing": 0.05,
    "lidar_range": 0.5,
    "centralized": False,
    "shared_rew": False,
}

CHKPT_PATH = f"{N_AGENTS}agents_{3}goals/policy.pt"  # ← your weights


# ──────────────────────────────────────────────────────────────────────
# Env factories
# ──────────────────────────────────────────────────────────────────────
def make_torchrl_env(num_envs: int, seed: int, **kwargs) -> TransformedEnv:
    """TorchRL wrapper – provides observation_spec and action_spec."""
    raw = VmasEnv(
        scenario=SCENARIO,
        num_envs=num_envs,
        device=DEVICE,
        continuous_actions=True,
        categorical_actions=False,
        seed=seed,
        **kwargs,
    )
    new_env = TransformedEnv(
        raw,
        RewardSum(in_keys=[raw.reward_key], out_keys=[("agents", "episode_reward")]),
    )
    check_env_specs(new_env)
    return new_env


def make_native_vmas_env(num_envs: int, seed: int, **kwargs):
    """Native VMAS env (used by heuristic policies)."""
    return make_native_env(
        scenario=SCENARIO,
        num_envs=num_envs,
        device=DEVICE,
        continuous_actions=True,
        wrapper=None,
        seed=seed,
        **kwargs,
    )


# ──────────────────────────────────────────────────────────────────────
# Network builder / loader
# ──────────────────────────────────────────────────────────────────────
def build_policy(env: TransformedEnv):
    obs_dim = env.observation_spec["agents", "observation"].shape[-1]
    act_dim = env.action_spec.shape[-1]

    policy_backbone = MultiAgentMLP(
        n_agent_inputs=obs_dim,
        n_agent_outputs=2 * act_dim,
        n_agents=N_AGENTS,
        centralized=False,
        share_params=False,
        device=DEVICE,
        depth=2,
        num_cells=256,
        activation_class=torch.nn.Tanh,
    )

    policy_module = TensorDictModule(
        torch.nn.Sequential(policy_backbone, NormalParamExtractor()),
        in_keys=[("agents", "observation")],
        out_keys=[("agents", "loc"), ("agents", "scale")],
    )

    policy = ProbabilisticActor(
        module=policy_module,
        spec=env.action_spec,  # env.action_spec_unbatched,
        in_keys=[("agents", "loc"), ("agents", "scale")],
        out_keys=[env.action_key],
        distribution_class=TanhNormal,
        distribution_kwargs={
            "low": env.full_action_spec[env.action_key].space.low,
            "high": env.full_action_spec[env.action_key].space.high,
        },
        # distribution_kwargs={
        #    "low": env.full_action_spec_unbatched[env.action_key].space.low,
        #    "high": env.full_action_spec_unbatched[env.action_key].space.high,
        # },
        return_log_prob=True,
    )

    return policy


def load_trained_policy(env: TransformedEnv, ckpt_path: str) -> ProbabilisticActor:
    policy = build_policy(env)
    state = torch.load(ckpt_path, map_location=DEVICE)
    sd = (
        state["state_dict"]
        if isinstance(state, dict) and "state_dict" in state
        else state
    )
    policy.load_state_dict(sd, strict=False)
    policy.eval()
    return policy


# ──────────────────────────────────────────────────────────────────────
# Roll-out helpers
# ──────────────────────────────────────────────────────────────────────
###############################################################################
# Learned policy rollout
###############################################################################
@torch.no_grad()
def rollout_learned(
    env: TransformedEnv,
    policy,
    steps: int,
) -> Tuple[List[float], List[float]]:
    """
    Evaluate `policy` for `steps` steps in `env` (TorchRL ≥ 0.6).

    Returns
    -------
    Tuple[List[float], List[float]]
        Two traces of length `steps`:
        • mean reward per step (first taking the mean across agents)
        • std-dev of that quantity across parallel envs
    """
    td = env.reset()
    act_key = env.action_key  # e.g. ('agents', 'action')
    rew_key = env.reward_key  # e.g. ('agents', 'episode_reward')
    batch_sz = env.batch_size  # torch.Size([num_envs])

    mean_trace: List[float] = []
    std_trace: List[float] = []

    for _ in tqdm(range(steps), desc="Learned"):
        # 1. policy → action ­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­
        td = policy(td)  # appends td[act_key]
        td_in = TensorDict(
            {act_key: td[act_key].to(env.device)},
            batch_size=batch_sz,
            device=env.device,
        )

        # 2. env step ­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­
        td_out = env.step(td_in)
        nxt = td_out.get("next", td_out)  # next-state container

        # 3. rewards → per-env mean, then global mean & std ­­­­­­­­­­­­­­
        rews = nxt.get(rew_key, None)
        if rews is None:  # fallback to raw env reward
            rews = nxt[("agents", "reward")]  # shape ≈ [n_envs, n_agents] *

        # * TorchRL keeps the batch (env) dimension first by default; if your
        #   transform/collector puts agents first, swap the axes:
        if rews.shape[0] == env.batch_size.numel():  # [n_envs, n_agents]
            per_env = rews.mean(dim=1)  # mean over agents
        else:  # [n_agents, n_envs]
            per_env = rews.mean(dim=0)

        step_mean = per_env.mean()  # scalar
        step_std = per_env.std(unbiased=False)  # sample-size-independent

        mean_trace.append(float(step_mean))
        std_trace.append(float(step_std))

        # 4. build next observation for policy ­­­­­­­­­­­­­­­­­­­­­­­­­­­­
        td = TensorDict(
            {("agents", "observation"): nxt[("agents", "observation")]},
            batch_size=batch_sz,
            device=env.device,
        )

    return mean_trace, std_trace


###############################################################################
# Hand-crafted / heuristic rollout
###############################################################################
def rollout_heuristic(
    env, heuristic_cls, steps: int
) -> Tuple[List[float], List[float]]:
    """
    Same output format as `rollout_learned`.
    """
    if heuristic_cls.__name__ == "VoronoiPolicy":
        pol = heuristic_cls(env=env, continuous_action=True)
    else:
        pol = heuristic_cls(continuous_action=True)

    obs = torch.stack(env.reset(), 0)  # [n_agents, n_envs, …]
    mean_trace, std_trace = [], []

    for _ in tqdm(range(steps), desc="Heuristic"):
        acts = [
            pol.compute_action(obs[i], u_range=env.agents[i].u_range)
            for i in range(N_AGENTS)
        ]
        obs, rews, *_ = env.step(acts)  # `rews`: list[n_agents][n_envs]
        obs = torch.stack(obs, 0)

        # convert to tensor: [n_agents, n_envs]
        rews_t = torch.stack(rews, 0)

        per_env = rews_t.mean(dim=0)  # mean over agents
        mean_trace.append(float(per_env.mean()))
        std_trace.append(float(per_env.std(unbiased=False)))

    return mean_trace, std_trace


# ──────────────────────────────────────────────────────────────────────
# Main comparison function
# ──────────────────────────────────────────────────────────────────────
def compare(
    heuristic: Type[BaseHeuristicPolicy],
    n_steps: int = 300,
    n_envs: int = 10,
    seed: int = 2,
    render: bool = False,
    save_vid: bool = True,
):
    env_h = make_native_vmas_env(n_envs, seed, **ENV_KWARGS)
    env_p = make_torchrl_env(n_envs, seed, **ENV_KWARGS)
    # print("TorchRL-env action dim:", env_p.action_spec.shape[-1])  # expect 2

    FONTSIZE = 20

    policy = load_trained_policy(env_p, CHKPT_PATH)

    mean_heur, std_heur = rollout_heuristic(env_h, heuristic, n_steps)
    mean_learned, std_learned = rollout_learned(env_p, policy, n_steps)

    # 1. cast to NumPy (or torch.Tensor)
    mh = np.asarray(mean_heur)  # shape [n_steps]
    sh = np.asarray(std_heur)
    ml = np.asarray(mean_learned)
    sl = np.asarray(std_learned)

    # 2. x-axis
    x = np.arange(len(mh))

    # 3. plot
    plt.figure(dpi=500)
    plt.plot(x, mh, label="Heuristic", lw=2, c="orange")
    plt.fill_between(x, mh - sh, mh + sh, alpha=0.2, color="orange")

    plt.plot(x, ml, label="PPO", lw=2, c="blue")
    plt.fill_between(x, ml - sl, ml + sl, alpha=0.2, color="blue")

    plt.xlabel("step", fontsize=FONTSIZE)
    plt.ylabel("mean global reward", fontsize=FONTSIZE)
    plt.xticks(fontsize=FONTSIZE - 4)
    plt.yticks(fontsize=FONTSIZE - 4)
    plt.title(f"{N_AGENTS} agents - {N_GAUSSIANS} gaussians", fontsize=FONTSIZE)
    plt.legend(loc="best", fontsize=FONTSIZE - 4)
    plt.tight_layout()
    plt.show()

    if render:
        frames = env_p.render(mode="gif", n_steps=n_steps)
        if save_vid:
            save_video(f"{SCENARIO}_learned", frames, 1 / env_p.scenario.world.dt)


# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from vmas.scenarios.voronoi import VoronoiPolicy

    compare(VoronoiPolicy)
