# evaluate_voronoi.py
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations

from typing import List, Tuple, Type

import cv2
import matplotlib.pyplot as plt
import numpy as np

import torch

from tensordict.nn import NormalParamExtractor, TensorDictModule
from torchrl.envs import check_env_specs, RewardSum, TransformedEnv
from torchrl.envs.libs.vmas import VmasEnv
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
    "n_rays": 25,
    "grid_spacing": 0.05,
    "lidar_range": 0.5,
    "centralized": False,
    "shared_rew": False,
}

CHKPT_PATH = "../../runs_2/icra26_b2/batch2/basic_3agents_3gauss/trained_policies/ippo_checkpoint_176.pt"  # ← your weights


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
        # categorical_actions=False,
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
def build_policy(env, n_agents: int):
    obs_dim = env.observation_spec["agents", "observation"].shape[-1]
    action_dim = env.action_spec.shape[-1]

    backbone = MultiAgentMLP(
        n_agent_inputs=obs_dim,
        n_agent_outputs=2 * action_dim,
        n_agents=n_agents,
        centralised=False,
        share_params=False,  # each agent head separate
        device=DEVICE,
        depth=2,
        num_cells=256,
        activation_class=torch.nn.Tanh,
    )

    policy_module = TensorDictModule(
        torch.nn.Sequential(backbone, NormalParamExtractor()),
        in_keys=[("agents", "observation")],
        out_keys=[("agents", "loc"), ("agents", "scale")],
    )

    actor = ProbabilisticActor(
        module=policy_module,
        spec=env.action_spec,
        in_keys=[("agents", "loc"), ("agents", "scale")],
        out_keys=[env.action_key],
        distribution_class=TanhNormal,
        distribution_kwargs={
            "low": env.full_action_spec[env.action_key].space.low,
            "high": env.full_action_spec[env.action_key].space.high,
        },
        return_log_prob=True,
    )
    return actor


# ──────────────────────────────────────────────────────────────────────
# State-dict loader with automatic matching
# ──────────────────────────────────────────────────────────────────────
def _load_matching(target_actor, source_state_dict):
    tgt_sd = target_actor.state_dict()
    matched = {}
    for k, v_src in source_state_dict.items():
        if k in tgt_sd and torch.is_tensor(v_src) and torch.is_tensor(tgt_sd[k]):
            if tgt_sd[k].shape == v_src.shape:
                matched[k] = v_src
            else:
                pass
    # merge and load non-strictly
    tgt_sd.update(matched)
    target_actor.load_state_dict(tgt_sd, strict=False)
    return target_actor


def load_trained_policy(env: TransformedEnv, ckpt_path: str) -> ProbabilisticActor:
    n_agents = env.n_agents
    policy = build_policy(env, n_agents)

    state = torch.load(ckpt_path, map_location=DEVICE)
    sd = (
        state["state_dict"]
        if isinstance(state, dict) and "state_dict" in state
        else state
    )

    policy = _load_matching(policy, sd)  # automatic check + matching
    policy.eval()
    return policy


# ──────────────────────────────────────────────────────────────────────
# Roll-out helpers
# ──────────────────────────────────────────────────────────────────────
###############################################################################
# Learned policy rollout
###############################################################################
def rollout_learned(
    env: TransformedEnv,
    policy,
    steps: int,
    video_path: str = None,
) -> Tuple[List[float], List[float]]:
    """
    Evaluate `policy` for `steps` steps in `env` (TorchRL ≥ 0.6).
    Optionally record a video of the rollout.
    """

    def save_video(frames: List[np.ndarray], filename: str, fps: int = 30):
        if not frames:
            print("[warning] no frames to write", filename)
            return
        h, w, _ = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        if ".mp4" not in filename:
            filename += ".mp4"
        vout = cv2.VideoWriter(filename, fourcc, fps, (w, h))
        for fr in frames:
            vout.write(fr)
        vout.release()

    rew_key = env.reward_key

    # ─── Video setup ──────────────────────────────────────────────
    frames = []
    callback = None
    if video_path is not None:
        callback = lambda e, td: frames.append(e.render(mode="rgb_array"))

    # ─── Rollout ──────────────────────────────────────────────────
    with torch.no_grad():
        td = env.rollout(
            max_steps=steps,
            policy=policy,
            callback=callback,
            break_when_any_done=False,
            auto_cast_to_device=True,
        )

    # ─── Rewards traces ───────────────────────────────────────────
    rewards = td.get(("next",) + rew_key, None)
    if rewards is None:
        rewards = td.get(rew_key, None)
    if rewards is None:
        raise RuntimeError("No rewards found in rollout")

    mean_trace = []
    std_trace = []

    for t in range(rewards.shape[1]):  # loop over time steps
        r_t = rewards[:, t, :]  # [n_envs, n_agents]
        r_tot = r_t.mean(dim=1).squeeze(-1)  # [n_envs]
        m, s = _iqm_and_iqrstd_1d(r_tot.cpu().numpy())
        mean_trace.append(float(m))
        std_trace.append(float(s))

    if video_path is not None:
        save_video(frames, video_path, fps=30)

    return mean_trace, std_trace


def _iqm_and_iqrstd_1d(x):
    """
    Return (IQM, IQRStd) for a 1D array with NaNs allowed.
    IQM = mean of values within [Q1, Q3]; IQRStd = (IQR of the middle values)/2.
    """
    assert x.ndim == 1, "array for iqm and iqrstd must be 1-dimensional"

    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return float("nan"), float("nan")
    q1, q3 = np.percentile(x, [25, 75])
    mid = x[(x >= q1) & (x <= q3)]
    if mid.size == 0:
        return float("nan"), float("nan")
    iqm = float(np.mean(mid))
    # IQR of the middle slice; divide by 2 as your convention
    iqr_std = float((np.percentile(mid, 75) - np.percentile(mid, 25)) / 2.0)

    return iqm, iqr_std


###############################################################################
# Hand-crafted / heuristic rollout
###############################################################################
def rollout_heuristic(
    env, heuristic_cls, steps: int, video_path: str = None
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

    frames_list = []

    for _ in tqdm(range(steps), desc="Heuristic"):
        acts = [
            pol.compute_action(obs[i], u_range=env.agents[i].u_range)
            for i in range(N_AGENTS)
        ]
        obs, rews, *_ = env.step(acts)  # `rews`: list[n_agents][n_envs]
        obs = torch.stack(obs, 0)

        # convert to tensor: [n_agents, n_envs]
        rews_t = torch.stack(rews, 0)

        per_env = rews_t.mean(dim=0)  # mean over agents [n_envs]
        m, s = _iqm_and_iqrstd_1d(per_env.cpu().numpy())
        mean_trace.append(float(m))
        std_trace.append(float(s))
        if video_path is not None:
            frames_list.append(
                env.render(
                    mode="rgb_array",
                    agent_index_focus=None,
                    visualize_when_rgb=True,
                )
            )

    if video_path is not None:
        save_video(video_path, frames_list, 1 / env.scenario.world.dt)

    return mean_trace, std_trace


# ──────────────────────────────────────────────────────────────────────
# Main comparison function
# ──────────────────────────────────────────────────────────────────────
def compare(
    heuristic: Type[BaseHeuristicPolicy],
    n_steps: int = 30,
    n_envs: int = 1,
    seed: int = 1,
    video_path: str = None,
):
    env_h = make_native_vmas_env(n_envs, seed, **ENV_KWARGS)
    env_p = make_torchrl_env(n_envs, seed, **ENV_KWARGS)
    # print("TorchRL-env action dim:", env_p.action_spec.shape[-1])  # expect 2

    FONTSIZE = 20

    policy = load_trained_policy(env_p, CHKPT_PATH)

    video_path_h = "video_heuristic"
    video_path_p = "video_ppo"

    mean_heur, std_heur = rollout_heuristic(env_h, heuristic, n_steps, video_path_h)
    mean_learned, std_learned = rollout_learned(env_p, policy, n_steps, video_path_p)

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


# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from vmas.scenarios.voronoi import VoronoiPolicy

    compare(VoronoiPolicy)
