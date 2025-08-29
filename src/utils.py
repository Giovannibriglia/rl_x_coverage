from __future__ import annotations

import csv
import os
import random
from datetime import datetime
from pathlib import Path
from string import Template
from typing import Any, Dict, List, Tuple, Union

import cv2
import numpy as np
import torch
import yaml
from tensordict.nn import set_composite_lp_aggregate
from torchrl.envs import check_env_specs, RewardSum, TransformedEnv, VmasEnv

from src import SCALARS_FOLDER, VIDEOS_FOLDER


def _deep_update(base: dict, override: dict) -> dict:
    """Recursively merge override into base (override wins)."""
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def _fill_placeholders(obj: Any, params: Dict[str, Any]) -> Any:
    """Recursively substitute ${var} in strings."""
    if isinstance(obj, dict):
        return {k: _fill_placeholders(v, params) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_fill_placeholders(v, params) for v in obj]
    if isinstance(obj, str):
        return Template(obj).safe_substitute(params)
    return obj


def _coerce_numbers(obj: Any) -> Any:
    """Try to convert numeric-looking strings to int/float."""
    if isinstance(obj, dict):
        return {k: _coerce_numbers(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_coerce_numbers(v) for v in obj]
    if isinstance(obj, str):
        s = obj.strip()
        try:
            if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
                return int(s)
            # float, including scientific notation
            return float(s)
        except ValueError:
            return obj
    return obj


def _load_yaml(path: Union[str, Path]) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def get_algo_dict(yaml_path: str, env_dict: Dict, env: TransformedEnv) -> Dict:
    algo_config = _load_yaml(yaml_path)

    frames_per_batch = env_dict["frames_per_batch"]
    algo_config["frames_per_batch"] = frames_per_batch

    n_iters = env_dict["max_steps"]
    algo_config["n_iters"] = n_iters

    n_agents = env_dict["n_agents"]
    algo_config["n_agents"] = n_agents

    algo_config["env"] = env

    return algo_config


def get_env_dict(
    kind: str,
    n_agents: int,
    n_gauss: int,
    max_steps_env: int = 500,
    n_envs: int = 10,
    seed: int = 42,
    cfg_dir: Union[str, Path] = "./config/envs",
) -> Tuple[Dict, str]:
    """
    Load base + kind YAML, fill placeholders, and return (env_dict, env_name).
    """
    cfg_dir = Path(cfg_dir)
    base_cfg = _load_yaml(cfg_dir / "base.yaml")
    kind_cfg_path = cfg_dir / f"{kind}.yaml"
    if not kind_cfg_path.exists():
        raise ValueError(f"unknown env kind or missing YAML: {kind_cfg_path}")

    override_cfg = _load_yaml(kind_cfg_path)

    merged = _deep_update(base_cfg, override_cfg)

    frames_per_batch = int(max_steps_env * n_envs)
    params = {
        "n_agents": n_agents,
        "n_gauss": n_gauss,
        "max_steps_env": max_steps_env,
        "seed": seed,
        "frames_per_batch": frames_per_batch,
    }

    filled = _fill_placeholders(merged, params)
    filled = _coerce_numbers(filled)

    env_name = f"{kind}_{n_agents}agents_{n_gauss}gauss"
    return filled, env_name


def get_torch_rl_env(
    env_config: Dict, device: str, fix_seed: bool = False
) -> TransformedEnv:

    frames_per_batch = env_config["frames_per_batch"]
    max_steps = env_config["max_steps"]
    scenario = env_config["scenario_name"]
    seed = env_config["seed"]
    n_agents = env_config["n_agents"]
    env_kwargs = env_config.get("env_kwargs", {})

    set_composite_lp_aggregate(False)
    num_envs = frames_per_batch // max_steps
    torch_rl_raw_env = VmasEnv(
        scenario=scenario,
        num_envs=num_envs,
        continuous_actions=True,
        max_steps=max_steps,
        device=device,
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

    if fix_seed:
        torch_rl_transformed_env.seed(seed)

    return torch_rl_transformed_env


def _iqm_and_iqrstd_1d(x):
    """
    Return (IQM, IQRStd) for a 1D array with NaNs allowed.
    IQM = mean of values within [Q1, Q3]; IQRStd = (IQR of the middle values)/2.
    """
    assert x.ndim == 1, "array for iqm and iqrstd must be 1-dimensional"
    if x.shape[0] > 4:
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
    else:
        return np.mean(x), np.std(x)


def save_csv(
    csv_path: Path,
    n_agents: int,
    checkpoints: List,
    rewards_np: np.ndarray,
    eta_np: np.ndarray,
    beta_np: np.ndarray,
    collisions_np: np.ndarray,
):
    # sanity checks
    assert rewards_np.ndim == 3, f"rewards_np must be 3D, got {rewards_np.shape}"
    assert eta_np.ndim == 3, f"eta_np must be 3D, got {eta_np.shape}"
    assert beta_np.ndim == 3, f"beta_np must be 3D, got {beta_np.shape}"
    assert (
        collisions_np.ndim == 3
    ), f"collisions_np must be 3D, got {collisions_np.shape}"

    for name, arr in [
        ("rewards", rewards_np),
        ("eta", eta_np),
        ("beta", beta_np),
        ("collisions", collisions_np),
    ]:
        assert (
            arr.shape[2] == n_agents
        ), f"{name}_np.shape[2]={arr.shape[2]} != n_agents={n_agents}"

    csv_path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = csv_path.exists()

    with csv_path.open("a", newline="") as f:
        writer = csv.writer(f)

        # Write header only if file did not exist
        if not file_exists:
            header = ["step"]
            # rewards
            for i in range(n_agents):
                header += [f"agent{i}_reward_iqm", f"agent{i}_reward_iqrstd"]
            header += ["team_reward_iqm", "team_reward_iqrstd"]
            # eta
            for i in range(n_agents):
                header += [f"agent{i}_eta_iqm", f"agent{i}_eta_iqrstd"]
            header += ["eta_iqm", "eta_iqrstd"]
            # beta
            for i in range(n_agents):
                header += [f"agent{i}_beta_iqm", f"agent{i}_beta_iqrstd"]
            header += ["beta_iqm", "beta_iqrstd"]
            # collisions
            for i in range(n_agents):
                header += [f"agent{i}_collisions_iqm", f"agent{i}_collisions_iqrstd"]
            header += ["collisions_iqm", "collisions_iqrstd"]

            writer.writerow(header)

        checkpoints = sorted(checkpoints)
        for idx_t, t in enumerate(checkpoints):
            row = [t]

            # agents' rewards
            for i in range(n_agents):
                rew_ag_iqm, rew_ag_iqrstd = _iqm_and_iqrstd_1d(rewards_np[:, idx_t, i])
                row += [rew_ag_iqm, rew_ag_iqrstd]

            # team's reward
            team_iqm, team_iqrstd = _iqm_and_iqrstd_1d(
                np.mean(rewards_np[:, idx_t, :], axis=1)
            )
            row += [team_iqm, team_iqrstd]

            # eta's agents
            for i in range(n_agents):
                eta_ag_iqm, eta_ag_iqrstd = _iqm_and_iqrstd_1d(eta_np[:, idx_t, i])
                row += [eta_ag_iqm, eta_ag_iqrstd]

            # eta's team
            team_eta_iqm, team_eta_iqrstd = _iqm_and_iqrstd_1d(
                np.mean(eta_np[:, idx_t, :], axis=1)
            )
            row += [team_eta_iqm, team_eta_iqrstd]

            # beta's agents
            for i in range(n_agents):
                beta_ag_iqm, beta_ag_iqrstd = _iqm_and_iqrstd_1d(beta_np[:, idx_t, i])
                row += [beta_ag_iqm, beta_ag_iqrstd]

            # beta's team
            team_beta_iqm, team_beta_iqrstd = _iqm_and_iqrstd_1d(
                np.mean(beta_np[:, idx_t, :], axis=1)
            )
            row += [team_beta_iqm, team_beta_iqrstd]

            # collisions' agents
            for i in range(n_agents):
                coll_ag_iqm, coll_ag_iqrstd = _iqm_and_iqrstd_1d(
                    collisions_np[:, idx_t, i]
                )
                row += [coll_ag_iqm, coll_ag_iqrstd]

            # collisions' team
            team_coll_iqm, team_coll_iqrstd = _iqm_and_iqrstd_1d(
                np.sum(collisions_np[:, idx_t, :], axis=1)
            )
            row += [team_coll_iqm, team_coll_iqrstd]

            writer.writerow(row)


def seed_everything(seed: int):
    r"""Sets the seed for generating random numbers in :pytorch:`PyTorch`,
    :obj:`numpy` and :python:`Python`.

    Args:
        seed (int): The desired seed.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def setup_folders(experiment_folder: str = "test", experiment_name: str = ""):

    root_dir = Path(f"runs/{experiment_folder}")
    os.makedirs(root_dir, exist_ok=True)

    if experiment_name != "":
        root_dir = root_dir / experiment_name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        root_dir = root_dir / f"{experiment_folder}test_{timestamp}"

    root_dir = Path(root_dir)
    os.makedirs(root_dir, exist_ok=True)

    print("\nExperiment root:", root_dir)
    return root_dir


def rollout_eval(
    policy,
    env: TransformedEnv,
    seed: int,
    main_dir: str | Path,
    filename: str,
    n_checkpoints_eval: int = 50,
    with_video: bool = False,
):

    def save_video(frames: List[torch.Tensor], filename: Path, fps: int = 30):
        if not frames:
            print("[warning] no frames to write", filename)
            return
        h, w, _ = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vout = cv2.VideoWriter(str(filename), fourcc, fps, (w, h))
        for fr in frames:
            vout.write(fr)
        vout.release()

    max_steps_evaluation = env.max_steps
    n_agents = env.n_agents

    seed_everything(seed)
    env.seed(seed)

    # prepare CSV
    csv_path = Path(main_dir) / SCALARS_FOLDER / f"{filename}_scalars.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure at least 2 checkpoints
    assert n_checkpoints_eval >= 2, "Need at least 2 checkpoints (first and last)."

    if n_checkpoints_eval >= max_steps_evaluation:
        checkpoints = range(max_steps_evaluation)
    else:
        checkpoints = [
            int(round(i * (max_steps_evaluation - 1) / (n_checkpoints_eval - 1)))
            for i in range(n_checkpoints_eval)
        ]

    if with_video:
        frames = []
        callback = lambda e, td: frames.append(e.render(mode="rgb_array"))
    else:
        callback = None

    policy.eval()  # disable dropout/BN etc.
    # rollout
    with torch.no_grad():
        td = env.rollout(
            max_steps=max_steps_evaluation,
            policy=policy,
            callback=callback,
            break_when_any_done=False,
            auto_cast_to_device=True,
        )

    # save video
    if with_video:
        video_file = Path(main_dir) / VIDEOS_FOLDER / f"{filename}_video.mp4"
        video_file.parent.mkdir(parents=True, exist_ok=True)
        save_video(frames, video_file, 1 / env.scenario.world.dt)

    # rewards [T, E, n_agents]
    rewards = td.get(("next",) + env.reward_key, None)
    if rewards is None:
        rewards = td.get(env.reward_key, None)
    if rewards is None:
        print("[warning] reward not found in rollout – skipping CSV")
        return 0.0

    rewards = rewards.squeeze(-1)
    # metrics from infos (shape [T, E, n_agents] or [T, E])
    eta = td.get(("next", "agents", "info", "eta"), None).squeeze(-1)
    beta = td.get(("next", "agents", "info", "beta"), None).squeeze(-1)
    collisions = td.get(("next", "agents", "info", "n_collisions"), None).squeeze(-1)

    """print("\neta: ", eta.shape)
    print("beta: ", beta.shape)
    print("collisions: ", collisions.shape)
    print("rewards: ", rewards.shape)"""

    eta_np = eta.cpu().numpy()  # shape: [n_envs, max_steps_evaluation, n_agents]
    beta_np = beta.cpu().numpy()  # shape: [n_envs, max_steps_evaluation, n_agents]
    collisions_np = (
        collisions.cpu().numpy()
    )  # shape: [n_envs, max_steps_evaluation, n_agents]
    rewards_np = (
        rewards.cpu().numpy()
    )  # shape: [n_envs, max_steps_evaluation, n_agents]

    save_csv(
        csv_path, n_agents, checkpoints, rewards_np, eta_np, beta_np, collisions_np
    )
