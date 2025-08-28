from __future__ import annotations

import csv
import os
import random
import re
from pathlib import Path
from string import Template
from typing import Any, Dict, Iterable, List, Tuple, Union

import cv2
import numpy as np
import pandas as pd
import torch
import yaml
from tensordict.nn import set_composite_lp_aggregate
from torchrl.envs import check_env_specs, RewardSum, TransformedEnv, VmasEnv

from src import SCALARS_FOLDER, VIDEOS_FOLDER


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


def get_torch_rl_env(env_config: Dict, device: str) -> TransformedEnv:

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


def evaluate_and_record(
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


def get_first_layer_folders(p: str | Path):
    if isinstance(p, str):
        return [name for name in os.listdir(p) if os.path.isdir(os.path.join(p, name))]
    elif isinstance(p, Path):
        return [x for x in p.iterdir() if x.is_dir()]
    else:
        raise TypeError("p must be either a Path or str")


def get_files_in_folder(folder: Union[str, Path], extension: str = None) -> List[Path]:
    """
    Get all files in the first layer of a folder, optionally filtered by extension.

    Args:
        folder (str | Path): Path to the folder.
        extension (str, optional): File extension (e.g., 'csv', 'json').
                                   If None, returns all files.

    Returns:
        List[Path]: List of file paths.
    """
    folder = Path(folder)  # ensure Path object
    if not folder.is_dir():
        raise ValueError(f"{folder} is not a valid directory")

    # Normalize extension (with or without dot)
    if extension:
        extension = extension.lower().lstrip(".")

    files = [
        f
        for f in folder.iterdir()
        if f.is_file()
        and (extension is None or f.suffix.lower().lstrip(".") == extension)
    ]

    return files


def group_by_checkpoints(
    paths: Iterable[Union[str, Path]],
    baseline_pred=lambda name: "voronoi_based" in name.lower(),
) -> Dict[int, List[Path]]:
    """
    Group paths by checkpoint number (from filenames like '*_checkpoint_13_*').
    Include any 'baseline' files (default: names containing 'voronoi') in every group.

    Args:
        paths: iterable of str/Path.
        baseline_pred: function(name:str)->bool to mark files replicated into each group.

    Returns:
        Dict: { checkpoint_id (int) : [list of Path files for that checkpoint + baselines] }
    """
    paths = [Path(p) for p in paths]
    ckpt_re = re.compile(r"chkpt_(\d+)")
    by_ckpt: Dict[int, List[Path]] = {}
    baselines: List[Path] = []

    # Separate baselines and checkpointed files
    for p in paths:
        name = p.name
        if baseline_pred(name):
            baselines.append(p)
            continue
        m = ckpt_re.search(name)
        if m:
            ck = int(m.group(1))
            by_ckpt.setdefault(ck, []).append(p)
        # else: ignore files that are neither baseline nor checkpointed

    # Add baselines to every checkpoint group (deduplicated)
    for ck, files in by_ckpt.items():
        # keep order: existing files first, then baselines not already there
        existing = set(map(lambda x: x.resolve(), files))
        for b in baselines:
            if b.resolve() not in existing:
                files.append(b)

    return by_ckpt


def read_csv_strict(csv_path: Union[str, Path]) -> pd.DataFrame:
    """
    Read a CSV even if some rows have extra/missing fields:
    - Parse the header with csv.reader (handles quotes correctly)
    - Force pandas to use exactly that set of columns
    - Skip/ignore extra columns per row to avoid header/data length mismatch
    """
    csv_path = Path(csv_path)

    # 1) Read header safely with csv.reader (respects quotechar, commas inside quotes)
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)  # first non-empty line is assumed header

    # 2) Normalize header names (strip spaces)
    header = [h.strip() for h in header if h is not None]

    # 3) Read the rest with pandas, enforcing this schema
    #    - header=None + skiprows=1 to prevent pandas from re-parsing the header
    #    - names=header to fix the number of columns
    #    - usecols=range(len(header)) to drop any extra trailing fields per row
    #    - engine='python' for more permissive parsing
    df = pd.read_csv(
        csv_path,
        header=None,
        names=header,
        skiprows=1,
        usecols=range(len(header)),
        engine="python",
        index_col=False,
        on_bad_lines="warn",  # or 'skip' to silently drop malformed lines
    )

    # 4) Drop accidental index columns if present
    for junk in ("Unnamed: 0", "index"):
        if junk in df.columns:
            df = df.drop(columns=[junk])

    return df
