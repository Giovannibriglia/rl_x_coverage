from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import List

import numpy as np


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


def save_csv(
    csv_path: Path,
    n_agents: int,
    checkpoints: List,
    rewards_np: np.ndarray,
    eta_np: np.ndarray,
    beta_np: np.ndarray,
    collisions_np: np.ndarray,
):
    assert (
        rewards_np.ndim == 3
    ), "rewards_np must be 3-dimensional, instead is {}".format(rewards_np.shape)
    assert eta_np.ndim == 3, "eta_np must be 3-dimensional, instead is {}".format(
        eta_np.shape
    )
    assert beta_np.ndim == 3, "beta_np must be 3-dimensional, instead is {}".format(
        beta_np.shape
    )
    assert (
        collisions_np.ndim == 3
    ), "collisions_np must be 3-dimensional, instead is {}".format(collisions_np.shape)

    assert (
        rewards_np.shape[2] == n_agents
    ), f"rewards_np shape 2 must be equal to n_agents ({n_agents}), instead is {rewards_np.shape[2]}"
    assert (
        eta_np.shape[2] == n_agents
    ), f"eta_np shape 2 must be equal to n_agents ({n_agents}), instead is {eta_np.shape[2]}"
    assert (
        beta_np.shape[2] == n_agents
    ), f"beta_np shape 2 must be equal to n_agents ({n_agents}), instead is {beta_np.shape[2]}"
    assert (
        collisions_np.shape[2] == n_agents
    ), f"collisions_np shape 2 must be equal to n_agents ({n_agents}), instead is {collisions_np.shape[2]}"

    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)

        # Header: agent-wise + team IQM/IQRStd for each metric
        header = ["step"]
        for i in range(n_agents):
            header += [f"agent{i}_reward_iqm", f"agent{i}_reward_iqrstd"]
        header += ["team_reward_iqm", "team_reward_iqrstd"]

        header += [f"agent{i}_eta" for i in range(n_agents)]
        header += ["eta_iqm", "eta_iqrstd"]

        header += [f"agent{i}_beta" for i in range(n_agents)]
        header += ["beta_iqm", "beta_iqrstd"]

        header += [f"agent{i}_collisions" for i in range(n_agents)]
        header += ["collisions_iqm", "collisions_iqrstd"]

        writer.writerow(header)

        for t in checkpoints:
            # checkpoint
            row = [t]
            idx_t = checkpoints.index(t)

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

            # eta's team
            team_beta_iqm, team_beta_iqrstd = _iqm_and_iqrstd_1d(
                np.mean(beta_np[:, idx_t, :], axis=1)
            )
            row += [team_beta_iqm, team_beta_iqrstd]

            # collision's agents
            for i in range(n_agents):
                coll_ag_iqm, coll_ag_iqrstd = _iqm_and_iqrstd_1d(
                    collisions_np[:, idx_t, i]
                )
                row += [coll_ag_iqm, coll_ag_iqrstd]

            # eta's team
            team_coll_iqm, team_coll_iqrstd = _iqm_and_iqrstd_1d(
                np.sum(collisions_np[:, idx_t, :], axis=1)
            )
            row += [team_coll_iqm, team_coll_iqrstd]

            writer.writerow(row)


def get_first_layer_folders(p: str | Path):
    if isinstance(p, str):
        return [name for name in os.listdir(p) if os.path.isdir(os.path.join(p, name))]
    elif isinstance(p, Path):
        return [x for x in p.iterdir() if x.is_dir()]
    else:
        raise TypeError("p must be either a Path or str")
