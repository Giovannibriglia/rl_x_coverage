from __future__ import annotations

import csv
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Union

import numpy as np
import pandas as pd


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
    ckpt_re = re.compile(r"checkpoint_(\d+)")
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
