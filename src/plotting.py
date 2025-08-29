from __future__ import annotations

import csv
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Union

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

from src import (
    IPPO_KEYWORD,
    MAPPO_KEYWORD,
    PLOTS_FOLDER,
    POLICIES_FOLDER,
    SCALARS_FOLDER,
    TEST_FOLDER,
    TRAIN_FOLDER,
    VORONOI_BASED_KEYWORD,
)


def _get_files_in_folder(folder: Union[str, Path], extension: str = None) -> List[Path]:
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


def _get_first_layer_folders(p: str | Path):
    if isinstance(p, str):
        return [name for name in os.listdir(p) if os.path.isdir(os.path.join(p, name))]
    elif isinstance(p, Path):
        return [x for x in p.iterdir() if x.is_dir()]
    else:
        raise TypeError("p must be either a Path or str")


def _group_by_checkpoints(
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


def plot_exp(exp_dir: Path):
    train_dir = exp_dir / TRAIN_FOLDER
    test_dir = exp_dir / TEST_FOLDER

    METRIC_FOR_BEST = "team_reward_iqm"

    # TRAIN
    list_csv_train = _get_files_in_folder(train_dir / SCALARS_FOLDER, "csv")
    dir_save = train_dir / SCALARS_FOLDER / PLOTS_FOLDER
    data, metrics = _sort_list_of_csv(list_csv_train)
    _plot_results(metrics, data, "train", dir_save)

    # TEST
    test_folders = _get_first_layer_folders(test_dir)
    # print("Test folders: ", test_folders)

    per_algo_rows = defaultdict(list)

    pbar = tqdm(test_folders)
    for test_dir in test_folders:
        pbar.set_description(desc=f"Plotting {str(test_dir)}...")
        # print(test_dir)

        test_res = test_dir / SCALARS_FOLDER
        list_all_csv_test = _get_files_in_folder(test_res, "csv")
        groups_by_chkpt = _group_by_checkpoints(list_all_csv_test)

        # For each algo we'll collect one 1-row DF per checkpoint
        _rows_per_algo = defaultdict(list)

        for chkpt, list_csv_test in groups_by_chkpt.items():
            test_on = str(test_res).split("/")[-2].replace("_", " ")
            title = "Test on " + test_on + " chkpt: " + str(chkpt)
            dir_save = test_res / PLOTS_FOLDER / f"checkpoint_{chkpt}"

            data, metrics = _sort_list_of_csv(list_csv_test)
            _plot_results(metrics, data, title, dir_save)

            for algo_name, df in data.items():
                if algo_name == VORONOI_BASED_KEYWORD:
                    continue

                mean_row = df.mean(axis=0, numeric_only=True).to_frame().T  # 1-row DF

                # drop "step" if it sneaks in
                if "step" in mean_row.columns:
                    mean_row = mean_row.drop(columns=["step"])

                mean_row.insert(
                    0, "checkpoint", chkpt
                )  # keep the checkpoint as a column
                _rows_per_algo[algo_name].append(mean_row)

        # Build the final dict: {algo_name: DataFrame with n_rows = n_checkpoints}
        df_mean_test = {
            algo: pd.concat(rows, ignore_index=True)
            .sort_values("checkpoint")
            .reset_index(drop=True)
            for algo, rows in _rows_per_algo.items()
        }

        # Replace the above with your actual loop. Below is the aggregation step:
        for algo, df in df_mean_test.items():
            # Keep only checkpoint + the metric (drop NaNs)
            sub = df[["checkpoint", METRIC_FOR_BEST]].copy()
            sub = sub.replace([np.inf, -np.inf], np.nan).dropna(
                subset=[METRIC_FOR_BEST]
            )

            per_algo_rows[algo].append(sub)

    # Now compute average metric per checkpoint across all test envs, pick best
    best_overall = {}  # algo -> dict with the best checkpoint info

    for algo, parts in per_algo_rows.items():
        if not parts:
            continue

        combined = pd.concat(parts, ignore_index=True)

        # Must have the metric column
        if METRIC_FOR_BEST not in combined.columns:
            print(f"\n[warn] {algo}: metric '{METRIC_FOR_BEST}' missing, skipping.")
            continue

        # Average over tests for each checkpoint
        avg_per_ckpt = (
            combined.groupby("checkpoint", as_index=False)[METRIC_FOR_BEST]
            .mean()
            .rename(columns={METRIC_FOR_BEST: f"avg_{METRIC_FOR_BEST}"})
        )

        if avg_per_ckpt.empty:
            print(f"\n[warn] {algo}: no checkpoints to average, skipping.")
            continue

        # Pick the best checkpoint (tie-breaker: smallest checkpoint id)
        best_row = avg_per_ckpt.sort_values(
            [f"avg_{METRIC_FOR_BEST}", "checkpoint"], ascending=[False, True]
        ).iloc[0]
        best_chkpt = int(best_row["checkpoint"])
        best_avg = float(best_row[f"avg_{METRIC_FOR_BEST}"])

        # Build paths (NO trailing comma!)
        src = train_dir / POLICIES_FOLDER / f"{algo}_chkpt_{best_chkpt}.pt"
        dst = train_dir / POLICIES_FOLDER / f"{algo}_best.pt"

        # Ensure folder exists
        dst.parent.mkdir(parents=True, exist_ok=True)

        if not src.exists():
            print(f"[warn] {algo}: source policy not found: {src}")
        else:
            shutil.copy(src, dst)

        best_overall[algo] = {
            "best_checkpoint": best_chkpt,
            "avg_metric": best_avg,
            "best_policy_path": dst,
        }

        print(
            f"\n{algo} - {METRIC_FOR_BEST}: {best_avg:.6f} - checkpoint: {best_chkpt} - policy path: {dst}"
        )


def _read_csv_strict(csv_path: Union[str, Path]) -> pd.DataFrame:
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


def _sort_list_of_csv(list_of_csv_path: List):
    """
    Plot mean ± std curves for each metric across multiple CSVs.
    'step' is always used as the x-axis, not plotted as a metric.

    Args:
        list_of_csv_path: list of CSV paths (str or Path).

    Returns:
        Dict mapping <metric_name> -> saved image Path.
    """
    csv_paths = [Path(p) for p in list_of_csv_path]

    # Load CSVs
    data = {}
    for csv_path in csv_paths:
        df = _read_csv_strict(csv_path)
        df.columns = [c.strip() for c in df.columns]
        if "step" not in df.columns:
            raise ValueError(f"'step' column not found in {csv_path}")

        name = csv_path.stem.lower()
        if "mappo" in name:
            algo_name = MAPPO_KEYWORD
        elif "ippo" in name:
            algo_name = IPPO_KEYWORD
        elif "voronoi" in name:
            algo_name = VORONOI_BASED_KEYWORD
        else:
            algo_name = csv_path.stem  # fallback

        data[algo_name] = df

    # Find all metrics (exclude "step")
    metrics = set()
    for df in data.values():
        for col in df.columns:
            if col.endswith("_iqm"):
                base = col[:-4]
                if f"{base}_iqrstd" in df.columns and base != "step":
                    metrics.add(base)

    if not metrics:
        raise ValueError("No valid metrics with *_iqm and *_iqrstd found.")

    return data, list(metrics)


def _plot_results(
    metrics: List,
    data: Dict[str, pd.DataFrame],
    title: str,
    dir_save: Path,
    img_format: str = "pdf",
):
    algo_colors = {
        "mappo": "orange",
        "ippo": "green",
        "voronoi": "blue",
    }

    os.makedirs(dir_save, exist_ok=True)

    for metric in metrics:
        plt.figure(dpi=500, figsize=(6, 4))
        plt.title(title)
        for algo_name, df in data.items():
            mean_col, std_col = f"{metric}_iqm", f"{metric}_iqrstd"
            if mean_col not in df.columns or std_col not in df.columns:
                continue

            x = df["step"].values
            y = df[mean_col].values
            s = df[std_col].values

            color = algo_colors.get(algo_name, None)

            plt.plot(x, y, label=algo_name, color=color, alpha=0.8)
            plt.fill_between(x, y - s, y + s, alpha=0.15, color=color)

        plt.xlabel("Step")
        plt.ylabel(metric.replace("_", " "))
        plt.legend(loc="best")
        plt.tight_layout()
        img_save = dir_save / f"{metric}.{img_format}"
        plt.savefig(img_save)
        # plt.show()
        plt.close()
