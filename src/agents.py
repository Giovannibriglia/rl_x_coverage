import csv
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List

import cv2
import torch

from src import SCALARS_FOLDER_NAME, VIDEOS_FOLDER_NAME


class MarlBase(ABC):
    def __init__(self, env, configs: Dict):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.env = env
        self.max_steps = None
        self.n_agents = None

        self._setup(configs)

    @abstractmethod
    def _setup(self, configs: Dict):
        raise NotImplementedError

    @abstractmethod
    def train_and_evaluate(
        self, env_train, envs_test, main_dir, n_checkpoints: int = 50
    ):
        raise NotImplementedError

    @staticmethod
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

    def evaluate_and_record(self, policy, env, iteration: int, main_dir, algo: str):
        """Run a rollout, save per‑step rewards CSV and an MP4 video."""
        frames: List[torch.Tensor] = []
        with torch.no_grad():
            td = env.rollout(
                max_steps=self.max_steps,
                policy=policy,
                callback=lambda e, td: frames.append(e.render(mode="rgb_array")),
                break_when_any_done=False,
                auto_cast_to_device=True,
            )

        # video
        video_file = main_dir / VIDEOS_FOLDER_NAME / f"{algo}_iter_{iteration}.mp4"
        os.makedirs(video_file.parent, exist_ok=True)
        self.save_video(frames, video_file)

        # --- reward extraction (robust across TorchRL versions) -----------------
        rewards = td.get(env.reward_key, None)  # try direct key (agents,reward)
        if rewards is None:
            # older rollout returns rewards under "next"
            rewards = td.get(("next",) + self.env.reward_key, None)
        if rewards is None:
            print("[warning] reward not found in rollout – skipping CSV")
            return 0.0

        # rewards shape: [T, E, A] or [T+1, E, A] (extra step) – unify length
        if rewards.shape[0] > self.max_steps:
            rewards = rewards[:-1]

        # mean across envs to keep file compact
        step_rewards = rewards.mean(dim=1).cpu().numpy()  # [T, A]
        team_step = step_rewards.mean(axis=1)

        csv_path = (
            Path(main_dir)
            / SCALARS_FOLDER_NAME
            / f"{algo}_eval_iter_{iteration}_reward_per_step.csv"
        )
        csv_path.parent.mkdir(parents=True, exist_ok=True)  # create folder(s) only

        # guard against the rare case that a directory with the CSV name already exists
        if csv_path.is_dir():
            raise RuntimeError(
                f"{csv_path!r} exists as a directory; remove or rename it before writing."
            )

        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["step"] + [f"agent{i}" for i in range(self.n_agents)] + ["team"]
            )
            for t, row in enumerate(step_rewards):
                writer.writerow([t, *row.tolist(), team_step[t]])
