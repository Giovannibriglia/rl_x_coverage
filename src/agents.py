from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List

import cv2

import torch

from src import SCALARS_FOLDER_NAME, VIDEOS_FOLDER_NAME
from src.utils import save_csv


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
        self,
        env_train,
        envs_test,
        main_dir,
        n_checkpoints: int = 50,
        n_checkpoints_metrics: int = 50,
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

    def evaluate_and_record(
        self, policy, env, main_dir, filename: str, n_checkpoints_metrics: int = 50
    ):
        max_steps_evaluation = env.max_steps
        n_agents = env.n_agents

        # prepare CSV
        csv_path = Path(main_dir) / SCALARS_FOLDER_NAME / f"{filename}_scalars.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        # Ensure at least 2 checkpoints
        assert (
            n_checkpoints_metrics >= 2
        ), "Need at least 2 checkpoints (first and last)."

        if n_checkpoints_metrics >= max_steps_evaluation:
            checkpoints = range(max_steps_evaluation)
        else:
            checkpoints = [
                int(round(i * (max_steps_evaluation - 1) / (n_checkpoints_metrics - 1)))
                for i in range(n_checkpoints_metrics)
            ]

        # rollout
        frames = []
        with torch.no_grad():
            td = env.rollout(
                max_steps=max_steps_evaluation,
                policy=policy,
                callback=lambda e, td: frames.append(e.render(mode="rgb_array")),
                break_when_any_done=False,
                auto_cast_to_device=True,
            )

        # video
        video_file = Path(main_dir) / VIDEOS_FOLDER_NAME / f"{filename}_video.mp4"
        video_file.parent.mkdir(parents=True, exist_ok=True)
        self.save_video(frames, video_file)

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
        collisions = td.get(("next", "agents", "info", "n_collisions"), None).squeeze(
            -1
        )

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
