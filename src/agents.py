import csv
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

    @staticmethod
    def _to_time_env_agent(rewards: torch.Tensor, env, steps: int) -> torch.Tensor:
        # drop trailing singleton dims
        while rewards.dim() > 0 and rewards.shape[-1] == 1:
            rewards = rewards.squeeze(-1)

        # 1) move time axis (== steps or steps+1) to front
        sizes = list(rewards.shape)
        cand_t = [i for i, s in enumerate(sizes) if s in (steps, steps + 1)]
        time_axis = cand_t[0] if cand_t else 0
        r = rewards.movedim(time_axis, 0)

        # 2) move agent axis (== env.n_agents) to last
        n_agents = getattr(env, "n_agents", None)
        agent_axis = None
        if n_agents is not None:
            cand_a = [i for i, s in enumerate(r.shape) if i != 0 and s == n_agents]
            agent_axis = cand_a[-1] if cand_a else None
        r = r.movedim(agent_axis if agent_axis is not None else -1, -1)

        # 3) flatten middle dims to env axis
        T, A = r.shape[0], r.shape[-1]
        mid = r.shape[1:-1]
        E = 1
        for m in mid:
            E *= int(m)
        r = r.reshape(T, E, A)

        # 4) trim/clip/pad to exactly `steps`
        if r.shape[0] == steps + 1:
            r = r[:-1]
        elif r.shape[0] > steps:
            r = r[:steps]
        elif r.shape[0] < steps:
            pad = torch.full(
                (steps - r.shape[0], E, A), float("nan"), device=r.device, dtype=r.dtype
            )
            r = torch.cat([r, pad], dim=0)
        return r  # [steps, E, A]

    def evaluate_and_record(
        self, policy, env, main_dir, filename: str, max_steps_evaluation: int = 5000
    ):
        """Write exactly `max_steps_evaluation` rows with per-agent and team mean."""

        max_steps_evaluation = env.max_steps

        # rollout
        frames = []
        with torch.no_grad():
            td = env.rollout(
                max_steps=self.env.max_steps,
                policy=policy,
                callback=lambda e, td: frames.append(e.render(mode="rgb_array")),
                break_when_any_done=False,
                auto_cast_to_device=True,
            )

        # video
        video_file = Path(main_dir) / VIDEOS_FOLDER_NAME / f"{filename}_video.mp4"
        video_file.parent.mkdir(parents=True, exist_ok=True)
        self.save_video(frames, video_file)

        rewards = td.get(("next",) + env.reward_key, None)
        if rewards is None:
            rewards = td.get(env.reward_key, None)
        if rewards is None:
            print("[warning] reward not found in rollout – skipping CSV")
            return 0.0

        # normalize to [T, E, A]
        rewards_tea = self._to_time_env_agent(rewards, env, max_steps_evaluation)

        # averages: over envs → [T, A], then team mean over agents → [T]
        per_agent_step = torch.nanmean(rewards_tea, dim=1)  # [steps, A]
        team_step = torch.nanmean(per_agent_step, dim=1)  # [steps]

        # write CSV with exactly `steps` rows
        A = per_agent_step.shape[1]
        csv_path = Path(main_dir) / SCALARS_FOLDER_NAME / f"{filename}_scalars.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step"] + [f"agent{i}" for i in range(A)] + ["team"])
            pa = per_agent_step.cpu().numpy()
            tm = team_step.cpu().numpy()
            for t in range(max_steps_evaluation):
                writer.writerow([t, *pa[t].tolist(), float(tm[t])])

        # return total team return (ignoring NaN pads)
        return float(torch.nan_to_num(team_step, nan=0.0).sum().item())
