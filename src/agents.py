import csv
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

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

    def evaluate_and_record(self, policy, env, main_dir, filename: str):
        """Write exactly `max_steps_evaluation` rows with per-agent and team mean + eta/beta/collisions."""

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

        # rewards [T, E, n_agents]
        rewards = td.get(("next",) + env.reward_key, None)
        if rewards is None:
            rewards = td.get(env.reward_key, None)
        if rewards is None:
            print("[warning] reward not found in rollout – skipping CSV")
            return 0.0
        rewards_tea = self._to_time_env_agent(rewards, env, max_steps_evaluation)

        # metrics from infos (shape [T, E, n_agents] or [T, E])
        eta = td.get(("next", "agents", "info", "eta"), None)
        beta = td.get(("next", "agents", "info", "beta"), None)
        collisions = td.get(("next", "agents", "info", "n_collisions"), None)

        # normalize to [T, E, n_agents] for consistency
        def _to_time_env_agent(x):
            if x is None:
                return None
            if x.ndim == 2:  # [T, E] -> [T, E, 1]
                x = x.unsqueeze(-1)
            return x

        eta = _to_time_env_agent(eta)
        beta = _to_time_env_agent(beta)
        collisions = _to_time_env_agent(collisions)
        """print("\neta: ", eta.shape)
        print("beta: ", beta.shape)
        print("collisions: ", collisions.shape)"""
        # averages: over envs → [T, n_agents], then team mean over agents → [T]
        per_agent_step = torch.nanmean(rewards_tea, dim=0)  # [T, n_agents]
        team_step = torch.nanmean(per_agent_step, dim=1)  # [T]

        # prepare CSV
        n_agents = per_agent_step.shape[1]
        csv_path = Path(main_dir) / SCALARS_FOLDER_NAME / f"{filename}_scalars.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            header = (
                ["step"]
                + [f"agent{i}_reward" for i in range(n_agents)]
                + ["team_reward"]
                + [f"agent{i}_eta" for i in range(n_agents)]
                + ["eta_mean"]
                + [f"agent{i}_beta" for i in range(n_agents)]
                + ["beta_mean"]
                + [f"agent{i}_collisions" for i in range(n_agents)]
                + ["collisions_mean"]
            )
            writer.writerow(header)

            pa = per_agent_step.cpu().numpy()  # [T, n_agents]
            tm = team_step.cpu().numpy()  # [T]

            eta_pa = (
                eta.squeeze(0).squeeze(-1).cpu().numpy() if eta is not None else None
            )  # [T, A]
            beta_pa = (
                beta.squeeze(0).squeeze(-1).cpu().numpy() if beta is not None else None
            )
            coll_pa = (
                collisions.squeeze(0).squeeze(-1).cpu().numpy()
                if collisions is not None
                else None
            )
            """print("eta_pa.shape: ", eta_pa.shape)
            print("beta_pa.shape: ", beta_pa.shape)
            print("coll_pa.shape: ", coll_pa.shape)
            print("pa.shape: ", pa.shape)"""
            for t in range(max_steps_evaluation):
                row = [t] + pa[t].tolist() + [float(tm[t])]

                if eta_pa is not None:
                    row += eta_pa[t].tolist()
                    row += [float(np.nanmean(eta_pa[t]))]

                if beta_pa is not None:
                    row += beta_pa[t].tolist()
                    row += [float(np.nanmean(beta_pa[t]))]

                if coll_pa is not None:
                    row += coll_pa[t].tolist()
                    row += [float(np.nanmean(coll_pa[t]))]

                writer.writerow(row)

        # return total team return
        return float(torch.nan_to_num(team_step, nan=0.0).sum().item())

    """def evaluate_and_record(
            self, policy, env, main_dir, filename: str, max_steps_evaluation: int = 5000
        ):

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
            return float(torch.nan_to_num(team_step, nan=0.0).sum().item())"""
