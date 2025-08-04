from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import List

import cv2  # required for mp4 encoding
import torch

# TorchRL / VMAS imports
from tensordict.nn import set_composite_lp_aggregate, TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import multiprocessing
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import RewardSum, TransformedEnv
from torchrl.envs.libs.vmas import VmasEnv
from torchrl.envs.utils import check_env_specs
from torchrl.modules import MultiAgentMLP, ProbabilisticActor, TanhNormal
from torchrl.objectives import ClipPPOLoss, ValueEstimators
from tqdm import tqdm

# ────────────────────────────────────────────────────────────────────────────
# Hyper‑parameters
# ────────────────────────────────────────────────────────────────────────────

EXP_NAME = "voronoi_ppo"

# devices
IS_FORK = multiprocessing.get_start_method() == "fork"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() and not IS_FORK else "cpu")
VMAS_DEVICE = DEVICE

# sampling / training
FRAMES_PER_BATCH = 6_000
N_ITERS = 500
NUM_EPOCHS = 50
MINIBATCH_SIZE = 400
LR = 3e-4
MAX_GRAD_NORM = 1.0
CLIP_EPSILON = 0.2
GAMMA = 0.99
LAMBDA = 0.9
ENTROPY_EPS = 1e-4
N_CHECKPOINTS = 50  # number of videos / checkpoints you want
LOG_EVERY = max(1, N_ITERS // N_CHECKPOINTS)

# environment
MAX_STEPS = 500
SCENARIO_NAME = "voronoi"
N_AGENTS = int(input("Number of agents: "))
N_GAUSSIANS = int(input("Number of gaussians: "))
SEED = 0

# ────────────────────────────────────────────────────────────────────────────
# Experiment folder layout
# ────────────────────────────────────────────────────────────────────────────


TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
ROOT_DIR = Path.cwd() / f"{EXP_NAME}_{TIMESTAMP}"
POLICY_DIR = ROOT_DIR / "policies"
VIDEO_DIR = ROOT_DIR / "videos"
SCALAR_DIR = ROOT_DIR / "scalars"
for d in (POLICY_DIR, VIDEO_DIR, SCALAR_DIR):
    d.mkdir(parents=True, exist_ok=True)

print("Experiment root:", ROOT_DIR)

# ────────────────────────────────────────────────────────────────────────────
# Build env & networks
# ────────────────────────────────────────────────────────────────────────────

set_composite_lp_aggregate(False)

NUM_VMAS_ENVS = FRAMES_PER_BATCH // MAX_STEPS
raw_env = VmasEnv(
    scenario=SCENARIO_NAME,
    num_envs=NUM_VMAS_ENVS,
    continuous_actions=True,
    max_steps=MAX_STEPS,
    device=VMAS_DEVICE,
    n_agents=N_AGENTS,
    n_gaussians=N_GAUSSIANS,
    seed=SEED,
    n_rays=50,
    lidar_range=0.5,
    grid_spacing=0.05,
    centralized=False,
    shared_rew=False,
    n_obstacles=0,
)

env = TransformedEnv(
    raw_env,
    RewardSum(in_keys=[raw_env.reward_key], out_keys=[("agents", "episode_reward")]),
)
check_env_specs(env)

obs_dim = env.observation_spec["agents", "observation"].shape[-1]
action_dim = env.action_spec.shape[-1]

policy_backbone = MultiAgentMLP(
    n_agent_inputs=obs_dim,
    n_agent_outputs=2 * action_dim,
    n_agents=N_AGENTS,
    centralised=False,
    share_params=False,
    device=DEVICE,
    depth=2,
    num_cells=256,
    activation_class=torch.nn.Tanh,
)


policy_module = TensorDictModule(
    torch.nn.Sequential(policy_backbone, NormalParamExtractor()),
    in_keys=[("agents", "observation")],
    out_keys=[("agents", "loc"), ("agents", "scale")],
)

policy = ProbabilisticActor(
    module=policy_module,
    spec=env.action_spec,  # env.action_spec_unbatched,
    in_keys=[("agents", "loc"), ("agents", "scale")],
    out_keys=[env.action_key],
    distribution_class=TanhNormal,
    distribution_kwargs={
        "low": env.full_action_spec[env.action_key].space.low,
        "high": env.full_action_spec[env.action_key].space.high,
    },
    # distribution_kwargs={
    #    "low": env.full_action_spec_unbatched[env.action_key].space.low,
    #    "high": env.full_action_spec_unbatched[env.action_key].space.high,
    # },
    return_log_prob=True,
)

critic_backbone = MultiAgentMLP(
    n_agent_inputs=obs_dim,
    n_agent_outputs=1,
    n_agents=N_AGENTS,
    centralised=False,
    share_params=True,
    device=DEVICE,
    depth=2,
    num_cells=256,
    activation_class=torch.nn.Tanh,
)

critic = TensorDictModule(
    critic_backbone,
    in_keys=[("agents", "observation")],
    out_keys=[("agents", "state_value")],
)

# ────────────────────────────────────────────────────────────────────────────
# Collector, buffer, loss
# ────────────────────────────────────────────────────────────────────────────

collector = SyncDataCollector(
    env,
    policy,
    frames_per_batch=FRAMES_PER_BATCH,
    total_frames=FRAMES_PER_BATCH * N_ITERS,
    device=VMAS_DEVICE,
    storing_device=DEVICE,
)

replay_buffer = ReplayBuffer(
    storage=LazyTensorStorage(FRAMES_PER_BATCH, device=DEVICE),
    sampler=SamplerWithoutReplacement(),
    batch_size=MINIBATCH_SIZE,
)

loss_module = ClipPPOLoss(
    actor_network=policy,
    critic_network=critic,
    clip_epsilon=CLIP_EPSILON,
    entropy_coef=ENTROPY_EPS,
    normalize_advantage=False,
)
loss_module.set_keys(
    reward=env.reward_key,
    action=env.action_key,
    value=("agents", "state_value"),
    done=("agents", "done"),
    terminated=("agents", "terminated"),
)
loss_module.make_value_estimator(ValueEstimators.GAE, gamma=GAMMA, lmbda=LAMBDA)
GAE = loss_module.value_estimator
optimizer = torch.optim.Adam(loss_module.parameters(), LR)

# ────────────────────────────────────────────────────────────────────────────
# CSV helpers
# ────────────────────────────────────────────────────────────────────────────

train_csv_path = SCALAR_DIR / "training_mean_reward.csv"
with train_csv_path.open("w", newline="") as f:
    writer = csv.writer(f)
    header = ["iteration"] + [f"agent{i}" for i in range(N_AGENTS)] + ["team_mean"]
    writer.writerow(header)

# evaluation CSV written inside evaluate_and_record()

# ────────────────────────────────────────────────────────────────────────────
# Video helper
# ────────────────────────────────────────────────────────────────────────────


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


def evaluate_and_record(policy, iteration: int):
    """Run a rollout, save per‑step rewards CSV and an MP4 video."""
    frames: List[torch.Tensor] = []
    with torch.no_grad():
        td = env.rollout(
            max_steps=MAX_STEPS,
            policy=policy,
            callback=lambda e, td: frames.append(e.render(mode="rgb_array")),
            break_when_any_done=False,
            auto_cast_to_device=True,
        )

    # video
    video_file = VIDEO_DIR / f"{SCENARIO_NAME}_iter_{iteration}.mp4"
    save_video(frames, video_file)

    # --- reward extraction (robust across TorchRL versions) -----------------
    rewards = td.get(env.reward_key, None)  # try direct key (agents,reward)
    if rewards is None:
        # older rollout returns rewards under "next"
        rewards = td.get(("next",) + env.reward_key, None)
    if rewards is None:
        print("[warning] reward not found in rollout – skipping CSV")
        return 0.0

    # rewards shape: [T, E, A] or [T+1, E, A] (extra step) – unify length
    if rewards.shape[0] > MAX_STEPS:
        rewards = rewards[:-1]

    # mean across envs to keep file compact
    step_rewards = rewards.mean(dim=1).cpu().numpy()  # [T, A]
    team_step = step_rewards.mean(axis=1)

    csv_path = SCALAR_DIR / f"eval_iter_{iteration}_reward_per_step.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step"] + [f"agent{i}" for i in range(N_AGENTS)] + ["team"])
        for t, row in enumerate(step_rewards):
            writer.writerow([t, *row.tolist(), team_step[t]])

    return team_step.mean()


# ────────────────────────────────────────────────────────────────────────────
# Training loop
# ────────────────────────────────────────────────────────────────────────────

pbar = tqdm(total=N_ITERS, desc="first iteration...")
checkpoint_set = set(range(0, N_ITERS, LOG_EVERY))

for it, data in enumerate(collector):
    # expand done / terminated to agent dim
    for key in ("done", "terminated"):
        data.set(
            ("next", "agents", key),
            data.get(("next", key))
            .unsqueeze(-1)
            .expand(data.get_item_shape(("next", env.reward_key))),
        )

    with torch.no_grad():
        GAE(
            data,
            params=loss_module.critic_network_params,
            target_params=loss_module.target_critic_network_params,
        )

    # push to replay buffer
    replay_buffer.extend(data.reshape(-1))

    # gradient updates
    for _ in range(NUM_EPOCHS):
        for _ in range(FRAMES_PER_BATCH // MINIBATCH_SIZE):
            batch = replay_buffer.sample()
            losses = loss_module(batch)
            loss = sum(losses.values())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(loss_module.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            optimizer.zero_grad()

    collector.update_policy_weights_()

    # training rewards (only finished episodes)
    ep_rew = data.get(("next", "agents", "episode_reward"))  # [envs, agents]
    done_mask = data.get(("next", "agents", "done"))
    agent_means = (
        ep_rew[done_mask].view(-1, N_AGENTS).mean(dim=0)
        if done_mask.any()
        else torch.zeros(N_AGENTS)
    )
    team_mean = agent_means.mean().item()

    # append to training CSV
    with train_csv_path.open("a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([it, *agent_means.tolist(), team_mean])

    # checkpoint?
    if it in checkpoint_set or it == N_ITERS - 1:
        pbar.set_description(f"evaluation... - train‑team‑mean = {team_mean:.3f}")
        # save policy
        policy_file = POLICY_DIR / f"policy_iter_{it}.pt"
        torch.save(policy.state_dict(), policy_file)
        # evaluation rollout + video + csv
        eval_mean = evaluate_and_record(policy, it)
        pbar.set_description(f"eval‑team‑mean = {eval_mean:.3f}")
    else:
        pbar.set_description(f"train‑team‑mean = {team_mean:.3f}")

    pbar.update()

# final artefacts
torch.save(policy.state_dict(), POLICY_DIR / "policy_final.pt")
evaluate_and_record(policy, N_ITERS)
print("Training finished. Data saved to", ROOT_DIR)
