from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import List

import cv2  # MP4 encoding
import torch

# ─── TorchRL / VMAS imports ─────────────────────────────────────────────────
from tensordict.nn import TensorDictModule, TensorDictSequential
from torch import multiprocessing, nn
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import RewardSum, TransformedEnv
from torchrl.envs.libs.vmas import VmasEnv
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
from torchrl.modules import EGreedyModule, MultiAgentMLP, QValueModule, SafeSequential
from torchrl.modules.models.multiagent import QMixer
from torchrl.objectives import SoftUpdate, ValueEstimators
from torchrl.objectives.multiagent.qmixer import QMixerLoss
from tqdm import tqdm

# ────────────────────────────────────────────────────────────────────────────
# Hyper-parameters
# ────────────────────────────────────────────────────────────────────────────

EXP_NAME = "voronoi_qmix"

IS_FORK = multiprocessing.get_start_method() == "fork"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() and not IS_FORK else "cpu")
VMAS_DEVICE = DEVICE

# sampling / training
FRAMES_PER_BATCH = 6_000  # steps collected before each update
N_ITERS = 500
MINIBATCH_SIZE = 400
LR = 3e-4
MAX_GRAD_NORM = 5.0
GAMMA = 0.99
TAU = 0.005  # target-net soft-update rate
LOG_EVERY = 10  # evaluation / checkpoint frequency

# ε-greedy exploration
EPS_INIT = 0.3
EPS_END = 0.0
EPS_FRAMES = FRAMES_PER_BATCH * N_ITERS // 2  # linear anneal over 50 %

# environment
MAX_STEPS = 500
SCENARIO_NAME = "voronoi"
N_AGENTS = int(input("Number of agents: "))
N_GAUSSIANS = int(input("Number of Gaussians: "))
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
# Build environment
# ────────────────────────────────────────────────────────────────────────────

NUM_VMAS_ENVS = FRAMES_PER_BATCH // MAX_STEPS
raw_env = VmasEnv(
    scenario=SCENARIO_NAME,
    num_envs=NUM_VMAS_ENVS,
    continuous_actions=False,  # QMIX needs discrete actions
    categorical_actions=True,  # one-hot encoded
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

# ────────────────────────────────────────────────────────────────────────────
# Specs & dimensions
# ────────────────────────────────────────────────────────────────────────────

obs_dim = env.observation_spec[("agents", "observation")].shape[-1]
action_spec = env.action_spec  # already a Categorical spec
num_actions = action_spec.space.n

# ────────────────────────────────────────────────────────────────────────────
# Per-agent Q-network
# ────────────────────────────────────────────────────────────────────────────

q_backbone = MultiAgentMLP(
    n_agent_inputs=obs_dim,
    n_agent_outputs=num_actions,  # one Q-value per discrete action
    n_agents=N_AGENTS,
    centralised=False,
    share_params=True,
    device=DEVICE,
    depth=2,
    num_cells=256,
    activation_class=nn.Tanh,
)

q_module = TensorDictModule(
    q_backbone,
    in_keys=[("agents", "observation")],
    out_keys=[("agents", "action_value")],
)

q_value = QValueModule(
    action_value_key=("agents", "action_value"),
    out_keys=[
        env.action_key,  # chosen discrete action
        ("agents", "action_value"),  # Q(s,a) for all a
        ("agents", "chosen_action_value"),  # Q(s,a*) for picked action
    ],
    spec=action_spec,
)

q_net = SafeSequential(q_module, q_value)

# ── ε-greedy exploration wrapper ────────────────────────────────────────────
q_net_explore = TensorDictSequential(
    q_net,
    EGreedyModule(
        eps_init=EPS_INIT,
        eps_end=EPS_END,
        annealing_num_steps=EPS_FRAMES,
        action_key=env.action_key,
        spec=action_spec,
    ),
)

# ── Monotonic mixing network ────────────────────────────────────────────────
qmixer = TensorDictModule(
    QMixer(
        state_shape=env.unbatched_observation_spec[("agents", "observation")].shape,
        mixing_embed_dim=32,
        n_agents=N_AGENTS,
        device=DEVICE,
    ),
    in_keys=[("agents", "chosen_action_value"), ("agents", "observation")],
    out_keys=["chosen_action_value"],  # joint Q-value
)

# ────────────────────────────────────────────────────────────────────────────
# Collector, replay buffer, loss
# ────────────────────────────────────────────────────────────────────────────

collector = SyncDataCollector(
    env,
    q_net_explore,
    frames_per_batch=FRAMES_PER_BATCH,
    total_frames=FRAMES_PER_BATCH * N_ITERS,
    device=VMAS_DEVICE,
    storing_device=DEVICE,
)

replay_buffer = TensorDictReplayBuffer(
    storage=LazyTensorStorage(FRAMES_PER_BATCH, device=DEVICE),
    sampler=SamplerWithoutReplacement(),
    batch_size=MINIBATCH_SIZE,
)

loss_module = QMixerLoss(
    local_value_network=q_net,
    mixer_network=qmixer,
    delay_value=True,
)
loss_module.set_keys(
    action_value=("agents", "action_value"),
    local_value=("agents", "chosen_action_value"),
    global_value="chosen_action_value",
    action=env.action_key,
)
loss_module.make_value_estimator(ValueEstimators.TD0, gamma=GAMMA)

optim = torch.optim.Adam(loss_module.parameters(), LR)
target_updater = SoftUpdate(loss_module, eps=1 - TAU)

# ────────────────────────────────────────────────────────────────────────────
# CSV helpers
# ────────────────────────────────────────────────────────────────────────────
train_csv_path = SCALAR_DIR / "training_mean_reward.csv"
with train_csv_path.open("w", newline="") as f:
    csv.writer(f).writerow(["iteration", "team_mean"])


# ────────────────────────────────────────────────────────────────────────────
# Video helper
# ────────────────────────────────────────────────────────────────────────────
def save_video(frames: List[torch.Tensor], filename: Path, fps: int = 30):
    if not frames:
        print("[warning] no frames to write", filename)
        return
    h, w, _ = frames[0].shape
    vout = cv2.VideoWriter(str(filename), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for fr in frames:
        vout.write(fr)
    vout.release()


def evaluate_and_record(policy, iteration: int) -> float:
    """Deterministic rollout → MP4 + per-step reward CSV → return mean."""
    frames: List[torch.Tensor] = []
    with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
        td = env.rollout(
            max_steps=MAX_STEPS,
            policy=policy,
            callback=lambda e, td: frames.append(e.render(mode="rgb_array")),
            break_when_any_done=False,
            auto_cast_to_device=True,
        )

    save_video(frames, VIDEO_DIR / f"{SCENARIO_NAME}_iter_{iteration}.mp4")

    rewards = td.get(env.reward_key, td.get(("next",) + env.reward_key))
    if rewards is None:
        return 0.0
    if rewards.shape[0] > MAX_STEPS:  # rollout can return T+1 steps
        rewards = rewards[:-1]
    team_step = rewards.mean(dim=[1, 2]).cpu().numpy()  # shape [T]

    with (SCALAR_DIR / f"eval_iter_{iteration}_reward_per_step.csv").open(
        "w", newline=""
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["step", "team"])
        for t, r in enumerate(team_step):
            writer.writerow([t, r])

    return float(team_step.mean())


# ────────────────────────────────────────────────────────────────────────────
# Training loop
# ────────────────────────────────────────────────────────────────────────────
pbar = tqdm(total=N_ITERS, desc="collecting…")
checkpoint_set = set(range(0, N_ITERS, LOG_EVERY))

for it, data in enumerate(collector):
    # Reduce per-agent reward → scalar team reward for QMIX loss
    data.set(("next", "reward"), data.get(("next", env.reward_key)).mean(-2))
    del data["next", env.reward_key]  # keep tensordict compact

    # Replay buffer
    replay_buffer.extend(data.reshape(-1))

    # One GD step per batch
    for _ in range(FRAMES_PER_BATCH // MINIBATCH_SIZE):
        batch = replay_buffer.sample()
        loss_td = loss_module(batch)
        loss_td["loss"].backward()
        torch.nn.utils.clip_grad_norm_(loss_module.parameters(), MAX_GRAD_NORM)
        optim.step()
        optim.zero_grad()
        target_updater.step()

    # Update ε schedule
    q_net_explore[1].step(frames=data.numel())
    collector.update_policy_weights_()

    # Mean finished-episode reward (for quick feedback)
    ep_rew = data.get(("next", "agents", "episode_reward"), None)
    done_msk = data.get(("next", "agents", "done"), None)
    if ep_rew is not None and done_msk is not None and done_msk.any():
        team_mean = float(ep_rew[done_msk].view(-1, N_AGENTS).mean())
    else:
        team_mean = 0.0
    with train_csv_path.open("a", newline="") as f:
        csv.writer(f).writerow([it, team_mean])

    # Checkpoint / evaluate?
    if it in checkpoint_set or it == N_ITERS - 1:
        pbar.set_description(f"evaluation… (train {team_mean:.3f})")
        torch.save(q_net.state_dict(), POLICY_DIR / f"qmix_policy_iter_{it}.pt")
        eval_mean = evaluate_and_record(q_net, it)
        pbar.set_description(f"eval mean {eval_mean:.3f}")
    else:
        pbar.set_description(f"train mean {team_mean:.3f}")
    pbar.update()

# ─── Final artefacts ────────────────────────────────────────────────────────
torch.save(q_net.state_dict(), POLICY_DIR / "qmix_policy_final.pt")
evaluate_and_record(q_net, N_ITERS)
print("Training finished. Data saved to", ROOT_DIR)
