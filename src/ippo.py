import csv
import os
from pathlib import Path
from typing import Dict

import torch
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor

from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.modules import MultiAgentMLP, ProbabilisticActor, TanhNormal
from torchrl.objectives import ClipPPOLoss, ValueEstimators
from tqdm import tqdm

from src import POLICIES_FOLDER_NAME, SCALARS_FOLDER_NAME
from src.agents import MarlBase


class MarlIPPO(MarlBase):
    def __init__(self, env, configs: Dict):
        super().__init__(env=env, configs=configs)
        self.algo_name = "ippo"

    def _setup(self, configs: Dict):

        obs_dim = self.env.observation_spec["agents", "observation"].shape[-1]
        action_dim = self.env.action_spec.shape[-1]

        self.n_agents = configs["n_agents"]
        self.frames_per_batch = configs["frames_per_batch"]
        self.n_iters = configs["n_iters"]
        self.max_steps = configs["max_steps"]
        self.num_epochs = configs["num_epochs"]
        self.minibatch_size = configs["minibatch_size"]
        self.max_grad_norm = configs["max_grad_norm"]
        clip_epsilon = configs["clip_epsilon"]
        entropy_eps = configs["entropy_eps"]
        gamma = configs["gamma"]
        lambda_estimator = configs["lambda"]
        lr = configs["lr"]

        policy_backbone = MultiAgentMLP(
            n_agent_inputs=obs_dim,
            n_agent_outputs=2 * action_dim,
            n_agents=self.n_agents,
            centralised=False,
            share_params=False,
            device=self.device,
            depth=2,
            num_cells=256,
            activation_class=torch.nn.Tanh,
        )

        policy_module = TensorDictModule(
            torch.nn.Sequential(policy_backbone, NormalParamExtractor()),
            in_keys=[("agents", "observation")],
            out_keys=[("agents", "loc"), ("agents", "scale")],
        )

        self.policy = ProbabilisticActor(
            module=policy_module,
            spec=self.env.action_spec,  # env.action_spec_unbatched,
            in_keys=[("agents", "loc"), ("agents", "scale")],
            out_keys=[self.env.action_key],
            distribution_class=TanhNormal,
            distribution_kwargs={
                "low": self.env.full_action_spec[self.env.action_key].space.low,
                "high": self.env.full_action_spec[self.env.action_key].space.high,
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
            n_agents=self.n_agents,
            centralised=False,
            share_params=True,
            device=self.device,
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

        self.collector = SyncDataCollector(
            self.env,
            self.policy,
            frames_per_batch=self.frames_per_batch,
            total_frames=self.frames_per_batch * self.n_iters,
            device=self.device,
            storing_device=self.device,
        )

        self.replay_buffer = ReplayBuffer(
            storage=LazyTensorStorage(self.frames_per_batch, device=self.device),
            sampler=SamplerWithoutReplacement(),
            batch_size=self.minibatch_size,
        )

        self.loss_module = ClipPPOLoss(
            actor_network=self.policy,
            critic_network=critic,
            clip_epsilon=clip_epsilon,
            entropy_coef=entropy_eps,
            normalize_advantage=False,
        )
        self.loss_module.set_keys(
            reward=self.env.reward_key,
            action=self.env.action_key,
            value=("agents", "state_value"),
            done=("agents", "done"),
            terminated=("agents", "terminated"),
        )
        self.loss_module.make_value_estimator(
            ValueEstimators.GAE, gamma=gamma, lmbda=lambda_estimator
        )
        self.GAE = self.loss_module.value_estimator
        self.optimizer = torch.optim.Adam(self.loss_module.parameters(), lr)

    def train_and_evaluate(
        self, env_train, env_test, main_dir, n_checkpoints: int = 50
    ):

        log_every = int(self.n_iters / n_checkpoints)
        pbar = tqdm(total=self.n_iters, desc="first iteration...")
        checkpoint_set = set(range(0, self.n_iters, log_every))

        for it, data in enumerate(self.collector):
            # expand done / terminated to agent dim
            for key in ("done", "terminated"):
                data.set(
                    ("next", "agents", key),
                    data.get(("next", key))
                    .unsqueeze(-1)
                    .expand(data.get_item_shape(("next", self.env.reward_key))),
                )

            with torch.no_grad():
                self.GAE(
                    data,
                    params=self.loss_module.critic_network_params,
                    target_params=self.loss_module.target_critic_network_params,
                )

            # push to replay buffer
            self.replay_buffer.extend(data.reshape(-1))

            # gradient updates
            for _ in range(self.num_epochs):
                for _ in range(self.frames_per_batch // self.minibatch_size):
                    batch = self.replay_buffer.sample()
                    losses = self.loss_module(batch)
                    loss = sum(losses.values())
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.loss_module.parameters(), self.max_grad_norm
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            self.collector.update_policy_weights_()

            # training rewards (only finished episodes)
            ep_rew = data.get(("next", "agents", "episode_reward"))  # [envs, agents]
            done_mask = data.get(("next", "agents", "done"))
            agent_means = (
                ep_rew[done_mask].view(-1, self.n_agents).mean(dim=0)
                if done_mask.any()
                else torch.zeros(self.n_agents)
            )
            team_mean = agent_means.mean().item()

            train_csv_path = (
                Path(main_dir) / SCALARS_FOLDER_NAME / "train.csv"
            )  # ensure main_dir is a Path or convertible
            train_csv_path.parent.mkdir(
                parents=True, exist_ok=True
            )  # create containing folder(s)

            with train_csv_path.open("a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([it, *agent_means.tolist(), team_mean])

            policy_dir = main_dir / POLICIES_FOLDER_NAME
            os.makedirs(policy_dir, exist_ok=True)
            # checkpoint?
            if it in checkpoint_set or it == self.n_iters - 1:
                pbar.set_description(
                    f"evaluation... - train‑team‑mean = {team_mean:.3f}"
                )
                # save policy
                policy_file = policy_dir / f"policy_iter_{it}.pt"
                torch.save(self.policy.state_dict(), policy_file)
                # evaluation rollout + video + csv
                eval_mean = self.evaluate_and_record(
                    self.policy, it, main_dir, self.algo_name
                )
                pbar.set_description(f"eval‑team‑mean = {eval_mean:.3f}")
            else:
                pbar.set_description(f"train‑team‑mean = {team_mean:.3f}")

            pbar.update()
