from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from tensordict.nn import NormalParamExtractor, TensorDictModule, TensorDictSequential
from torch import nn

from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.modules import MultiAgentMLP, ProbabilisticActor, TanhNormal
from torchrl.objectives import ClipPPOLoss, ValueEstimators
from tqdm import tqdm

from src import POLICIES_FOLDER_NAME, TRAIN_SCALARS_FOLDER_NAME
from src.agents import MarlBase
from src.utils import evaluate_and_record, save_csv


class _CentralisedValueHead(nn.Module):
    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        hidden: int = 256,
        depth: int = 2,
        broadcast_to_agents: bool = True,
    ):
        super().__init__()
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.broadcast_to_agents = broadcast_to_agents

        layers = [nn.Linear(n_agents * obs_dim, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, obs_agents: torch.Tensor) -> torch.Tensor:
        # obs_agents: [*, A, obs_dim]
        x = obs_agents.reshape(*obs_agents.shape[:-2], self.n_agents * self.obs_dim)
        v_team = self.net(x).unsqueeze(-1)  # [*, 1, 1]
        if self.broadcast_to_agents:
            expand_shape = list(v_team.shape)
            expand_shape[-2] = self.n_agents  # agent dim
            return v_team.expand(*expand_shape)  # [*, A, 1]
        return v_team


class MarlMAPPO(MarlBase):
    """
    MAPPO: Decentralized continuous actors + centralized critic.

    - Actors: per-agent Gaussian policies (TanhNormal), identical to IPPO.
    - Critic: centralised value function over concatenated per-agent observations,
      broadcast to per-agent value tensor to fit ClipPPOLoss API.
    - Everything else (collector, buffer, loss) mirrors your IPPO pipeline.
    """

    def __init__(self, env, configs: Dict):
        super().__init__(env=env, configs=configs)
        self.algo_name = "mappo"

    def _setup(self, configs: Dict):
        # ── read env dims
        obs_dim = self.env.observation_spec["agents", "observation"].shape[-1]
        action_dim = self.env.action_spec.shape[-1]

        # ── training hyperparams
        self.n_agents = self.env.n_agents
        self.frames_per_batch = int(configs["frames_per_batch"])
        self.n_iters = int(configs["n_iters"])
        self.max_steps = int(configs["max_steps"])
        self.num_epochs = int(configs.get("num_epochs", 1))
        self.minibatch_size = int(configs["minibatch_size"])
        self.max_grad_norm = float(configs["max_grad_norm"])
        clip_epsilon = float(configs["clip_epsilon"])
        entropy_eps = float(configs["entropy_eps"])
        gamma = float(configs["gamma"])
        lambda_estimator = float(configs["lambda"])
        lr = float(configs["lr"])

        # ─────────────────────────────────────────────────────────────────────
        # Actor: decentralized Gaussian policy (TanhNormal), iterable pipeline
        # ─────────────────────────────────────────────────────────────────────
        backbone = MultiAgentMLP(
            n_agent_inputs=obs_dim,
            n_agent_outputs=2 * action_dim,  # params for (loc, scale)
            n_agents=self.n_agents,
            centralised=False,
            share_params=False,  # set True if you want param sharing
            device=self.device,
            depth=2,
            num_cells=256,
            activation_class=nn.Tanh,
        )

        policy_net = TensorDictSequential(
            # ("agents","observation") -> ("agents","params")
            TensorDictModule(
                backbone,
                in_keys=[("agents", "observation")],
                out_keys=[("agents", "params")],
            ),
            # ("agents","params") -> ("agents","loc"), ("agents","scale")
            TensorDictModule(
                NormalParamExtractor(),
                in_keys=[("agents", "params")],
                out_keys=[("agents", "loc"), ("agents", "scale")],
            ),
        ).to(self.device)

        self.policy = ProbabilisticActor(
            module=policy_net,  # iterable → OK for ProbabilisticActor
            spec=self.env.action_spec,
            in_keys=[("agents", "loc"), ("agents", "scale")],
            out_keys=[self.env.action_key],
            distribution_class=TanhNormal,
            distribution_kwargs={
                "low": self.env.full_action_spec[self.env.action_key].space.low,
                "high": self.env.full_action_spec[self.env.action_key].space.high,
            },
            return_log_prob=True,
        ).to(self.device)
        self.policy.cfg = configs

        # ─────────────────────────────────────────────────────────────────────
        # Centralized Critic: concat per-agent obs → team value → broadcast
        # NOTE: _CentralisedValueHead must NOT access .parameters() in forward
        # ─────────────────────────────────────────────────────────────────────
        centr_value = _CentralisedValueHead(
            n_agents=self.n_agents,
            obs_dim=obs_dim,
            hidden=256,
            depth=2,
            broadcast_to_agents=True,  # outputs [*, A, 1]
        ).to(self.device)

        critic = TensorDictModule(
            centr_value,
            in_keys=[("agents", "observation")],
            out_keys=[("agents", "state_value")],  # shape [*, A, 1]
        ).to(self.device)

        # ─────────────────────────────────────────────────────────────────────
        # Collector, Replay, Loss (PPO + GAE), Optimizer
        # ─────────────────────────────────────────────────────────────────────
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
        ).to(self.device)

        # Map keys expected by TorchRL objectives
        self.loss_module.set_keys(
            reward=self.env.reward_key,  # expects [*, A, 1]
            action=self.env.action_key,
            value=("agents", "state_value"),
            done=("agents", "done"),
            terminated=("agents", "terminated"),
        )

        # Generalized Advantage Estimator on the same device
        self.loss_module.make_value_estimator(
            ValueEstimators.GAE, gamma=gamma, lmbda=lambda_estimator
        )
        self.GAE = self.loss_module.value_estimator

        # Optimizer
        self.optimizer = torch.optim.Adam(self.loss_module.parameters(), lr)

        # (Optional) quick assertions
        # assert next(self.policy.parameters()).device == self.device
        # assert next(critic.parameters()).device == self.device

    def _load_matching(self, target_actor, source_actor):
        tgt_sd = target_actor.state_dict()
        src_sd = source_actor.state_dict()

        matched = {}
        for k, v_src in src_sd.items():
            if k not in tgt_sd:
                continue
            v_tgt = tgt_sd[k]
            if torch.is_tensor(v_src) and torch.is_tensor(v_tgt):
                if v_tgt.shape == v_src.shape:
                    matched[k] = v_src
        tgt_sd.update(matched)
        target_actor.load_state_dict(tgt_sd, strict=False)

    def train_and_evaluate(
        self,
        env_train,
        envs_test: dict[str, Any],
        main_dir: Path,
        seed: int,
        n_checkpoints_train: int = 50,
        n_checkpoints_eval: int = 50,
    ):
        assert n_checkpoints_train >= 2, "Need at least 2 checkpoints (first and last)."

        if n_checkpoints_train >= self.n_iters:
            checkpoint_iters = range(self.n_iters)
        else:
            checkpoint_iters = [
                int(round(i * (self.n_iters - 1) / (n_checkpoints_train - 1)))
                for i in range(n_checkpoints_train)
            ]

        pbar = tqdm(total=self.n_iters, desc=f"training {self.algo_name}...")

        scalars_train_dir = Path(main_dir) / TRAIN_SCALARS_FOLDER_NAME
        policies_dir = Path(main_dir) / POLICIES_FOLDER_NAME
        scalars_train_dir.mkdir(parents=True, exist_ok=True)
        policies_dir.mkdir(parents=True, exist_ok=True)

        # swap in training env
        self.env = env_train
        self.n_agents = env_train.n_agents

        train_metrics = {
            "rewards": np.zeros((self.env.num_envs, self.env.max_steps, self.n_agents)),
            "eta": np.zeros((self.env.num_envs, self.env.max_steps, self.n_agents)),
            "beta": np.zeros((self.env.num_envs, self.env.max_steps, self.n_agents)),
            "collisions": np.zeros(
                (self.env.num_envs, self.env.max_steps, self.n_agents)
            ),
        }

        n_chkpt = 0

        for it, data in enumerate(self.collector):
            # expand done/terminated to agent dim
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

            # optimize PPO
            self.replay_buffer.extend(data.reshape(-1))
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

            # ——— log training metrics ———
            ep_rew = data.get(("next", "agents", "episode_reward"))
            eta_td = data.get(("next", "agents", "info", "eta"), None)
            beta_td = data.get(("next", "agents", "info", "beta"), None)
            coll_td = data.get(("next", "agents", "info", "n_collisions"), None)

            # Helper: squeeze last dim if size-1, then mean over time dim=1 → [n_envs, n_agents]
            def _mean_over_steps(t: torch.Tensor) -> torch.Tensor:
                if t.dim() >= 3 and t.size(-1) == 1:
                    t = t.squeeze(-1)
                return t.mean(dim=1)

            # Rewards (always present)
            m = _mean_over_steps(ep_rew)  # torch [n_envs, n_agents]
            train_metrics["rewards"][:, it, :] = (
                m.detach().to("cpu", non_blocking=True).contiguous().numpy()
            )

            # Optional metrics
            if eta_td is not None:
                m = _mean_over_steps(eta_td)
                train_metrics["eta"][:, it, :] = (
                    m.detach().to("cpu", non_blocking=True).contiguous().numpy()
                )

            if beta_td is not None:
                m = _mean_over_steps(beta_td)
                train_metrics["beta"][:, it, :] = (
                    m.detach().to("cpu", non_blocking=True).contiguous().numpy()
                )

            if coll_td is not None:
                m = _mean_over_steps(coll_td)
                train_metrics["collisions"][:, it, :] = (
                    m.detach().to("cpu", non_blocking=True).contiguous().numpy()
                )

            # ——— checkpoint & evaluate ———
            if it in checkpoint_iters:
                n_chkpt += 1
                pbar.set_description(f"evaluating {self.algo_name}...")

                # save actor (policy) weights
                policy_path = policies_dir / f"{self.algo_name}_checkpoint_{it}.pt"
                torch.save(self.policy.state_dict(), policy_path)

                # evaluate on test envs (rebuild actor when team size differs)
                for env_idx, (env_test_name, env_test_obj) in enumerate(
                    envs_test.items(), start=1
                ):
                    pbar.set_postfix(
                        checkpoint=f"{n_chkpt}/{n_checkpoints_eval}",
                        env_test=f"{env_idx}/{len(envs_test)}",
                    )

                    if env_test_obj.n_agents == self.n_agents:
                        eval_policy = copy.deepcopy(self.policy).to(self.device).eval()
                    else:
                        # different team size → build fresh actor for that env & copy shared params
                        obs_dim_te = env_test_obj.observation_spec[
                            "agents", "observation"
                        ].shape[-1]
                        action_dim_te = env_test_obj.action_spec.shape[-1]

                        backbone_te = MultiAgentMLP(
                            n_agent_inputs=obs_dim_te,
                            n_agent_outputs=2 * action_dim_te,
                            n_agents=env_test_obj.n_agents,
                            centralised=False,
                            share_params=False,
                            device=self.device,
                            depth=2,
                            num_cells=256,
                            activation_class=nn.Tanh,
                        )
                        policy_module_te = TensorDictModule(
                            nn.Sequential(backbone_te, NormalParamExtractor()),
                            in_keys=[("agents", "observation")],
                            out_keys=[("agents", "loc"), ("agents", "scale")],
                        )
                        eval_policy = ProbabilisticActor(
                            module=policy_module_te,
                            spec=env_test_obj.action_spec,
                            in_keys=[("agents", "loc"), ("agents", "scale")],
                            out_keys=[env_test_obj.action_key],
                            distribution_class=TanhNormal,
                            distribution_kwargs={
                                "low": env_test_obj.full_action_spec[
                                    env_test_obj.action_key
                                ].space.low,
                                "high": env_test_obj.full_action_spec[
                                    env_test_obj.action_key
                                ].space.high,
                            },
                            return_log_prob=True,
                        ).to(self.device)

                        self._load_matching(eval_policy, self.policy)
                        eval_policy.eval()

                    filename = f"{self.algo_name}_checkpoint_{it}"
                    with torch.no_grad():
                        evaluate_and_record(
                            eval_policy,
                            seed=seed,
                            main_dir=main_dir / env_test_name,
                            filename=filename,
                            env=env_test_obj,
                            n_checkpoints_eval=n_checkpoints_eval,
                        )
            else:
                pbar.set_description(f"training {self.algo_name}...")
                pbar.set_postfix()
            pbar.update()

        # store train metrics
        csv_path = scalars_train_dir / f"{self.algo_name}_train.csv"
        save_csv(
            csv_path,
            self.n_agents,
            checkpoint_iters,
            train_metrics["rewards"],
            train_metrics["eta"],
            train_metrics["beta"],
            train_metrics["collisions"],
        )

        pbar.close()
