from typing import Dict

import torch
from tensordict.nn import TensorDictModule
from torch import nn
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyTensorStorage, ReplayBuffer, SamplerWithoutReplacement
from torchrl.modules import MultiAgentMLP, ProbabilisticActor, TanhNormal
from torchrl.objectives import ClipPPOLoss, ValueEstimators

from src.algos.utils import SafeParamHead
from src.base_algo import MarlAlgo


# ─────────────────────────────────────────────────────────────────────────────
# Helpers for centralized critic
# ─────────────────────────────────────────────────────────────────────────────
class CentralizedCriticToAgents(nn.Module):
    """
    Centralized critic over concatenated observations, but outputs
    per-agent values shaped [..., n_agents, 1] so keys match IPPO.
    """

    def __init__(self, n_agents: int, obs_dim: int, hidden: int = 256, depth: int = 2):
        super().__init__()
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        in_dim = n_agents * obs_dim

        layers = [nn.Linear(in_dim, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, obs_agents: torch.Tensor) -> torch.Tensor:
        # obs_agents: [..., n_agents, obs_dim]
        x = obs_agents.reshape(*obs_agents.shape[:-2], self.n_agents * self.obs_dim)
        v = self.net(x)  # [..., 1]
        v_agents = v.unsqueeze(-2).expand(
            *v.shape[:-1], self.n_agents, 1
        )  # [..., n_agents, 1]
        return v_agents


# ─────────────────────────────────────────────────────────────────────────────
# MAPPO
# ─────────────────────────────────────────────────────────────────────────────


class MAPPO(MarlAlgo):
    def __init__(self):
        super().__init__()
        self.algo_name = "mappo"

    # ── hooks so base.transfer_policy works out-of-the-box ───────────────────
    def _build_policy_for_env(self, env, n_agents: int, share_params_actor: bool):
        obs_dim = env.observation_spec["agents", "observation"].shape[-1]
        action_dim = env.action_spec.shape[-1]

        backbone = MultiAgentMLP(
            n_agent_inputs=obs_dim,
            n_agent_outputs=2 * action_dim,
            n_agents=n_agents,
            centralised=False,  # actor is decentralized
            share_params=share_params_actor,
            device=self.device,
            depth=2,
            num_cells=256,
            activation_class=torch.nn.Tanh,
        )

        policy_module = TensorDictModule(
            torch.nn.Sequential(backbone, SafeParamHead(min_std=1e-3, max_std=2.0)),
            in_keys=[("agents", "observation")],
            out_keys=[("agents", "loc"), ("agents", "scale")],
        )

        policy = ProbabilisticActor(
            module=policy_module,
            spec=env.action_spec,
            in_keys=[("agents", "loc"), ("agents", "scale")],
            out_keys=[env.action_key],
            distribution_class=TanhNormal,
            distribution_kwargs={
                "low": env.full_action_spec[env.action_key].space.low,
                "high": env.full_action_spec[env.action_key].space.high,
            },
            return_log_prob=True,
        )
        return policy, backbone

    def _extract_actor_backbone(self, policy) -> nn.Module:
        # Our policy is Sequential[ MultiAgentMLP , NormalParamExtractor ]
        return policy.module[0]

    # ── standard TorchRL setup (actor decentralized, critic centralized) ─────
    def setup(self, configs: Dict, **kwargs):
        lr = configs["lr"]
        gamma = configs["gamma"]
        lb = configs["lambda"]
        self.frames_per_batch = configs["frames_per_batch"]
        self.n_iters = configs["n_iters"]
        self.minibatch_size = configs["minibatch_size"]
        self.max_grad_norm = configs["max_grad_norm"]
        clip_eps = configs["clip_eps"]
        entropy_eps = configs["entropy_eps"]
        self.n_epochs = configs["n_epochs"]

        env = configs["env"]
        self.n_agents = env.n_agents
        self.env_reward_key = env.reward_key

        # ── actor (same style as IPPO; often shared in MAPPO) ────────────────
        self.policy, actor_backbone = self._build_policy_for_env(
            env,
            n_agents=self.n_agents,
            share_params_actor=True,  # MAPPO typically shares actor
        )

        # ── centralized critic: build central value then repeat to per-agent ─
        obs_dim = env.observation_spec["agents", "observation"].shape[-1]

        critic = TensorDictModule(
            CentralizedCriticToAgents(self.n_agents, obs_dim, hidden=256, depth=2),
            in_keys=[("agents", "observation")],
            out_keys=[("agents", "state_value")],  # <-- matches IPPO
        ).to(self.device)

        # ──────────────────────────────────────────────────────────────
        # Collector, buffer, loss (GAE on per-agent repeated values)
        # ──────────────────────────────────────────────────────────────
        self.collector = SyncDataCollector(
            env,
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
            clip_epsilon=clip_eps,
            entropy_coef=entropy_eps,
            normalize_advantage=True,
        )
        self.loss_module.set_keys(
            reward=env.reward_key,
            action=env.action_key,
            value=("agents", "state_value"),
            done=("agents", "done"),
            terminated=("agents", "terminated"),
        )
        self.loss_module.make_value_estimator(
            ValueEstimators.GAE, gamma=gamma, lmbda=lb
        )
        self.GAE = self.loss_module.value_estimator
        self.optimizer = torch.optim.Adam(self.loss_module.parameters(), lr)

        self.policy.train()
        critic.train()
