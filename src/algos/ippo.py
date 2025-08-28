from typing import Dict

import torch
from tensordict.nn import TensorDictModule
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyTensorStorage, ReplayBuffer, SamplerWithoutReplacement
from torchrl.modules import MultiAgentMLP, ProbabilisticActor, TanhNormal
from torchrl.objectives import ClipPPOLoss, ValueEstimators

from src.algos.utils import SafeParamHead
from src.base_algo import MarlAlgo


class IPPO(MarlAlgo):
    def __init__(self):
        super().__init__()
        self.algo_name = "ippo"

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

        obs_dim = env.observation_spec["agents", "observation"].shape[-1]
        action_dim = env.action_spec.shape[-1]

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
            torch.nn.Sequential(
                policy_backbone, SafeParamHead(min_std=1e-3, max_std=2.0)
            ),
            in_keys=[("agents", "observation")],
            out_keys=[("agents", "loc"), ("agents", "scale")],
        )

        self.policy = ProbabilisticActor(
            module=policy_module,
            spec=env.action_spec,  # env.action_spec_unbatched,
            in_keys=[("agents", "loc"), ("agents", "scale")],
            out_keys=[env.action_key],
            distribution_class=TanhNormal,
            distribution_kwargs={
                "low": env.full_action_spec[env.action_key].space.low,
                "high": env.full_action_spec[env.action_key].space.high,
            },
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
        ).to(self.device)

        # ────────────────────────────────────────────────────────────────────────────
        # Collector, buffer, loss
        # ────────────────────────────────────────────────────────────────────────────

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

    def _build_policy_for_env(self, env, n_agents: int, share_params_actor: bool):
        obs_dim = env.observation_spec["agents", "observation"].shape[-1]
        action_dim = env.action_spec.shape[-1]

        backbone = MultiAgentMLP(
            n_agent_inputs=obs_dim,
            n_agent_outputs=2 * action_dim,
            n_agents=n_agents,
            centralised=False,
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

    # ── NEW: extractor hook used by the base transfer_policy ─────────────────
    def _extract_actor_backbone(self, policy) -> torch.nn.Module:
        # Our policy is Sequential[ MultiAgentMLP , NormalParamExtractor ]
        return policy.module[0]
