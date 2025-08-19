import copy

from pathlib import Path
from typing import Any, Dict

import numpy as np

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

from src import POLICIES_FOLDER_NAME, TRAIN_SCALARS_FOLDER_NAME
from src.agents import MarlBase
from src.utils import save_csv


class MarlIPPO(MarlBase):
    def __init__(self, env, configs: Dict):
        super().__init__(env=env, configs=configs)
        self.algo_name = "ippo"

    def _setup(self, configs: Dict):

        obs_dim = self.env.observation_spec["agents", "observation"].shape[-1]

        self.n_agents = self.env.n_agents
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

        self.policy = self._make_actor_for_env(self.env, self.n_agents)

        self.policy.cfg = configs

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

    def _make_actor_for_env(self, env, n_agents: int):
        obs_dim = env.observation_spec["agents", "observation"].shape[-1]
        action_dim = env.action_spec.shape[-1]

        backbone = MultiAgentMLP(
            n_agent_inputs=obs_dim,
            n_agent_outputs=2 * action_dim,
            n_agents=n_agents,
            centralised=False,
            share_params=False,
            device=self.device,
            depth=2,
            num_cells=256,
            activation_class=torch.nn.Tanh,
        )

        policy_module = TensorDictModule(
            torch.nn.Sequential(backbone, NormalParamExtractor()),
            in_keys=[("agents", "observation")],
            out_keys=[("agents", "loc"), ("agents", "scale")],
        )

        actor = ProbabilisticActor(
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
        return actor

    def _load_matching(self, target_actor, source_actor):
        tgt_sd = target_actor.state_dict()
        src_sd = source_actor.state_dict()

        matched = {}
        for k, v_src in src_sd.items():
            if k not in tgt_sd:
                continue
            v_tgt = tgt_sd[k]

            # only copy when both sides are tensors and shapes match
            if torch.is_tensor(v_src) and torch.is_tensor(v_tgt):
                if v_tgt.shape == v_src.shape:
                    matched[k] = v_src
            # otherwise skip (buffers like torch.Size, ints, etc.)

        # update and load non-strictly to ignore the rest
        tgt_sd.update(matched)
        target_actor.load_state_dict(tgt_sd, strict=False)

    def train_and_evaluate(
        self,
        env_train,
        envs_test: dict[str, Any],
        main_dir: Path,
        n_checkpoints_train: int = 50,
        n_checkpoints_eval: int = 50,
    ):
        # Ensure at least 2 checkpoints
        assert n_checkpoints_train >= 2, "Need at least 2 checkpoints (first and last)."

        if n_checkpoints_train >= self.n_iters:
            checkpoint_iters = range(self.n_iters)
        else:
            checkpoint_iters = [
                int(round(i * (self.n_iters - 1) / (n_checkpoints_train - 1)))
                for i in range(n_checkpoints_train)
            ]
        print(checkpoint_iters)
        pbar = tqdm(total=self.n_iters, desc=f"training {self.algo_name}...")

        # prepare directories
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
            # —————— collect & GAE ——————
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

            # —————— replay & optimize ——————
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

            # —————— checkpoint & evaluate ——————
            if it in checkpoint_iters:
                n_chkpt += 1
                pbar.set_description(f"evaluating {self.algo_name}...")

                # save your *training* policy weights
                policy_path = policies_dir / f"{self.algo_name}_checkpoint_{it}.pt"
                torch.save(self.policy.state_dict(), policy_path)

                # evaluate on each test env WITHOUT touching self.policy
                for env_idx, (env_test_name, env_test_obj) in enumerate(
                    envs_test.items(), start=1
                ):
                    pbar.set_postfix(
                        checkpoint=f"{n_chkpt}/{n_checkpoints_eval}",
                        env_test=f"{env_idx}/{len(envs_test)}",
                    )
                    if env_test_obj.n_agents == self.n_agents:
                        # same team size → deep copy to avoid aliasing / accidental mutation
                        eval_policy = copy.deepcopy(self.policy).to(self.device).eval()
                    else:
                        # different team size → build fresh actor for *that* env
                        eval_policy = self._make_actor_for_env(
                            env_test_obj, env_test_obj.n_agents
                        ).to(self.device)
                        # copy what matches (shared blocks) and leave per-agent heads fresh
                        self._load_matching(eval_policy, self.policy)
                        eval_policy.eval()

                    # keep a cfg on the eval copy (optional, handy for logs)
                    try:
                        eval_policy.cfg = {
                            **getattr(self.policy, "cfg", {}),
                            "n_agents": env_test_obj.n_agents,
                        }
                    except Exception:
                        pass

                    filename = f"{self.algo_name}_checkpoint_{it}"

                    # run evaluation with a separate actor instance
                    with torch.no_grad():
                        self.evaluate_and_record(
                            eval_policy,
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
