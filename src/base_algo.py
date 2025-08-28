from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict

import numpy as np
import torch
from tqdm import tqdm

from src.utils import save_csv


class MarlAlgo(ABC):
    def __init__(self, **kwargs):
        self.algo_name = None
        self.collector = None
        self.loss_module = None
        self.env_reward_key = None
        self.replay_buffer = None
        self.n_epochs = None
        self.frames_per_batch = None
        self.minibatch_size = None
        self.max_grad_norm = None
        self.optimizer = None
        self.n_agents = None
        self.policy = None
        self.n_iters = None
        self.GAE = None
        self.device = kwargs.get(
            "device", torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        )

    @abstractmethod
    def setup(self, configs: Dict, **kwargs):
        raise NotImplementedError

    # ── NEW: required hooks for policy construction / extraction ────────────
    @abstractmethod
    def _build_policy_for_env(self, env, n_agents: int, share_params_actor: bool):
        """
        Return (policy, actor_backbone_module) for the given env.
        `actor_backbone_module` must be the underlying MultiAgent actor MLP.
        """
        raise NotImplementedError

    @abstractmethod
    def _extract_actor_backbone(self, policy) -> torch.nn.Module:
        """
        Given a policy built by this algo, return the underlying actor backbone
        (MultiAgentMLP) whose state_dict we can copy.
        """
        raise NotImplementedError

    @staticmethod
    def _normalize_old_key_for_agent0(k: str) -> str:
        # remove the first ".0." occurrence (agent-0 -> shared style)
        return k.replace(".0.", ".", 1) if ".0." in k else k

    @staticmethod
    def _possible_old_keys_for_new_key(k_new: str):
        """Yield candidate old keys that could match k_new (handles shared/non-shared)."""
        yield k_new  # exact match
        parts = k_new.split(".")
        if len(parts) > 1 and parts[1].isdigit():
            # new non-shared -> shared
            yield ".".join([parts[0]] + parts[2:])  # remove index
            yield ".".join([parts[0], "0"] + parts[2:])  # map to old agent-0
        else:
            # new shared -> old non-shared (agent-0)
            yield ".".join([parts[0], "0"] + parts[1:])

    def _copy_actor_weights(
        self, old_backbone: torch.nn.Module, new_backbone: torch.nn.Module
    ):
        """Version-agnostic, param-by-param copy with explicit None checks."""
        with torch.no_grad():
            old_params = dict(old_backbone.named_parameters())
            old_buffers = dict(old_backbone.named_buffers())

            # also build a “shared-style” view by stripping ".0."
            old_params_shared = {
                self._normalize_old_key_for_agent0(k): v for k, v in old_params.items()
            }
            old_buffers_shared = {
                self._normalize_old_key_for_agent0(k): v for k, v in old_buffers.items()
            }

            # ---- copy parameters
            for k_new, p_new in new_backbone.named_parameters():
                for k_cand in self._possible_old_keys_for_new_key(k_new):
                    src = old_params.get(k_cand)
                    if src is None:
                        src = old_params_shared.get(k_cand)
                    if src is not None and tuple(src.shape) == tuple(p_new.shape):
                        p_new.copy_(src)
                        break  # next param

            # ---- copy buffers (e.g., running stats)
            for k_new, b_new in new_backbone.named_buffers():
                for k_cand in self._possible_old_keys_for_new_key(k_new):
                    src = old_buffers.get(k_cand)
                    if src is None:
                        src = old_buffers_shared.get(k_cand)
                    if src is not None and tuple(src.shape) == tuple(b_new.shape):
                        b_new.copy_(src)
                        break  # next buffer

    def transfer_policy(
        self,
        target_env,
        checkpoint_path: str,
        new_share_params_actor: bool = True,
        old_share_params_actor: bool | None = None,  # allow auto-detect
    ):
        """
        Build a NEW policy for `target_env` and init it from the current policy
        (or from a checkpoint). Returns the new policy.

        - If `old_share_params_actor` is None, we auto-detect by trying both.
        """
        n_agents_new = target_env.n_agents

        # 1) Source (old) backbone to copy FROM
        if checkpoint_path is not None:
            # we need a temp policy to host the checkpoint (to get a backbone)
            # first guess non-shared (your IPPO default); fallback to shared
            tried = []
            errors = []

            for guess_shared in (
                [False, True]
                if old_share_params_actor is None
                else [old_share_params_actor]
            ):
                try:
                    temp_old_policy, temp_old_backbone = self._build_policy_for_env(
                        target_env,
                        n_agents=self.n_agents,
                        share_params_actor=guess_shared,
                    )
                    ckpt = torch.load(checkpoint_path, map_location=self.device)
                    temp_old_policy.load_state_dict(ckpt, strict=False)
                    old_backbone = temp_old_backbone
                    break
                except Exception as e:
                    tried.append(guess_shared)
                    errors.append(repr(e))
                    if old_share_params_actor is not None:
                        raise
                    # else: will try the other setting
            else:
                # both attempts failed → surface a concise hint
                msg = (
                    "transfer_policy: failed to load checkpoint with both "
                    f"share_params_actor={tried}. Last error:\n{errors[-1]}"
                )
                raise RuntimeError(msg)
        else:
            # use current in-memory policy
            old_backbone = self._extract_actor_backbone(self.policy)

        # 2) Target (new) policy to copy TO
        new_policy, new_backbone = self._build_policy_for_env(
            target_env, n_agents=n_agents_new, share_params_actor=new_share_params_actor
        )

        # 3) Copy weights param-by-param (your robust copier)
        self._copy_actor_weights(old_backbone, new_backbone)

        # 4) Done
        return new_policy

    def train_algo(
        self,
        n_checkpoints: int = 10,
        train_csv_dir: str = None,
        policy_dir: str = None,
    ):
        assert n_checkpoints >= 2, "checkpoints must be at least 2"

        pbar = tqdm(total=self.n_iters, desc=f"training {self.algo_name}...")

        checkpoint_set = [
            int(round(i * (self.n_iters - 1) / (n_checkpoints - 1)))
            for i in range(n_checkpoints)
        ]

        if train_csv_dir is not None:
            # n_envs = int(self.frames_per_batch / self.n_iters)
            n_envs = self.collector.env.num_envs  # or store from configs["env"]

            train_metrics = {
                "rewards": np.zeros((n_envs, n_checkpoints, self.n_agents)),
                "eta": np.zeros((n_envs, n_checkpoints, self.n_agents)),
                "beta": np.zeros((n_envs, n_checkpoints, self.n_agents)),
                "collisions": np.zeros((n_envs, n_checkpoints, self.n_agents)),
            }

        t = 0
        for it, data in enumerate(self.collector):
            # expand done / terminated to agent dim
            for key in ("done", "terminated"):
                data.set(
                    ("next", "agents", key),
                    data.get(("next", key))
                    .unsqueeze(-1)
                    .expand(data.get_item_shape(("next", self.env_reward_key))),
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
            for _ in range(self.n_epochs):
                for _ in range(self.frames_per_batch // self.minibatch_size):
                    batch = self.replay_buffer.sample()
                    losses = self.loss_module(batch)
                    loss = sum(losses.values())
                    loss.backward()

                    # if any grad is NaN/Inf, skip this optimizer step
                    grads_finite = True
                    for p in self.loss_module.parameters():
                        if p.grad is not None and not torch.isfinite(p.grad).all():
                            grads_finite = False
                            break

                    loss_finite = torch.isfinite(loss)

                    if grads_finite and loss_finite:
                        self.optimizer.step()
                    else:
                        # skip update on bad numerics
                        # (optional) you can log a warning counter here
                        pass

                    self.optimizer.zero_grad(set_to_none=True)

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

            # checkpoint?
            if it in checkpoint_set:

                if train_csv_dir is not None:
                    self._update_train_metrics(data, t, train_metrics)
                    t += 1

                if policy_dir is not None:
                    # save policy
                    policy_file = policy_dir / f"{self.algo_name}_chkpt_{it}.pt"
                    policy_file.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(self.policy.state_dict(), policy_file)

            pbar.set_postfix(team_mean=team_mean)
            pbar.update()

        rew_np = train_metrics["rewards"]
        eta_np = train_metrics["eta"]
        beta_np = train_metrics["beta"]
        collisions_np = train_metrics["collisions"]

        train_csv_path = train_csv_dir / f"{self.algo_name}_train.csv"
        save_csv(
            train_csv_path,
            self.n_agents,
            checkpoint_set,
            rew_np,
            eta_np,
            beta_np,
            collisions_np,
        )

        return checkpoint_set

    @staticmethod
    def _update_train_metrics(data, it, train_metrics):
        # ——— log training metrics ———
        ep_rew = data.get(("next", "agents", "episode_reward"))
        done_mask = data.get(("next", "agents", "done"))  # [n_envs, T, n_agents]

        eta_td = data.get(("next", "agents", "info", "eta"), None)
        beta_td = data.get(("next", "agents", "info", "beta"), None)
        coll_td = data.get(("next", "agents", "info", "n_collisions"), None)

        # Helper: squeeze last dim if size-1
        def _squeeze(t: torch.Tensor) -> torch.Tensor:
            if t.dim() >= 3 and t.size(-1) == 1:
                t = t.squeeze(-1)
            return t

        # Mean over steps with masking
        def _mean_over_steps(
            t: torch.Tensor, mask: torch.Tensor | None = None
        ) -> torch.Tensor:
            t = _squeeze(t)
            if mask is not None:
                mask = _squeeze(mask)
                t = t * mask
                denom = mask.sum(dim=1).clamp(min=1)  # avoid div by 0
                return t.sum(dim=1) / denom
            return t.mean(dim=1)

        # Sum over steps with masking (for collisions)
        def _sum_over_steps(
            t: torch.Tensor, mask: torch.Tensor | None = None
        ) -> torch.Tensor:
            t = _squeeze(t)
            if mask is not None:
                mask = _squeeze(mask)
                t = t * mask
            return t.sum(dim=1)

        # Rewards (always present)
        m = _mean_over_steps(ep_rew, done_mask)  # [n_envs, n_agents]
        train_metrics["rewards"][:, it, :] = m.detach().cpu().numpy()

        # Optional metrics
        if eta_td is not None:
            m = _mean_over_steps(eta_td, done_mask)
            train_metrics["eta"][:, it, :] = m.detach().cpu().numpy()

        if beta_td is not None:
            m = _mean_over_steps(beta_td, done_mask)
            train_metrics["beta"][:, it, :] = m.detach().cpu().numpy()

        if coll_td is not None:
            m = _sum_over_steps(coll_td, done_mask)  # <── sum, not mean
            train_metrics["collisions"][:, it, :] = m.detach().cpu().numpy()
