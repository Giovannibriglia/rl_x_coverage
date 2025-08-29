from typing import Callable, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from torch import Tensor
from torch.distributions import MultivariateNormal

from torchrl.envs import TransformedEnv

from vmas import render_interactively
from vmas.simulator.core import Agent, Box, Entity, Landmark, Line, Sphere, World
from vmas.simulator.heuristic_policy import BaseHeuristicPolicy
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.sensors import Lidar
from vmas.simulator.utils import Color, ScenarioUtils, X, Y


class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        self.n_agents = kwargs.pop("n_agents", 1)
        self.shared_rew = kwargs.pop("shared_rew", False)

        self.comms_range = kwargs.pop("comms_range", 0.0)
        self.lidar_range = kwargs.pop("lidar_range", 0.2)
        self.agent_radius = kwargs.pop("agent_radius", 0.025)
        self.xdim = kwargs.pop("xdim", 1)
        self.ydim = kwargs.pop("ydim", 1)
        self.grid_spacing = kwargs.pop("grid_spacing", 0.05)

        self.min_collision_distance = kwargs.pop("min_collision_distance", 0.05)
        self.agent_collision_penalty = kwargs.pop("agent_collision_penalty", -0.05)
        self.n_obstacles = kwargs.pop("n_obstacles", 0)
        self.L_env = kwargs.pop("L_env", False)
        self.n_gaussians = kwargs.pop("n_gaussians", 3)
        self.cov = kwargs.pop("cov", 0.1)
        self.collisions = kwargs.pop("collisions", True)
        self.spawn_same_pos = kwargs.pop("spawn_same_pos", False)
        self.norm = kwargs.pop("norm", True)
        self.dynamic = kwargs.pop("dynamic", False)

        self.n_collisions = torch.zeros(
            batch_dim, self.n_agents, dtype=torch.long, device=device
        )
        self._min_dist_between_entities = kwargs.pop(
            "min_dist_between_entities", self.agent_radius * 2 + 0.05
        )
        self.last_centroid = torch.zeros((batch_dim, self.n_agents, 2), device=device)

        self.angle_start = kwargs.pop("angle_start", 0.05)
        self.angle_end = kwargs.pop("angle_end", 2 * torch.pi + 0.05)
        self.n_rays = kwargs.pop("n_rays", 50)
        self.cells_range = kwargs.pop(
            "cells_range", 3
        )  # number of cells sensed on each side
        self.centralized = kwargs.pop("centralized", False)
        ScenarioUtils.check_kwargs_consumed(kwargs)

        assert not (self.spawn_same_pos and self.collisions)
        assert (self.xdim / self.grid_spacing) % 1 == 0 and (
            self.ydim / self.grid_spacing
        ) % 1 == 0
        self.covs = (
            [self.cov] * self.n_gaussians if isinstance(self.cov, float) else self.cov
        )
        assert len(self.covs) == self.n_gaussians

        self.plot_grid = False
        self.visualize_semidims = False
        self.n_x_cells = int((2 * self.xdim) / self.grid_spacing)
        self.n_y_cells = int((2 * self.ydim) / self.grid_spacing)
        self.max_pdf = torch.zeros((batch_dim,), device=device, dtype=torch.float32)
        self.alpha_plot: float = 0.5

        self.agent_xspawn_range = 0 if self.spawn_same_pos else self.xdim
        self.agent_yspawn_range = 0 if self.spawn_same_pos else self.ydim
        self.x_semidim = self.xdim - self.agent_radius
        self.y_semidim = self.ydim - self.agent_radius

        self.steps = 0

        self.pdf = [None] * batch_dim
        self.Kp = 0.8  # proportional gain

        self.voronoi = VoronoiCoverage(
            self.grid_spacing,
            self.cells_range,
            self.xdim,
            self.ydim,
            device,
            self.centralized,
        )

        # Make world
        world = World(
            batch_dim,
            device,
            x_semidim=self.x_semidim,
            y_semidim=self.y_semidim,
        )
        # entity_filter_agents: Callable[[Entity], bool] = lambda e: isinstance(e, Agent)
        entity_filter_agents: Callable[[Entity], bool] = lambda e: isinstance(
            e, (Agent, Landmark)
        )
        for i in range(self.n_agents):
            agent = Agent(
                name=f"agent_{i}",
                render_action=True,
                collide=self.collisions,
                shape=Sphere(radius=self.agent_radius),
                sensors=(
                    [
                        Lidar(
                            world,
                            angle_start=self.angle_start,
                            angle_end=self.angle_end,
                            n_rays=self.n_rays,
                            max_range=self.lidar_range,
                            entity_filter=entity_filter_agents,
                        ),
                    ]
                    if self.collisions
                    else None
                ),
            )

            world.add_agent(agent)

        self.sampled = torch.zeros(
            (batch_dim, self.n_x_cells, self.n_y_cells),
            device=device,
            dtype=torch.bool,
        )

        self.locs = [
            torch.zeros((batch_dim, world.dim_p), device=device, dtype=torch.float32)
            for _ in range(self.n_gaussians)
        ]
        self.cov_matrices = [
            torch.tensor(
                [[cov, 0], [0, cov]], dtype=torch.float32, device=device
            ).expand(batch_dim, world.dim_p, world.dim_p)
            for cov in self.covs
        ]

        # Add landmarks
        self.obstacles = []
        for i in range(self.n_obstacles):
            obstacle = Landmark(
                name=f"obstacle_{i}",
                collide=True,
                movable=False,
                shape=Sphere(radius=0.1),
                color=Color.RED,
            )
            world.add_landmark(obstacle)
            self.obstacles.append(obstacle)

        # set obstacle in a corner to make the env non-convex (L-shaped)
        if self.L_env:
            obstacle = Landmark(
                name=f"obstacle_{i}",
                collide=True,
                movable=False,
                shape=Box(length=1.0, width=1.0),
                color=Color.RED,
            )
            world.add_landmark(obstacle)
            self.obstacles.append(obstacle)

        x_grid = torch.linspace(-self.xdim, self.xdim, self.n_x_cells)
        y_grid = torch.linspace(-self.ydim, self.ydim, self.n_y_cells)
        xg, yg = torch.meshgrid(x_grid, y_grid)
        self.xy_grid = torch.vstack((xg.ravel(), yg.ravel())).T.to(world.device)

        return world

    def reset_world_at(self, env_index: int = None):
        for i in range(len(self.locs)):
            x = torch.zeros(
                (1,) if env_index is not None else (self.world.batch_dim, 1),
                device=self.world.device,
                dtype=torch.float32,
            ).uniform_(-self.xdim, self.xdim)
            y = torch.zeros(
                (1,) if env_index is not None else (self.world.batch_dim, 1),
                device=self.world.device,
                dtype=torch.float32,
            ).uniform_(-self.ydim, self.ydim)
            new_loc = torch.cat([x, y], dim=-1)
            # new_loc = torch.tensor([0.0, 0.0]).to(self.world.device)
            if env_index is None:
                self.locs[i] = new_loc
            else:
                self.locs[i][env_index] = new_loc

        self.gaussians = [
            MultivariateNormal(
                loc=loc,
                covariance_matrix=cov_matrix,
            )
            for loc, cov_matrix in zip(self.locs, self.cov_matrices)
        ]

        # xy_grid = xy_grid.unsqueeze(0).expand(self.world.batch_dim, -1, -1)
        self.pdf = [
            self.sample_single_env(self.xy_grid, i) for i in range(self.world.batch_dim)
        ]

        if env_index is None:
            self.max_pdf[:] = 0
            self.sampled[:] = False
            self.n_collisions.zero_()
        else:
            self.max_pdf[env_index] = 0
            self.sampled[env_index] = False
            self.n_collisions[env_index].zero_()
        self.normalize_pdf(env_index=env_index)

        # random obstacles
        ScenarioUtils.spawn_entities_randomly(
            self.obstacles,
            self.world,
            env_index,
            self._min_dist_between_entities,
            x_bounds=(-self.xdim, self.xdim),
            y_bounds=(-self.ydim, self.ydim),
            # occupied_positions=target_pos.unsqueeze(1),
        )

        for agent in self.world.agents:
            agent.set_pos(
                torch.cat(
                    [
                        torch.zeros(
                            (
                                (1, 1)
                                if env_index is not None
                                else (self.world.batch_dim, 1)
                            ),
                            device=self.world.device,
                            dtype=torch.float32,
                        ).uniform_(-self.agent_xspawn_range, self.agent_xspawn_range),
                        torch.zeros(
                            (
                                (1, 1)
                                if env_index is not None
                                else (self.world.batch_dim, 1)
                            ),
                            device=self.world.device,
                            dtype=torch.float32,
                        ).uniform_(-self.agent_yspawn_range, self.agent_yspawn_range),
                    ],
                    dim=-1,
                ),
                batch_index=env_index,
            )
            agent.sample = self.sample(agent.state.pos, norm=self.norm)

    def sample(
        self,
        pos,
        update_sampled_flag: bool = False,
        norm: bool = True,
    ):
        out_of_bounds = (
            (pos[:, X] < -self.xdim)
            + (pos[:, X] > self.xdim)
            + (pos[:, Y] < -self.ydim)
            + (pos[:, Y] > self.ydim)
        )
        pos[:, X].clamp_(-self.world.x_semidim, self.world.x_semidim)
        pos[:, Y].clamp_(-self.world.y_semidim, self.world.y_semidim)

        index = pos / self.grid_spacing
        index[:, X] += self.n_x_cells / 2
        index[:, Y] += self.n_y_cells / 2
        index = index.to(torch.long)
        v = torch.stack(
            [gaussian.log_prob(pos).exp() for gaussian in self.gaussians],
            dim=-1,
        ).sum(-1)
        if norm:
            v = v / (self.max_pdf + 1e-8)

        sampled = self.sampled[
            torch.arange(self.world.batch_dim), index[:, 0], index[:, 1]
        ]

        v[sampled + out_of_bounds] = 0
        if update_sampled_flag:
            self.sampled[
                torch.arange(self.world.batch_dim), index[:, 0], index[:, 1]
            ] = True

        return v

    def sample_single_env(
        self,
        pos,
        env_index,
        norm: bool = True,
    ):
        pos = pos.view(-1, self.world.dim_p)

        out_of_bounds = (
            (pos[:, X] < -self.xdim)
            + (pos[:, X] > self.xdim)
            + (pos[:, Y] < -self.ydim)
            + (pos[:, Y] > self.ydim)
        )
        pos[:, X].clamp_(-self.x_semidim, self.x_semidim)
        pos[:, Y].clamp_(-self.y_semidim, self.y_semidim)

        index = pos / self.grid_spacing
        index[:, X] += self.n_x_cells / 2
        index[:, Y] += self.n_y_cells / 2
        index = index.to(torch.long)

        pos = pos.unsqueeze(1).expand(pos.shape[0], self.world.batch_dim, 2)

        v = torch.stack(
            [gaussian.log_prob(pos).exp() for gaussian in self.gaussians],
            dim=-1,
        ).sum(-1)[:, env_index]
        if norm:
            v = v / (self.max_pdf[env_index] + 1e-8)

        sampled = self.sampled[env_index, index[:, 0], index[:, 1]]

        v[sampled + out_of_bounds] = 0

        return v

    def normalize_pdf(self, env_index: int = None):
        xpoints = torch.arange(
            -self.xdim, self.xdim, self.grid_spacing, device=self.world.device
        )
        ypoints = torch.arange(
            -self.ydim, self.ydim, self.grid_spacing, device=self.world.device
        )
        if env_index is not None:
            ygrid, xgrid = torch.meshgrid(ypoints, xpoints, indexing="ij")
            pos = torch.stack((xgrid, ygrid), dim=-1).reshape(-1, 2)
            sample = self.sample_single_env(pos, env_index, norm=False)
            self.max_pdf[env_index] = sample.max()
        else:
            for x in xpoints:
                for y in ypoints:
                    pos = torch.tensor(
                        [x, y], device=self.world.device, dtype=torch.float32
                    ).repeat(self.world.batch_dim, 1)
                    sample = self.sample(pos, norm=False)
                    self.max_pdf = torch.maximum(self.max_pdf, sample)

    """def reward(self, agent: Agent) -> torch.Tensor:
        # reward basica
        # Step counter maintained once per env-step
        if self.world.agents.index(agent) == 0:
            self.steps += 1

        device = self.world.device
        B = self.world.batch_dim
        N = self.n_agents
        P = self.xy_grid.shape[0]  # #grid points

        # Agents' positions: [B, N, 2]
        pos_agents = torch.stack(
            [ag.state.pos for ag in self.world.agents], dim=1
        )  # [B,N,2]

        # Global density per env on the same grid: [B, P]
        phi = torch.stack(self.pdf, dim=0)
        phi = torch.nan_to_num(phi, nan=0.0, posinf=0.0, neginf=0.0)

        # Pairwise sq distances grid <-> agents: [B,N,P]
        diff = self.xy_grid.view(1, 1, P, 2) - pos_agents.unsqueeze(2)  # [B,N,P,2]
        d2 = (diff**2).sum(-1)  # [B,N,P]

        # Nearest agent per grid point (Voronoi assignment)
        nearest = d2.argmin(dim=1)  # [B,P]

        # Index of current agent
        idx = self.world.agents.index(agent)

        # -------- Coverage term (≤ 0, best is 0) --------
        # Individual (non-shared): integrate φ * 1{nearest==idx} * ||x - p_idx||² * dA
        mask_agent = nearest == idx  # [B,P]
        d2_agent = d2[:, idx, :]  # [B,P]
        cost_agent = (mask_agent.float() * phi * d2_agent).sum(-1) * (
            self.grid_spacing**2
        )  # [B]
        pos_rew_agent = -cost_agent  # [B] ≤ 0

        # Shared: team uses min-distance field directly
        d2_min = d2.gather(dim=1, index=nearest.unsqueeze(1)).squeeze(1)  # [B,P]
        cost_team = (phi * d2_min).sum(-1) * (self.grid_spacing**2)  # [B]
        pos_rew_team = -cost_team  # [B] ≤ 0

        # -------- Collision penalties (vectorized) --------
        pdiff = pos_agents[:, :, None, :] - pos_agents[:, None, :, :]  # [B,N,N,2]
        pd2 = (pdiff**2).sum(-1)  # [B,N,N]
        eye = torch.eye(N, dtype=torch.bool, device=device).unsqueeze(0)  # [1,N,N]
        coll_mask = (pd2 <= (self.min_collision_distance**2)) & ~eye  # [B,N,N]

        # #collisions seen by each agent (neighbors within thresh)
        collisions_per_agent = coll_mask.sum(dim=-1)  # [B,N]

        # Update persistent per-agent counters exactly once per step
        if idx == 0:
            self.n_collisions += collisions_per_agent.to(torch.long)

        # Penalties
        pen_per_agent = (
            collisions_per_agent[:, idx].float() * self.agent_collision_penalty
        )  # [B]
        pen_team = (
            collisions_per_agent.sum(dim=-1).float() * self.agent_collision_penalty
        )  # [B]

        # Final per-agent reward
        if self.shared_rew:
            return pos_rew_team + pen_team  # same team reward for all agents
        else:
            return pos_rew_agent + pen_per_agent"""

    def reward(self, agent: Agent) -> torch.Tensor:
        # reward pesata meglio
        # step counter as before
        if self.world.agents.index(agent) == 0:
            self.steps += 1

        device = self.world.device
        # B = self.world.batch_dim
        N = self.n_agents
        P = self.xy_grid.shape[0]
        dA = self.grid_spacing**2

        # Positions [B,N,2]
        pos_agents = torch.stack(
            [ag.state.pos for ag in self.world.agents], dim=1
        )  # [B,N,2]

        # φ over grid [B,P]
        phi = torch.stack(self.pdf, dim=0)
        phi = torch.nan_to_num(phi, nan=0.0, posinf=0.0, neginf=0.0)

        # --- Voronoi assignments via nearest agent (vectorized) ---
        diff = self.xy_grid.view(1, 1, P, 2) - pos_agents.unsqueeze(2)  # [B,N,P,2]
        d2 = (diff**2).sum(-1)  # [B,N,P]
        nearest = d2.argmin(dim=1)  # [B,P]
        mask = (
            F.one_hot(nearest, num_classes=N).permute(0, 2, 1).to(phi.dtype)
        )  # [B,N,P]

        # --- Cell weights and centroids (φ-weighted) ---
        w = mask * phi.unsqueeze(1)  # [B,N,P]
        W = w.sum(dim=2) * dA  # [B,N]   total weight (with area)
        num = w.unsqueeze(-1) * self.xy_grid  # [B,N,P,2]
        num = num.sum(dim=2) * dA  # [B,N,2]
        centroids = torch.where(
            (W > 0).unsqueeze(-1), num / (W.unsqueeze(-1) + 1e-8), pos_agents
        )  # [B,N,2]

        # --- Centered cost: W_i * ||p_i - c_i||^2  (zero at optimum) ---
        d2_pc = ((pos_agents - centroids) ** 2).sum(-1)  # [B,N]

        # Per-agent centered reward (≤ 0)
        centered_rewards = -(W * d2_pc)  # [B,N]

        # Index of the current agent
        idx = self.world.agents.index(agent)
        pos_rew_agent = centered_rewards[:, idx]  # [B]
        pos_rew_team = centered_rewards.sum(dim=1)  # [B]  (team = sum over agents)

        # --- Collision penalty (unchanged, vectorized) ---
        pdiff = pos_agents[:, :, None, :] - pos_agents[:, None, :, :]  # [B,N,N,2]
        pd2 = (pdiff**2).sum(-1)  # [B,N,N]
        eye = torch.eye(N, dtype=torch.bool, device=device).unsqueeze(0)
        coll_mask = (pd2 <= (self.min_collision_distance**2)) & ~eye  # [B,N,N]

        collisions_per_agent = coll_mask.sum(dim=-1)  # [B,N]
        if idx == 0:
            self.n_collisions += collisions_per_agent.to(torch.long)

        pen_per_agent = (
            collisions_per_agent[:, idx].float() * self.agent_collision_penalty
        )  # [B]
        pen_team = (
            collisions_per_agent.sum(dim=-1).float() * self.agent_collision_penalty
        )  # [B]

        # --- Final ---
        if self.shared_rew:
            return pos_rew_team + pen_team
        else:
            return pos_rew_agent + pen_per_agent

    def observation(self, agent: Agent) -> Tensor:
        if self.dynamic and self.steps % 100 == 0.0:
            for env_index in range(self.world.batch_dim):
                for i in range(len(self.locs)):
                    x = torch.zeros(
                        (1,) if env_index is not None else (self.world.batch_dim, 1),
                        device=self.world.device,
                        dtype=torch.float32,
                    ).uniform_(-self.xdim, self.xdim)
                    y = torch.zeros(
                        (1,) if env_index is not None else (self.world.batch_dim, 1),
                        device=self.world.device,
                        dtype=torch.float32,
                    ).uniform_(-self.ydim, self.ydim)
                    new_loc = torch.cat([x, y], dim=-1)
                    # new_loc = torch.tensor([0.0, 0.0]).to(self.world.device)
                    if env_index is None:
                        self.locs[i] = new_loc
                    else:
                        self.locs[i][env_index] = new_loc

                self.gaussians = [
                    MultivariateNormal(
                        loc=loc,
                        covariance_matrix=cov_matrix,
                    )
                    for loc, cov_matrix in zip(self.locs, self.cov_matrices)
                ]

            # xy_grid = xy_grid.unsqueeze(0).expand(self.world.batch_dim, -1, -1)
            self.pdf = [
                self.sample_single_env(self.xy_grid, env)
                for env in range(self.world.batch_dim)
            ]

            if env_index is None:
                self.max_pdf[:] = 0
                self.sampled[:] = False
            else:
                self.max_pdf[env_index] = 0
                self.sampled[env_index] = False
            self.normalize_pdf(env_index=env_index)

        observations = [
            agent.state.pos,
            agent.state.vel,
            agent.sensors[0].measure(),
        ]

        # deltas = [[0, 0],
        #     [-self.grid_spacing, -self.grid_spacing],
        #     [0, -self.grid_spacing],
        #     [self.grid_spacing, -self.grid_spacing],
        #     [self.grid_spacing, 0],
        #     [self.grid_spacing, self.grid_spacing],
        #     [0, self.grid_spacing],
        #     [-self.grid_spacing, self.grid_spacing],
        #     [-self.grid_spacing, 0]]

        if not self.centralized:
            deltas = []
            for i in range(-self.cells_range, self.cells_range + 1):
                for j in range(-self.cells_range, self.cells_range + 1):
                    deltas.append([i * self.grid_spacing, j * self.grid_spacing])

            for delta in deltas:
                # occupied cell + ccw cells from bottom left
                pos = agent.state.pos + torch.tensor(
                    delta,
                    device=self.world.device,
                    dtype=torch.float32,
                )
                sample = self.sample(
                    pos,
                    update_sampled_flag=False,
                ).unsqueeze(-1)
                observations.append(sample)
        else:
            for x in np.linspace(-self.xdim, self.xdim, self.n_x_cells):
                for y in np.linspace(-self.ydim, self.ydim, self.n_y_cells):
                    xy = torch.tensor(
                        [[x, y]],
                        device=self.world.device,
                        dtype=torch.float32,
                    )
                    sample = self.sample(
                        xy,
                        update_sampled_flag=False,
                    ).unsqueeze(-1)
                    observations.append(sample)

        return torch.cat(
            observations,
            dim=-1,
        )

    def info(self, agent: Agent) -> Dict[str, Tensor]:
        eta, beta = self._get_coverage_metrics()
        idx = self.world.agents.index(agent)
        return {
            "agent_sample": agent.sample,
            "eta": eta,
            "beta": beta,
            "n_collisions": self.n_collisions[:, idx],  # shape [B]
        }

    def _get_coverage_metrics(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        eta  : torch.Tensor  # shape [num_envs] – coverage effectiveness in [0,1]
        beta : torch.Tensor  # shape [num_envs] – area collectively covered [m²]
        """
        # ------------------------------------------------------------------
        # constants and cached grids
        # ------------------------------------------------------------------
        r = self.lidar_range  # sensing radius
        dA = self.grid_spacing**2  # area of one grid cell
        xy = self.xy_grid  # [P,2] grid points (device‑correct)

        # every agent position → [B, N, 2]
        pos_agents = torch.stack([ag.state.pos for ag in self.world.agents], dim=1)

        # ------------------------------------------------------------------
        # Boolean mask: cell is inside *any* agent disk
        # ------------------------------------------------------------------
        diff = xy.unsqueeze(0).unsqueeze(0) - pos_agents.unsqueeze(2)  # [B,N,P,2]
        dist2 = (diff**2).sum(-1)  # [B,N,P]
        covered_any = (dist2 <= r**2).any(dim=1)  # [B,P]

        # ------------------------------------------------------------------
        # β (area)  – one value per environment
        # ------------------------------------------------------------------
        beta = covered_any.float().sum(dim=-1) * dA  # [B]

        # ------------------------------------------------------------------
        # η (effectiveness)  – one value per environment
        # ------------------------------------------------------------------
        # stack pre‑computed PDFs  → [B,P]   (they might contain NaNs if
        # max_pdf was zero somewhere during normalisation!)
        phi = torch.stack(self.pdf, dim=0)

        # Replace non‑finite entries with 0
        phi = torch.nan_to_num(phi, nan=0.0, posinf=1, neginf=0.0)

        num = (phi * covered_any.float()).sum(dim=-1)  # ∑_B φ
        den = phi.sum(dim=-1)  # ∑_Q φ

        # Avoid divide‑by‑zero – if den==0 just set η:=0
        eta = torch.where(den > 0, num / den, torch.zeros_like(den))

        return eta, beta

    def density_for_plot(self, env_index):
        def f(x):
            sample = self.sample_single_env(
                torch.tensor(x, dtype=torch.float32, device=self.world.device),
                env_index=env_index,
            )

            return sample

        return f

    def extra_render(self, env_index: int = 0):
        from vmas.simulator import rendering
        from vmas.simulator.rendering import render_function_util

        # Function
        geoms = [
            render_function_util(
                f=self.density_for_plot(env_index=env_index),
                plot_range=(self.xdim, self.ydim),
                cmap_alpha=self.alpha_plot,
                cmap_name="plasma",
            )
        ]

        try:
            # Compute Voronoi regions
            vor = self._compute_voronoi_regions(env_index)
            geoms = self._plot_voronoi_regions(vor, geoms)
        except Exception:
            pass
            # print(f"Unable to compute and plot voronoi regions: {e}")

        # Communication lines
        for i, agent1 in enumerate(self.world.agents):
            for j, agent2 in enumerate(self.world.agents):
                if j <= i:
                    continue
                agent_dist = torch.linalg.vector_norm(
                    agent1.state.pos - agent2.state.pos, dim=-1
                )
                if agent_dist[env_index] <= self.comms_range:
                    color = Color.BLACK.value
                    line = rendering.Line(
                        (agent1.state.pos[env_index]),
                        (agent2.state.pos[env_index]),
                        width=1,
                    )
                    xform = rendering.Transform()
                    line.add_attr(xform)
                    line.set_color(*color)
                    geoms.append(line)

        # Perimeter
        for i in range(4):
            geom = Line(
                length=2
                * ((self.ydim if i % 2 == 0 else self.xdim) - self.agent_radius)
                + self.agent_radius * 2
            ).get_geometry()
            xform = rendering.Transform()
            geom.add_attr(xform)

            xform.set_translation(
                (
                    0.0
                    if i % 2
                    else (
                        self.x_semidim + self.agent_radius
                        if i == 0
                        else -self.x_semidim - self.agent_radius
                    )
                ),
                (
                    0.0
                    if not i % 2
                    else (
                        self.y_semidim + self.agent_radius
                        if i == 1
                        else -self.y_semidim - self.agent_radius
                    )
                ),
            )
            xform.set_rotation(torch.pi / 2 if not i % 2 else 0.0)
            color = Color.BLACK.value
            if isinstance(color, torch.Tensor) and len(color.shape) > 1:
                color = color[env_index]
            geom.set_color(*color)
            geoms.append(geom)

        return geoms

    def _compute_voronoi_regions(self, env_index):
        """
        Computes Voronoi regions based on agent positions for a specific environment.

        Args:
            env_index: Index of the environment.

        Returns:
            Voronoi object from scipy.spatial.
        """
        from scipy.spatial import Voronoi

        agent_positions = [
            agent.state.pos[env_index].cpu().numpy() for agent in self.world.agents
        ]
        points = np.array(agent_positions)

        # Compute Voronoi regions
        vor = Voronoi(points)
        return vor

    @staticmethod
    def _clip_line_to_bounds(p1, p2, x_bounds, y_bounds):
        """
        Clips a line segment to fit within the specified rectangular bounds.

        Args:
            p1, p2: Endpoints of the line segment as [x, y].
            x_bounds: Tuple of (x_min, x_max) for x-coordinates.
            y_bounds: Tuple of (y_min, y_max) for y-coordinates.

        Returns:
            Clipped line segment as a list of two points, or None if outside bounds.
        """
        from shapely.geometry import box, LineString

        bbox = box(x_bounds[0], y_bounds[0], x_bounds[1], y_bounds[1])
        line = LineString([p1, p2])
        clipped_line = line.intersection(bbox)

        if clipped_line.is_empty:
            return None
        elif clipped_line.geom_type == "LineString":
            return list(clipped_line.coords)
        else:
            return None

    def _plot_voronoi_regions(self, vor, geoms):
        """
        Plots Voronoi regions with finite and clipped infinite edges.

        Args:
            vor: Voronoi object from scipy.spatial.
            geoms: List of geometric shapes for rendering.

        Returns:
            Updated list of geometries including Voronoi regions.
        """
        from vmas.simulator.rendering import PolyLine

        x_min, x_max = -self.xdim, self.xdim
        y_min, y_max = -self.ydim, self.ydim
        ptp_bound = np.array([x_max - x_min, y_max - y_min])

        center = vor.points.mean(axis=0)
        finite_segments = []
        infinite_segments = []

        for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
            simplex = np.asarray(simplex)
            if np.all(simplex >= 0):
                finite_segments.append(vor.vertices[simplex])
            else:
                i = simplex[simplex >= 0][0]
                t = vor.points[pointidx[1]] - vor.points[pointidx[0]]
                t /= np.linalg.norm(t)
                n = np.array([-t[1], t[0]])
                midpoint = vor.points[pointidx].mean(axis=0)
                direction = np.sign(np.dot(midpoint - center, n)) * n
                far_point = vor.vertices[i] + direction * ptp_bound.max()
                infinite_segments.append([vor.vertices[i], far_point])

        # Render finite segments
        for segment in finite_segments:
            line = PolyLine(segment.tolist(), close=False)
            line.set_color(0.2, 0.8, 0.2)
            geoms.append(line)

        # Render clipped infinite segments
        for segment in infinite_segments:
            clipped_segment = self._clip_line_to_bounds(
                segment[0], segment[1], (x_min, x_max), (y_min, y_max)
            )
            if clipped_segment:
                line = PolyLine(clipped_segment, close=False)
                line.set_color(0.8, 0.2, 0.2)
                geoms.append(line)

        return geoms


class VoronoiPolicy(BaseHeuristicPolicy):
    def __init__(self, env, continuous_action: bool):
        super().__init__(continuous_action=continuous_action)
        self.env = env
        self.scenario = env.scenario
        self.device = env.world.device
        self.Kp = getattr(self.scenario, "Kp", 0.8)
        self.voronoi = self.scenario.voronoi

    @torch.no_grad()
    def compute_action(self, observation: torch.Tensor, u_range: float) -> torch.Tensor:
        """
        observation: [B, obs_dim] (first 2 entries are the agent's (x,y))
        u_range: float scalar (same range for x and y)
        returns: [B, 2] action for *this* agent across the batch, fully vectorized
        """
        # current agent's positions from its observation
        pos_self = observation[:, :2]  # [B,2]

        # all agents' true positions (needed to define Voronoi regions)
        pos_agents = torch.stack(
            [ag.state.pos for ag in self.scenario.world.agents], dim=1
        )  # [B,N,2]

        # identify which agent in the world this observation belongs to (per env)
        # (pick the index whose position matches pos_self)
        d2_self = ((pos_agents - pos_self.unsqueeze(1)) ** 2).sum(-1)  # [B,N]
        idx_self = d2_self.argmin(dim=1)  # [B]

        # global φ over the arena grid for each env
        phi = torch.stack(self.scenario.pdf, dim=0)  # [B,P]
        phi = torch.nan_to_num(phi, nan=0.0, posinf=0.0, neginf=0.0)

        # compute centroids for every agent in every env (implicit Voronoi via nearest-assignment)
        centroids = self.voronoi.centroids(phi=phi, pos_agents=pos_agents)  # [B,N,2]

        # gather centroid for THIS agent in each env
        gather_idx = idx_self.view(-1, 1, 1).expand(-1, 1, 2)  # [B,1,2]
        centroid_self = torch.gather(centroids, dim=1, index=gather_idx).squeeze(
            1
        )  # [B,2]

        # proportional control toward centroid, clamped by u_range
        action = self.Kp * (centroid_self - pos_self)  # [B,2]
        action = torch.clamp(action, min=-u_range, max=u_range)
        return action

    # (Optional) fully-batched helper to compute actions for *all* agents at once (no Python loops).
    @torch.no_grad()
    def compute_actions_all(self) -> torch.Tensor:
        """
        returns: [B, N, 2] actions for all agents at once (vectorized)
        """
        pos_agents = torch.stack(
            [ag.state.pos for ag in self.scenario.world.agents], dim=1
        )  # [B,N,2]
        phi = torch.stack(self.scenario.pdf, dim=0)
        phi = torch.nan_to_num(phi, nan=0.0, posinf=0.0, neginf=0.0)
        centroids = self.voronoi.centroids(phi=phi, pos_agents=pos_agents)  # [B,N,2]
        actions = self.Kp * (centroids - pos_agents)  # [B,N,2]

        # clamp using each agent's u_range
        u_ranges = torch.tensor(
            [ag.u_range for ag in self.scenario.world.agents],
            device=self.device,
            dtype=actions.dtype,
        )  # [N]
        actions = torch.clamp(
            actions, min=-u_ranges.view(1, -1, 1), max=u_ranges.view(1, -1, 1)
        )
        return actions


class VoronoiBasedActor(torch.nn.Module):
    def __init__(self, env: TransformedEnv, continuous_actions: bool = True):
        super().__init__()
        self.env = env
        self.heuristic = VoronoiPolicy(
            env=env.base_env, continuous_action=continuous_actions
        )
        self.n_agents = env.n_agents

    def forward(self, td):
        # Compute all agents' actions in one go (ignore obs; use true positions + global φ)
        acts = self.heuristic.compute_actions_all()  # [B,N,2]
        td.set(("agents", "action"), acts.to(td.device))
        return td


class VoronoiCoverage:
    """
    Vectorized helpers for coverage with implicit Voronoi via nearest-agent assignment.
    Works with the global grid (xy_grid) and per-env PDF φ (stacked as [B,P]).
    """

    def __init__(
        self,
        grid_spacing,
        cells_range,
        xdim=10,
        ydim=10,
        device="cpu",
        centralized=True,
    ):
        self.centralized = centralized
        self.grid_spacing = float(grid_spacing)
        self.cells_range = int(cells_range)
        self.device = device

        self.xmin = -float(xdim)
        self.xmax = float(xdim)
        self.ymin = -float(ydim)
        self.ymax = float(ydim)

        # Full global grid (for centralized mode or general use)
        nxcells = int((self.xmax - self.xmin) / self.grid_spacing)
        nycells = int((self.ymax - self.ymin) / self.grid_spacing)
        xg = torch.linspace(self.xmin, self.xmax, nxcells, device=device)
        yg = torch.linspace(self.ymin, self.ymax, nycells, device=device)
        X, Y = torch.meshgrid(xg, yg, indexing="xy")
        self.xy_grid = torch.stack((X.reshape(-1), Y.reshape(-1)), dim=-1)  # [P,2]

    @torch.no_grad()
    def nearest_assignments(self, pos_agents: torch.Tensor) -> torch.Tensor:
        """
        pos_agents: [B, N, 2]
        Returns: assignments [B, P] with values in [0..N-1]
        """
        B, N, _ = pos_agents.shape
        P = self.xy_grid.shape[0]
        diff = self.xy_grid.view(1, 1, P, 2) - pos_agents.unsqueeze(2)  # [B,N,P,2]
        d2 = (diff**2).sum(-1)  # [B,N,P]
        return d2.argmin(dim=1)  # [B,P]

    @torch.no_grad()
    def centroids(self, phi: torch.Tensor, pos_agents: torch.Tensor) -> torch.Tensor:
        """
        phi:        [B, P]   non-negative weights per grid cell (can be unnormalized)
        pos_agents: [B, N, 2]
        Returns:    [B, N, 2] weighted centroids of each agent's Voronoi region
        """
        B, N, _ = pos_agents.shape
        # P = self.xy_grid.shape[0]

        # Assign each grid cell to its nearest agent
        assignments = self.nearest_assignments(pos_agents)  # [B,P]
        # One-hot mask per agent: [B,N,P]
        mask = F.one_hot(assignments, num_classes=N).permute(0, 2, 1).to(phi.dtype)

        # Weighted masks: φ * mask
        w = phi.unsqueeze(1) * mask  # [B,N,P]

        # Numerator: sum_p w * x_p , sum_p w * y_p
        xy = self.xy_grid  # [P,2]
        num = w.unsqueeze(-1) * xy  # [B,N,P,2]
        num = num.sum(dim=2)  # [B,N,2]

        # Denominator: sum_p w
        den = w.sum(dim=2, keepdim=True)  # [B,N,1]

        # Avoid divide by zero; fall back to current positions when region empty
        safe = den > 0
        centroids = torch.where(safe, num / (den + 1e-8), pos_agents)  # [B,N,2]
        return centroids

    @torch.no_grad()
    def coverage_costs(
        self, phi: torch.Tensor, pos_agents: torch.Tensor
    ) -> torch.Tensor:
        """
        φ-weighted quadratic costs per agent (non-shared), integrated over its Voronoi cell.
        Returns: [B, N] costs (non-negative). Reward would be -costs.
        """
        B, N, _ = pos_agents.shape
        P = self.xy_grid.shape[0]
        diff = self.xy_grid.view(1, 1, P, 2) - pos_agents.unsqueeze(2)  # [B,N,P,2]
        d2 = (diff**2).sum(-1)  # [B,N,P]
        assignments = d2.argmin(dim=1)  # [B,P]
        mask = (
            F.one_hot(assignments, num_classes=N).permute(0, 2, 1).to(phi.dtype)
        )  # [B,N,P]
        costs = (mask * phi.unsqueeze(1) * d2).sum(dim=2) * (
            self.grid_spacing**2
        )  # [B,N]
        return costs

    @torch.no_grad()
    def team_cost(self, phi: torch.Tensor, pos_agents: torch.Tensor) -> torch.Tensor:
        """
        Team cost using min-distance field: [B], non-negative.
        """
        P = self.xy_grid.shape[0]
        diff = self.xy_grid.view(1, 1, P, 2) - pos_agents.unsqueeze(2)  # [B,N,P,2]
        d2 = (diff**2).sum(-1)  # [B,N,P]
        nearest = d2.min(dim=1).values  # [B,P]
        return (phi * nearest).sum(dim=1) * (self.grid_spacing**2)  # [B]


if __name__ == "__main__":
    render_interactively(__file__, control_two_agents=True)
