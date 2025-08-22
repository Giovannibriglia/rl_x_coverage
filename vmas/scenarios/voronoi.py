from typing import Callable, Dict, Tuple

import numpy as np
import torch
from matplotlib.path import Path

from scipy.spatial import Voronoi
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
        self.agent_collision_penalty = kwargs.pop("agent_collision_penalty", -0.01)
        self.n_obstacles = kwargs.pop("n_obstacles", 0)
        self.L_env = kwargs.pop("L_env", False)
        self.n_gaussians = kwargs.pop("n_gaussians", 3)
        self.cov = kwargs.pop("cov", 0.1)
        self.collisions = kwargs.pop("collisions", True)
        self.spawn_same_pos = kwargs.pop("spawn_same_pos", False)
        self.norm = kwargs.pop("norm", True)
        self.dynamic = kwargs.pop("dynamic", False)

        self.n_collisions = torch.zeros(batch_dim, device=device)
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
        else:
            self.max_pdf[env_index] = 0
            self.sampled[env_index] = False
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
            v = v / self.max_pdf

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
            v = v / self.max_pdf[env_index]

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

    def reward(self, agent: Agent) -> Tensor:
        if self.world.agents.index(agent) == 0:
            self.steps += 1

        """if eval:
            return torch.zeros(self.world.batch_dim)
        else:"""
        observation = self.observation(
            agent
        )  # pos, vel, lidar, pdf(centralized or decentralized)
        # extract info from observation
        pdf = observation[:, (4 + self.n_rays) :]
        pos = observation[:, :2]
        # vel = observation[:, 2:4]
        lidar_values = observation[:, 4 : 4 + self.n_rays]  # [n_envs, n_rays]

        # get detected robots relative positions
        angles = torch.linspace(
            self.angle_start,
            self.angle_end,
            lidar_values.shape[1],
            device=self.world.device,
        )
        x = lidar_values * torch.cos(angles)
        y = lidar_values * torch.sin(angles)
        robots_rel = torch.stack((x, y), dim=-1)  # [n_envs, n_rays, 2]
        # detected_mask = lidar_values != self.lidar_range  # [n_envs, n_rays]

        indices_robots_too_far = torch.where(lidar_values == self.lidar_range)

        for id, index_1 in enumerate(indices_robots_too_far[1]):
            index_0 = int(indices_robots_too_far[0][id])
            robots_rel[index_0, index_1, :] = torch.rand((2,)) * 3 + max(
                self.xdim, self.ydim
            )

        points = pos.unsqueeze(1).expand(-1, self.n_rays, -1)  # [n_envs, n_rays, 2]
        robots = points + robots_rel
        points = torch.cat(
            (pos.unsqueeze(1), robots), dim=1
        )  # [n_envs, n_robot_tot, 2]

        if self.shared_rew:
            is_first = self.world.agents.index(agent) == 0
            rewards = torch.zeros(
                (self.world.batch_dim, self.n_agents), device=self.world.device
            )
            if is_first:
                for env_idx in range(self.world.batch_dim):
                    robots_j = points[env_idx, :, :]
                    voro = self.voronoi.partitioning_single_env(robots_j)
                    for ag_idx in range(self.n_agents):
                        rewards[env_idx, ag_idx] = (
                            self.voronoi.computeHalfRangeCoverageFunctionSingleEnv(
                                voro, pdf[env_idx], 0, env_idx
                            )
                        )

                self.sampling_rew = rewards.sum(-1)
        else:
            rewards = torch.zeros(self.world.batch_dim, device=self.world.device)
            for env_idx in range(self.world.batch_dim):
                robots_j = points[env_idx, :, :]
                vor = self.voronoi.partitioning_single_env(robots_j)
                rewards[env_idx] = (
                    self.voronoi.computeHalfRangeCoverageFunctionSingleEnv(
                        vor, pdf[env_idx], 0, env_idx
                    )
                )
            # print(
            # f"Agent: {self.world.agents.index(agent)} - \n reward: {rewards} - \n robots_j: {points[0, :, :]} - \n pdf[0]: {pdf[0]}"
            # )

        for a in self.world.agents:
            if not hasattr(a, "agent_collision_rew"):
                a.agent_collision_rew = torch.zeros(
                    self.world.batch_dim, device=self.world.device
                )
            a.agent_collision_rew[:] = 0

        for i, a in enumerate(self.world.agents):
            for j, b in enumerate(self.world.agents):
                if i <= j:
                    continue
                if self.world.collides(a, b):
                    distance = self.world.get_distance(a, b)
                    a.agent_collision_rew[
                        distance <= self.min_collision_distance
                    ] += self.agent_collision_penalty
                    b.agent_collision_rew[
                        distance <= self.min_collision_distance
                    ] += self.agent_collision_penalty

        pos_rew = self.sampling_rew if self.shared_rew else rewards

        return pos_rew + agent.agent_collision_rew

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
        # return also metrics:
        # eta: coverage effectiveness
        # beta: beta(p,r) area collectively covered by team
        eta, beta = self._get_coverage_metrics()
        return {
            "agent_sample": agent.sample,
            "eta": eta,
            "beta": beta,
            "n_collisions": self.n_collisions,
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
    def __init__(self, env, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # env = kwargs["env"]
        self.grid_spacing = env.scenario.grid_spacing
        self.cells_range = env.scenario.cells_range
        self.xdim = env.scenario.xdim
        self.ydim = env.scenario.ydim
        self.device = env.world.device
        self.n_rays = env.scenario.n_rays
        self.angle_start = env.scenario.angle_start
        self.angle_end = env.scenario.angle_end
        self.lidar_range = env.scenario.lidar_range
        self.Kp = 0.8
        # self.voronoi = VoronoiCoverage(self.grid_spacing, self.cells_range, self.xdim, self.ydim, self.device, self.centralized)
        # env.scenario.voronoi = self.voronoi
        self.voronoi = env.scenario.voronoi
        self.centralized = env.scenario.centralized

    def compute_action(self, observation: torch.Tensor, u_range: float) -> torch.Tensor:
        # extract info from observation
        pdf = observation[:, (4 + self.n_rays) :]
        pos = observation[:, :2]
        # vel = observation[:, 2:4]
        lidar_values = observation[:, 4 : 4 + self.n_rays]  # [n_envs, n_rays]

        if lidar_values.numel() > 0:
            # get detected robots relative positions
            angles = torch.linspace(
                self.angle_start,
                self.angle_end,
                lidar_values.shape[1],
                device=self.device,
            )
            x = lidar_values * torch.cos(angles)
            y = lidar_values * torch.sin(angles)
            robots_rel = torch.stack((x, y), dim=-1)  # [n_envs, n_rays, 2]
            # detected_mask = lidar_values != self.lidar_range  # [n_envs, n_rays]

            indices_robots_too_far = torch.where(lidar_values == self.lidar_range)

            for id, index_1 in enumerate(indices_robots_too_far[1]):
                index_0 = int(indices_robots_too_far[0][id])
                robots_rel[index_0, index_1, :] = torch.rand((2,)) * 3 + max(
                    self.xdim, self.ydim
                )

            """for id, index_1 in enumerate(indices_robots_too_far[1]):
                index_0 = int(indices_robots_too_far[0][id])
                random_far_value = (
                    100 + id * 10
                )  # Ensures different values, starting from 100
                robots_rel[index_0, index_1, :] = torch.tensor(
                    [random_far_value, random_far_value], device=self.device
                )"""
            """ids = torch.arange(
                len(indices_robots_too_far[1]), device=self.device, dtype=torch.float32
            )  # Ensure float type
            random_far_values = (100 + ids * 10).to(
                robots_rel.dtype
            )  # Ensure dtype matches `robots_rel`

            index_0 = indices_robots_too_far[0].long()  # Ensure indices are long type
            index_1 = indices_robots_too_far[1].long()

            # Create a tensor of shape [num_ids, 2] with [n_robots_too_far, n_robots_too_far] values
            random_far_values = random_far_values.unsqueeze(1).repeat(
                1, 2
            )  # Expand to [n_robots_too_far, n_robots_too_far] format

            # Assign in one operation
            robots_rel[index_0, index_1, :] = random_far_values"""

            points = pos.unsqueeze(1).expand(-1, self.n_rays, -1)  # [n_envs, n_rays, 2]
            robots = points + robots_rel
            points = torch.cat(
                (pos.unsqueeze(1), robots), dim=1
            )  # [n_envs, n_robot_tot, 2]
        else:
            points = pos.unsqueeze(1)

        actions = torch.zeros((pos.shape[0], 2))
        for i in range(pos.shape[0]):
            voro = self.voronoi.partitioning_single_env(points[i])
            centroid = self.voronoi.computeCentroidSingleEnv(voro, pdf[i])

            res_action = self.Kp * (centroid - pos[i, :])

            if torch.isnan(res_action).any().item():
                print("action=nan: ", res_action)
            actions[i, :] = (
                torch.zeros_like(
                    res_action, device=res_action.device, dtype=res_action.dtype
                )
                if torch.isnan(res_action).any().item()
                else res_action
            )

        return torch.clip(actions, -u_range, u_range)  # output: [n_envs, 2]


class VoronoiBasedActor(torch.nn.Module):
    def __init__(self, env: TransformedEnv, continuous_actions: bool = True):
        super().__init__()
        self.env = env
        self.heuristic = VoronoiPolicy(
            env=env.base_env, continuous_action=continuous_actions
        )
        self.n_agents = env.n_agents

    def forward(self, td):
        obs = td[("agents", "observation")]  # [n_envs, n_agents, obs_dim]

        acts = []
        for i in range(self.n_agents):
            agent_obs = obs[:, i, :]  # [n_envs, obs_dim]
            u_range = self.env.base_env.scenario.world.agents[i].u_range
            act_i = self.heuristic.compute_action(agent_obs, u_range=u_range)
            acts.append(act_i)

        acts = torch.stack(acts, dim=1)  # [n_envs, n_agents, act_dim]
        td.set(("agents", "action"), acts.to(obs.device))
        return td


if __name__ == "__main__":
    render_interactively(__file__, control_two_agents=True)


class VoronoiCoverage:
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
        self.xmin = -xdim
        self.xmax = xdim
        self.ymin = -ydim
        self.ymax = ydim
        self.cells_range = cells_range
        self.grid_spacing = grid_spacing
        self.nxcells = (
            int(2 * xdim / self.grid_spacing)
            if centralized
            else 2 * self.cells_range + 1
        )
        self.nycells = (
            int(2 * ydim / self.grid_spacing)
            if centralized
            else 2 * self.cells_range + 1
        )
        self.device = device

        self.x_grid = torch.linspace(self.xmin, self.xmax, self.nxcells)
        self.y_grid = torch.linspace(self.ymin, self.ymax, self.nycells)
        xg, yg = torch.meshgrid(self.x_grid, self.y_grid)
        self.xy_grid = torch.vstack((xg.ravel(), yg.ravel())).T.to(device)

        nxcells_tot = int((self.xmax - self.xmin) / self.grid_spacing)
        nycells_tot = int((self.ymax - self.ymin) / self.grid_spacing)
        xg = torch.linspace(self.xmin, self.xmax, nxcells_tot)
        yg = torch.linspace(self.ymin, self.ymax, nycells_tot)
        xg, yg = torch.meshgrid(xg, yg)
        self.xy_grid_tot = (
            torch.vstack((xg.ravel(), yg.ravel())).T.to(self.device)
            if centralized
            else self.xy_grid
        )

        # self.matrices = torch.

    """def mirror(self, points, xmin, xmax, ymin, ymax):
        square_corners = torch.tensor(
            [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)],
            device=self.device,
            dtype=torch.float32,
        )
        edges = torch.roll(square_corners, shifts=-1, dims=0) - square_corners

        # normalize edge vectors
        edge_lengths = torch.norm(edges, dim=1, keepdim=True)
        normalized_edges = edges / edge_lengths

        n_points = points.shape[0]
        n_edges = square_corners.shape[0]

        points_expanded = points.unsqueeze(1).expand(
            -1, n_edges, -1
        )  # [n_pts, n_edges, 2]
        edge_starts_expanded = square_corners.unsqueeze(0).expand(
            n_points, -1, -1
        )  # [n_pts, n_edges, 2]
        edge_directions_expanded = normalized_edges.unsqueeze(0).expand(
            n_points, -1, -1
        )

        relative_points = points_expanded - edge_starts_expanded
        projections = torch.sum(
            relative_points * edge_directions_expanded, dim=2, keepdim=True
        )
        reflected_points = relative_points - 2 * projections
        mirrored_points = reflected_points + edge_starts_expanded

        mirrored_points = mirrored_points.reshape(-1, 2)
        return mirrored_points.tolist()"""

    def mirror(self, points, x_min, x_max, y_min, y_max):
        points_np = (
            points.cpu().detach().numpy()
        )  # Convert PyTorch tensor to NumPy array

        # Define the square corners and edges
        square_corners = np.array(
            [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]
        )

        edges_start = square_corners
        edges_end = np.roll(
            square_corners, shift=-1, axis=0
        )  # Circular shift to form edges

        # Compute edge vectors
        edge_vectors = edges_end - edges_start  # Shape: (4, 2)

        # Compute vectors from edge start points to given points (broadcasting)
        point_vectors = (
            points_np[:, None, :] - edges_start[None, :, :]
        )  # Shape: (n_points, 4, 2)

        # Compute dot products between point_vectors and edge_vectors
        dot_products = np.sum(
            point_vectors * edge_vectors, axis=-1
        )  # Shape: (n_points, 4)
        edge_lengths_sq = np.sum(edge_vectors**2, axis=-1)  # Shape: (4,)

        # Compute reflection coefficients
        coeffs = 2 * (dot_products / edge_lengths_sq)  # Shape: (n_points, 4)

        # Compute mirrored vectors
        mirrored_vectors = (
            point_vectors - coeffs[:, :, None] * edge_vectors
        )  # Shape: (n_points, 4, 2)

        # Compute final mirrored points
        mirrored_points = (
            edges_start[None, :, :] + mirrored_vectors
        )  # Shape: (n_points, 4, 2)

        return mirrored_points.reshape(-1, 2)  # Flatten to return a list of points

    def partitioning(self, agents: torch.Tensor):
        self.worlds_num = agents.shape[0]
        robots_num = agents.shape[1]
        regions_single_env = [None for i in range(robots_num)]
        self.regions = [regions_single_env] * self.worlds_num
        self.vertices = [regions_single_env] * self.worlds_num
        self.voronois = [None] * self.worlds_num

        dummy_points = torch.zeros((5 * robots_num, self.worlds_num, 2))
        dummy_points[:robots_num, :, :] = self.agents
        for i in range(self.worlds_num):
            mirrored_points = self.mirror(
                self.agents[:, i, :], self.xmin, self.xmax, self.ymin, self.ymax
            )
            mir_pts = torch.tensor(mirrored_points)
            dummy_points[robots_num:, i, :] = mir_pts

        # Voronoi diagram
        for j in range(self.worlds_num):
            vor = Voronoi(dummy_points[:, j, :].cpu().detach().numpy())
            # vor.filtered_points = self.agents[:, j, :].cpu().detach().numpy()
            # regions = [vor.point_region[i] for i in range(self.robots_num)]
            self.regions[j] = [vor.regions[i] for i in vor.point_region[:robots_num]]
            self.vertices[j] = [
                [vor.vertices[v] for v in self.regions[j]]
            ]  # (n_env, n_robots, n_verts)
            # for v in range(self.regions):
            #     self.vertices[j][i].append(vor.vertices[i])
            self.voronois[j] = vor

    def partitioning_single_env(self, agents: torch.Tensor):

        mask = (
            (agents[:, 0] >= self.xmin)
            & (agents[:, 0] <= self.xmax)
            & (agents[:, 1] >= self.ymin)
            & (agents[:, 1] <= self.ymax)
        )
        detected_agents = agents[mask]

        robots_num = detected_agents.shape[0]

        dummy_points = torch.zeros((5 * robots_num, 2))
        dummy_points[:robots_num, :] = detected_agents
        mirrored_points = self.mirror(
            detected_agents, self.xmin, self.xmax, self.ymin, self.ymax
        )
        mir_pts = torch.tensor(mirrored_points)
        dummy_points[robots_num:, :] = mir_pts

        # Voronoi diagram
        vor = Voronoi(dummy_points.cpu().detach().numpy())
        # vor.filtered_points = self.agents[:, j, :].cpu().detach().numpy()
        # regions = [vor.point_region[i] for i in range(self.robots_num)]
        # regions = [vor.regions[i] for i in vor.point_region[:robots_num]]
        # vertices = [vor.vertices[v] for v in regions]  # (n_env, n_robots, n_verts)
        # for v in range(self.regions):
        #     self.vertices[j][i].append(vor.vertices[i])
        return vor

    def getPointsInRegion(self, region, xy_grid):
        p = Path(region)
        bool_val = p.contains_points(xy_grid.cpu().detach().numpy())
        return bool_val

    def extract_central_cells(self, matrix):
        matrix = matrix.reshape((self.nxcells, self.nycells))
        Nx = matrix.shape[0]
        Ny = matrix.shape[1]
        size_x = Nx // 2  # floor division
        size_y = Ny // 2  # floor division
        start_x = (Nx - size_x) // 2
        end_x = start_x + size_x
        start_y = (Ny - size_y) // 2
        end_y = start_y + size_y
        bool_matrix = np.zeros((self.nxcells, self.nycells))
        bool_matrix[start_x:end_x, start_y:end_y] = 1
        return bool_matrix.ravel()

    def computeCoverageFunction(self, agent_id):
        reward = torch.zeros((self.worlds_num))
        for i in range(self.worlds_num):
            vor = self.voronois[i]
            region = vor.point_region[agent_id]
            verts = [vor.vertices[v] for v in vor.regions[region]]
            # regions = [vor.regions[j] for j in vor.point_region[:self.robots_num]]
            # verts = [[vor.vertices[v] for v in vor.regions[region]] for region in regions]
            bool_val = self.getPointsInRegion(verts)
            weights = self.pdf[i][bool_val]
            reward[i] = (
                torch.sum(
                    weights
                    * torch.linalg.norm(
                        self.xy_grid[bool_val] - self.agents[agent_id, i], axis=1
                    )
                    ** 2
                )
                * self.grid_spacing**2
            )

        return -reward

    def computeCoverageFunctionSingleEnv(self, voro, pdf_global, agent_id, env_id):
        region = voro.point_region[agent_id]
        verts = [voro.vertices[v] for v in voro.regions[region]]
        robot = voro.points[agent_id]
        robot = torch.from_numpy(robot).to(self.device)
        bool_val = self.getPointsInRegion(verts, self.xy_grid_tot)
        weights = pdf_global[bool_val]
        reward = (
            torch.sum(
                weights
                * torch.linalg.norm(self.xy_grid_tot[bool_val] - robot, axis=1) ** 2
            )
            * self.grid_spacing**2
        )
        return -reward

    def computeHalfRangeCoverageFunctionSingleEnv(self, voro, pdf, agent_id, env_id):
        region = voro.point_region[agent_id]
        verts = [voro.vertices[v] for v in voro.regions[region]]
        robot = voro.points[agent_id]
        robot = torch.from_numpy(robot).to(self.device)
        pdf_grid_x = torch.linspace(
            robot[0] - self.cells_range * self.grid_spacing,
            robot[0] + self.cells_range * self.grid_spacing,
            self.nxcells,
        )
        pdf_grid_y = torch.linspace(
            robot[1] - self.cells_range * self.grid_spacing,
            robot[1] + self.cells_range * self.grid_spacing,
            self.nycells,
        )
        xg, yg = torch.meshgrid(pdf_grid_x, pdf_grid_y)
        pdf_grid = torch.vstack((xg.ravel(), yg.ravel())).T.to(self.device)
        bool_val = self.getPointsInRegion(verts, pdf_grid)
        pdf_int = self.extract_central_cells(pdf)
        # bool_val = self.getPointsInRegion(verts, self.xy_grid_tot)
        # print("bool val: ", bool_val.reshape((self.nxcells, self.nycells)))
        in_and_int = np.logical_and(bool_val, pdf_int)
        # print("int and int: ", in_and_int.reshape((self.nxcells, self.nycells)))
        # in_and_ext = np.logical_and(bool_val, np.logical_not(pdf_int))
        weights_in = pdf[in_and_int]

        # n_cells_normalization = in_and_int.sum().item()

        # weights_ext = pdf[np.logical_not(pdf_int)]
        reward = (
            torch.sum(
                weights_in
                * torch.linalg.norm(pdf_grid[in_and_int] - robot, axis=1) ** 2
                * self.grid_spacing**2
            )
            # + (0.5 * self.cells_range * self.grid_spacing) ** 2
            # * torch.sum(weights_ext)
            # * self.grid_spacing**2
        )
        return reward

    """def computeCentroid(self, agent_id):
        centroids = torch.zeros((self.worlds_num, 2), device=self.device)
        for i in range(self.worlds_num):
            vor = self.voronois[i]
            region = vor.point_region[agent_id]
            verts = [vor.vertices[v] for v in vor.regions[region]]
            bool_val = self.getPointsInRegion(verts)
            weights = self.pdf[i][bool_val]
            dA = self.grid_spacing**2
            A = torch.sum(weights) * dA
            Cx = torch.sum(weights * self.xy_grid[:, 0][bool_val]) * dA
            Cy = torch.sum(weights * self.xy_grid[:, 1][bool_val]) * dA
            centroids[i, :] = torch.tensor([Cx / A, Cy / A])

        return centroids"""

    def computeCentroidSingleEnv(self, vor, pdf):
        region = vor.point_region[0]
        robot = vor.points[0]
        verts = [vor.vertices[v] for v in vor.regions[region]]
        if self.centralized:
            xy_grid = self.xy_grid
        else:
            x_grid = torch.linspace(
                robot[0] - self.cells_range * self.grid_spacing,
                robot[0] + self.cells_range * self.grid_spacing,
                self.nxcells,
            )
            y_grid = torch.linspace(
                robot[1] - self.cells_range * self.grid_spacing,
                robot[1] + self.cells_range * self.grid_spacing,
                self.nycells,
            )
            xg, yg = torch.meshgrid(x_grid, y_grid)
            xy_grid = torch.vstack((xg.ravel(), yg.ravel())).T.to(self.device)
        bool_val = self.getPointsInRegion(verts, xy_grid)
        weights = pdf[bool_val]
        dA = self.grid_spacing**2
        A = torch.sum(weights) * dA

        Cx = torch.sum(weights * xy_grid[:, 0][bool_val]) * dA
        Cy = torch.sum(weights * xy_grid[:, 1][bool_val]) * dA

        if A.item() == 0:
            # uniform distribution
            A = weights.shape[0] * dA
            Cx = 0  # torch.sum(xy_grid[:, 0][bool_val]) * dA
            Cy = 0  # torch.sum(xy_grid[:, 1][bool_val]) * dA

        try:
            centroid = torch.tensor([Cx / A, Cy / A], device=self.device)
        except Exception:
            centroid = torch.tensor([0.0, 0.0], device=self.device)

        assert not torch.isnan(centroid).any().item(), ValueError(
            "centroids have nan values: ", centroid
        )

        return centroid
