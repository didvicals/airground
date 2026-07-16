"""Gymnasium environment: Layer-3 mode arbitration as an RL problem.

The agent navigates a (procedurally generated) multi-floor building from
entry to an overwatch goal, choosing movement direction AND locomotion mode
each step. Reward is the negative mission cost — the same currency the
Dijkstra oracle optimizes — so learned policies are directly comparable:

    r = -(w_E * energy + w_T * time + w_N * acoustic_exposure) / SCALE

Phase weights are part of the observation -> one conditioned policy covers
all mission phases. The Dijkstra planner (full map knowledge) is the oracle
upper bound; the agent only sees a local egocentric crop, so the research
question is how close partial-observation arbitration gets to the oracle.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from airground.environment import CELL_M, STAIR_TRANSIT_S, Pose
from airground.mission import Phase, phase_weights
from airground.models import Mode, Params, power_w, transition_cost
from airground.planner import Weights

DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))
REWARD_SCALE = 100.0
GOAL_BONUS = 50.0
DEAD_BATTERY_PENALTY = 50.0
BUMP_TIME_S = 0.5


class PoliceEnv(gym.Env):
    """Actions: Discrete(8) = direction (4) x target mode (2: ground, flight)."""

    metadata = {"render_modes": ["ansi"]}

    def __init__(self, building_fn=None, phase: Phase = Phase.APPROACH,
                 crop: int = 9, max_steps: int = 500,
                 params: Params | None = None,
                 shaping_coef: float = 1.0, gamma: float = 0.995):
        super().__init__()
        self.p = params or Params.load()
        self.building_fn = building_fn  # callable(seed) -> Building; None = fixed townhouse
        self.phase = phase
        self.crop = crop
        self.max_steps = max_steps
        # potential-based reward shaping (Ng et al. 1999): dense gradient
        # toward goal, provably policy-invariant. shaping_coef=0 disables.
        self.shaping_coef = shaping_coef
        self.gamma = gamma

        n_scalar = 11
        self.observation_space = spaces.Box(
            -1.0, 1.0, shape=(4 * crop * crop + n_scalar,), dtype=np.float32)
        self.action_space = spaces.Discrete(8)

    # ------------------------------------------------------------ helpers

    def _goal_distance_field(self) -> list[np.ndarray]:
        """Per-floor min step-distance to goal over the flight graph.

        Flight-passable 4-connectivity plus bidirectional stair transit.
        Flight reachability is a superset of ground's, so this is an
        admissible-ish shaping potential (PBRS is valid for any potential,
        so approximation only affects guidance strength, never correctness).
        Unreachable cells get a large finite value.
        """
        from collections import deque
        big = float(self.b.n_floors * self.b.rows * self.b.cols)
        dist = [np.full((self.b.rows, self.b.cols), big, dtype=np.float32)
                for _ in range(self.b.n_floors)]
        g = self.b.goal
        dist[g.floor][g.r, g.c] = 0.0
        q = deque([(g.floor, g.r, g.c)])
        while q:
            f, r, c = q.popleft()
            d = dist[f][r, c]
            neigh = [(f, r + dr, c + dc) for dr, dc in
                     ((1, 0), (-1, 0), (0, 1), (0, -1))]
            if self.b.is_stair(Pose(f, r, c)):
                neigh += [(f + 1, r, c), (f - 1, r, c)]
            for nf, nr, nc in neigh:
                p = Pose(nf, nr, nc)
                if self.b.passable(p, Mode.FLIGHT) and dist[nf][nr, nc] > d + 1:
                    dist[nf][nr, nc] = d + 1
                    q.append((nf, nr, nc))
        return dist

    def _potential(self, pose: Pose) -> float:
        return -float(self.dist[pose.floor][pose.r, pose.c])

    def _audible_fly_map(self) -> list[np.ndarray]:
        """Per floor: 1.0 where flying would be audible at the suspect."""
        maps = []
        for f in range(self.b.n_floors):
            m = np.zeros((self.b.rows, self.b.cols), dtype=np.float32)
            for r in range(self.b.rows):
                for c in range(self.b.cols):
                    pose = Pose(f, r, c)
                    if (self.b.passable(pose, Mode.FLIGHT)
                            and self.b.exposure_dbs(self.p, pose, Mode.FLIGHT, 1.0) > 0):
                        m[r, c] = 1.0
            maps.append(m)
        return maps

    def _obs(self) -> np.ndarray:
        k = self.crop // 2
        ch = np.zeros((4, self.crop, self.crop), dtype=np.float32)
        f, r0, c0 = self.pose.floor, self.pose.r, self.pose.c
        for i in range(self.crop):
            for j in range(self.crop):
                r, c = r0 - k + i, c0 - k + j
                pose = Pose(f, r, c)
                if not self.b.in_bounds(pose):
                    ch[0, i, j] = 1.0  # out of bounds reads as wall
                    continue
                cell = self.b.cell(pose)
                ch[0, i, j] = 1.0 if cell == "#" else 0.0
                ch[1, i, j] = 1.0 if cell == "." else 0.0  # ground-passable
                ch[2, i, j] = 1.0 if cell == "S" else 0.0
                ch[3, i, j] = self.audible_fly[f][r, c]
        g = self.b.goal
        w = self.weights
        scalars = np.array([
            np.clip((g.r - r0) / 10.0, -1, 1),
            np.clip((g.c - c0) / 10.0, -1, 1),
            np.clip(g.floor - f, -1, 1),
            self.battery_j / self.usable_j,
            1.0 if self.mode is Mode.FLIGHT else 0.0,
            1.0 if self.b.landable(self.pose) else 0.0,
            self.audible_fly[f][r0, c0],
            w.w_energy / 5.0,
            w.w_time / 50.0,
            w.w_noise / 200.0,
            self.steps / self.max_steps,
        ], dtype=np.float32)
        return np.concatenate([ch.ravel(), scalars])

    def _cost(self, e_j: float, dt: float, exp: float) -> float:
        w = self.weights
        return (w.w_energy * e_j + w.w_time * dt + w.w_noise * exp) / REWARD_SCALE

    # ---------------------------------------------------------------- API

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self.building_fn is not None:
            self.b = self.building_fn(int(self.np_random.integers(2**31)))
        else:
            from airground.scenarios import townhouse
            self.b = townhouse()
        self.weights: Weights = phase_weights(self.phase)
        self.pose: Pose = self.b.start
        self.mode = Mode.GROUND
        self.usable_j = self.p.battery_wh * 3600.0 * self.p.usable_fraction
        self.battery_j = self.usable_j
        self.steps = 0
        self.total_energy = 0.0
        self.total_time = 0.0
        self.total_exposure = 0.0
        self.audible_fly = self._audible_fly_map()
        self.dist = self._goal_distance_field()
        self._prev_potential = self._potential(self.pose)
        return self._obs(), {}

    def step(self, action: int):
        self.steps += 1
        direction = DIRS[action % 4]
        target_mode = Mode.FLIGHT if action >= 4 else Mode.GROUND
        reward = 0.0

        # mode switch requested
        if target_mode is not self.mode:
            if self.b.landable(self.pose):
                e, dt = transition_cost(self.p, self.mode, target_mode)
                exp = self.b.exposure_dbs(self.p, self.pose, Mode.FLIGHT, dt)
                reward -= self._cost(e, dt, exp)
                self._account(e, dt, exp)
                self.mode = target_mode
            else:
                reward -= 1.0  # cannot land here / invalid switch

        # move
        nxt = Pose(self.pose.floor, self.pose.r + direction[0],
                   self.pose.c + direction[1])
        stair_up = (self.mode is Mode.FLIGHT and self.b.is_stair(self.pose)
                    and not self.b.passable(nxt, self.mode))
        if self.b.passable(nxt, self.mode):
            v = (self.p.flight_nominal_ms if self.mode is Mode.FLIGHT
                 else self.p.ground_nominal_ms)
            dt = CELL_M / v
            e = power_w(self.p, self.mode, v) * dt
            exp = self.b.exposure_dbs(self.p, nxt, self.mode, dt)
            reward -= self._cost(e, dt, exp)
            self._account(e, dt, exp)
            self.pose = nxt
        elif stair_up:
            for df in (1, -1):
                up = Pose(self.pose.floor + df, self.pose.r, self.pose.c)
                if self.b.in_bounds(up) and self.b.is_stair(up):
                    dt = STAIR_TRANSIT_S
                    e = power_w(self.p, Mode.FLIGHT,
                                self.p.flight_nominal_ms) * dt
                    exp = self.b.exposure_dbs(self.p, up, Mode.FLIGHT, dt)
                    reward -= self._cost(e, dt, exp)
                    self._account(e, dt, exp)
                    self.pose = up
                    break
        else:
            # bump: pay idle time in current mode
            e = power_w(self.p, self.mode, 0.0) * BUMP_TIME_S
            reward -= self._cost(e, BUMP_TIME_S, 0.0) + 0.1
            self._account(e, BUMP_TIME_S, 0.0)

        # potential-based shaping: coef * (gamma*Phi(s') - Phi(s))
        phi = self._potential(self.pose)
        reward += self.shaping_coef * (self.gamma * phi - self._prev_potential)
        self._prev_potential = phi

        terminated = reached_goal = False
        g = self.b.goal
        if (self.pose.floor, self.pose.r, self.pose.c) == (g.floor, g.r, g.c):
            reward += GOAL_BONUS
            terminated = reached_goal = True
        elif self.battery_j <= self.usable_j * self.p.reserve_return_fraction:
            reward -= DEAD_BATTERY_PENALTY
            terminated = True

        truncated = self.steps >= self.max_steps
        info = {"energy_j": self.total_energy, "time_s": self.total_time,
                "exposure_dbs": self.total_exposure, "success": reached_goal}
        return self._obs(), reward, terminated, truncated, info

    def _account(self, e: float, dt: float, exp: float) -> None:
        self.battery_j -= e
        self.total_energy += e
        self.total_time += dt
        self.total_exposure += exp

    def render(self) -> str:
        mark = "f" if self.mode is Mode.FLIGHT else "g"
        out = []
        for f in range(self.b.n_floors):
            out.append(f"floor {f}:")
            for r in range(self.b.rows):
                row = "".join(self.b.grid[f][r])
                if f == self.pose.floor and r == self.pose.r:
                    row = row[:self.pose.c] + mark + row[self.pose.c + 1:]
                out.append("  " + row)
        return "\n".join(out)
