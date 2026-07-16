"""Mode-aware path planner over (floor, r, c, mode) states.

Dijkstra with edge cost J = w_E * energy_J + w_T * time_s + w_N * exposure_dBs.
Mode switches are explicit edges (transition energy/time, takeoff/landing noise).
Weights come from the Layer-3 mission policy (mission.py); baselines lock the
mode or zero out weights.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field

from airground.environment import CELL_M, STAIR_TRANSIT_S, Building, Pose
from airground.models import Mode, Params, power_w, transition_cost


@dataclass(frozen=True)
class Weights:
    w_energy: float = 1.0    # cost per joule
    w_time: float = 0.0      # joule-equivalents per second
    w_noise: float = 0.0     # joule-equivalents per dB*s of exposure
    lock_mode: Mode | None = None
    forbid_audible: bool = False  # hard stealth constraint


@dataclass
class Plan:
    feasible: bool
    states: list[tuple[Pose, Mode]] = field(default_factory=list)
    energy_j: float = 0.0
    time_s: float = 0.0
    exposure_dbs: float = 0.0
    transitions: int = 0

    def battery_fraction(self, p: Params) -> float:
        return self.energy_j / (p.battery_wh * 3600.0 * p.usable_fraction)


State = tuple[int, int, int, Mode]  # floor, r, c, mode


def _edges(b: Building, p: Params, pose: Pose, mode: Mode):
    """Yield (next_pose, next_mode, energy_j, time_s)."""
    v = p.flight_nominal_ms if mode is Mode.FLIGHT else p.ground_nominal_ms
    # planar moves
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nxt = Pose(pose.floor, pose.r + dr, pose.c + dc)
        if b.passable(nxt, mode):
            dt = CELL_M / v
            yield nxt, mode, power_w(p, mode, v) * dt, dt
    # stair transit (flight only, same (r, c) on adjacent floor)
    if mode is Mode.FLIGHT and b.is_stair(pose):
        for df in (1, -1):
            nxt = Pose(pose.floor + df, pose.r, pose.c)
            if b.in_bounds(nxt) and b.is_stair(nxt):
                dt = STAIR_TRANSIT_S
                yield nxt, mode, power_w(p, mode, v) * dt, dt
    # mode switch (landable floor cells only)
    if b.landable(pose):
        other = Mode.GROUND if mode is Mode.FLIGHT else Mode.FLIGHT
        e, dt = transition_cost(p, mode, other)
        yield pose, other, e, dt


def plan(b: Building, p: Params, w: Weights, start: Pose, goal: Pose,
         start_mode: Mode = Mode.GROUND) -> Plan:
    if w.lock_mode is not None:
        start_mode = w.lock_mode
    s0: State = (start.floor, start.r, start.c, start_mode)
    dist: dict[State, float] = {s0: 0.0}
    prev: dict[State, tuple[State, float, float, float]] = {}
    pq: list[tuple[float, int, State]] = [(0.0, 0, s0)]
    tie = 0

    while pq:
        d, _, s = heapq.heappop(pq)
        if d > dist.get(s, float("inf")):
            continue
        f, r, c, mode = s
        if (f, r, c) == (goal.floor, goal.r, goal.c):
            return _reconstruct(prev, s0, s, p)
        pose = Pose(f, r, c)
        for nxt, nmode, e_j, dt in _edges(b, p, pose, mode):
            if w.lock_mode is not None and nmode is not w.lock_mode:
                continue
            # noise during the edge: transition uses flight-level noise
            noisy_mode = Mode.FLIGHT if nmode is not mode else nmode
            exp = b.exposure_dbs(p, nxt, noisy_mode, dt)
            if w.forbid_audible and exp > 0.0:
                continue
            cost = w.w_energy * e_j + w.w_time * dt + w.w_noise * exp
            ns: State = (nxt.floor, nxt.r, nxt.c, nmode)
            nd = d + cost
            if nd < dist.get(ns, float("inf")):
                dist[ns] = nd
                prev[ns] = (s, e_j, dt, exp)
                tie += 1
                heapq.heappush(pq, (nd, tie, ns))

    return Plan(feasible=False)


def _reconstruct(prev, s0: State, s_goal: State, p: Params) -> Plan:
    plan_ = Plan(feasible=True)
    chain: list[State] = [s_goal]
    s = s_goal
    while s != s0:
        s_prev, e_j, dt, exp = prev[s]
        plan_.energy_j += e_j
        plan_.time_s += dt
        plan_.exposure_dbs += exp
        if s_prev[3] is not s[3]:
            plan_.transitions += 1
        chain.append(s_prev)
        s = s_prev
    chain.reverse()
    plan_.states = [(Pose(f, r, c), m) for f, r, c, m in chain]
    return plan_
