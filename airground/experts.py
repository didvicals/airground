"""Scripted expert: greedy descent on the goal-distance field.

Reaches the goal on 100% of solvable maps, so it serves three roles:
  - solvability oracle for diagnostics,
  - behavior-cloning demonstration source (warm-start PPO),
  - reference policy for DAgger.

The expert uses the same distance field the env computes for reward shaping,
so its actions are consistent with the shaped objective.
"""

from __future__ import annotations

import numpy as np

from airground.environment import Pose
from airground.models import Mode

DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def env_successor(b, pose: Pose, mode: Mode, action: int):
    """Mirror PoliceEnv.step transition (pose/mode only, no reward)."""
    tmode = Mode.FLIGHT if action >= 4 else Mode.GROUND
    d = DIRS[action % 4]
    m = mode
    if tmode is not mode and b.landable(pose):
        m = tmode
    nxt = Pose(pose.floor, pose.r + d[0], pose.c + d[1])
    stair_up = m is Mode.FLIGHT and b.is_stair(pose) and not b.passable(nxt, m)
    if b.passable(nxt, m):
        return nxt, m
    if stair_up:
        for df in (1, -1):
            up = Pose(pose.floor + df, pose.r, pose.c)
            if b.in_bounds(up) and b.is_stair(up):
                return up, m
    return pose, m  # bump


def greedy_action(env) -> int:
    """Action minimizing successor distance-to-goal; prefer ground (cheaper)."""
    best_a, best_d = 0, 1e18
    for a in range(8):
        nxt, _ = env_successor(env.b, env.pose, env.mode, a)
        d = env.dist[nxt.floor][nxt.r, nxt.c] + (0.0 if a < 4 else 0.1)
        if d < best_d:
            best_d, best_a = d, a
    return best_a


def greedy_eval(building_fn, n: int = 100, seed: int = 0):
    """Run the greedy expert; return (success_rate, mean_shaped_return)."""
    from airground.envs import PoliceEnv
    rng = np.random.default_rng(seed)
    env = PoliceEnv(building_fn=building_fn)
    hits, returns = 0, []
    for _ in range(n):
        env.reset(seed=int(rng.integers(2**31)))
        done = trunc = False
        ep_ret = 0.0
        while not (done or trunc):
            _, r, done, trunc, info = env.step(greedy_action(env))
            ep_ret += r
        hits += info["success"]
        returns.append(ep_ret)
    return hits / n, float(np.mean(returns))


def collect_demos(building_fn, n_episodes: int = 2000, seed: int = 0):
    """Behavior-cloning dataset: (obs, action) pairs from the greedy expert.

    Returns (obs_array, action_array). Feed to an imitation-learning BC
    trainer to warm-start the PPO policy before RL fine-tuning.
    """
    from airground.envs import PoliceEnv
    rng = np.random.default_rng(seed)
    env = PoliceEnv(building_fn=building_fn)
    obs_buf, act_buf = [], []
    for _ in range(n_episodes):
        obs, _ = env.reset(seed=int(rng.integers(2**31)))
        done = trunc = False
        while not (done or trunc):
            a = greedy_action(env)
            obs_buf.append(obs)
            act_buf.append(a)
            obs, _, done, trunc, _ = env.step(a)
    return np.asarray(obs_buf, dtype=np.float32), np.asarray(act_buf, dtype=np.int64)
