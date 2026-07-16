"""Root-cause diagnostic: is the env even solvable, and how hard?

BFS over the env's OWN transition dynamics (exactly mirroring PoliceEnv.step,
reward aside). Answers:
  1. Is the goal reachable by any action sequence? (env bug vs learning problem)
  2. Minimum action-steps to goal (exploration difficulty).
  3. Does a random policy ever stumble onto it? (sparse-reward severity).
"""

from __future__ import annotations

from collections import deque

import numpy as np

from airground.environment import Pose
from airground.envs import PoliceEnv, generate
from airground.experts import env_successor, greedy_eval
from airground.models import Mode
from airground.scenarios import townhouse


def solvable(b) -> tuple[bool, int]:
    start = (b.start.floor, b.start.r, b.start.c, Mode.GROUND)
    goal = (b.goal.floor, b.goal.r, b.goal.c)
    seen = {start}
    q = deque([(start, 0)])
    while q:
        (f, r, c, m), d = q.popleft()
        if (f, r, c) == goal:
            return True, d
        for a in range(8):
            np_, nm = env_successor(b, Pose(f, r, c), m, a)
            key = (np_.floor, np_.r, np_.c, nm)
            if key not in seen:
                seen.add(key)
                q.append((key, d + 1))
    return False, -1


def random_success_rate(building_fn, n=200, seed=0):
    rng = np.random.default_rng(seed)
    env = PoliceEnv(building_fn=building_fn)
    hits = 0
    for _ in range(n):
        obs, _ = env.reset(seed=int(rng.integers(2**31)))
        done = trunc = False
        while not (done or trunc):
            obs, _, done, trunc, info = env.step(int(rng.integers(8)))
        hits += info["success"]
    return hits / n


def main() -> None:
    print("=== Fixed townhouse ===")
    b = townhouse()
    ok, steps = solvable(b)
    print(f"  solvable by BFS over env dynamics: {ok}, min steps: {steps}")

    print("\n=== Generated maps (train distribution) ===")
    reach, dists = 0, []
    for s in range(30):
        gb = generate(s)
        ok, steps = solvable(gb)
        reach += ok
        if ok:
            dists.append(steps)
    print(f"  reachable: {reach}/30")
    if dists:
        print(f"  min steps to goal: mean {np.mean(dists):.0f}, "
              f"max {max(dists)}, min {min(dists)}")

    print("\n=== Random policy success (sparse-reward severity) ===")
    print(f"  townhouse : {random_success_rate(None, 100):.2%}")
    print(f"  generated : {random_success_rate(generate, 100):.2%}")

    print("\n=== Greedy distance-descent (solvable-by-sane-policy + shaping) ===")
    sr_t, ret_t = greedy_eval(None, 50)
    sr_g, ret_g = greedy_eval(generate, 100)
    print(f"  townhouse : success {sr_t:.2%}, mean shaped return {ret_t:+.1f}")
    print(f"  generated : success {sr_g:.2%}, mean shaped return {ret_g:+.1f}")


if __name__ == "__main__":
    main()
