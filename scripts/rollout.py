"""Inspect a trained policy: replay one episode and print the trajectory.

Shows the mode chosen at each visited cell overlaid on the building
(g ground, f flight), plus per-episode metrics and the oracle comparison.
Works on Colab or any machine with the trained model + torch.

Run: python scripts/rollout.py --model runs/ppo_v0/final.zip --seed 100001
"""

from __future__ import annotations

import argparse

from stable_baselines3 import PPO

from airground.envs import PoliceEnv, generate
from airground.mission import Phase, phase_weights
from airground.models import Mode, Params
from airground.planner import plan


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--seed", type=int, default=100_001)
    ap.add_argument("--townhouse", action="store_true",
                    help="use the fixed townhouse instead of a generated map")
    args = ap.parse_args()

    model = PPO.load(args.model)
    env = PoliceEnv(building_fn=None if args.townhouse else generate)
    obs, _ = env.reset(seed=args.seed)

    trail: dict[tuple[int, int, int], str] = {}
    done = trunc = False
    while not (done or trunc):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, trunc, info = env.step(int(action))
        key = (env.pose.floor, env.pose.r, env.pose.c)
        trail[key] = "f" if env.mode is Mode.FLIGHT else "g"

    for f in range(env.b.n_floors):
        print(f"floor {f}:")
        for r in range(env.b.rows):
            row = ""
            for c in range(env.b.cols):
                row += trail.get((f, r, c), env.b.grid[f][r][c])
            print("  " + row)

    print(f"\nsuccess={info['success']}  energy={info['energy_j']:.0f} J  "
          f"time={info['time_s']:.0f} s  exposure={info['exposure_dbs']:.0f} dB*s")

    w = phase_weights(Phase.APPROACH)
    oracle = plan(env.b, Params.load(), w, env.b.start, env.b.goal)
    if oracle.feasible:
        print(f"oracle:  energy={oracle.energy_j:.0f} J  "
              f"time={oracle.time_s:.0f} s  exposure={oracle.exposure_dbs:.0f} dB*s")


if __name__ == "__main__":
    main()
