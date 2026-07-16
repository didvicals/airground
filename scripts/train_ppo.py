"""PPO training for the Layer-3 mode-arbitration policy.

Designed to run on Colab (GPU) or any CPU box:
    python scripts/train_ppo.py --steps 2000000 --nenv 8 --out runs/ppo_v0

After training, evaluates on held-out generator seeds against the
Dijkstra oracle (full-map optimal) and prints the optimality gap.
Requires: pip install -e .[train]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

from airground.envs import PoliceEnv, generate
from airground.mission import Phase, phase_weights
from airground.models import Params
from airground.planner import plan


def make_env():
    return PoliceEnv(building_fn=generate)


def evaluate(model: PPO, n_episodes: int = 50) -> None:
    p = Params.load()
    env = PoliceEnv(building_fn=generate)
    gaps, successes = [], 0
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=100_000 + ep)  # held-out seeds
        done = trunc = False
        while not (done or trunc):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, trunc, info = env.step(int(action))
        w = phase_weights(Phase.APPROACH)
        oracle = plan(env.b, p, w, env.b.start, env.b.goal)
        if info["success"] and oracle.feasible:
            successes += 1
            agent_cost = (w.w_energy * info["energy_j"]
                          + w.w_time * info["time_s"]
                          + w.w_noise * info["exposure_dbs"])
            oracle_cost = (w.w_energy * oracle.energy_j
                           + w.w_time * oracle.time_s
                           + w.w_noise * oracle.exposure_dbs)
            gaps.append(agent_cost / oracle_cost)
    print(f"success rate: {successes}/{n_episodes}")
    if gaps:
        print(f"mean cost vs oracle: {sum(gaps) / len(gaps):.2f}x "
              f"(1.0 = matches full-map optimal)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2_000_000)
    ap.add_argument("--nenv", type=int, default=8)
    ap.add_argument("--out", type=str, default="runs/ppo_v0")
    ap.add_argument("--resume", type=str, default=None)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    venv = SubprocVecEnv([make_env for _ in range(args.nenv)])
    if args.resume:
        model = PPO.load(args.resume, env=venv)
    else:
        model = PPO("MlpPolicy", venv, verbose=1,
                    tensorboard_log=str(out / "tb"),
                    n_steps=512, batch_size=1024, learning_rate=3e-4,
                    gamma=0.995, ent_coef=0.01)
    model.learn(total_timesteps=args.steps, progress_bar=True)
    model.save(out / "final")
    evaluate(model)


if __name__ == "__main__":
    main()
