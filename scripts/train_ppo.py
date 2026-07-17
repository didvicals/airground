"""PPO training for the Layer-3 mode-arbitration policy.

Designed to run on Colab (GPU) or any CPU box:
    python scripts/train_ppo.py --steps 2000000 --nenv 8 --out runs/ppo_v0

After training, evaluates on held-out generator seeds against the
Dijkstra oracle (full-map optimal) and prints the optimality gap.
Requires: pip install -e .[train]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from airground.envs import PoliceEnv, generate
from airground.experts import collect_demos
from airground.mission import Phase, phase_weights
from airground.models import Params
from airground.planner import plan


def make_env():
    return PoliceEnv(building_fn=generate)


def bc_warmstart(model: PPO, n_episodes: int, epochs: int,
                 batch: int = 512, lr: float = 1e-3) -> None:
    """Supervised warm-start of the PPO policy from the greedy expert.

    Hand-rolled behavior cloning on SB3's own ActorCriticPolicy (cross-entropy
    on the action distribution) — no external imitation-learning dependency,
    so it never conflicts with Colab's preinstalled package set.
    """
    import numpy as np
    import torch

    print(f"BC: collecting {n_episodes} expert episodes...")
    obs, acts = collect_demos(generate, n_episodes=n_episodes)
    print(f"BC: {len(obs)} state-action pairs; cloning {epochs} epochs")
    device = model.device
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
    act_t = torch.as_tensor(acts, dtype=torch.long, device=device)
    opt = torch.optim.Adam(model.policy.parameters(), lr=lr)
    n = len(obs_t)
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        total, nb = 0.0, 0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            dist = model.policy.get_distribution(obs_t[idx])
            loss = -dist.log_prob(act_t[idx]).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            nb += 1
        acc = float((model.policy.predict(obs, deterministic=True)[0]
                     == np.asarray(acts)).mean())
        print(f"  BC epoch {ep + 1}/{epochs}: loss {total / nb:.3f}, "
              f"expert-action acc {acc:.2%}")


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
    ap.add_argument("--bc-episodes", type=int, default=0,
                    help="expert episodes for BC warm-start (0 = shaping only)")
    ap.add_argument("--bc-epochs", type=int, default=15)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # run metadata: reproduce any checkpoint from commit + args
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit = "unknown"
    (out / "run_info.json").write_text(json.dumps({
        "commit": commit,
        "args": vars(args),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, indent=2))

    venv = SubprocVecEnv([make_env for _ in range(args.nenv)])
    # normalize reward: step costs and shaping span orders of magnitude;
    # unnormalized returns make PPO's value target unstable
    venv = VecNormalize(venv, norm_obs=False, norm_reward=True, gamma=0.995)
    if args.resume:
        model = PPO.load(args.resume, env=venv)
    else:
        model = PPO("MlpPolicy", venv, verbose=1,
                    tensorboard_log=str(out / "tb"),
                    n_steps=512, batch_size=1024, learning_rate=3e-4,
                    gamma=0.995, ent_coef=0.01)
        if args.bc_episodes > 0:
            bc_warmstart(model, args.bc_episodes, args.bc_epochs)
    # periodic checkpoints: survive Colab session drops mid-run
    ckpt = CheckpointCallback(save_freq=max(250_000 // args.nenv, 1),
                              save_path=str(out / "ckpt"), name_prefix="ppo")
    model.learn(total_timesteps=args.steps, progress_bar=True, callback=ckpt)
    model.save(out / "final")
    evaluate(model)


if __name__ == "__main__":
    main()
