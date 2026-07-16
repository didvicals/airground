"""Behavior-cloning warm-start from the scripted greedy expert.

Fix 2 for the sparse-reward collapse: instead of hoping PPO discovers the
40-step goal sequence by exploration, clone the expert's (obs -> action)
mapping first, then hand the warmed policy to PPO for fine-tuning under the
mission-cost reward. The expert reaches goal on 100% of maps, so the cloned
policy starts near-competent and PPO only has to trade off energy/time/noise.

Requires: pip install -e .[train]   (adds `imitation`)
Run:      python scripts/bc_pretrain.py --episodes 3000 --out runs/bc_v0
Then:     python scripts/train_ppo.py --bc-init runs/bc_v0/bc_policy.zip ...
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from imitation.algorithms import bc
from imitation.data.types import Transitions

from airground.envs import PoliceEnv, generate
from airground.experts import collect_demos


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=3000)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--out", type=str, default="runs/bc_v0")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"collecting {args.episodes} expert episodes...")
    obs, acts = collect_demos(generate, n_episodes=args.episodes)
    print(f"  {len(obs)} state-action pairs")

    # BC only needs (obs, acts); pad next_obs/dones to satisfy the dataclass
    transitions = Transitions(
        obs=obs, acts=acts, next_obs=np.roll(obs, -1, axis=0),
        dones=np.zeros(len(obs), dtype=bool),
        infos=np.array([{}] * len(obs)),
    )

    probe = PoliceEnv(building_fn=generate)
    rng = np.random.default_rng(0)
    trainer = bc.BC(
        observation_space=probe.observation_space,
        action_space=probe.action_space,
        demonstrations=transitions,
        rng=rng,
    )
    trainer.train(n_epochs=args.epochs)
    trainer.policy.save(str(out / "bc_policy"))
    print(f"saved BC policy -> {out / 'bc_policy.zip'}")


if __name__ == "__main__":
    main()
