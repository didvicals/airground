# airground — Hybrid Ground-Air Mode Arbitration

Research codebase for mission-aware flight/drive mode arbitration of an indoor
police tactical hybrid robot (Lemur 2-class airframe + actuated wheels).

## Layout

- `docs/reference-platform.md` — market-derived reference platform ("RefBot-1"):
  spec comparison of BRINC Lemur 2, CapsuleBot, TABV, and the derived parameter set.
- `airground/platform_params.yaml` — single source of truth for model parameters.
- `airground/models.py` — mode-wise energy model and acoustic (detectability) model.
- `airground/environment.py` / `planner.py` / `mission.py` / `scenarios.py` —
  2.5D building world, mode-aware Dijkstra (oracle), Layer-3 phase policy.
- `airground/envs/` — `building_gen.py` procedural scenario generator (validated:
  flight-reachable + stealth-feasible), `police_env.py` Gymnasium env for RL.
- `scripts/` — `demo_breakeven.py`, `evaluate.py` (baseline table),
  `train_ppo.py` (SB3 PPO + oracle-gap eval).
- `notebooks/colab_train.ipynb` — Colab GPU training bootstrap.
- `tests/test_framework.py` — generator/planner/env smoke tests.

## Workflow (local laptop + Colab)

Local (no CUDA): env development, tests, evaluation, visualization.
Colab (T4): PPO training via the notebook; checkpoints persist to Drive.
Sync: push to GitHub, notebook clones/pulls.

Env throughput ~7k steps/s/process (pure Python) -> 2M-step PPO run is
hours-scale on Colab CPU/GPU with 8 vectorized envs.

## RL formulation (Layer-3 arbitration as POMDP)

- Obs: 9x9 egocentric crop (walls / ground-passable / stairs / fly-audible
  zone) + battery, mode, goal offset, phase weights.
- Action: Discrete(8) = 4 directions x {ground, flight}; switching modes
  requires landable cell and pays transition energy/time/noise.
- Reward: -(w_E*E + w_T*dt + w_N*exposure)/100, +50 goal, -50 battery floor.
  Same currency as the Dijkstra oracle -> direct optimality-gap metric.
- Research question: how close does partial-observation arbitration get to
  the full-map oracle, and does it generalize to unseen buildings?

## Roadmap (dev plan)

1. ~~Survey + market spec collection~~ done
2. ~~Reference platform model (energy, acoustics)~~ done — calibration items open
3. ~~Simulation environment~~ done — `environment.py` (2.5D multi-floor grid,
   acoustic ray-cast), `scenarios.py` (townhouse: stairs, debris, barricaded suspect)
4. ~~Layer-3 arbitration prototype~~ done — `mission.py` (phase -> weights),
   `planner.py` (mode-aware Dijkstra, stealth constraint), `evaluate.py` (baselines)
5. Continuous-space upgrade: replace grid Dijkstra with the TABV unified-NMPC
   stack (arXiv:2403.00322) fed by Layer-3 weights; richer scenarios; MCTS for
   multi-goal search legs (arXiv:2507.21338)
6. Hardware calibration + real-platform validation

## Run

```
pip install -e .            # or .[train] for PPO + imitation
python tests/test_framework.py
python scripts/diagnose.py   # solvability + sparse-reward severity
python scripts/evaluate.py   # oracle baseline comparison table
python scripts/train_ppo.py --bc-episodes 3000 --steps 2000000    # BC + PPO
python scripts/rollout.py --model runs/ppo_v0/final.zip           # inspect
```

## Training notes

First PPO run collapsed to 0% success (worse than random's 10%). Root cause:
sparse terminal reward + per-step negative cost -> the policy minimized
activity instead of reaching the goal; the +50 goal bonus was never
discovered by exploration (townhouse needs a 40-step precise sequence).
`scripts/diagnose.py` confirmed the env itself is fully solvable (greedy
descent on the goal-distance field reaches goal 100%).

Fixes applied:
1. **Potential-based reward shaping** (`shaping_coef` in `PoliceEnv`): dense
   gradient toward goal via `gamma*Phi(s') - Phi(s)`, `Phi = -dist_to_goal`.
   Policy-invariant (Ng et al. 1999). Turns goal-reaching return positive.
2. **Reward normalization** (`VecNormalize`) in training.
3. **BC warm-start** (`--bc-episodes` in `train_ppo.py`): hand-rolled behavior
   cloning of the 100%-success greedy expert (`airground/experts.py`) directly
   on the SB3 policy — no `imitation` dependency — then PPO fine-tunes for cost
   trade-offs.

If success stays low, escalate: curriculum (2-floor maps first), then
`RecurrentPPO` (sb3-contrib) since the 9x9 crop is a partial observation.
