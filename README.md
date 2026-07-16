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
pip install -e .            # or .[train] for PPO
python tests/test_framework.py
python scripts/evaluate.py  # baseline comparison table
python scripts/train_ppo.py --steps 200000 --nenv 4   # CPU smoke train
```
