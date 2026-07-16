# Lab notes

One line per experiment: hypothesis -> change -> result. Newest last.

## 2026-07-17

- **run ppo_v0** (commit abf47ea): PPO 2M steps, generated maps, no shaping.
  Hypothesis: MLP over 9x9 crop learns mode arbitration. Result: **0/50
  success**, worse than random (10%). Diagnosis: sparse reward + negative
  step cost -> policy collapse to inactivity.
- **diagnostic**: BFS over env dynamics -> all maps solvable (townhouse 40
  steps, generated 6-42). Greedy distance-descent expert -> 100% success.
  Conclusion: not an env bug; exploration/reward problem.
- **fix**: added potential-based reward shaping (Phi = -dist_to_goal),
  VecNormalize, BC warm-start from greedy expert. Shaped greedy return
  +36.6 (townhouse) / +13.4 (generated). Next: re-run PPO with these.
