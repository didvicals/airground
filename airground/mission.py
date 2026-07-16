"""Layer-3 mission policy: mission phase -> arbitration weights.

This is the novel layer of the pipeline. Phases follow police indoor
tactics; weights convert seconds and acoustic exposure into
joule-equivalents so all objectives share one currency.

Weight rationale (initial, to be tuned against operator feedback):
- APPROACH: stealth dominates. Audible edges forbidden outright when a
  suspect position estimate exists; time mildly valued.
- SEARCH: no confirmed suspect, speed matters (life safety), noise
  moderately penalized (may alert an unlocated suspect).
- OVERWATCH: hold position; energy dominates (endurance = mission value).
- RETURN: energy-optimal with battery-reserve enforcement.
"""

from __future__ import annotations

from enum import Enum

from airground.planner import Weights


class Phase(Enum):
    APPROACH = "approach"
    SEARCH = "search"
    OVERWATCH = "overwatch"
    RETURN = "return"


def phase_weights(phase: Phase, battery_frac_remaining: float = 1.0) -> Weights:
    """Map mission phase (+ battery state) to cost weights.

    battery_frac_remaining below 0.3 shifts any phase toward energy
    conservation (mission abort threshold is the operator's call; the
    planner just gets stingy).
    """
    low_batt = battery_frac_remaining < 0.3

    if phase is Phase.APPROACH:
        return Weights(w_energy=1.0, w_time=20.0, w_noise=200.0,
                       forbid_audible=True)
    if phase is Phase.SEARCH:
        w_t = 40.0 if not low_batt else 10.0
        return Weights(w_energy=1.0, w_time=w_t, w_noise=30.0)
    if phase is Phase.OVERWATCH:
        return Weights(w_energy=5.0, w_time=1.0, w_noise=100.0)
    # RETURN
    return Weights(w_energy=3.0, w_time=5.0, w_noise=0.0)
