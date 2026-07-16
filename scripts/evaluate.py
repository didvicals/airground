"""Baseline comparison on the townhouse scenario.

Baselines (APPROACH leg, entry -> observation point outside suspect room):
  ground-only   : mode locked to ground (expected infeasible: stairs)
  flight-only   : mode locked to flight (fast, loud, energy-hungry)
  time-optimal  : minimize time only
  energy-optimal: minimize energy only
  proposed      : Layer-3 mission policy (APPROACH weights, stealth constraint)

Then a full mission (approach + 10 min overwatch + return) under the
proposed policy, with battery accounting.

Run: python sim/evaluate.py
"""

from __future__ import annotations

from airground.environment import Building
from airground.mission import Phase, phase_weights
from airground.models import Mode, Params, power_w
from airground.planner import Plan, Weights, plan
from airground.scenarios import townhouse

OVERWATCH_HOLD_S = 600.0


def render(b: Building, p: Plan) -> str:
    """Overlay plan on maps: g=ground, f=flight, *=mode switch."""
    canvas = [[row[:] for row in floor] for floor in b.grid]
    prev_mode = None
    for pose, mode in p.states:
        ch = "g" if mode is Mode.GROUND else "f"
        if prev_mode is not None and mode is not prev_mode:
            ch = "*"
        canvas[pose.floor][pose.r][pose.c] = ch
        prev_mode = mode
    out = []
    for f in range(b.n_floors):
        out.append(f"  floor {f}:")
        out.extend("    " + "".join(row) for row in canvas[f])
    return "\n".join(out)


def describe(name: str, pl: Plan, p: Params) -> str:
    if not pl.feasible:
        return f"  {name:14s} INFEASIBLE"
    return (f"  {name:14s} E={pl.energy_j:7.0f} J  T={pl.time_s:5.0f} s  "
            f"noise={pl.exposure_dbs:6.0f} dB*s  switches={pl.transitions}  "
            f"battery={100 * pl.battery_fraction(p):4.1f}%")


def main() -> None:
    b = townhouse()
    p = Params.load()

    print("=== Townhouse scenario: APPROACH leg (entry -> overwatch point) ===\n")

    baselines = {
        "ground-only": Weights(w_energy=1.0, lock_mode=Mode.GROUND),
        "flight-only": Weights(w_energy=1.0, lock_mode=Mode.FLIGHT),
        "time-optimal": Weights(w_energy=0.0, w_time=1.0),
        "energy-optimal": Weights(w_energy=1.0),
        "proposed": phase_weights(Phase.APPROACH),
    }
    plans: dict[str, Plan] = {}
    for name, w in baselines.items():
        plans[name] = plan(b, p, w, b.start, b.goal)
        print(describe(name, plans[name], p))

    print("\nProposed plan path (g ground, f flight, * switch):")
    print(render(b, plans["proposed"]))

    print("\n=== Full mission, proposed policy ===\n")
    approach = plans["proposed"]

    # Overwatch: hold at goal, pick quieter/cheaper sustainable mode.
    hold_mode = Mode.GROUND  # perch: near-zero acoustic signature, 12x endurance
    hold_e = power_w(p, hold_mode, 0.0) * OVERWATCH_HOLD_S
    hold_alt = power_w(p, Mode.FLIGHT, 0.0) * OVERWATCH_HOLD_S
    print(f"  overwatch 10 min perched : {hold_e:7.0f} J "
          f"(hovering instead would cost {hold_alt:.0f} J)")

    ret = plan(b, p, phase_weights(Phase.RETURN), b.goal, b.start)
    print(describe("return", ret, p))

    total = approach.energy_j + hold_e + ret.energy_j
    usable = p.battery_wh * 3600.0 * p.usable_fraction
    reserve = p.reserve_return_fraction
    print(f"\n  mission total: {total:.0f} J = {100 * total / usable:.1f}% of usable"
          f" battery (reserve floor {100 * reserve:.0f}%)"
          f" -> {'OK' if total <= usable * (1 - reserve) else 'OVER BUDGET'}")
    flight_only_hold = plans["flight-only"].energy_j + hold_alt
    print(f"  flight-only equivalent (approach + hover overwatch): "
          f"{flight_only_hold:.0f} J = "
          f"{100 * flight_only_hold / usable:.1f}% of usable battery")


if __name__ == "__main__":
    main()
