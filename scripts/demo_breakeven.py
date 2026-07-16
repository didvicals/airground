"""Break-even analysis: when is flying worth it?

Compares flight vs ground per segment under the market-calibrated model,
including transition overhead and acoustic detection radius. This is the
sanity check that grounds the mode-arbitration cost-function weights.

Run: python sim/demo_breakeven.py
"""

from airground.models import (
    Mode,
    Params,
    detection_radius_m,
    endurance_min,
    power_w,
    received_spl_db,
    segment_energy_j,
    transition_cost,
)


def main() -> None:
    p = Params.load()

    print("=== RefBot-1 market-calibrated model ===\n")

    print("Steady-state power (avionics included):")
    for mode, v in ((Mode.FLIGHT, p.flight_nominal_ms),
                    (Mode.GROUND, p.ground_nominal_ms)):
        print(f"  {mode.value:7s} @ {v:.1f} m/s : {power_w(p, mode, v):7.1f} W")

    print("\nEndurance on usable battery:")
    print(f"  flight (hover)      : {endurance_min(p, Mode.FLIGHT):6.1f} min")
    print(f"  ground @0.7 m/s     : {endurance_min(p, Mode.GROUND, 0.7):6.1f} min")

    print("\nAcoustic detection radius (3 dB above 43 dB ambient):")
    for walls in (0, 1, 2):
        rf = detection_radius_m(p, Mode.FLIGHT, walls)
        rg = detection_radius_m(p, Mode.GROUND, walls)
        print(f"  {walls} wall(s): flight {rf:7.1f} m   ground {rg:5.1f} m")

    print("\nSegment cost, ground-capable path (energy J / time s),")
    print("flight includes takeoff+land transition overhead:")
    cols = ("dist", "fly E", "fly T", "drive E", "drive T")
    print("  {:>6s} | {:>8s} {:>6s} | {:>8s} {:>7s} | cheaper".format(*cols))
    for d in (5, 10, 20, 50, 100):
        ef, tf = segment_energy_j(p, Mode.FLIGHT, d)
        te, tt = transition_cost(p, Mode.GROUND, Mode.FLIGHT)
        le, lt = transition_cost(p, Mode.FLIGHT, Mode.GROUND)
        ef, tf = ef + te + le, tf + tt + lt
        eg, tg = segment_energy_j(p, Mode.GROUND, d)
        winner = "drive" if eg < ef else "fly"
        print(f"  {d:4d} m | {ef:8.0f} {tf:6.0f} | {eg:8.0f} {tg:7.0f} | {winner}"
              f"  (fly {ef/eg:4.0f}x energy, saves {tg-tf:4.0f} s)")

    print("\nExample: suspect behind 1 wall, robot at 8 m:")
    for mode in Mode:
        spl = received_spl_db(p, mode, 8.0, n_walls=1)
        audible = "AUDIBLE" if spl > p.ambient_db + 3 else "quiet"
        print(f"  {mode.value:7s}: {spl:5.1f} dB received -> {audible}")


if __name__ == "__main__":
    main()
