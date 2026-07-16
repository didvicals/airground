"""Mode-wise energy and acoustic cost models for RefBot-1.

Parameters are market-calibrated (BRINC Lemur 2 mission profile,
CapsuleBot measured power/noise, TABV speeds). See docs/reference-platform.md.

These models feed the mode-arbitration cost function:
    J(segment, mode) = w_E * E(mode) + w_T * T(mode) + w_N * exposure(mode)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import yaml

PARAMS_PATH = Path(__file__).parent / "platform_params.yaml"


class Mode(Enum):
    FLIGHT = "flight"
    GROUND = "ground"


@dataclass(frozen=True)
class Params:
    mass_kg: float
    battery_wh: float
    usable_fraction: float
    reserve_return_fraction: float
    avionics_base_w: float
    hover_w: float
    k_v: float
    ground_idle_w: float
    k_roll: float
    flight_nominal_ms: float
    ground_nominal_ms: float
    flight_spl_db_1m: float
    ground_spl_db_1m: float
    ambient_db: float
    wall_tl_db: float
    land_duration_s: float
    land_energy_j: float
    takeoff_duration_s: float
    takeoff_energy_j: float

    @classmethod
    def load(cls, path: Path = PARAMS_PATH) -> "Params":
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cls(
            mass_kg=raw["platform"]["mass_kg"],
            battery_wh=raw["battery"]["capacity_wh"],
            usable_fraction=raw["battery"]["usable_fraction"],
            reserve_return_fraction=raw["battery"]["reserve_return_fraction"],
            avionics_base_w=raw["power"]["avionics_base_w"],
            hover_w=raw["power"]["flight"]["hover_w"],
            k_v=raw["power"]["flight"]["k_v_per_m2s2"],
            ground_idle_w=raw["power"]["ground"]["idle_w"],
            k_roll=raw["power"]["ground"]["k_roll_w_per_ms"],
            flight_nominal_ms=raw["speed"]["flight_nominal_ms"],
            ground_nominal_ms=raw["speed"]["ground_nominal_ms"],
            flight_spl_db_1m=raw["acoustics"]["flight_spl_db_1m"],
            ground_spl_db_1m=raw["acoustics"]["ground_spl_db_1m"],
            ambient_db=raw["acoustics"]["ambient_db"],
            wall_tl_db=raw["acoustics"]["wall_transmission_loss_db"],
            land_duration_s=raw["transition"]["land_duration_s"],
            land_energy_j=raw["transition"]["land_energy_j"],
            takeoff_duration_s=raw["transition"]["takeoff_duration_s"],
            takeoff_energy_j=raw["transition"]["takeoff_energy_j"],
        )


# ---------------------------------------------------------------- energy

def power_w(p: Params, mode: Mode, speed_ms: float) -> float:
    """Total electrical draw (avionics included) at steady speed."""
    if mode is Mode.FLIGHT:
        return p.avionics_base_w + p.hover_w * (1.0 + p.k_v * speed_ms**2)
    return p.avionics_base_w + p.ground_idle_w + p.k_roll * speed_ms


def segment_energy_j(p: Params, mode: Mode, distance_m: float,
                     speed_ms: float | None = None) -> tuple[float, float]:
    """Energy (J) and time (s) to cover a segment at nominal or given speed."""
    v = speed_ms or (p.flight_nominal_ms if mode is Mode.FLIGHT
                     else p.ground_nominal_ms)
    t = distance_m / v
    return power_w(p, mode, v) * t, t


def transition_cost(p: Params, from_mode: Mode, to_mode: Mode) -> tuple[float, float]:
    """(energy J, time s) for a mode switch; zero if no switch."""
    if from_mode is to_mode:
        return 0.0, 0.0
    if to_mode is Mode.GROUND:
        return p.land_energy_j, p.land_duration_s
    return p.takeoff_energy_j, p.takeoff_duration_s


def endurance_min(p: Params, mode: Mode, speed_ms: float = 0.0) -> float:
    """Minutes of operation at given speed on usable battery."""
    usable_j = p.battery_wh * 3600.0 * p.usable_fraction
    return usable_j / power_w(p, mode, speed_ms) / 60.0


# -------------------------------------------------------------- acoustics

def received_spl_db(p: Params, mode: Mode, distance_m: float,
                    n_walls: int = 0) -> float:
    """SPL at listener: inverse-square spreading + wall transmission loss.

    Free-field approximation; indoor reverberation makes this optimistic
    at long range — calibrate with measurements (open item).
    """
    src = p.flight_spl_db_1m if mode is Mode.FLIGHT else p.ground_spl_db_1m
    spreading = 20.0 * math.log10(max(distance_m, 1.0))
    return src - spreading - n_walls * p.wall_tl_db


def is_audible(p: Params, mode: Mode, distance_m: float,
               n_walls: int = 0, margin_db: float = 3.0) -> bool:
    """Audible if received level exceeds ambient by margin."""
    return received_spl_db(p, mode, distance_m, n_walls) > p.ambient_db + margin_db


def detection_radius_m(p: Params, mode: Mode, n_walls: int = 0,
                       margin_db: float = 3.0) -> float:
    """Range inside which the robot is audible above ambient."""
    src = p.flight_spl_db_1m if mode is Mode.FLIGHT else p.ground_spl_db_1m
    excess = src - n_walls * p.wall_tl_db - (p.ambient_db + margin_db)
    if excess <= 0:
        return 0.0
    return 10.0 ** (excess / 20.0)
