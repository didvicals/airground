"""2.5D multi-floor grid environment for indoor police scenarios.

Cell legend:
    '#' wall            (impassable both modes)
    '.' floor           (both modes; landable)
    'o' debris/step>3cm (flight only — exceeds ground step limit)
    'S' stairwell       (flight only; connects floors at same (r, c))
    'E' start           (floor cell)
    'G' goal            (floor cell)
    'X' suspect         (floor cell marker, suspect position)

Acoustic propagation between robot and suspect: inverse-square spreading,
wall transmission loss counted by ray-cast on the robot's floor, plus a
floor-slab loss per floor of separation.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log10, sqrt

from airground.models import Mode, Params

CELL_M = 1.0            # grid resolution
FLOOR_SLAB_TL_DB = 45.0  # concrete slab transmission loss
STAIR_TRANSIT_S = 4.0    # vertical stair flight, one floor


@dataclass(frozen=True)
class Pose:
    floor: int
    r: int
    c: int


class Building:
    def __init__(self, floors: list[list[str]]):
        self.grid = [[list(row) for row in f] for f in floors]
        self.n_floors = len(floors)
        self.rows = len(floors[0])
        self.cols = len(floors[0][0])
        self.start = self._find("E")
        self.goal = self._find("G")
        self.suspect = self._find("X")
        for mark in "EGX":  # markers are ordinary floor cells for planning
            p = self._find(mark)
            if p:
                self.grid[p.floor][p.r][p.c] = "."

    def _find(self, ch: str) -> Pose | None:
        for f in range(self.n_floors):
            for r in range(self.rows):
                for c in range(self.cols):
                    if self.grid[f][r][c] == ch:
                        return Pose(f, r, c)
        return None

    def cell(self, p: Pose) -> str:
        return self.grid[p.floor][p.r][p.c]

    def in_bounds(self, p: Pose) -> bool:
        return (0 <= p.floor < self.n_floors and 0 <= p.r < self.rows
                and 0 <= p.c < self.cols)

    def passable(self, p: Pose, mode: Mode) -> bool:
        if not self.in_bounds(p):
            return False
        ch = self.cell(p)
        if ch == "#":
            return False
        if mode is Mode.GROUND:
            return ch == "."
        return ch in ".oS"

    def landable(self, p: Pose) -> bool:
        return self.in_bounds(p) and self.cell(p) == "."

    def is_stair(self, p: Pose) -> bool:
        return self.in_bounds(p) and self.cell(p) == "S"

    # ---------------------------------------------------------- acoustics

    def walls_between(self, a: Pose, b_r: int, b_c: int) -> int:
        """Wall cells crossed by a straight ray on floor a.floor (Bresenham)."""
        walls = 0
        r0, c0, r1, c1 = a.r, a.c, b_r, b_c
        dr, dc = abs(r1 - r0), abs(c1 - c0)
        sr = 1 if r1 > r0 else -1
        sc = 1 if c1 > c0 else -1
        err = dr - dc
        r, c = r0, c0
        while (r, c) != (r1, c1):
            e2 = 2 * err
            if e2 > -dc:
                err -= dc
                r += sr
            if e2 < dr:
                err += dr
                c += sc
            if (r, c) != (r1, c1) and self.grid[a.floor][r][c] == "#":
                walls += 1
        return walls

    def received_spl_at_suspect(self, p: Params, robot: Pose,
                                mode: Mode) -> float:
        """SPL (dB) at suspect position for robot operating at `robot`."""
        s = self.suspect
        dist = CELL_M * sqrt((robot.r - s.r) ** 2 + (robot.c - s.c) ** 2)
        dist = max(dist, 1.0)
        src = (p.flight_spl_db_1m if mode is Mode.FLIGHT
               else p.ground_spl_db_1m)
        floors_apart = abs(robot.floor - s.floor)
        walls = self.walls_between(robot, s.r, s.c) if floors_apart == 0 else 0
        return (src - 20.0 * log10(dist)
                - walls * p.wall_tl_db - floors_apart * FLOOR_SLAB_TL_DB)

    def exposure_dbs(self, p: Params, robot: Pose, mode: Mode,
                     dt_s: float, margin_db: float = 3.0) -> float:
        """Audibility exposure (excess dB x seconds) accrued over dt at pose."""
        excess = (self.received_spl_at_suspect(p, robot, mode)
                  - (p.ambient_db + margin_db))
        return max(excess, 0.0) * dt_s
