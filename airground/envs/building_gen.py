"""Procedural indoor police-scenario generator.

Layout: corridor spine with rooms above/below, identical stairwell column on
every floor (flight-only), debris patches, suspect in a random top-floor room,
goal in the corridor outside the suspect's door (in the door's acoustic line
of sight -> flying to the goal is audible, driving is not).

Every sampled building is validated: flight-mode BFS must connect entry to
goal. Used for RL domain randomization and for hunting scenarios where
mission-aware arbitration beats single-objective baselines.
"""

from __future__ import annotations

import random
from collections import deque

from functools import lru_cache

from airground.environment import Building, Pose
from airground.models import Mode, Params


@lru_cache(maxsize=1)
def _params() -> Params:
    return Params.load()


def _empty_floor(rows: int, cols: int) -> list[list[str]]:
    g = [["." for _ in range(cols)] for _ in range(rows)]
    for r in range(rows):
        g[r][0] = g[r][cols - 1] = "#"
    for c in range(cols):
        g[0][c] = g[rows - 1][c] = "#"
    return g


def _floor_layout(rng: random.Random, rows: int, cols: int,
                  stair_c: int) -> tuple[list[list[str]], list[tuple[int, int]]]:
    """One floor: corridor at mid-row, rooms above/below, random doors.

    Returns (grid, room_door_cols) where each entry is (door_col, band_sign)
    for later suspect/goal placement. band_sign: -1 room above corridor, +1 below.
    """
    g = _empty_floor(rows, cols)
    mid = rows // 2
    wall_top, wall_bot = mid - 1, mid + 1

    doors: list[tuple[int, int]] = []
    for wall_r, sign in ((wall_top, -1), (wall_bot, +1)):
        for c in range(1, cols - 1):
            g[wall_r][c] = "#"
        # vertical dividers split the band into rooms (min width 4)
        dividers = [1]
        c = 1
        while c < cols - 6:
            c += rng.randint(4, 7)
            if c < cols - 2:
                dividers.append(c)
        dividers.append(cols - 2)
        band = range(1, wall_top) if sign < 0 else range(wall_bot + 1, rows - 1)
        for d in dividers[1:-1]:
            for r in band:
                g[r][d] = "#"
        # one door per room, not through the stairwell column
        for left, right in zip(dividers, dividers[1:]):
            cands = [c for c in range(left + 1, right) if c != stair_c]
            if cands:
                dc = rng.choice(cands)
                g[wall_r][dc] = "."
                doors.append((dc, sign))

    g[mid][stair_c] = "S"
    return g, doors


def _add_debris(rng: random.Random, g: list[list[str]], n_patches: int) -> None:
    rows, cols = len(g), len(g[0])
    for _ in range(n_patches):
        h, w = rng.randint(1, 2), rng.randint(2, 4)
        r0 = rng.randint(1, rows - 2 - h)
        c0 = rng.randint(1, cols - 2 - w)
        for r in range(r0, r0 + h):
            for c in range(c0, c0 + w):
                if g[r][c] == ".":
                    g[r][c] = "o"


def _flight_reachable(b: Building, a: Pose, target: Pose) -> bool:
    seen = {(a.floor, a.r, a.c)}
    q = deque([a])
    while q:
        p = q.popleft()
        if (p.floor, p.r, p.c) == (target.floor, target.r, target.c):
            return True
        nxt = [Pose(p.floor, p.r + dr, p.c + dc)
               for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))]
        if b.is_stair(p):
            nxt += [Pose(p.floor + 1, p.r, p.c), Pose(p.floor - 1, p.r, p.c)]
        for n in nxt:
            key = (n.floor, n.r, n.c)
            if key not in seen and b.passable(n, Mode.FLIGHT):
                seen.add(key)
                q.append(n)
    return False


def generate(seed: int | None = None, max_tries: int = 50) -> Building:
    rng = random.Random(seed)
    for _ in range(max_tries):
        rows = rng.choice((11, 13))
        cols = rng.randint(20, 26)
        n_floors = rng.randint(2, 3)
        mid = rows // 2
        stair_c = rng.choice((2, cols - 3))

        floors, top_doors = [], []
        for f in range(n_floors):
            g, doors = _floor_layout(rng, rows, cols, stair_c)
            _add_debris(rng, g, rng.randint(1, 4))
            if f == n_floors - 1:
                top_doors = doors
            floors.append(g)

        if not top_doors:
            continue
        # suspect room on top floor; goal in corridor outside its door
        door_c, sign = rng.choice(top_doors)
        top = floors[-1]
        # suspect >= 3 cells from the goal: ground approach stays below the
        # audibility threshold (53 dB - 20*log10(3) < ambient + 3), flight
        # through the door line of sight stays loud
        room_r = mid - 3 if sign < 0 else mid + 3
        room_r = max(1, min(rows - 2, room_r))
        if top[room_r][door_c] != "." or top[mid][door_c] != ".":
            continue
        top[room_r][door_c] = "X"
        top[mid][door_c] = "G"
        # entry: corridor end on floor 0
        e_c = 1 if stair_c != 1 else 3
        if floors[0][mid][e_c] != ".":
            continue
        floors[0][mid][e_c] = "E"

        b = Building([["".join(row) for row in g] for g in floors])
        if not (b.start and b.goal and b.suspect
                and _flight_reachable(b, b.start, b.goal)):
            continue
        # stealth-feasibility: an APPROACH-phase (no audible edges) route
        # must exist, else the scenario is unwinnable for any policy
        from airground.mission import Phase, phase_weights
        from airground.planner import plan
        if plan(b, _params(), phase_weights(Phase.APPROACH),
                b.start, b.goal).feasible:
            return b
    raise RuntimeError("failed to generate a valid building")
