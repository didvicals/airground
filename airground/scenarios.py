"""Test scenarios: indoor police buildings.

Scenario 'townhouse': two-floor residence, barricaded suspect upstairs.
- Floor 0: entry (E), hallway, debris-strewn room (o), stairwell (S).
- Floor 1: two bedrooms off a corridor; suspect (X) in the right bedroom;
  goal (G) is an observation point outside the suspect's door.
Stairs are flight-only (wheels cannot climb) -> ground-only baseline is
infeasible, demonstrating why the hybrid platform exists.
"""

from airground.environment import Building

TOWNHOUSE_F0 = [
    "####################",
    "#E........#.......S#",
    "#.........#........#",
    "#...####..#..oooo..#",
    "#...#..#..#..oooo..#",
    "#...#..#..#..oooo..#",
    "#...####..#........#",
    "#..................#",
    "#.........#........#",
    "####################",
]

TOWNHOUSE_F1 = [
    "####################",
    "#.................S#",
    "#..######..######..#",
    "#..#....#..#....#..#",
    "#..#....#..#..X.#..#",
    "#..#.......#....#..#",
    "#..######..###.##..#",
    "#.............G....#",
    "#..................#",
    "####################",
]


def townhouse() -> Building:
    return Building([TOWNHOUSE_F0, TOWNHOUSE_F1])
