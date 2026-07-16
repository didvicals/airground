"""Smoke tests: generator validity, env API contract, planner regression.

Run: python -m pytest tests/ -q   (or python tests/test_framework.py)
"""

import numpy as np

from airground.envs import PoliceEnv, generate
from airground.mission import Phase, phase_weights
from airground.models import Mode, Params
from airground.planner import plan
from airground.scenarios import townhouse


def test_generator_valid_buildings():
    for seed in range(20):
        b = generate(seed)
        assert b.start and b.goal and b.suspect
        assert b.start.floor == 0
        assert b.goal.floor == b.n_floors - 1
        w = phase_weights(Phase.APPROACH)
        # flight-reachability was checked inside generate(); the full planner
        # must also find a route under approach weights
        pl = plan(b, Params.load(), w, b.start, b.goal)
        assert pl.feasible, f"seed {seed}: no stealth-feasible plan"


def test_planner_townhouse_regression():
    b = townhouse()
    p = Params.load()
    pl = plan(b, p, phase_weights(Phase.APPROACH), b.start, b.goal)
    assert pl.feasible
    assert pl.exposure_dbs == 0.0
    assert pl.transitions == 2
    assert 3000 < pl.energy_j < 6000


def test_env_api_random_rollout():
    env = PoliceEnv(building_fn=generate)
    obs, _ = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape
    for _ in range(200):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        assert np.isfinite(r)
        assert obs.shape == env.observation_space.shape
        if term or trunc:
            obs, _ = env.reset()
    env.close()


def test_env_ground_step_costs_less_than_flight():
    env = PoliceEnv()  # fixed townhouse
    env.reset(seed=0)
    assert env.mode is Mode.GROUND
    _, r_ground, *_ = env.step(0)  # move in ground mode
    env.reset(seed=0)
    env.step(4)  # same direction, flight mode (includes takeoff)
    _, r_flight, *_ = env.step(4)
    assert r_ground > r_flight  # less negative = cheaper


if __name__ == "__main__":
    test_generator_valid_buildings()
    test_planner_townhouse_regression()
    test_env_api_random_rollout()
    test_env_ground_step_costs_less_than_flight()
    print("all tests passed")
