import gymnasium as gym
import numpy as np
import pytest

import gym_lunar_rover  # noqa: F401  (registers 'lunar-rover-v0')
from gym_lunar_rover.envs import Test_env
from gym_lunar_rover.envs.Lunar_Rover_env import LunarEnv
from gym_lunar_rover.envs.Utils import RoverAction, RoverRewards


def make_env(n_agents: int = 2, grid_size: int = 8, vision_range: int = 2) -> LunarEnv:
    env = LunarEnv(n_agents, grid_size, vision_range)
    env.reset(seed=0)
    return env


def place_rover(env: LunarEnv, rover_index: int, position: tuple[int, int]) -> None:
    """Moves a rover to an empty cell so tests can control its surroundings."""
    rover = env.rovers[rover_index]
    env.grid[rover.position] = env.initial_grid[rover.position]
    env.grid[position] = rover.agent_id
    rover.position = position


def test_reset_returns_one_observation_per_agent() -> None:
    env = LunarEnv(3, 10, 2)
    obs, info = env.reset(seed=1)

    assert len(obs) == 3
    assert all(o.shape == (5, 5) for o in obs)
    assert info == {}


def test_reset_with_same_seed_is_reproducible() -> None:
    env = LunarEnv(2, 10, 2)
    env.reset(seed=123)
    first_grid = env.grid.copy()
    env.reset(seed=456)
    env.reset(seed=123)

    np.testing.assert_array_equal(env.grid, first_grid)


def test_test_env_layouts_are_reproducible() -> None:
    env = Test_env.TestEnv(3, 12, 3)
    env.reset(seed=4)
    first_grid = env.grid.copy()
    env.reset(seed=4)

    np.testing.assert_array_equal(env.grid, first_grid)


def test_get_movements_at_corner_excludes_out_of_bounds() -> None:
    env = make_env()
    corner = (0, 0)
    env.grid[corner] = 0
    env.initial_grid[corner] = 0
    place_rover(env, 0, corner)

    movements = env.rovers[0].get_movements()

    assert RoverAction.UP.value not in movements
    assert RoverAction.LEFT.value not in movements
    assert RoverAction.DOWN.value in movements
    assert RoverAction.RIGHT.value in movements


def test_invalid_action_is_penalised_without_moving() -> None:
    env = make_env()
    corner = (0, 0)
    env.grid[corner] = 0
    env.initial_grid[corner] = 0
    place_rover(env, 0, corner)
    rover = env.rovers[0]
    visits_before = rover.visits_maps[corner]

    _, _, reward, done, _ = rover.step(RoverAction.UP.value)

    assert reward == RoverRewards.INVALID.value
    assert not done
    assert rover.position == corner
    assert rover.visits_maps[corner] == visits_before + 1


def test_env_step_returns_scalar_reward_and_per_rover_rewards() -> None:
    env = make_env()
    actions = [RoverAction.WAIT.value] * env.n_agents

    _, reward, terminated, truncated, info = env.step(actions)

    assert np.isscalar(reward)
    assert len(info["rewards"]) == env.n_agents
    assert all(np.isscalar(r) for r in info["rewards"])
    assert reward == sum(info["rewards"])
    assert terminated is False
    assert truncated is False


def test_env_total_reward_matches_rovers() -> None:
    env = make_env()
    env.step([RoverAction.WAIT.value] * env.n_agents)

    assert env._get_total_reward() == sum(r.total_reward for r in env.rovers)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_random_episode_through_gym_make_runs(seed: int) -> None:
    env = gym.make("lunar-rover-v0", n_agents=2, grid_size=8, vision_range=2)
    env.reset(seed=seed)
    env.action_space.seed(seed)

    for _ in range(200):
        _, reward, terminated, _, _ = env.step(env.action_space.sample())
        assert np.isscalar(reward)
        if terminated:
            break
    env.close()
