import numpy as np

from gym_lunar_rover.algorithms.DDDQL import DoubleDuelingDQNAgent


def make_agent(warm_up_steps: int = 0) -> DoubleDuelingDQNAgent:
    return DoubleDuelingDQNAgent(
        7, 7, 5, buffer_size=100, batch_size=8, warm_up_steps=warm_up_steps,
        clip_rewards=False, epsilon=1.0, min_epsilon=0.1, epsilon_decay=1e-3, gamma=0.9,
        lr=1e-3, min_lr=1e-5, lr_decay_factor=0.5, patience=10, cooldown=5,
        dropout_rate=0.0, l1_rate=0.0, l2_rate=0.0, update_target_freq=2,
    )


def fill_buffer(agent: DoubleDuelingDQNAgent, rng: np.random.Generator, n: int) -> None:
    for i in range(n):
        done = i % 4 == 3
        agent.add_experience(
            rng.random((7, 7)), rng.random((7, 7)), rng.random(7), int(rng.integers(5)),
            float(rng.normal()), rng.random((7, 7)), rng.random((7, 7)), rng.random(7), done,
            # A finished rover has no available actions.
            [] if done else [0, 1, 2],
        )


def test_act_only_returns_available_actions() -> None:
    agent = make_agent()
    agent.epsilon = 0.0
    rng = np.random.default_rng(0)

    for _ in range(5):
        action = agent.act(rng.random((7, 7)), rng.random((7, 7)), rng.random(7), [1, 3])
        assert action in (1, 3)


def test_train_returns_finite_loss_and_decays_epsilon() -> None:
    agent = make_agent()
    rng = np.random.default_rng(0)
    fill_buffer(agent, rng, 16)

    losses = [agent.train() for _ in range(3)]

    assert all(loss is not None and np.isfinite(loss) for loss in losses)
    assert agent.update_counter == 3
    assert agent.epsilon < 1.0


def test_train_skips_updates_during_warm_up() -> None:
    agent = make_agent(warm_up_steps=2)
    fill_buffer(agent, np.random.default_rng(0), 4)

    assert agent.train() is None
    assert agent.train() is None
    assert agent.train() is not None
