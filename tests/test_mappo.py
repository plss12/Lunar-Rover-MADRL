import numpy as np
import pytest
import tensorflow as tf

from gym_lunar_rover.algorithms.MAPPO import MAPPOAgent

AVAILABLE_ACTIONS = [0, 1, 2, 3, 4]


def make_agent(map_shape: int = 8) -> MAPPOAgent:
    return MAPPOAgent(
        map_shape, 7, 7, 5, 2,
        clip_rewards=False, gamma=0.9, lamda=0.95, clip=0.2, entropy_coef=0.01,
        lr=1e-3, min_lr=1e-5, lr_decay_factor=0.5, patience=10, cooldown=5,
        dropout_rate=0.0, l1_rate=0.0, l2_rate=0.0,
    )


def random_step_inputs(rng: np.random.Generator) -> tuple[np.ndarray, ...]:
    observation = rng.random((7, 7), dtype=np.float32)
    visits = rng.random((7, 7), dtype=np.float32)
    rovers_state = rng.random((8, 8), dtype=np.float32)
    no_rovers_state = rng.random((8, 8), dtype=np.float32)
    info = rng.random(7, dtype=np.float32)
    return observation, visits, rovers_state, no_rovers_state, info


def collect(agent: MAPPOAgent, agent_ids: list[int], steps: int, last_done: bool) -> None:
    """Stores the same `steps` random transitions for every agent in `agent_ids`."""
    rng = np.random.default_rng(0)
    for step in range(steps):
        observation, visits, rovers_state, no_rovers_state, info = random_step_inputs(rng)
        action, prob, value = agent.act(
            observation, visits, rovers_state, no_rovers_state, info, AVAILABLE_ACTIONS
        )
        reward = float(rng.normal())
        done = last_done and step == steps - 1
        for agent_id in agent_ids:
            agent.add_experience(
                agent_id, observation, visits, info, action, reward, done,
                AVAILABLE_ACTIONS, rovers_state, no_rovers_state, value, prob,
            )


def test_discounted_rewards_reset_at_episode_end() -> None:
    agent = make_agent()
    rewards = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    dones = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)

    result = agent.compute_discounted_rewards(rewards, dones, 0.5)

    np.testing.assert_allclose(result, [1.5, 1.0, 1.5, 1.0])


def test_discounted_rewards_bootstrap_cut_trajectories() -> None:
    agent = make_agent()
    rewards = np.array([1.0, 1.0], dtype=np.float32)

    cut = agent.compute_discounted_rewards(rewards, np.array([0.0, 0.0]), 0.5, last_value=4.0)
    finished = agent.compute_discounted_rewards(rewards, np.array([0.0, 1.0]), 0.5, last_value=4.0)

    np.testing.assert_allclose(cut, [2.5, 3.0])
    # The bootstrap value is ignored when the trajectory really ends.
    np.testing.assert_allclose(finished, [1.5, 1.0])


def test_advantages_bootstrap_cut_trajectories() -> None:
    agent = make_agent()
    rewards = tf.constant([1.0, 1.0])
    values = tf.constant([2.0, 2.0])
    dones = tf.constant([0.0, 0.0])

    advantages = agent.compute_advantages(rewards, values, dones, 0.5, 1.0, last_value=4.0)
    returns = agent.compute_discounted_rewards(rewards.numpy(), dones.numpy(), 0.5, last_value=4.0)

    np.testing.assert_allclose(advantages, [0.5, 1.0])
    # With lambda = 1, GAE reduces to the bootstrapped return minus the value.
    np.testing.assert_allclose(advantages, returns - values.numpy())


def test_critic_loss_is_elementwise_mse() -> None:
    agent = make_agent()
    states = tf.random.uniform((4, 8, 8, 2), seed=0)
    returns = np.array([1.0, -2.0, 3.0, 0.5], dtype=np.float32)

    values = tf.squeeze(agent.critic(states, training=True), axis=-1)
    expected = tf.reduce_mean(tf.square(returns - values))

    loss = agent.compute_critic_loss(states, returns)

    np.testing.assert_allclose(loss.numpy(), expected.numpy(), rtol=1e-5)


def test_act_samples_a_valid_action() -> None:
    agent = make_agent()
    rng = np.random.default_rng(0)
    available_actions = [0, 2, 4]

    for _ in range(10):
        action, prob, value = agent.act(*random_step_inputs(rng), available_actions)
        assert isinstance(action, int)
        assert action in available_actions
        assert 0.0 < prob <= 1.0
        assert np.isfinite(value)


def test_train_updates_from_collected_experiences() -> None:
    agent = make_agent()
    collect(agent, [0, 1], steps=6, last_done=False)
    rng = np.random.default_rng(1)
    last_states = [random_step_inputs(rng)[2:4] for _ in range(2)]

    actor_loss, critic_loss = agent.train(last_states)

    assert np.isfinite(actor_loss)
    assert np.isfinite(critic_loss)
    assert len(agent.buffer) == 0


def test_train_requires_last_state_of_unfinished_agents() -> None:
    agent = make_agent()
    collect(agent, [0], steps=3, last_done=False)

    with pytest.raises(ValueError, match="agente 0"):
        agent.train([None, None])


def test_train_does_not_need_last_state_of_finished_agents() -> None:
    agent = make_agent()
    collect(agent, [0], steps=3, last_done=True)

    actor_loss, _ = agent.train([None, None])

    assert np.isfinite(actor_loss)


def test_gradients_are_averaged_over_agents_with_experiences(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def captured_actor_gradients(agent_ids: list[int]) -> list[np.ndarray]:
        tf.keras.utils.set_random_seed(0)
        agent = make_agent()
        captured: list[list[np.ndarray]] = []
        monkeypatch.setattr(
            agent.act_optimizer, "apply_gradients",
            lambda grads_and_vars: captured.append([g.numpy() for g, _ in grads_and_vars]),
        )
        monkeypatch.setattr(agent.cri_optimizer, "apply_gradients", lambda grads_and_vars: None)
        collect(agent, agent_ids, steps=4, last_done=True)
        agent.train([None, None])
        return captured[0]

    # Both agents hold identical experiences, so their average must equal a single agent's
    # gradient: a finished agent without experiences must not shrink the update.
    one_agent = captured_actor_gradients([0])
    two_agents = captured_actor_gradients([0, 1])

    for single, averaged in zip(one_agent, two_agents):
        np.testing.assert_allclose(single, averaged, rtol=1e-5, atol=1e-7)

