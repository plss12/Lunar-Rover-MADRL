from pathlib import Path

import numpy as np
import pytest

from gym_lunar_rover.algorithms.DDDQL import InferenceDDDQNAgent
from gym_lunar_rover.algorithms.MAPPO import InferenceMAPPOAgent, critic_model

SAVED = Path(__file__).resolve().parent.parent / "saved_trains"
DDDQL_WEIGHTS = SAVED / "DDDQL" / "model_weights_steps_3750000.weights.h5"
MAPPO_ACTOR = SAVED / "MAPPO" / "actor_weights_steps_6000000.weights.h5"
MAPPO_CRITIC = SAVED / "MAPPO" / "critic_weights_steps_6000000.weights.h5"

# Sum of absolute values (rounded to 4 decimals) of each layer's (kernel, bias), in
# creation order, as read by Keras 2.10, the version the checkpoint was saved with.
DDDQL_REFERENCE_WEIGHTS = [
    ("conv2d", [40.9886, 1.2211]),
    ("conv2d", [510.272, 2.655]),
    ("conv2d", [1203.4287, 5.5347]),
    ("dense", [44.6948, 2.5216]),
    ("dense", [133.4898, 3.7751]),
    ("dense", [382.6011, 7.3879]),
    ("dense", [1097.8025, 8.079]),
    ("dense", [1251.7164, 17.7199]),
    ("dense", [2464.2053, 13.4618]),
    ("dense", [57.1089, 0.0825]),
    ("dense", [2072.825, 15.1307]),
    ("dense", [160.5778, 1.388]),
]
# Q-values for `reference_input()` under Keras 2.10. Float32 noise between Keras
# versions is around 0.3% (Keras 2.10 itself differs that much between
# `model()` and `model.predict()`), so they are compared loosely.
DDDQL_REFERENCE_Q = [-4.255886077880859, -4.777458190917969, -3.2820627689361572,
                     -6.365819931030273, -3.4309256076812744]


def reference_input() -> tuple[np.ndarray, np.ndarray]:
    obs_visits = np.linspace(0, 1, 7 * 7 * 2, dtype=np.float32).reshape(1, 7, 7, 2)
    info = np.linspace(0, 1, 7, dtype=np.float32).reshape(1, 7)
    return obs_visits, info


def creation_order(name: str) -> tuple[str, int]:
    kind, _, suffix = name.rpartition("_")
    return (kind, int(suffix)) if suffix.isdigit() else (name, 0)


@pytest.mark.skipif(not DDDQL_WEIGHTS.exists(), reason="DDDQL checkpoint not available")
def test_legacy_keras2_dddql_checkpoint_loads_every_layer() -> None:
    agent = InferenceDDDQNAgent(7, 7, 5, str(DDDQL_WEIGHTS))
    layers = sorted(
        (layer for layer in agent.primary_network.layers if layer.weights),
        key=lambda layer: creation_order(layer.name),
    )

    loaded = [
        (creation_order(layer.name)[0], [float(np.abs(w).sum()) for w in layer.get_weights()])
        for layer in layers
    ]

    assert [kind for kind, _ in loaded] == [kind for kind, _ in DDDQL_REFERENCE_WEIGHTS]
    for (_, sums), (_, expected) in zip(loaded, DDDQL_REFERENCE_WEIGHTS):
        np.testing.assert_allclose(sums, expected, rtol=0, atol=1e-4)

    q_values = agent.primary_network(list(reference_input()), training=False).numpy()
    np.testing.assert_allclose(q_values[0], DDDQL_REFERENCE_Q, rtol=1e-2)


@pytest.mark.skipif(not MAPPO_ACTOR.exists(), reason="MAPPO checkpoint not available")
def test_mappo_checkpoints_load() -> None:
    agent = InferenceMAPPOAgent(7, 7, 5, str(MAPPO_ACTOR))
    critic = critic_model(12)
    critic.load_weights(str(MAPPO_CRITIC))

    probs = agent.actor(list(reference_input()), training=False).numpy()

    np.testing.assert_allclose(probs.sum(), 1.0, rtol=1e-5)
