import os
from pathlib import Path

import numpy as np
import pytest

from gym_lunar_rover.envs.Utils import (
    RoversObjects,
    csv_save_train_dddql,
    csv_save_train_mappo,
    normalize_obs,
    normalize_pos,
    normalize_visits,
)


def test_rover_and_mine_ids_do_not_collide_with_lunar_objects() -> None:
    ids = RoversObjects(3).get_agents_mines()

    assert ids == {4: 5, 6: 7, 8: 9}


def test_normalize_obs_maps_range_to_unit_interval() -> None:
    obs = np.array([-1, 0, 8])

    np.testing.assert_allclose(normalize_obs(obs), [0.0, 1 / 9, 1.0])


def test_normalize_pos_maps_unknown_to_zero() -> None:
    np.testing.assert_allclose(normalize_pos((-1, 11), 12), [0.0, 1.0])


def test_normalize_visits_constant_map_is_zero() -> None:
    np.testing.assert_array_equal(normalize_visits(np.full((3, 3), 4)), np.zeros((3, 3)))


@pytest.mark.parametrize(
    ("save", "filename", "extra"),
    [
        (csv_save_train_dddql, "training_metrics_dddql.csv", (0.5,)),
        (csv_save_train_mappo, "training_metrics_mappo.csv", (0.5, 0.25)),
    ],
)
def test_training_metrics_are_saved_where_graphs_read_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, save, filename: str, extra: tuple
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "saved_trains").mkdir()

    save("ALG", 0, 10, -5, -0.5, *extra, 1, 10)
    save("ALG", 10, 20, -5, -0.5, *extra, 1, 10)

    lines = (tmp_path / "saved_trains" / filename).read_text().splitlines()
    assert len(lines) == 3
    assert not os.path.exists(tmp_path / filename)
