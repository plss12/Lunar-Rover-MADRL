"""Smoke tests for the training and evaluation scripts at the repository root."""

from pathlib import Path

import numpy as np
import pytest

import Test
import Train


@pytest.fixture
def workdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Runs the scripts in an empty directory so they never touch saved_trains/."""
    for algorithm in ("DDDQL", "MAPPO"):
        (tmp_path / "saved_trains" / algorithm).mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_training_does_not_render_by_default() -> None:
    assert Train.env.unwrapped.render_mode is None


def test_last_states_are_none_only_for_finished_rovers() -> None:
    Train.env.reset(seed=0)
    Train.env.unwrapped.rovers[0].done = True

    states = Train.get_last_states()

    assert states[0] is None
    for rovers_state, no_rovers_state in states[1:]:
        assert rovers_state.shape == no_rovers_state.shape == (Train.grid_size, Train.grid_size)


def test_dddql_training_during_warm_up_reports_zero_loss(workdir: Path) -> None:
    # 300 steps never leave the 2500 warm-up steps, so no loss is ever computed.
    _, _, average_loss, _, _ = Train.train_dddql(300, 0)

    assert average_loss == 0
    assert (workdir / "saved_trains/DDDQL/model_weights_steps_300.weights.h5").exists()


def test_mappo_training_runs_and_saves(workdir: Path) -> None:
    # 600 steps train once mid-episode (every 500 steps) and once at the end.
    _, _, actor_loss, critic_loss, _, _ = Train.train_mappo(600, 0)

    assert np.isfinite(actor_loss)
    assert np.isfinite(critic_loss)
    assert (workdir / "saved_trains/MAPPO/actor_weights_steps_600.weights.h5").exists()


def test_evaluation_with_unknown_algorithm_exits(capsys: pytest.CaptureFixture[str]) -> None:
    Test.test("UNKNOWN", 0)

    assert "no existe" in capsys.readouterr().out


@pytest.mark.parametrize(("algorithm", "steps"), [("DDDQL", 3750000), ("MAPPO", 6000000)])
def test_evaluation_of_saved_models_finishes(
    algorithm: str, steps: int, capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(Path(__file__).resolve().parent.parent)
    monkeypatch.setattr(Test.test_env, "render_mode", None)

    Test.test(algorithm, steps)

    # Every seed reports a result, including those where the policy gets stuck.
    lines = [line for line in capsys.readouterr().out.splitlines() if "seed" in line]
    assert len(lines) == 4
