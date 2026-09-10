"""Guards against settings that go nowhere and policies that break quietly.

Each of these was a real defect. None of them raised an error; they produced
wrong results while looking like they worked, which is the only kind of bug
this project keeps finding.
"""

from __future__ import annotations

import argparse

import pytest

from humanoid_rl.cli import _positive, build_parser

# --------------------------------------------------- missing normalisation


@pytest.mark.slow
def test_playback_without_statistics_warns(capsys):
    """Silently handing a normalised policy raw observations ruins it.

    Measured on this project's walking policy: 7,753 reward with statistics,
    154 without -- below the 125 a random policy scores. Returning that quietly
    makes a working policy look like a failed one.
    """
    from humanoid_rl.config import Config
    from humanoid_rl.envs import build_env

    cfg = Config.build("ppo", overrides={"n_envs": 1})
    assert cfg.normalize, "this test is meaningless if normalisation is off"

    env = build_env(cfg, n_envs=1, training=False, stats_path=None, force_dummy=True)
    env.close()

    warning = capsys.readouterr().out
    assert "No observation statistics" in warning
    assert "random" in warning


@pytest.mark.slow
def test_playback_with_statistics_is_silent(tmp_path, capsys):
    from humanoid_rl.config import Config
    from humanoid_rl.envs import build_env

    cfg = Config.build("ppo", overrides={"n_envs": 1})
    seed_env = build_env(cfg, n_envs=1, training=True, force_dummy=True)
    stats = tmp_path / "vecnormalize.pkl"
    seed_env.save(str(stats))
    seed_env.close()
    capsys.readouterr()

    env = build_env(cfg, n_envs=1, training=False, stats_path=stats, force_dummy=True)
    env.close()
    assert "No observation statistics" not in capsys.readouterr().out


@pytest.mark.slow
def test_training_without_statistics_does_not_warn(capsys):
    """A fresh run legitimately starts with no statistics."""
    from humanoid_rl.config import Config
    from humanoid_rl.envs import build_env

    cfg = Config.build("ppo", overrides={"n_envs": 1})
    env = build_env(cfg, n_envs=1, training=True, stats_path=None, force_dummy=True)
    env.close()
    assert "No observation statistics" not in capsys.readouterr().out


# ------------------------------------------------------- dropped task kwargs


@pytest.mark.slow
def test_flat_kwargs_with_combined_tasks_is_rejected():
    """They used to be discarded, so config.json disagreed with the run."""
    import gymnasium as gym

    from humanoid_rl.tasks import wrap_task

    env = gym.make("Humanoid-v5")
    try:
        with pytest.raises(ValueError, match="keyed by task name"):
            wrap_task(env, "velocity,natural", {"arm_cost_weight": 0.5})
    finally:
        env.close()


@pytest.mark.slow
def test_kwargs_for_an_unused_task_is_rejected():
    import gymnasium as gym

    from humanoid_rl.tasks import wrap_task

    env = gym.make("Humanoid-v5")
    try:
        with pytest.raises(ValueError, match="not part of task"):
            wrap_task(env, "natural", {"goal": {"tolerance": 99}})
    finally:
        env.close()


@pytest.mark.slow
def test_kwargs_on_plain_walk_is_rejected():
    """`walk` applies no wrapper, so any options given would vanish."""
    import gymnasium as gym

    from humanoid_rl.tasks import wrap_task

    env = gym.make("Humanoid-v5")
    try:
        with pytest.raises(ValueError, match="keyed by task name"):
            wrap_task(env, "walk", {"arm_cost_weight": 0.5})
    finally:
        env.close()


@pytest.mark.slow
def test_a_flat_option_holding_a_dict_is_not_mistaken_for_keying():
    """Keying is detected by task names, not by "the values are dicts"."""
    import gymnasium as gym

    from humanoid_rl.tasks import wrap_task

    env = gym.make("Humanoid-v5")
    try:
        # `nonsense` is not a task name, so this stays flat and the wrapper
        # rejects it as an unexpected argument rather than silently ignoring it.
        with pytest.raises(TypeError):
            wrap_task(env, "natural", {"nonsense": {"a": 1}})
    finally:
        env.close()


# ------------------------------------------------------------------- resume


@pytest.fixture
def saved_run(tmp_path):
    """A run directory whose config can be resumed."""
    from humanoid_rl import runs
    from humanoid_rl.config import Config

    run_dir = tmp_path / "20260101-000000-humanoid-v5-ppo"
    run_dir.mkdir()
    Config.build("ppo", overrides={"n_envs": 8, "seed": 3}).save(run_dir / runs.CONFIG_FILE)
    return run_dir


def _resumed(saved_run, *flags):
    from humanoid_rl.cli import _resume_config, build_parser

    args = build_parser().parse_args(["train", "--resume", "latest", *flags])
    return _resume_config(saved_run, args)


def test_resume_without_flags_keeps_the_run_as_it_was(saved_run):
    cfg = _resumed(saved_run)
    assert cfg.n_envs == 8
    assert cfg.seed == 3
    assert cfg.algo == "ppo"


@pytest.mark.parametrize(
    "flags,field,expected",
    [
        (["--n-envs", "4"], "n_envs", 4),
        (["--steps", "2e6"], "total_timesteps", 2_000_000),
        (["--device", "cpu"], "device", "cpu"),
        (["--eval-freq", "5000"], "eval_freq", 5000),
        (["--checkpoint-freq", "1000"], "checkpoint_freq", 1000),
        (["--seed", "11"], "seed", 11),
    ],
)
def test_resume_applies_the_flags_it_can(saved_run, flags, field, expected):
    """These used to be discarded without a word."""
    assert getattr(_resumed(saved_run, *flags), field) == expected


def test_resume_applies_hyperparameter_overrides(saved_run):
    cfg = _resumed(saved_run, "--set", "learning_rate=1e-5")
    assert cfg.hyperparams["learning_rate"] == 1e-5
    assert cfg.hyperparams["batch_size"] == 256, "other hyperparameters survive"


@pytest.mark.parametrize(
    "flags",
    [
        ["--task", "goal"],
        ["--algo", "sac"],
        ["--env", "Walker2d-v5"],
        ["--no-normalize"],
    ],
)
def test_resume_refuses_what_would_invalidate_the_policy(saved_run, flags):
    """Silently ignoring these is worse than refusing them."""
    with pytest.raises(SystemExit, match="cannot change when resuming"):
        _resumed(saved_run, *flags)


def test_resume_allows_a_locked_flag_that_matches(saved_run):
    """Restating what the run already uses is not a conflict."""
    cfg = _resumed(saved_run, "--algo", "ppo", "--env", "Humanoid-v5")
    assert cfg.algo == "ppo"


def test_the_conflict_message_names_both_values(saved_run):
    with pytest.raises(SystemExit) as excinfo:
        _resumed(saved_run, "--task", "goal")
    message = str(excinfo.value)
    assert "--task" in message
    assert "'walk'" in message and "'goal'" in message


# ------------------------------------------------------------ episode counts


@pytest.mark.parametrize("value", ["1", "20", "500"])
def test_positive_accepts_counts(value):
    assert _positive(value) == int(value)


@pytest.mark.parametrize("value", ["0", "-1", "-100"])
def test_positive_rejects_zero_and_below(value):
    with pytest.raises(argparse.ArgumentTypeError, match="at least 1"):
        _positive(value)


@pytest.mark.parametrize("value", ["", "many", "1.5"])
def test_positive_rejects_non_integers(value):
    with pytest.raises(argparse.ArgumentTypeError):
        _positive(value)


@pytest.mark.parametrize(
    "argv",
    [
        ["eval", "--episodes", "0"],
        ["play", "--episodes", "0"],
        ["record", "--every", "0"],
        ["compare", "--limit", "0"],
        ["compare", "--episodes", "-3"],
        ["train", "--eval-episodes", "0"],
    ],
)
def test_cli_rejects_counts_below_one(argv):
    """These used to reach numpy and die inside an empty reduction."""
    with pytest.raises(SystemExit):
        build_parser().parse_args(argv)
