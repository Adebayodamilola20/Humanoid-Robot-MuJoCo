"""CLI parsing: the flags are the contract, so they get pinned down here."""

from __future__ import annotations

import pytest

from humanoid_rl.cli import _override, _steps, build_parser


@pytest.mark.parametrize(
    "text,expected",
    [("2000000", 2_000_000), ("2_000_000", 2_000_000), ("2e6", 2_000_000), ("1.5e3", 1500)],
)
def test_steps_accepts_every_spelling(text, expected):
    assert _steps(text) == expected


@pytest.mark.parametrize("text", ["lots", "", "3 million"])
def test_steps_rejects_prose(text):
    import argparse

    with pytest.raises(argparse.ArgumentTypeError):
        _steps(text)


@pytest.mark.parametrize(
    "text,key,value",
    [
        ("learning_rate=1e-4", "learning_rate", 1e-4),
        ("batch_size=512", "batch_size", 512),
        ("net_arch=[64, 64]", "net_arch", [64, 64]),
        ("activation_fn=tanh", "activation_fn", "tanh"),  # bare word stays a string
    ],
)
def test_override_parses_key_equals_value(text, key, value):
    assert _override(text) == (key, value)


def test_override_without_equals_is_rejected():
    import argparse

    with pytest.raises(argparse.ArgumentTypeError):
        _override("learning_rate")


def test_override_keeps_equals_signs_in_the_value():
    assert _override("note=a=b") == ("note", "a=b")


def test_train_defaults_leave_config_choices_to_the_algo_table():
    """Unset flags must arrive as None so Config.build keeps its tuned defaults."""
    args = build_parser().parse_args(["train"])
    assert args.total_timesteps is None
    assert args.n_envs is None
    assert args.normalize is None
    assert args.device is None


def test_train_flags_are_collected():
    args = build_parser().parse_args(
        ["train", "--algo", "sac", "--steps", "2e6", "--n-envs", "4", "--no-normalize"]
    )
    assert args.algo == "sac"
    assert args.total_timesteps == 2_000_000
    assert args.n_envs == 4
    assert args.normalize is False


def test_normalize_flags_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["train", "--normalize", "--no-normalize"])


def test_repeated_set_flags_accumulate():
    args = build_parser().parse_args(
        ["train", "--set", "learning_rate=1e-4", "--set", "batch_size=512"]
    )
    assert dict(args.hyperparams) == {"learning_rate": 1e-4, "batch_size": 512}


def test_resume_without_a_value_means_latest():
    assert build_parser().parse_args(["train", "--resume"]).resume == "latest"
    assert build_parser().parse_args(["train", "--resume", "some-run"]).resume == "some-run"


@pytest.mark.parametrize("cmd,episodes", [("play", 5), ("record", 1), ("eval", 20)])
def test_playback_commands_default_to_best_of_the_latest_run(cmd, episodes):
    args = build_parser().parse_args([cmd])
    assert args.run is None  # None means "latest"
    assert args.model == "best"
    assert args.episodes == episodes


def test_record_size_flags_default_to_the_environment():
    args = build_parser().parse_args(["record"])
    assert args.width is None and args.height is None
    assert args.every == 1
    assert args.fps == 30


def test_record_size_flags_are_collected():
    args = build_parser().parse_args(
        ["record", "--width", "320", "--height", "240", "--every", "3"]
    )
    assert (args.width, args.height, args.every) == (320, 240, 3)


@pytest.mark.parametrize(
    "fps,every,expected",
    [(30, 1, 30), (30, 2, 15), (30, 3, 10), (60, 4, 15), (30, 100, 1)],
)
def test_subsampling_divides_the_writer_frame_rate(fps, every, expected):
    """Dropping frames without slowing the writer would speed up playback."""
    from humanoid_rl.rollout import writer_fps

    assert writer_fps(fps, every) == expected


def test_bad_model_choice_is_rejected():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["play", "--model", "greatest"])


def test_bare_invocation_prints_help_and_succeeds(capsys):
    from humanoid_rl.cli import main

    assert main([]) == 0
    assert "usage" in capsys.readouterr().out.lower()
