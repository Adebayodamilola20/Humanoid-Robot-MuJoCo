"""Model construction and a genuine end-to-end smoke train.

The smoke test is slow-ish but it is the only thing that catches "the pieces
all import fine and still cannot train a single step".
"""

from __future__ import annotations

import pytest
import torch.nn as nn

from humanoid_rl import runs
from humanoid_rl.config import Config
from humanoid_rl.train import _resolve_policy_kwargs, build_model


def test_activation_names_become_torch_classes():
    assert _resolve_policy_kwargs({"activation_fn": "relu"})["activation_fn"] is nn.ReLU
    assert _resolve_policy_kwargs({"activation_fn": "TANH"})["activation_fn"] is nn.Tanh


def test_already_resolved_activations_pass_through():
    assert _resolve_policy_kwargs({"activation_fn": nn.ELU})["activation_fn"] is nn.ELU


def test_unknown_activation_lists_the_valid_ones():
    with pytest.raises(SystemExit, match="Unknown activation_fn"):
        _resolve_policy_kwargs({"activation_fn": "sigmoid"})


def test_missing_policy_kwargs_is_fine():
    assert _resolve_policy_kwargs(None) == {}


def test_resolve_does_not_mutate_the_caller_dict():
    raw = {"activation_fn": "relu"}
    _resolve_policy_kwargs(raw)
    assert raw["activation_fn"] == "relu"


@pytest.mark.slow
def test_build_model_applies_the_configured_hyperparameters():
    from humanoid_rl.envs import build_env

    cfg = Config.build(
        "ppo", overrides={"n_envs": 1}, hyperparams={"batch_size": 64, "n_steps": 64}
    )
    env = build_env(cfg, force_dummy=True)
    try:
        model = build_model(cfg, env)
        assert model.batch_size == 64
        assert model.n_steps == 64
        assert model.gamma == cfg.hyperparams["gamma"]
    finally:
        env.close()


def test_resume_after_a_hard_kill_still_finds_the_matching_stats(tmp_path):
    """A killed run has no final_model/vecnormalize.pkl pair, only best_*.

    Resuming must pick up `best_vecnormalize.pkl` rather than silently starting
    normalisation from scratch, which would discard the run's statistics.
    """
    run_dir = tmp_path / "20260101-000000-humanoid-v5-ppo"
    (run_dir / "checkpoints").mkdir(parents=True)
    (run_dir / runs.BEST_MODEL).write_bytes(b"")
    best_stats = run_dir / "best_vecnormalize.pkl"
    best_stats.write_bytes(b"")
    assert not (run_dir / "vecnormalize.pkl").exists()

    model = runs.find_model(run_dir, prefer="final")  # falls back to best
    assert model.name == runs.BEST_MODEL
    assert runs.stats_path(run_dir, model) == best_stats


@pytest.mark.slow
def test_a_short_run_leaves_a_replayable_run_directory(tmp_path, monkeypatch):
    """Train briefly, then confirm the run has everything play/eval need."""
    monkeypatch.chdir(tmp_path)
    from humanoid_rl.train import train

    cfg = Config.build(
        "ppo",
        overrides={"n_envs": 1, "total_timesteps": 256, "eval_freq": 128, "eval_episodes": 1,
                   "checkpoint_freq": 128},
        hyperparams={"n_steps": 128, "batch_size": 64},
    )
    run_dir = train(cfg)

    assert (run_dir / runs.CONFIG_FILE).is_file()
    assert (run_dir / runs.FINAL_MODEL).is_file()
    assert runs.find_model(run_dir, "final").is_file()
    assert runs.stats_path(run_dir, run_dir / runs.FINAL_MODEL) is not None
    assert Config.load(run_dir / runs.CONFIG_FILE).total_timesteps == 256
    assert runs.resolve_run("latest", root=tmp_path / "runs") == run_dir.resolve()
