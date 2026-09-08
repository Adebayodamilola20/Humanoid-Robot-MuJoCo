"""Config is what makes a run reproducible -- if it drifts, every replay lies."""

from __future__ import annotations

import json

import pytest

from humanoid_rl.config import ALGO_DEFAULTS, DEFAULT_ENV, Config


def test_defaults_come_from_the_algo_table():
    cfg = Config.build("ppo")
    assert cfg.algo == "ppo"
    assert cfg.env_id == DEFAULT_ENV
    assert cfg.n_envs == ALGO_DEFAULTS["ppo"]["n_envs"]
    assert cfg.total_timesteps == ALGO_DEFAULTS["ppo"]["total_timesteps"]
    assert cfg.hyperparams["batch_size"] == 256


def test_sac_defaults_differ_from_ppo():
    ppo, sac = Config.build("ppo"), Config.build("sac")
    assert sac.n_envs == 1
    assert sac.normalize is False
    assert ppo.normalize is True
    assert "buffer_size" in sac.hyperparams
    assert "buffer_size" not in ppo.hyperparams


def test_build_does_not_mutate_the_shared_defaults():
    first = Config.build("ppo")
    first.hyperparams["batch_size"] = 999
    first.hyperparams["policy_kwargs"]["net_arch"]["pi"] = [8]

    second = Config.build("ppo")
    assert second.hyperparams["batch_size"] == 256
    assert second.hyperparams["policy_kwargs"]["net_arch"]["pi"] == [256, 256]
    assert ALGO_DEFAULTS["ppo"]["hyperparams"]["batch_size"] == 256


def test_overrides_apply_and_none_is_ignored():
    cfg = Config.build("ppo", overrides={"n_envs": 2, "seed": None, "env_id": "Walker2d-v5"})
    assert cfg.n_envs == 2
    assert cfg.env_id == "Walker2d-v5"
    assert cfg.seed == 0  # None left the default alone


def test_hyperparam_overrides_merge_rather_than_replace():
    cfg = Config.build("ppo", hyperparams={"learning_rate": 1e-4})
    assert cfg.hyperparams["learning_rate"] == 1e-4
    assert cfg.hyperparams["batch_size"] == 256  # untouched


@pytest.mark.parametrize("algo", ["dqn", "PPO2", ""])
def test_unknown_algo_is_rejected(algo):
    with pytest.raises(ValueError, match="unknown algo"):
        Config.build(algo)


def test_unknown_config_field_is_rejected():
    with pytest.raises(ValueError, match="unknown config field"):
        Config.build("ppo", overrides={"lerning_rate": 1e-4})


@pytest.mark.parametrize("bad", [{"n_envs": 0}, {"n_envs": -1}, {"total_timesteps": 0}])
def test_validate_rejects_nonsense(bad):
    with pytest.raises(ValueError):
        Config.build("ppo", overrides=bad)


def test_sac_with_parallel_envs_scales_gradient_steps():
    cfg = Config.build("sac", overrides={"n_envs": 4})
    assert cfg.hyperparams["gradient_steps"] == 4


def test_sac_respects_an_explicit_gradient_steps():
    cfg = Config.build("sac", overrides={"n_envs": 4}, hyperparams={"gradient_steps": 2})
    assert cfg.hyperparams["gradient_steps"] == 2


def test_save_load_roundtrip(tmp_path):
    cfg = Config.build("ppo", overrides={"n_envs": 3, "seed": 7}, hyperparams={"gamma": 0.98})
    path = tmp_path / "config.json"
    cfg.save(path)

    assert json.loads(path.read_text())["seed"] == 7
    assert Config.load(path).to_dict() == cfg.to_dict()


def test_load_tolerates_unknown_keys_from_a_future_version(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"algo": "ppo", "seed": 3, "invented_later": True}))
    assert Config.load(path).seed == 3


def test_slug_is_filesystem_safe():
    assert Config.build("ppo").slug == "humanoid-v5-ppo"
    assert Config.build("sac", overrides={"env_id": "myoLeg_Walk-v0"}).slug == "myoleg-walk-v0-sac"
