"""Training loop: build the agent, learn, and leave a run directory behind."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch.nn as nn
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import VecEnv, VecNormalize

from humanoid_rl import console, runs
from humanoid_rl.config import Config
from humanoid_rl.envs import VECNORM_FILE, build_env

ALGO_CLASSES: dict[str, type[BaseAlgorithm]] = {"ppo": PPO, "sac": SAC}

ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "elu": nn.ELU,
    "gelu": nn.GELU,
    "leaky_relu": nn.LeakyReLU,
}


class SaveVecNormalize(BaseCallback):
    """Snapshot observation statistics whenever a new best policy is found.

    Without this, `best_model.zip` would be paired with statistics from the end
    of training instead of the moment it was saved.
    """

    def __init__(self, save_path: Path):
        super().__init__()
        self.save_path = save_path

    def _on_step(self) -> bool:
        env = self.model.get_vec_normalize_env()
        if env is not None:
            env.save(str(self.save_path))
        return True


def _resolve_policy_kwargs(raw: dict[str, Any] | None) -> dict[str, Any]:
    """Turn the JSON-serialisable policy spec from `config.json` into objects."""
    kwargs = dict(raw or {})
    activation = kwargs.get("activation_fn")
    if isinstance(activation, str):
        try:
            kwargs["activation_fn"] = ACTIVATIONS[activation.lower()]
        except KeyError:
            raise SystemExit(
                f"Unknown activation_fn {activation!r}. Choose from: "
                f"{', '.join(sorted(ACTIVATIONS))}"
            ) from None
    return kwargs


def build_model(cfg: Config, env: VecEnv, tb_log: Path | None = None) -> BaseAlgorithm:
    algo_cls = ALGO_CLASSES[cfg.algo]
    hyperparams = dict(cfg.hyperparams)
    hyperparams["policy_kwargs"] = _resolve_policy_kwargs(hyperparams.pop("policy_kwargs", None))
    return algo_cls(
        "MlpPolicy",
        env,
        seed=cfg.seed,
        device=cfg.device,
        verbose=0,
        tensorboard_log=str(tb_log) if tb_log else None,
        **hyperparams,
    )


def load_model(
    path: Path, cfg: Config, env: VecEnv | None = None, device: str | None = None
) -> BaseAlgorithm:
    algo_cls = ALGO_CLASSES[cfg.algo]
    return algo_cls.load(str(path), env=env, device=device or cfg.device)


def _build_callbacks(cfg: Config, run_dir: Path) -> CallbackList:
    # SB3 counts callback frequencies per worker, while users think in total
    # environment steps -- divide so `--eval-freq 100000` means what it says.
    per_env = lambda freq: max(freq // cfg.n_envs, 1)  # noqa: E731

    eval_cfg = Config(**{**cfg.to_dict(), "n_envs": 1})
    eval_env = build_env(
        eval_cfg,
        n_envs=1,
        seed=cfg.seed + 10_000,
        training=False,
        force_dummy=True,
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(run_dir),
        log_path=str(run_dir / "tb"),
        eval_freq=per_env(cfg.eval_freq),
        n_eval_episodes=cfg.eval_episodes,
        deterministic=True,
        render=False,
        callback_on_new_best=(
            SaveVecNormalize(run_dir / f"best_{VECNORM_FILE}") if cfg.normalize else None
        ),
        verbose=1,
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=per_env(cfg.checkpoint_freq),
        save_path=str(run_dir / "checkpoints"),
        name_prefix=cfg.algo,
        save_vecnormalize=cfg.normalize,
        verbose=0,
    )
    return CallbackList([eval_cb, checkpoint_cb])


def _save(model: BaseAlgorithm, run_dir: Path) -> None:
    model.save(str(run_dir / runs.FINAL_MODEL))
    vec_env = model.get_vec_normalize_env()
    if isinstance(vec_env, VecNormalize):
        vec_env.save(str(run_dir / VECNORM_FILE))


def train(cfg: Config, resume: Path | None = None) -> Path:
    """Run training to completion and return the run directory.

    Ctrl-C is treated as "stop here, keep what you have" rather than as an
    error: the policy is saved before the process exits.
    """
    set_random_seed(cfg.seed)

    if resume is not None:
        run_dir = resume
        console.rule(f"Resuming {runs.relative(run_dir)}")
    else:
        run_dir = runs.create_run_dir(cfg)
        console.rule("Training")

    console.key_values(
        "Run",
        {
            "environment": cfg.env_id,
            "algorithm": cfg.algo.upper(),
            "timesteps": cfg.total_timesteps,
            "parallel envs": cfg.n_envs,
            "normalize obs": cfg.normalize,
            "device": cfg.device,
            "seed": cfg.seed,
            "run dir": runs.relative(run_dir),
        },
    )

    env = build_env(
        cfg,
        training=True,
        monitor_dir=run_dir / "monitor",
        stats_path=run_dir / VECNORM_FILE if resume else None,
    )

    try:
        if resume is not None:
            model_path = runs.find_model(run_dir, prefer="final")
            console.info(f"Loading policy from [bold]{runs.relative(model_path)}[/]")
            model = load_model(model_path, cfg, env=env)
            model.tensorboard_log = str(run_dir / "tb")
            if cfg.algo != "ppo":
                console.warn("Replay buffer is not restored; expect a short warm-up dip.")
        else:
            model = build_model(cfg, env, tb_log=run_dir / "tb")

        console.info(f"TensorBoard: [bold]{runs.tensorboard_hint(run_dir)}[/]\n")

        started = time.monotonic()
        try:
            model.learn(
                total_timesteps=cfg.total_timesteps,
                callback=_build_callbacks(cfg, run_dir),
                reset_num_timesteps=resume is None,
                progress_bar=console.has_progress_bar() and not runs.is_ci(),
            )
        except KeyboardInterrupt:
            console.warn("Interrupted - saving the policy trained so far.")
        finally:
            _save(model, run_dir)

        elapsed = time.monotonic() - started
        console.success(
            f"Finished {model.num_timesteps:,} steps in {elapsed / 60:.1f} min "
            f"({model.num_timesteps / max(elapsed, 1e-9):,.0f} steps/s)"
        )
        console.info(f"Saved to [bold]{runs.relative(run_dir)}[/]")
        console.info(f"Watch it: [bold]python -m humanoid_rl play --run {runs.relative(run_dir)}[/]")
    finally:
        env.close()

    return run_dir
