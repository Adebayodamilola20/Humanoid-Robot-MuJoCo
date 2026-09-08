"""Run configuration: everything needed to reproduce a training run.

A `Config` is written to `config.json` inside every run directory, so `play`,
`record` and `eval` can rebuild the exact environment the policy was trained in
without the user having to repeat a single flag.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from humanoid_rl.tasks import DEFAULT_TASK, parse_tasks

DEFAULT_ENV = "Humanoid-v5"

# Tuned starting points, adapted from the RL Baselines3 Zoo results for the
# MuJoCo humanoid. PPO is stable but sample-hungry (~10M steps to walk well);
# SAC reaches a decent gait in ~2M steps at the cost of slower wall-clock steps.
ALGO_DEFAULTS: dict[str, dict[str, Any]] = {
    "ppo": {
        "n_envs": 8,
        "normalize": True,
        "device": "cpu",  # MLP policies are faster on CPU than on GPU/MPS
        "total_timesteps": 10_000_000,
        "hyperparams": {
            "n_steps": 512,
            "batch_size": 256,
            "n_epochs": 5,
            "gamma": 0.95,
            "gae_lambda": 0.9,
            "learning_rate": 3.56987e-05,
            "ent_coef": 0.00238306,
            "clip_range": 0.3,
            "vf_coef": 0.431892,
            "max_grad_norm": 2.0,
            "policy_kwargs": {
                "log_std_init": -2,
                "ortho_init": False,
                "activation_fn": "relu",
                "net_arch": {"pi": [256, 256], "vf": [256, 256]},
            },
        },
    },
    "sac": {
        "n_envs": 1,
        "normalize": False,  # off-policy replay + running stats do not mix well
        "device": "auto",
        "total_timesteps": 2_000_000,
        "hyperparams": {
            "learning_rate": 3e-4,
            "buffer_size": 1_000_000,
            "batch_size": 256,
            "learning_starts": 10_000,
            "tau": 0.005,
            "gamma": 0.99,
            "train_freq": 1,
            "gradient_steps": 1,
            "policy_kwargs": {"net_arch": [256, 256]},
        },
    },
}

ALGOS = tuple(ALGO_DEFAULTS)


@dataclass
class Config:
    """A single training run, fully described."""

    env_id: str = DEFAULT_ENV
    task: str = DEFAULT_TASK
    algo: str = "ppo"
    total_timesteps: int = 10_000_000
    n_envs: int = 8
    seed: int = 0
    normalize: bool = True
    device: str = "cpu"

    eval_freq: int = 100_000
    eval_episodes: int = 5
    checkpoint_freq: int = 250_000

    # How far the run actually got. `total_timesteps` is the budget that was
    # asked for, and on a resume it means "this many *more*" -- neither tells
    # you how much training the saved policy has behind it. Written after every
    # save, so it stays true even for a run that was stopped early.
    trained_steps: int = 0

    hyperparams: dict[str, Any] = field(default_factory=dict)
    task_kwargs: dict[str, Any] = field(default_factory=dict)

    # ---------------------------------------------------------------- builders

    @classmethod
    def build(
        cls,
        algo: str = "ppo",
        *,
        overrides: dict[str, Any] | None = None,
        hyperparams: dict[str, Any] | None = None,
    ) -> Config:
        """Start from the tuned defaults for `algo`, then apply user overrides.

        `overrides` may contain any top-level Config field; keys whose value is
        `None` are ignored so the CLI can pass unset flags through untouched.
        """
        algo = algo.lower()
        if algo not in ALGO_DEFAULTS:
            raise ValueError(f"unknown algo {algo!r}, expected one of {', '.join(ALGOS)}")

        defaults = ALGO_DEFAULTS[algo]
        cfg = cls(algo=algo, hyperparams=json.loads(json.dumps(defaults["hyperparams"])))
        for key, value in defaults.items():
            if key != "hyperparams":
                setattr(cfg, key, value)

        for key, value in (overrides or {}).items():
            if value is None:
                continue
            if key not in {f.name for f in fields(cls)}:
                raise ValueError(f"unknown config field {key!r}")
            setattr(cfg, key, value)

        # SAC's zoo hyperparameters assume a single collector. With more
        # environments every step gathers more transitions, so the gradient
        # steps have to scale with them or the replay ratio quietly drops and
        # SAC becomes n_envs times less sample-efficient. An explicit
        # `--set gradient_steps=N` still wins, since user hyperparameters are
        # merged on top of this.
        if cfg.algo == "sac" and cfg.n_envs > 1:
            cfg.hyperparams["gradient_steps"] = cfg.n_envs

        cfg.hyperparams.update(hyperparams or {})
        cfg.validate()
        return cfg

    @classmethod
    def load(cls, path: str | Path) -> Config:
        data = json.loads(Path(path).read_text())
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    # -------------------------------------------------------------- validation

    def validate(self) -> None:
        if self.algo not in ALGO_DEFAULTS:
            raise ValueError(f"unknown algo {self.algo!r}")
        if self.n_envs < 1:
            raise ValueError("n_envs must be >= 1")
        if self.total_timesteps < 1:
            raise ValueError("total_timesteps must be >= 1")
        parse_tasks(self.task)  # raises with a useful message on a bad spec

    @property
    def slug(self) -> str:
        """Filesystem-friendly identity, e.g. `humanoid-v5-velocity-natural-ppo`.

        Combined tasks are comma-separated in the config but hyphenated here --
        a comma in a directory name is legal and a nuisance to type or quote.
        """
        env = self.env_id.lower().replace("_", "-")
        names = parse_tasks(self.task)
        task = f"-{'-'.join(names)}" if names else ""
        return f"{env}{task}-{self.algo}"
