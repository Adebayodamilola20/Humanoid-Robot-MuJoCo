"""Environment construction, shared by training and playback.

Training and playback must agree on the observation pipeline or a policy that
looked great in TensorBoard will flop on screen. Both paths go through
`build_env` here, and observation-normalisation statistics travel with the
checkpoint as `vecnormalize.pkl`.
"""

from __future__ import annotations

import sys
import threading
from collections.abc import Callable
from pathlib import Path

import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecEnv,
    VecNormalize,
)

from humanoid_rl.config import Config

VECNORM_FILE = "vecnormalize.pkl"


def ensure_registered(env_id: str) -> None:
    """MyoSuite environments only enter the Gym registry once imported."""
    if env_id in gym.registry:
        return
    if env_id.lower().startswith("myo"):
        try:
            import myosuite  # noqa: F401  (import registers the envs)
        except ImportError as exc:  # pragma: no cover - depends on extras
            raise SystemExit(
                f"'{env_id}' is a MyoSuite environment. Install it with:\n"
                f"    pip install 'humanoid-rl[myo]'"
            ) from exc
    if env_id not in gym.registry:
        raise SystemExit(f"Unknown environment id: {env_id!r}")


def make_env_fn(
    env_id: str,
    seed: int,
    rank: int = 0,
    render_mode: str | None = None,
    monitor_dir: str | Path | None = None,
) -> Callable[[], gym.Env]:
    """Return a thunk that builds one seeded, monitored environment."""

    def _init() -> gym.Env:
        ensure_registered(env_id)
        env = gym.make(env_id, render_mode=render_mode)
        monitor_path = str(Path(monitor_dir) / f"worker-{rank}") if monitor_dir else None
        env = Monitor(env, filename=monitor_path)
        env.reset(seed=seed + rank)
        env.action_space.seed(seed + rank)
        return env

    return _init


def build_env(
    cfg: Config,
    *,
    n_envs: int | None = None,
    seed: int | None = None,
    render_mode: str | None = None,
    training: bool = True,
    stats_path: str | Path | None = None,
    monitor_dir: str | Path | None = None,
    force_dummy: bool = False,
) -> VecEnv:
    """Build the vectorised environment, optionally wrapped in `VecNormalize`.

    `stats_path` loads normalisation statistics saved next to a checkpoint. When
    `training` is False the statistics are frozen and rewards are left raw so the
    numbers printed on screen are real environment rewards.
    """
    n_envs = cfg.n_envs if n_envs is None else n_envs
    seed = cfg.seed if seed is None else seed

    ensure_registered(cfg.env_id)
    fns = [make_env_fn(cfg.env_id, seed, i, render_mode, monitor_dir) for i in range(n_envs)]

    if n_envs > 1 and not force_dummy:
        # MuJoCo + fork is a known source of silent hangs on macOS.
        start_method = "spawn" if sys.platform == "darwin" else None
        venv: VecEnv = SubprocVecEnv(fns, start_method=start_method)
    else:
        venv = DummyVecEnv(fns)

    venv.seed(seed)

    if not cfg.normalize:
        return venv

    if stats_path is not None and Path(stats_path).exists():
        norm = VecNormalize.load(str(stats_path), venv)
        norm.training = training
        norm.norm_reward = training
        return norm

    return VecNormalize(
        venv,
        training=training,
        norm_obs=True,
        norm_reward=training,
        clip_obs=10.0,
        gamma=float(cfg.hyperparams.get("gamma", 0.99)),
    )


def env_dt(venv: VecEnv, fallback: float = 1 / 60) -> float:
    """Simulated seconds per environment step, used to pace human rendering."""
    try:
        return float(venv.get_attr("dt")[0])
    except Exception:
        return fallback


def require_interactive_viewer() -> None:
    """Fail early and helpfully when the MuJoCo window cannot open.

    Gymnasium renders `human` mode through its own GLFW window rather than
    `mujoco.viewer.launch_passive`, and GLFW must create that window on the
    process's main thread. `mjpython` runs the script on a *secondary* thread
    (it keeps the main one for MuJoCo's Cocoa event loop), so it is the one
    interpreter that cannot work here -- the opposite of the usual advice.
    """
    if threading.current_thread() is not threading.main_thread():
        raise SystemExit(
            "The MuJoCo window must be opened from the main thread.\n"
            "Run 'play' directly rather than from a worker thread."
        )

    if sys.platform != "darwin":
        return

    try:
        import mujoco.viewer as viewer
    except ImportError:  # pragma: no cover - mujoco is a hard dependency
        return

    if getattr(viewer, "_MJPYTHON", None) is not None:
        raise SystemExit(
            "Run 'play' with plain 'python', not 'mjpython'.\n"
            "Gymnasium opens its own GLFW window, which needs the main thread;\n"
            "mjpython runs your script on a secondary thread and will crash with\n"
            "'NSWindow should only be instantiated on the main thread!'.\n\n"
            "    python -m humanoid_rl play"
        )
