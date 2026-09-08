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
from humanoid_rl.tasks import DEFAULT_TASK, wrap_task

VECNORM_FILE = "vecnormalize.pkl"

ASSETS_DIR = Path(__file__).parent / "assets"

# Alternative scenes for the MuJoCo humanoid. These change appearance only --
# see the header of `humanoid_street.xml` and `tests/test_scene.py`, which
# asserts the physics is identical to the stock model. That is what lets a
# policy trained on `default` be viewed in `street` without retraining.
SCENES: dict[str, Path | None] = {
    "default": None,
    "street": ASSETS_DIR / "humanoid_street.xml",
}


def scene_kwargs(scene: str | None) -> dict:
    """`gym.make` keyword arguments selecting a scene."""
    if not scene or scene == "default":
        return {}
    try:
        path = SCENES[scene]
    except KeyError:
        raise SystemExit(
            f"Unknown scene {scene!r}. Choose from: {', '.join(SCENES)}"
        ) from None
    if path is None:
        return {}
    if not path.is_file():  # pragma: no cover - only if the package is broken
        raise SystemExit(f"Scene file is missing: {path}")
    return {"xml_file": str(path.resolve())}


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
    task: str = DEFAULT_TASK,
    task_kwargs: dict | None = None,
    make_kwargs: dict | None = None,
) -> Callable[[], gym.Env]:
    """Return a thunk that builds one seeded, monitored environment.

    `make_kwargs` reaches `gym.make` -- MuJoCo environments accept `width`
    and `height` there, which renders straight to the target size instead of
    downscaling afterwards.
    """

    def _init() -> gym.Env:
        ensure_registered(env_id)
        env = gym.make(env_id, render_mode=render_mode, **(make_kwargs or {}))
        # Task wrapper goes on *inside* Monitor, so the episode rewards written
        # to `monitor/` are the shaped rewards actually being optimised rather
        # than the stock forward-velocity ones.
        env = wrap_task(env, task, task_kwargs)
        monitor_path = str(Path(monitor_dir) / f"worker-{rank}") if monitor_dir else None
        # `override_existing=True` (SB3's default) truncates the CSV, so
        # resuming a run silently destroys every episode logged before it. Run
        # directories are timestamped, so these files only ever pre-exist on a
        # resume -- appending is correct in both cases.
        env = Monitor(env, filename=monitor_path, override_existing=False)
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
    make_kwargs: dict | None = None,
) -> VecEnv:
    """Build the vectorised environment, optionally wrapped in `VecNormalize`.

    `stats_path` loads normalisation statistics saved next to a checkpoint. When
    `training` is False the statistics are frozen and rewards are left raw so the
    numbers printed on screen are real environment rewards.
    """
    n_envs = cfg.n_envs if n_envs is None else n_envs
    seed = cfg.seed if seed is None else seed

    ensure_registered(cfg.env_id)
    fns = [
        make_env_fn(
            cfg.env_id, seed, i, render_mode, monitor_dir, cfg.task, cfg.task_kwargs,
            make_kwargs,
        )
        for i in range(n_envs)
    ]

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


def patch_viewer_camera_controls() -> None:
    """Make the viewer's mouse controls survive the MuJoCo 3.12 API change.

    Gymnasium 1.3 calls `mjv_moveCamera(model, action, dx, dy, scene, camera)`,
    the signature MuJoCo used before 3.12; 3.12 dropped the scene argument. The
    mismatch only bites inside GLFW's mouse callbacks, so the window opens fine
    and then dies with a `TypeError` the moment you drag or scroll -- which
    looks like a crash in the policy rather than in the camera.

    Wrapping the function to accept both shapes fixes it without pinning either
    package to an older release.
    """
    import mujoco

    original = mujoco.mjv_moveCamera
    if getattr(original, "_humanoid_rl_patched", False):
        return

    def move_camera(model, action, reldx, reldy, *rest):
        try:
            return original(model, action, reldx, reldy, rest[-1])
        except TypeError:
            return original(model, action, reldx, reldy, *rest)

    move_camera._humanoid_rl_patched = True
    mujoco.mjv_moveCamera = move_camera


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
