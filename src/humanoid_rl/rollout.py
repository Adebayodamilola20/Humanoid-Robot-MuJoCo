"""Watching the policy: play on screen, record to a file, score it numerically."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from humanoid_rl import console, runs
from humanoid_rl.config import Config
from humanoid_rl.envs import (
    build_env,
    env_dt,
    patch_viewer_camera_controls,
    require_interactive_viewer,
    scene_kwargs,
)


def _load(run_dir: Path, prefer: str):
    """Rebuild the policy together with the environment it was trained in."""
    cfg = Config.load(run_dir / runs.CONFIG_FILE)
    model_path = runs.find_model(run_dir, prefer=prefer)
    return cfg, model_path


def writer_fps(fps: int, every: int) -> int:
    """Frame rate for a subsampled recording, so playback stays real-time.

    Keeping every Nth frame and writing at the original rate would play the
    video back N times too fast.
    """
    return max(fps // max(every, 1), 1)


def _make_env(
    cfg: Config,
    run_dir: Path,
    model_path: Path,
    render_mode: str | None,
    seed: int,
    make_kwargs: dict | None = None,
):
    return build_env(
        cfg,
        n_envs=1,
        seed=seed,
        render_mode=render_mode,
        training=False,
        stats_path=runs.stats_path(run_dir, model_path),
        force_dummy=True,
        make_kwargs=make_kwargs,
    )


def _rollout(
    model, env, episodes: int, deterministic: bool, on_step=None, quiet: bool = False
) -> list[tuple[float, int]]:
    """Run `episodes` full episodes, returning (reward, length) for each.

    Rewards are read from the Monitor wrapper so they are true environment
    rewards even when the policy sees normalised ones.
    """
    results: list[tuple[float, int]] = []
    obs = env.reset()
    state, episode_start = None, np.ones((1,), dtype=bool)
    reward_sum, length = 0.0, 0

    while len(results) < episodes:
        action, state = model.predict(
            obs, state=state, episode_start=episode_start, deterministic=deterministic
        )
        obs, reward, done, infos = env.step(action)
        episode_start = done
        reward_sum += float(reward[0])
        length += 1
        if on_step is not None:
            on_step(env)
        if done[0]:
            episode = infos[0].get("episode")
            true_reward = float(episode["r"]) if episode else reward_sum
            true_length = int(episode["l"]) if episode else length
            results.append((true_reward, true_length))
            if not quiet:
                console.episode_row(len(results), episodes, true_reward, true_length)
            reward_sum, length = 0.0, 0
            state = None
    return results


def _summarise(results: list[tuple[float, int]]) -> dict[str, float]:
    rewards = np.array([r for r, _ in results], dtype=float)
    lengths = np.array([length for _, length in results], dtype=float)
    return {
        "episodes": float(len(results)),
        "reward_mean": float(rewards.mean()),
        "reward_std": float(rewards.std()),
        "reward_min": float(rewards.min()),
        "reward_max": float(rewards.max()),
        "length_mean": float(lengths.mean()),
    }


def _report(title: str, stats: dict[str, float]) -> None:
    console.key_values(
        title,
        {
            "episodes": int(stats["episodes"]),
            "mean reward": f"{stats['reward_mean']:.1f} ± {stats['reward_std']:.1f}",
            "best / worst": f"{stats['reward_max']:.1f} / {stats['reward_min']:.1f}",
            "mean length": f"{stats['length_mean']:.0f} steps",
        },
    )


# --------------------------------------------------------------------- play


def play(
    run: str | Path | None = None,
    episodes: int = 5,
    deterministic: bool = True,
    prefer: str = "best",
    seed: int = 0,
    realtime: bool = True,
    scene: str | None = None,
) -> dict[str, float]:
    """Open a MuJoCo window and watch the policy act.

    `scene` swaps the model's appearance only, so any trained policy can be
    viewed in any scene without retraining.
    """
    require_interactive_viewer()
    patch_viewer_camera_controls()
    run_dir = runs.resolve_run(run)
    cfg, model_path = _load(run_dir, prefer)

    console.rule(f"Playing {runs.relative(model_path)}")
    env = _make_env(cfg, run_dir, model_path, "human", seed, scene_kwargs(scene, cfg.env_id))

    from humanoid_rl.train import load_model

    model = load_model(model_path, cfg, device="cpu")
    dt = env_dt(env)

    pace = (lambda _env: time.sleep(dt)) if realtime else None
    try:
        results = _rollout(model, env, episodes, deterministic, on_step=pace)
    except KeyboardInterrupt:
        console.warn("Stopped.")
        return {}
    finally:
        env.close()

    stats = _summarise(results)
    _report("Playback", stats)
    return stats


# ------------------------------------------------------------------- record


def record(
    run: str | Path | None = None,
    episodes: int = 1,
    out: str | Path | None = None,
    prefer: str = "best",
    seed: int = 0,
    fps: int = 30,
    width: int | None = None,
    height: int | None = None,
    every: int = 1,
    scene: str | None = None,
) -> Path:
    """Render episodes to a video file (no window needed - works over SSH).

    `width`/`height` render straight to that size, and `every` keeps only every
    Nth frame -- together they make a GIF small enough to sit in a README. The
    writer's frame rate is divided by `every` so playback stays real-time.
    """
    import imageio.v2 as imageio

    if every < 1:
        raise SystemExit("--every must be >= 1")

    run_dir = runs.resolve_run(run)
    cfg, model_path = _load(run_dir, prefer)

    out_path = Path(out) if out else run_dir / f"{cfg.slug}.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".mp4":
        try:
            import imageio_ffmpeg  # noqa: F401
        except ImportError:
            out_path = out_path.with_suffix(".gif")
            console.warn("imageio-ffmpeg not installed - writing a GIF instead.")

    console.rule(f"Recording {runs.relative(model_path)}")
    make_kwargs = {k: v for k, v in (("width", width), ("height", height)) if v}
    make_kwargs.update(scene_kwargs(scene, cfg.env_id))
    env = _make_env(cfg, run_dir, model_path, "rgb_array", seed, make_kwargs)

    from humanoid_rl.train import load_model

    model = load_model(model_path, cfg, device="cpu")
    writer = imageio.get_writer(str(out_path), fps=writer_fps(fps, every))

    frame_index = 0

    def grab(vec_env) -> None:
        nonlocal frame_index
        if frame_index % every == 0:
            writer.append_data(np.asarray(vec_env.env_method("render")[0]))
        frame_index += 1

    try:
        results = _rollout(model, env, episodes, deterministic=True, on_step=grab)
    finally:
        writer.close()
        env.close()

    _report("Recording", _summarise(results))
    console.success(f"Wrote [bold]{runs.relative(out_path)}[/]")
    return out_path


# --------------------------------------------------------------------- eval


def score_checkpoint(
    run_dir: Path,
    model_path: Path,
    episodes: int = 10,
    deterministic: bool = True,
    seed: int = 0,
    quiet: bool = True,
) -> dict[str, float]:
    """Score one specific checkpoint file, without printing every episode.

    `evaluate` selects a checkpoint by preference (best/final/last); this takes
    an exact path, which is what comparing several of them needs. Each is paired
    with its own normalisation statistics via `runs.stats_path` -- comparing a
    policy against the wrong statistics would rank them by accident rather than
    by quality.
    """
    cfg = Config.load(run_dir / runs.CONFIG_FILE)
    env = _make_env(cfg, run_dir, model_path, None, seed)

    from humanoid_rl.train import load_model

    model = load_model(model_path, cfg, device="cpu")
    try:
        results = _rollout(model, env, episodes, deterministic, quiet=quiet)
    finally:
        env.close()
    return _summarise(results)


def evaluate(
    run: str | Path | None = None,
    episodes: int = 20,
    deterministic: bool = True,
    prefer: str = "best",
    seed: int = 0,
) -> dict[str, float]:
    """Score the policy over several episodes, headless and fast."""
    run_dir = runs.resolve_run(run)
    cfg, model_path = _load(run_dir, prefer)

    console.rule(f"Evaluating {runs.relative(model_path)}")
    env = _make_env(cfg, run_dir, model_path, None, seed)

    from humanoid_rl.train import load_model

    model = load_model(model_path, cfg, device="cpu")
    try:
        results = _rollout(model, env, episodes, deterministic)
    finally:
        env.close()

    stats = _summarise(results)
    _report("Evaluation", stats)
    return stats
