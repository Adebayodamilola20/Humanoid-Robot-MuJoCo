"""Watching the policy: play on screen, record to a file, score it numerically."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from humanoid_rl import console, runs
from humanoid_rl.config import Config
from humanoid_rl.envs import build_env, env_dt, require_interactive_viewer


def _load(run_dir: Path, prefer: str):
    """Rebuild the policy together with the environment it was trained in."""
    cfg = Config.load(run_dir / runs.CONFIG_FILE)
    model_path = runs.find_model(run_dir, prefer=prefer)
    return cfg, model_path


def _make_env(cfg: Config, run_dir: Path, model_path: Path, render_mode: str | None, seed: int):
    return build_env(
        cfg,
        n_envs=1,
        seed=seed,
        render_mode=render_mode,
        training=False,
        stats_path=runs.stats_path(run_dir, model_path),
        force_dummy=True,
    )


def _rollout(model, env, episodes: int, deterministic: bool, on_step=None) -> list[tuple[float, int]]:
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
) -> dict[str, float]:
    """Open a MuJoCo window and watch the policy act."""
    require_interactive_viewer()
    run_dir = runs.resolve_run(run)
    cfg, model_path = _load(run_dir, prefer)

    console.rule(f"Playing {runs.relative(model_path)}")
    env = _make_env(cfg, run_dir, model_path, "human", seed)

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
) -> Path:
    """Render episodes to a video file (no window needed - works over SSH)."""
    import imageio.v2 as imageio

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
    env = _make_env(cfg, run_dir, model_path, "rgb_array", seed)

    from humanoid_rl.train import load_model

    model = load_model(model_path, cfg, device="cpu")
    writer = imageio.get_writer(str(out_path), fps=fps)

    def grab(vec_env) -> None:
        writer.append_data(np.asarray(vec_env.env_method("render")[0]))

    try:
        results = _rollout(model, env, episodes, deterministic=True, on_step=grab)
    finally:
        writer.close()
        env.close()

    _report("Recording", _summarise(results))
    console.success(f"Wrote [bold]{runs.relative(out_path)}[/]")
    return out_path


# --------------------------------------------------------------------- eval


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
