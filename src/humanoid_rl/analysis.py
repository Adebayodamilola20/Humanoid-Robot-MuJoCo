"""Looking at a finished run: which checkpoint is best, and how it got there.

Two questions come up constantly and neither had an answer in the CLI:

    "Is the newest checkpoint actually the best one?"
    "What did the learning curve look like?"

The first matters more than it sounds. PPO does not converge smoothly -- it
oscillates around a plateau -- so a later checkpoint is regularly *worse* than an
earlier one. On this project's own run, 17.4M steps scored 6,040 while 15.1M
scored 7,679. Discovering that by hand meant running two evaluations and
remembering to compare them.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from humanoid_rl import console, runs


@dataclass
class Episode:
    """One finished episode, as recorded by SB3's Monitor wrapper."""

    wall_time: float
    reward: float
    length: int


def read_episodes(run_dir: Path) -> list[Episode]:
    """Every episode logged under `monitor/`, ordered by when it finished.

    Monitor writes one CSV per parallel worker, each with a JSON comment line
    first and then `r,l,t` columns. Interleaving them by wall time reconstructs
    the run's actual chronology.
    """
    episodes: list[Episode] = []
    for path in sorted((run_dir / "monitor").glob("*.csv")):
        try:
            with path.open() as handle:
                handle.readline()  # the JSON header comment
                for row in csv.DictReader(handle):
                    try:
                        episodes.append(
                            Episode(float(row["t"]), float(row["r"]), int(row["l"]))
                        )
                    except (KeyError, TypeError, ValueError):
                        # A run killed mid-write leaves one truncated line.
                        continue
        except OSError:
            continue
    episodes.sort(key=lambda e: e.wall_time)
    return episodes


def smooth(values: list[float], window: int) -> list[float]:
    """Running mean over `window` points, for a curve you can actually read."""
    if window <= 1 or len(values) < 2:
        return list(values)
    window = min(window, len(values))
    out, total = [], 0.0
    for i, value in enumerate(values):
        total += value
        if i >= window:
            total -= values[i - window]
        out.append(total / min(i + 1, window))
    return out


def checkpoint_steps(path: Path) -> int:
    """Step count encoded in an SB3 checkpoint filename, or 0."""
    stem = path.stem
    if stem.endswith("_steps"):
        _, _, steps = stem[: -len("_steps")].rpartition("_")
        if steps.isdigit():
            return int(steps)
    return 0


def list_checkpoints(run_dir: Path, limit: int | None = None) -> list[Path]:
    """Saved policies worth comparing, oldest first, plus best and final."""
    found = sorted(
        (p for p in (run_dir / "checkpoints").glob("*.zip")), key=checkpoint_steps
    )
    if limit is not None and len(found) > limit:
        # Evenly spaced across the run rather than only the tail, so the shape
        # of the curve is visible instead of just its end.
        step = (len(found) - 1) / (limit - 1) if limit > 1 else 1
        found = [found[round(i * step)] for i in range(limit)]
    for name in (runs.BEST_MODEL, runs.FINAL_MODEL):
        path = run_dir / name
        if path.is_file():
            found.append(path)
    return found


# ------------------------------------------------------------------- compare


def compare(
    run: str | Path | None = None,
    episodes: int = 10,
    limit: int = 6,
    seed: int = 0,
) -> list[tuple[Path, dict[str, float]]]:
    """Score several checkpoints of one run and rank them."""
    from humanoid_rl.rollout import score_checkpoint

    run_dir = runs.resolve_run(run)
    checkpoints = list_checkpoints(run_dir, limit=limit)
    if not checkpoints:
        raise SystemExit(f"No checkpoints found in {runs.relative(run_dir)}")

    console.rule(f"Comparing {len(checkpoints)} checkpoints · {episodes} episodes each")

    results: list[tuple[Path, dict[str, float]]] = []
    for path in checkpoints:
        stats = score_checkpoint(run_dir, path, episodes=episodes, seed=seed)
        results.append((path, stats))
        console.info(
            f"  {path.name:<36} reward [bold]{stats['reward_mean']:8.0f}[/] "
            f"± {stats['reward_std']:<7.0f} length {stats['length_mean']:6.0f}"
        )

    ranked = sorted(results, key=lambda item: item[1]["reward_mean"], reverse=True)
    best_path, best_stats = ranked[0]
    console.key_values(
        "Best",
        {
            "checkpoint": best_path.name,
            "mean reward": f"{best_stats['reward_mean']:.0f} ± {best_stats['reward_std']:.0f}",
            "mean length": f"{best_stats['length_mean']:.0f} steps",
        },
    )

    newest = max(results, key=lambda item: checkpoint_steps(item[0]))
    if newest[0] != best_path and checkpoint_steps(newest[0]):
        console.warn(
            f"The newest checkpoint ({newest[0].name}) is not the best. "
            "More training is not always an improvement."
        )
    return ranked


# ---------------------------------------------------------------------- plot


def plot(
    run: str | Path | None = None,
    out: str | Path | None = None,
    window: int = 200,
) -> Path:
    """Write a learning-curve PNG from the monitor logs.

    Plots reward and episode length together, because reward alone hides the
    thing that matters: a policy scoring well by surviving 400 of 1000 steps is
    still falling over.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")  # no display needed
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "Plotting needs matplotlib. Install it with:\n"
            "    pip install -e '.[dev]'"
        ) from exc

    run_dir = runs.resolve_run(run)
    episodes = read_episodes(run_dir)
    if len(episodes) < 2:
        raise SystemExit(
            f"Not enough episodes logged in {runs.relative(run_dir)} to plot."
        )

    out_path = Path(out) if out else run_dir / "learning_curve.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Episode index is a better x-axis than wall time: it is comparable between
    # runs on different machines.
    index = list(range(1, len(episodes) + 1))
    rewards = [e.reward for e in episodes]
    lengths = [float(e.length) for e in episodes]

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

    top.plot(index, rewards, linewidth=0.5, alpha=0.25, color="#4C78A8")
    top.plot(index, smooth(rewards, window), linewidth=1.8, color="#1F4E79")
    top.set_ylabel("episode reward")
    top.set_title(f"{run_dir.name} — {len(episodes):,} episodes")
    top.grid(alpha=0.25)

    bottom.plot(index, lengths, linewidth=0.5, alpha=0.25, color="#E45756")
    bottom.plot(index, smooth(lengths, window), linewidth=1.8, color="#8B2C2A")
    bottom.set_ylabel("episode length (steps)")
    bottom.set_xlabel("episode")
    bottom.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)

    final = smooth(rewards, window)[-1]
    final_length = smooth(lengths, window)[-1]
    console.key_values(
        "Learning curve",
        {
            "episodes": f"{len(episodes):,}",
            "final reward (smoothed)": f"{final:,.0f}",
            "final length (smoothed)": f"{final_length:.0f} steps",
            "written to": runs.relative(out_path),
        },
    )
    return out_path
