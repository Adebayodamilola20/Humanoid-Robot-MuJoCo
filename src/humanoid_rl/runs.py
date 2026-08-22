"""Run directories: where checkpoints, stats and logs live.

Every training run gets its own timestamped folder under `runs/`:

    runs/20260821-142530-humanoid-v5-ppo/
    ├── config.json          # exact settings used
    ├── best_model.zip       # highest eval reward so far
    ├── final_model.zip      # last policy, written even on Ctrl-C
    ├── vecnormalize.pkl     # observation statistics for that policy
    ├── checkpoints/         # periodic snapshots
    ├── monitor/             # per-episode CSV logs
    └── tb/                  # TensorBoard event files

`runs/latest` points at the most recent one, so every command defaults to it.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from humanoid_rl.config import Config
from humanoid_rl.envs import VECNORM_FILE

RUNS_DIR = Path("runs")
CONFIG_FILE = "config.json"
BEST_MODEL = "best_model.zip"
FINAL_MODEL = "final_model.zip"
LATEST_LINK = "latest"


def create_run_dir(cfg: Config, root: str | Path = RUNS_DIR) -> Path:
    """Create (and mark as latest) a fresh run directory for `cfg`."""
    root = Path(root)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = root / f"{stamp}-{cfg.slug}"
    run_dir.mkdir(parents=True, exist_ok=False)
    for sub in ("checkpoints", "monitor", "tb"):
        (run_dir / sub).mkdir()
    cfg.save(run_dir / CONFIG_FILE)
    _point_latest_at(run_dir, root)
    return run_dir


def _point_latest_at(run_dir: Path, root: Path) -> None:
    link = root / LATEST_LINK
    try:
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(run_dir.name)
    except OSError:
        # Filesystems without symlink support (or Windows without dev mode):
        # fall back to a plain pointer file that `resolve_run` also understands.
        (root / "LATEST").write_text(run_dir.name + "\n")


def resolve_run(spec: str | Path | None, root: str | Path = RUNS_DIR) -> Path:
    """Turn a user-supplied run reference into a directory.

    Accepts a path, a bare run name, `latest` (the default), or a path to a
    `.zip` checkpoint, in which case its containing run directory is returned.
    """
    root = Path(root)
    if spec is None:
        spec = LATEST_LINK

    candidate = Path(spec)
    if candidate.suffix == ".zip" and candidate.is_file():
        return candidate.parent.parent if candidate.parent.name == "checkpoints" else candidate.parent

    for path in (candidate, root / candidate):
        if path.is_dir():
            return path.resolve()

    pointer = root / "LATEST"
    if str(spec) == LATEST_LINK and pointer.is_file():
        fallback = root / pointer.read_text().strip()
        if fallback.is_dir():
            return fallback.resolve()

    if str(spec) == LATEST_LINK:
        raise SystemExit(
            "No runs found. Train a policy first:\n    python -m humanoid_rl train"
        )
    raise SystemExit(f"Run not found: {spec}")


def list_runs(root: str | Path = RUNS_DIR) -> list[Path]:
    """Existing run directories, newest first."""
    root = Path(root)
    if not root.is_dir():
        return []
    runs = [p for p in root.iterdir() if p.is_dir() and not p.is_symlink()]
    return sorted(runs, key=lambda p: p.name, reverse=True)


def find_model(run_dir: Path, prefer: str = "best") -> Path:
    """Locate a policy inside a run.

    `prefer` is "best", "final", or "last" (the newest periodic checkpoint);
    each falls back to the others so a run interrupted before its first
    evaluation is still playable.
    """
    checkpoints = sorted(
        (run_dir / "checkpoints").glob("*.zip"), key=lambda p: p.stat().st_mtime
    )
    options: dict[str, Path | None] = {
        "best": run_dir / BEST_MODEL,
        "final": run_dir / FINAL_MODEL,
        "last": checkpoints[-1] if checkpoints else None,
    }
    order = [prefer] + [k for k in ("best", "final", "last") if k != prefer]
    for key in order:
        path = options.get(key)
        if path is not None and path.is_file():
            return path
    raise SystemExit(f"No saved model in {run_dir}. Has training produced a checkpoint yet?")


def stats_path(run_dir: Path, model_path: Path | None = None) -> Path | None:
    """Normalisation statistics matching `model_path`, if any were saved.

    `best_model.zip` and periodic checkpoints each get their own snapshot of the
    running statistics; pairing the wrong ones degrades the policy silently.
    """
    candidates: list[Path] = []
    if model_path is not None:
        if model_path.name == BEST_MODEL:
            candidates.append(run_dir / f"best_{VECNORM_FILE}")
        elif model_path.parent.name == "checkpoints":
            stem = model_path.stem.replace("_steps", "")
            candidates += sorted(model_path.parent.glob(f"{stem}*vecnormalize*.pkl"))
    candidates.append(run_dir / VECNORM_FILE)
    for path in candidates:
        if path.is_file():
            return path
    return None


def relative(path: Path) -> str:
    """Path shown to the user, relative to the working directory when possible."""
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def tensorboard_hint(run_dir: Path) -> str:
    return f"tensorboard --logdir {relative(run_dir / 'tb')}"


def is_ci() -> bool:
    return bool(os.environ.get("CI"))
