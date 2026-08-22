"""Command line interface: `python -m humanoid_rl <command>`."""

from __future__ import annotations

import argparse
import ast
import sys
from typing import Any

from humanoid_rl import __version__, console, runs
from humanoid_rl.config import ALGOS, DEFAULT_ENV, Config

EPILOG = """\
examples:
  python -m humanoid_rl train                       train PPO on Humanoid-v5
  python -m humanoid_rl train --algo sac --steps 2e6
  python -m humanoid_rl train --resume latest       continue the last run
  mjpython -m humanoid_rl play                      watch the latest policy (macOS)
  python -m humanoid_rl record --out walk.mp4       render a video instead
  python -m humanoid_rl eval --episodes 20          score the policy
  python -m humanoid_rl ls                          list saved runs
"""


def _steps(value: str) -> int:
    """Accept 2000000, 2_000_000 and 2e6 alike."""
    try:
        return int(float(value.replace("_", "")))
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a number of steps, got {value!r}") from None


def _override(value: str) -> tuple[str, Any]:
    """Parse `key=value` hyperparameter overrides, e.g. --set learning_rate=1e-4."""
    if "=" not in value:
        raise argparse.ArgumentTypeError(f"expected key=value, got {value!r}")
    key, raw = value.split("=", 1)
    try:
        parsed = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        parsed = raw
    return key.strip(), parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="humanoid-rl",
        description="Teach a MuJoCo humanoid to walk with deep reinforcement learning.",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"humanoid-rl {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    def add_playback_args(p: argparse.ArgumentParser, episodes: int) -> None:
        p.add_argument("--run", default=None, metavar="RUN",
                       help="run directory or name (default: latest)")
        p.add_argument("--episodes", type=int, default=episodes, help=f"default: {episodes}")
        p.add_argument("--model", choices=("best", "final", "last"), default="best",
                       help="which checkpoint to load (default: best)")
        p.add_argument("--seed", type=int, default=0)

    # ---------------------------------------------------------------- train
    train_p = sub.add_parser("train", help="train a policy")
    train_p.add_argument("--env", default=DEFAULT_ENV, metavar="ID",
                         help=f"Gymnasium environment id (default: {DEFAULT_ENV})")
    train_p.add_argument("--algo", choices=ALGOS, default="ppo", help="default: ppo")
    train_p.add_argument("--steps", type=_steps, default=None, dest="total_timesteps",
                         help="total environment steps (default: per-algorithm)")
    train_p.add_argument("--n-envs", type=int, default=None, dest="n_envs",
                         help="parallel environments (default: per-algorithm)")
    train_p.add_argument("--seed", type=int, default=0)
    train_p.add_argument("--device", default=None, help="cpu, cuda, mps or auto")
    norm = train_p.add_mutually_exclusive_group()
    norm.add_argument("--normalize", dest="normalize", action="store_true", default=None,
                      help="normalise observations and rewards")
    norm.add_argument("--no-normalize", dest="normalize", action="store_false",
                      help="disable observation normalisation")
    train_p.add_argument("--eval-freq", type=_steps, default=None, dest="eval_freq")
    train_p.add_argument("--eval-episodes", type=int, default=None, dest="eval_episodes")
    train_p.add_argument("--checkpoint-freq", type=_steps, default=None, dest="checkpoint_freq")
    train_p.add_argument("--resume", nargs="?", const="latest", default=None, metavar="RUN",
                         help="continue an existing run (default: latest)")
    train_p.add_argument("--set", type=_override, action="append", default=[], metavar="KEY=VALUE",
                         dest="hyperparams", help="override a hyperparameter, repeatable")

    # ----------------------------------------------------------------- play
    play_p = sub.add_parser("play", help="watch a trained policy in a window")
    add_playback_args(play_p, episodes=5)
    play_p.add_argument("--stochastic", action="store_true",
                        help="sample actions instead of using the policy mean")
    play_p.add_argument("--no-realtime", dest="realtime", action="store_false",
                        help="run as fast as the simulator allows")

    # --------------------------------------------------------------- record
    rec_p = sub.add_parser("record", help="render a policy to a video file")
    add_playback_args(rec_p, episodes=1)
    rec_p.add_argument("--out", default=None, metavar="PATH", help="output .mp4 or .gif")
    rec_p.add_argument("--fps", type=int, default=30)

    # ----------------------------------------------------------------- eval
    eval_p = sub.add_parser("eval", help="score a policy over several episodes")
    add_playback_args(eval_p, episodes=20)
    eval_p.add_argument("--stochastic", action="store_true")

    # ------------------------------------------------------------------- ls
    sub.add_parser("ls", help="list saved runs")

    return parser


def _cmd_train(args: argparse.Namespace) -> None:
    from humanoid_rl.train import train

    resume_dir = runs.resolve_run(args.resume) if args.resume else None

    if resume_dir is not None:
        cfg = Config.load(resume_dir / runs.CONFIG_FILE)
        if args.total_timesteps is not None:
            cfg.total_timesteps = args.total_timesteps
    else:
        cfg = Config.build(
            args.algo,
            overrides={
                "env_id": args.env,
                "total_timesteps": args.total_timesteps,
                "n_envs": args.n_envs,
                "seed": args.seed,
                "normalize": args.normalize,
                "device": args.device,
                "eval_freq": args.eval_freq,
                "eval_episodes": args.eval_episodes,
                "checkpoint_freq": args.checkpoint_freq,
            },
            hyperparams=dict(args.hyperparams),
        )

    train(cfg, resume=resume_dir)


def _cmd_ls(_: argparse.Namespace) -> None:
    found = runs.list_runs()
    if not found:
        console.warn("No runs yet. Start one with: python -m humanoid_rl train")
        return
    console.rule("Runs")
    latest = None
    try:
        latest = runs.resolve_run("latest")
    except SystemExit:
        pass
    for run_dir in found:
        try:
            cfg = Config.load(run_dir / runs.CONFIG_FILE)
            detail = f"{cfg.env_id} · {cfg.algo.upper()} · {cfg.total_timesteps:,} steps"
        except (OSError, ValueError):
            detail = "(unreadable config)"
        marker = " [green]← latest[/]" if latest and run_dir.resolve() == latest else ""
        console.info(f"  [bold]{run_dir.name}[/]  {detail}{marker}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    try:
        if args.command == "train":
            _cmd_train(args)
        elif args.command == "ls":
            _cmd_ls(args)
        else:
            from humanoid_rl import rollout

            if args.command == "play":
                rollout.play(
                    run=args.run,
                    episodes=args.episodes,
                    deterministic=not args.stochastic,
                    prefer=args.model,
                    seed=args.seed,
                    realtime=args.realtime,
                )
            elif args.command == "record":
                rollout.record(
                    run=args.run,
                    episodes=args.episodes,
                    out=args.out,
                    prefer=args.model,
                    seed=args.seed,
                    fps=args.fps,
                )
            elif args.command == "eval":
                rollout.evaluate(
                    run=args.run,
                    episodes=args.episodes,
                    deterministic=not args.stochastic,
                    prefer=args.model,
                    seed=args.seed,
                )
    except KeyboardInterrupt:
        console.warn("Interrupted.")
        return 130
    except SystemExit as exc:  # our own guard rails carry a friendly message
        if exc.code and not isinstance(exc.code, int):
            console.warn(str(exc.code))
            return 1
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
