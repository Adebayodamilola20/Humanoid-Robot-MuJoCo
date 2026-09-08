"""Terminal output. Uses `rich` when available, plain text when it is not."""

from __future__ import annotations

from typing import Any

try:  # pragma: no cover - presentation only
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    _console: Console | None = Console()
except ImportError:  # pragma: no cover
    _console = None


def has_progress_bar() -> bool:
    """Whether SB3's `progress_bar=True` can be used (needs rich + tqdm)."""
    if _console is None:
        return False
    try:
        import tqdm  # noqa: F401
    except ImportError:
        return False
    return True


def rule(title: str) -> None:
    if _console:
        _console.rule(f"[bold cyan]{title}")
    else:
        print(f"\n=== {title} ===")


def info(message: str) -> None:
    if _console:
        _console.print(message)
    else:
        print(_strip(message))


def success(message: str) -> None:
    if _console:
        _console.print(f"[bold green]✓[/] {message}")
    else:
        print(f"[ok] {_strip(message)}")


def warn(message: str) -> None:
    if _console:
        _console.print(f"[bold yellow]![/] {message}")
    else:
        print(f"[warn] {_strip(message)}")


def key_values(title: str, data: dict[str, Any]) -> None:
    """Render a settings summary."""
    if _console:
        table = Table.grid(padding=(0, 2))
        table.add_column(style="dim", justify="right")
        table.add_column(style="bold")
        for key, value in data.items():
            table.add_row(key, _fmt(value))
        _console.print(Panel(table, title=f"[bold cyan]{title}", border_style="cyan", expand=False))
    else:
        print(f"\n--- {title} ---")
        width = max((len(k) for k in data), default=0)
        for key, value in data.items():
            print(f"  {key:>{width}} : {_fmt(value)}")
        print()


def episode_row(index: int, total: int, reward: float, length: int) -> None:
    info(
        f"  episode [bold]{index}/{total}[/]  reward [bold]{reward:9.1f}[/]  "
        f"steps [bold]{length:5d}[/]"
    )


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, int) and not isinstance(value, bool) and value >= 10_000:
        return f"{value:,}"
    return str(value)


def _strip(message: str) -> str:
    """Remove rich markup for the plain-text path."""
    out, depth = [], 0
    for char in message:
        if char == "[":
            depth += 1
        elif char == "]" and depth:
            depth -= 1
        elif not depth:
            out.append(char)
    return "".join(out)
