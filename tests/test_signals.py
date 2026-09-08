"""Being stopped must not throw away the run.

Two multi-hour runs on this project were killed and left no `final_model.zip`,
because `model.learn` only saves on Ctrl-C. These tests drive the real signal
path rather than mocking it -- a mocked handler would pass while the actual
`kill` still lost the policy.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import textwrap
import time

import pytest

from humanoid_rl.train import save_on_termination


def test_sigterm_becomes_a_keyboard_interrupt():
    with pytest.raises(KeyboardInterrupt):
        with save_on_termination():
            os.kill(os.getpid(), signal.SIGTERM)


def test_the_previous_handler_is_restored():
    original = signal.getsignal(signal.SIGTERM)
    with save_on_termination():
        assert signal.getsignal(signal.SIGTERM) is not original
    assert signal.getsignal(signal.SIGTERM) is original


def test_restored_even_when_the_body_raises():
    original = signal.getsignal(signal.SIGTERM)
    with pytest.raises(ValueError):
        with save_on_termination():
            raise ValueError("boom")
    assert signal.getsignal(signal.SIGTERM) is original


def test_no_handler_off_the_main_thread_is_survivable():
    """Signal handlers are main-thread only; a worker must not crash."""
    import threading

    result = {}

    def body():
        try:
            with save_on_termination():
                result["ran"] = True
        except Exception as exc:  # pragma: no cover - the bug this guards
            result["error"] = exc

    thread = threading.Thread(target=body)
    thread.start()
    thread.join()
    assert result.get("ran") is True
    assert "error" not in result


@pytest.mark.slow
def test_a_real_kill_leaves_a_loadable_policy(tmp_path):
    """End to end: start training, SIGTERM it, and load what it wrote."""
    script = textwrap.dedent(
        """
        import sys
        from humanoid_rl.config import Config
        from humanoid_rl.train import train

        cfg = Config.build(
            "ppo",
            overrides={"n_envs": 1, "total_timesteps": 10_000_000,
                       "eval_freq": 10_000_000, "checkpoint_freq": 10_000_000},
            hyperparams={"n_steps": 64, "batch_size": 32},
        )
        print("READY", flush=True)
        train(cfg)
        """
    )
    script_path = tmp_path / "run.py"
    script_path.write_text(script)

    proc = subprocess.Popen(
        [sys.executable, str(script_path)],
        cwd=tmp_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        # Wait for training to actually be underway before signalling.
        deadline = time.time() + 120
        while time.time() < deadline:
            line = proc.stdout.readline()
            if not line:  # EOF: the child died before it got going
                break
            if "READY" in line:
                break
        time.sleep(12)
        assert proc.poll() is None, "training exited before it could be signalled"

        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=120)
    finally:
        if proc.poll() is None:  # pragma: no cover - cleanup path
            proc.kill()

    runs_dir = tmp_path / "runs"
    run_dirs = [p for p in runs_dir.iterdir() if p.is_dir() and not p.is_symlink()]
    assert run_dirs, "no run directory was created"
    final = run_dirs[0] / "final_model.zip"
    assert final.is_file(), "SIGTERM did not save a final policy"

    from stable_baselines3 import PPO

    model = PPO.load(str(final), device="cpu")
    assert model.num_timesteps > 0
