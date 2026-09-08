"""Run directories: resolution and, crucially, pairing a policy with its stats.

Loading `best_model.zip` against end-of-training statistics degrades a policy
silently -- no error, just a worse gait -- so the pairing rules get real tests.
"""

from __future__ import annotations

import pytest

from humanoid_rl import runs
from humanoid_rl.config import Config


@pytest.fixture
def root(tmp_path):
    d = tmp_path / "runs"
    d.mkdir()
    return d


def make_run(root, name="20260101-000000-humanoid-v5-ppo", cfg=None):
    run_dir = root / name
    for sub in ("checkpoints", "monitor", "tb"):
        (run_dir / sub).mkdir(parents=True)
    (cfg or Config.build("ppo")).save(run_dir / runs.CONFIG_FILE)
    return run_dir


# ------------------------------------------------------------------- creation


def test_create_run_dir_lays_out_the_documented_structure(root):
    cfg = Config.build("ppo")
    run_dir = runs.create_run_dir(cfg, root=root)

    assert run_dir.is_dir()
    assert run_dir.name.endswith("-humanoid-v5-ppo")
    for sub in ("checkpoints", "monitor", "tb"):
        assert (run_dir / sub).is_dir()
    assert Config.load(run_dir / runs.CONFIG_FILE).to_dict() == cfg.to_dict()


def test_create_run_dir_marks_it_latest(root):
    run_dir = runs.create_run_dir(Config.build("ppo"), root=root)
    assert runs.resolve_run("latest", root=root) == run_dir.resolve()


# ----------------------------------------------------------------- resolution


def test_resolve_accepts_a_bare_name_and_a_path(root):
    run_dir = make_run(root)
    assert runs.resolve_run(run_dir.name, root=root) == run_dir.resolve()
    assert runs.resolve_run(run_dir, root=root) == run_dir.resolve()


def test_resolve_accepts_a_checkpoint_zip(root):
    run_dir = make_run(root)
    ckpt = run_dir / "checkpoints" / "ppo_100_steps.zip"
    ckpt.write_bytes(b"")
    assert runs.resolve_run(ckpt, root=root) == run_dir


def test_resolve_accepts_a_top_level_model_zip(root):
    run_dir = make_run(root)
    model = run_dir / runs.BEST_MODEL
    model.write_bytes(b"")
    assert runs.resolve_run(model, root=root) == run_dir


def test_resolve_falls_back_to_the_pointer_file_without_symlinks(root):
    run_dir = make_run(root)
    (root / "LATEST").write_text(run_dir.name + "\n")
    assert runs.resolve_run("latest", root=root) == run_dir.resolve()


def test_resolve_latest_with_no_runs_explains_how_to_train(root):
    with pytest.raises(SystemExit, match="No runs found"):
        runs.resolve_run("latest", root=root)


def test_resolve_missing_run_names_it(root):
    with pytest.raises(SystemExit, match="nope"):
        runs.resolve_run("nope", root=root)


# ---------------------------------------------------------------------- listing


def test_list_runs_is_newest_first_and_skips_the_latest_symlink(root):
    make_run(root, "20260101-000000-humanoid-v5-ppo")
    make_run(root, "20260303-000000-humanoid-v5-sac")
    runs._point_latest_at(root / "20260303-000000-humanoid-v5-sac", root)

    found = runs.list_runs(root)
    assert [p.name for p in found] == [
        "20260303-000000-humanoid-v5-sac",
        "20260101-000000-humanoid-v5-ppo",
    ]


def test_list_runs_on_a_fresh_checkout_is_empty(tmp_path):
    assert runs.list_runs(tmp_path / "absent") == []


# ------------------------------------------------------------ model selection


def test_find_model_honours_the_preference(root):
    run_dir = make_run(root)
    (run_dir / runs.BEST_MODEL).write_bytes(b"")
    (run_dir / runs.FINAL_MODEL).write_bytes(b"")

    assert runs.find_model(run_dir, "best").name == runs.BEST_MODEL
    assert runs.find_model(run_dir, "final").name == runs.FINAL_MODEL


def test_find_model_falls_back_when_the_preferred_one_is_missing(root):
    """A run killed before its first eval has no best_model -- still playable."""
    run_dir = make_run(root)
    (run_dir / runs.FINAL_MODEL).write_bytes(b"")
    assert runs.find_model(run_dir, "best").name == runs.FINAL_MODEL


def test_find_model_last_picks_the_newest_checkpoint(root):
    import os
    import time

    run_dir = make_run(root)
    older = run_dir / "checkpoints" / "ppo_100_steps.zip"
    newer = run_dir / "checkpoints" / "ppo_200_steps.zip"
    older.write_bytes(b"")
    newer.write_bytes(b"")
    now = time.time()
    os.utime(older, (now - 100, now - 100))
    os.utime(newer, (now, now))

    assert runs.find_model(run_dir, "last") == newer


def test_find_model_with_nothing_saved_is_a_friendly_error(root):
    with pytest.raises(SystemExit, match="No saved model"):
        runs.find_model(make_run(root))


# -------------------------------------------------------------- stats pairing


def test_best_model_prefers_its_own_snapshot(root):
    run_dir = make_run(root)
    best_stats = run_dir / "best_vecnormalize.pkl"
    best_stats.write_bytes(b"")
    (run_dir / "vecnormalize.pkl").write_bytes(b"")

    assert runs.stats_path(run_dir, run_dir / runs.BEST_MODEL) == best_stats


def test_best_model_falls_back_to_the_run_level_stats(root):
    run_dir = make_run(root)
    end_stats = run_dir / "vecnormalize.pkl"
    end_stats.write_bytes(b"")
    assert runs.stats_path(run_dir, run_dir / runs.BEST_MODEL) == end_stats


def test_a_checkpoint_prefers_the_stats_saved_beside_it(root):
    run_dir = make_run(root)
    ckpt = run_dir / "checkpoints" / "ppo_100_steps.zip"
    ckpt.write_bytes(b"")
    beside = run_dir / "checkpoints" / "ppo_vecnormalize_100_steps.pkl"
    beside.write_bytes(b"")
    (run_dir / "vecnormalize.pkl").write_bytes(b"")

    assert runs.stats_path(run_dir, ckpt) == beside


def test_no_stats_at_all_returns_none(root):
    run_dir = make_run(root)
    assert runs.stats_path(run_dir, run_dir / runs.BEST_MODEL) is None
