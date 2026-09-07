"""Task wrappers: the command must reach the observation and drive the reward.

The failure mode worth guarding against is a task that *looks* wired up but
whose command never influences anything -- the policy then quietly learns to
ignore it and the task variant is decorative.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

from humanoid_rl.tasks import (
    DEFAULT_TASK,
    TASKS,
    GoalReaching,
    NaturalWalk,
    TargetVelocity,
    parse_tasks,
    wrap_task,
)

pytestmark = pytest.mark.slow


@pytest.fixture
def base():
    env = gym.make("Humanoid-v5")
    yield env
    env.close()


def test_walk_is_the_untouched_environment(base):
    assert wrap_task(base, "walk") is base
    assert wrap_task(base, DEFAULT_TASK) is base


def test_unknown_task_is_rejected(base):
    with pytest.raises(ValueError, match="unknown task"):
        wrap_task(base, "backflip")


@pytest.mark.parametrize("task,extra", [("velocity", 1), ("goal", 3)])
def test_command_widens_the_observation(base, task, extra):
    base_dim = base.observation_space.shape[0]
    env = wrap_task(base, task)
    assert env.observation_space.shape == (base_dim + extra,)

    obs, _ = env.reset(seed=0)
    assert obs.shape == (base_dim + extra,)
    obs, *_ = env.step(env.action_space.sample())
    assert obs.shape == (base_dim + extra,)


@pytest.mark.parametrize("task", [t for t in TASKS if t != DEFAULT_TASK])
def test_seeded_episodes_are_reproducible(base, task):
    env = wrap_task(base, task)
    first, _ = env.reset(seed=123)
    second, _ = env.reset(seed=123)
    np.testing.assert_allclose(first, second)


def test_velocity_command_lands_in_the_observation(base):
    env = TargetVelocity(base, speed_range=(1.5, 1.5))
    obs, _ = env.reset(seed=0)
    assert obs[-1] == pytest.approx(1.5)


def test_velocity_command_varies_between_episodes(base):
    env = TargetVelocity(base, speed_range=(0.5, 3.0))
    commands = set()
    for seed in range(8):
        obs, _ = env.reset(seed=seed)
        commands.add(round(float(obs[-1]), 6))
    assert len(commands) > 1, "the command must be resampled, or the policy can ignore it"


def test_velocity_reward_peaks_at_the_commanded_speed(base):
    """Overshooting must be penalised like undershooting -- otherwise it sprints."""
    env = TargetVelocity(base, speed_range=(2.0, 2.0), tracking_weight=2.0, tracking_sigma=0.5)
    env.reset(seed=0)

    on_target = env._task_reward({"x_velocity": 2.0})
    too_slow = env._task_reward({"x_velocity": 1.0})
    too_fast = env._task_reward({"x_velocity": 3.0})

    assert on_target > too_slow
    assert on_target > too_fast
    assert too_slow == pytest.approx(too_fast)  # symmetric about the command


def test_velocity_replaces_rather_than_stacks_the_forward_reward(base):
    """The stock forward term must be gone, or the task is just a speed bonus."""
    env = TargetVelocity(base, speed_range=(2.0, 2.0))
    env.reset(seed=0)
    _, reward, _, _, info = env.step(np.zeros(env.action_space.shape))

    expected = (
        info["reward_survive"] - info["reward_ctrl"] - info["reward_contact"] + info["reward_task"]
    )
    assert reward == pytest.approx(expected, rel=1e-6)


def test_goal_is_reachable_and_encoded_relative_to_the_humanoid(base):
    env = GoalReaching(base, goal_radius=(5.0, 5.0), obs_scale=10.0)
    obs, _ = env.reset(seed=0)

    dx, dy, distance = obs[-3:] * env.obs_scale
    assert distance == pytest.approx(np.hypot(dx, dy), rel=1e-6)
    assert distance == pytest.approx(5.0, abs=0.1)  # starts near the origin


def test_goal_rewards_closing_the_distance(base):
    env = GoalReaching(base, goal_radius=(5.0, 5.0), progress_weight=5.0, tolerance=0.0)
    env.reset(seed=0)
    env.goal = np.array([5.0, 0.0])
    env._previous_distance = 5.0

    # Moved one metre closer.
    closer = env._task_reward({"x_position": 1.0, "y_position": 0.0})
    assert closer == pytest.approx(5.0)

    # Moved back to where it started: the gain is given back, not kept.
    env._previous_distance = 4.0
    further = env._task_reward({"x_position": 0.0, "y_position": 0.0})
    assert further == pytest.approx(-5.0)


def test_goal_pays_a_bonus_inside_the_tolerance(base):
    """Holding progress at zero isolates the bonus from the progress term."""
    env = GoalReaching(base, goal_radius=(5.0, 5.0), tolerance=0.5, arrival_bonus=2.0)
    env.reset(seed=0)
    env.goal = np.array([1.0, 0.0])

    env._previous_distance = 0.1  # standing still, inside tolerance
    inside = env._task_reward({"x_position": 0.9, "y_position": 0.0})

    env._previous_distance = 0.9  # standing still, outside tolerance
    outside = env._task_reward({"x_position": 0.1, "y_position": 0.0})

    assert inside == pytest.approx(2.0)
    assert outside == pytest.approx(0.0)
    assert inside - outside == pytest.approx(env.arrival_bonus)


def test_goal_varies_between_episodes(base):
    env = GoalReaching(base)
    goals = {tuple(np.round(env.reset(seed=s)[0][-3:], 6)) for s in range(8)}
    assert len(goals) > 1


# ------------------------------------------------------------- natural walk


def test_natural_does_not_change_the_observation(base):
    """No command, so a policy trained on `walk` can be scored under it."""
    base_dim = base.observation_space.shape[0]
    env = NaturalWalk(base)
    assert env.observation_space.shape == (base_dim,)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (base_dim,)


def test_natural_finds_the_arm_joints_by_name(base):
    env = NaturalWalk(base)
    assert len(env._arm_dofs) == 6, "both shoulders (x2) and both elbows"
    assert env._torso_id >= 0


def test_natural_only_ever_subtracts(base):
    """A gait penalty that could go negative would reward flailing."""
    env = NaturalWalk(base)
    env.reset(seed=0)
    for _ in range(20):
        _, reward, terminated, _, info = env.step(env.action_space.sample())
        assert info["gait_penalty"] >= 0.0
        assert info["cost_arm"] >= 0.0
        assert info["cost_tilt"] >= 0.0
        assert info["cost_drift"] >= 0.0
        if terminated:
            break


def test_natural_penalty_matches_the_reported_components(base):
    env = NaturalWalk(base)
    env.reset(seed=0)
    _, _, _, _, info = env.step(np.zeros(env.action_space.shape))
    assert info["gait_penalty"] == pytest.approx(
        info["cost_arm"] + info["cost_tilt"] + info["cost_drift"]
    )


def test_natural_reward_is_the_stock_reward_minus_the_penalty(base):
    import gymnasium as gym

    stock = gym.make("Humanoid-v5")
    tidy = NaturalWalk(gym.make("Humanoid-v5"))
    try:
        stock.reset(seed=11)
        tidy.reset(seed=11)
        action = np.zeros(stock.action_space.shape)
        _, stock_reward, _, _, _ = stock.step(action)
        _, tidy_reward, _, _, info = tidy.step(action)
        assert tidy_reward == pytest.approx(stock_reward - info["gait_penalty"])
    finally:
        stock.close()
        tidy.close()


def test_zero_weights_leave_the_reward_untouched(base):
    env = NaturalWalk(base, arm_cost_weight=0, tilt_cost_weight=0, drift_cost_weight=0)
    env.reset(seed=0)
    _, _, _, _, info = env.step(np.zeros(env.action_space.shape))
    assert info["gait_penalty"] == pytest.approx(0.0)


def test_uprightness_is_one_at_reset(base):
    """The humanoid starts standing, so its torso up-axis is near vertical."""
    env = NaturalWalk(base)
    env.reset(seed=0)
    assert env._uprightness() > 0.95


# -------------------------------------------------------------- composition


@pytest.mark.parametrize(
    "spec,expected",
    [
        (None, []),
        ("walk", []),
        ("natural", ["natural"]),
        ("velocity,natural", ["velocity", "natural"]),
        (" Velocity , Natural ", ["velocity", "natural"]),
        ("goal,natural", ["goal", "natural"]),
        ("walk,natural", ["natural"]),
    ],
)
def test_parse_tasks(spec, expected):
    assert parse_tasks(spec) == expected


@pytest.mark.parametrize(
    "spec,message",
    [
        ("velocity,goal", "pick one"),
        ("natural,natural", "repeated"),
        ("backflip", "unknown task"),
        ("velocity,backflip", "unknown task"),
    ],
)
def test_parse_tasks_rejects(spec, message):
    with pytest.raises(ValueError, match=message):
        parse_tasks(spec)


def test_combining_stacks_both_wrappers(base):
    """velocity,natural must command a speed AND penalise posture."""
    env = wrap_task(base, "velocity,natural")
    assert isinstance(env, NaturalWalk)
    assert isinstance(env.env, TargetVelocity)

    # The command still reaches the observation through the outer wrapper.
    obs, _ = env.reset(seed=0)
    assert obs.shape == (base.observation_space.shape[0] + 1,)

    _, _, _, _, info = env.step(np.zeros(env.action_space.shape))
    assert "reward_task" in info      # from TargetVelocity
    assert "gait_penalty" in info     # from NaturalWalk


def test_keyed_kwargs_reach_the_right_wrapper(base):
    env = wrap_task(
        base,
        "velocity,natural",
        {"velocity": {"speed_range": (1.4, 1.4)}, "natural": {"arm_cost_weight": 0.9}},
    )
    assert env.arm_cost_weight == 0.9
    assert env.env.speed_range == (1.4, 1.4)
    obs, _ = env.reset(seed=0)
    assert obs[-1] == pytest.approx(1.4)


def test_flat_kwargs_still_work_for_a_single_task(base):
    env = wrap_task(base, "natural", {"arm_cost_weight": 0.7})
    assert env.arm_cost_weight == 0.7
