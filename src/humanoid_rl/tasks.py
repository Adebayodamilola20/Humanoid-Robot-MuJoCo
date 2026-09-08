"""Task variants: change what the humanoid is asked to do, not how it learns.

Stock `Humanoid-v5` rewards one thing -- run forward as fast as possible. That
is a single point in a large space of things a humanoid could be asked to do,
and a policy trained on it has no notion of a *command*: you cannot ask it to
walk slowly, or to walk somewhere in particular.

Each task here replaces the environment's forward-velocity term with a
command-tracking term and appends the command to the observation, so one policy
learns to follow instructions rather than to sprint. The command is resampled
every episode, which is what forces the policy to actually read it instead of
memorising an average.

The task name is recorded in `config.json`, so `play`, `record` and `eval`
rebuild the same task without the user repeating a flag -- the same guarantee
the rest of the project makes about observation normalisation.

Stand-up is deliberately absent: Gymnasium already ships `HumanoidStandup-v5`,
so it needs no wrapper here.

    python -m humanoid_rl train --task velocity
    python -m humanoid_rl train --task goal
    python -m humanoid_rl train --env HumanoidStandup-v5
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

TASKS = ("walk", "natural", "velocity", "goal")

ARM_JOINTS = (
    "right_shoulder1",
    "right_shoulder2",
    "right_elbow",
    "left_shoulder1",
    "left_shoulder2",
    "left_elbow",
)

DEFAULT_TASK = "walk"


class _CommandedTask(gym.Wrapper):
    """Shared plumbing: widen the observation by the command, and swap rewards.

    Subclasses provide `command_size`, `_sample_command`, `_command_obs` and
    `_task_reward`. The environment's own forward-velocity reward is subtracted
    out via `info["reward_forward"]`, so the healthy bonus, control cost and
    contact cost all survive untouched -- only the objective changes.
    """

    command_size: int = 0

    def __init__(self, env: gym.Env):
        super().__init__(env)
        base = env.observation_space
        assert isinstance(base, gym.spaces.Box)
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(base.shape[0] + self.command_size,),
            dtype=base.dtype,
        )

    # ------------------------------------------------------------- overrides

    def _sample_command(self) -> None:
        raise NotImplementedError

    def _command_obs(self, info: dict[str, Any]) -> np.ndarray:
        raise NotImplementedError

    def _task_reward(self, info: dict[str, Any]) -> float:
        raise NotImplementedError

    # ------------------------------------------------------------- gym hooks

    def _extend(self, obs: np.ndarray, info: dict[str, Any]) -> np.ndarray:
        return np.concatenate([obs, self._command_obs(info)]).astype(obs.dtype)

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)
        # After reset so the command is drawn from the seeded env RNG, which
        # keeps a seeded run byte-for-byte reproducible.
        self._sample_command()
        self._on_reset(info)
        return self._extend(obs, info), info

    def _on_reset(self, info: dict[str, Any]) -> None:
        """Hook for tasks that track state across steps."""

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Keep the healthy bonus and the control/contact costs; replace only the
        # "run forward" term. `_task_reward` is called exactly once per step --
        # goal-reaching advances its progress state inside it.
        without_forward = float(reward) - float(info.get("reward_forward", 0.0))
        task_reward = self._task_reward(info)
        info["reward_task"] = task_reward
        return self._extend(obs, info), without_forward + task_reward, terminated, truncated, info


class TargetVelocity(_CommandedTask):
    """Walk at a commanded forward speed rather than as fast as possible.

    Reward peaks when `x_velocity` matches the command and decays as a Gaussian
    either side, so overshooting is penalised just like undershooting -- the
    difference between a controllable gait and a sprint.
    """

    command_size = 1

    def __init__(
        self,
        env: gym.Env,
        *,
        speed_range: tuple[float, float] = (0.5, 3.0),
        tracking_weight: float = 2.0,
        tracking_sigma: float = 0.5,
    ):
        super().__init__(env)
        self.speed_range = speed_range
        self.tracking_weight = tracking_weight
        self.tracking_sigma = tracking_sigma
        self.target_speed = float(np.mean(speed_range))

    def _sample_command(self) -> None:
        low, high = self.speed_range
        self.target_speed = float(self.unwrapped.np_random.uniform(low, high))

    def _command_obs(self, info: dict[str, Any]) -> np.ndarray:
        return np.array([self.target_speed], dtype=np.float64)

    def _task_reward(self, info: dict[str, Any]) -> float:
        error = float(info.get("x_velocity", 0.0)) - self.target_speed
        return self.tracking_weight * float(np.exp(-((error / self.tracking_sigma) ** 2)))


class GoalReaching(_CommandedTask):
    """Walk to a point on the ground, then stop there.

    Reward is progress made toward the goal this step, which stays informative
    at every distance, plus a bonus for holding position once inside the
    tolerance. A pure distance penalty would give a policy that has not yet
    learned to walk almost no gradient to follow.
    """

    command_size = 3

    def __init__(
        self,
        env: gym.Env,
        *,
        goal_radius: tuple[float, float] = (3.0, 8.0),
        tolerance: float = 0.5,
        progress_weight: float = 5.0,
        arrival_bonus: float = 2.0,
        obs_scale: float = 10.0,
    ):
        super().__init__(env)
        self.goal_radius = goal_radius
        self.tolerance = tolerance
        self.progress_weight = progress_weight
        self.arrival_bonus = arrival_bonus
        self.obs_scale = obs_scale
        self.goal = np.zeros(2)
        self._previous_distance = 0.0

    def _sample_command(self) -> None:
        rng = self.unwrapped.np_random
        angle = float(rng.uniform(-np.pi, np.pi))
        radius = float(rng.uniform(*self.goal_radius))
        self.goal = np.array([radius * np.cos(angle), radius * np.sin(angle)])

    def _position(self, info: dict[str, Any]) -> np.ndarray:
        if "x_position" in info:
            return np.array([info["x_position"], info["y_position"]])
        return np.asarray(self.unwrapped.data.qpos[:2], dtype=float)

    def _on_reset(self, info: dict[str, Any]) -> None:
        self._previous_distance = float(np.linalg.norm(self.goal - self._position(info)))

    def _command_obs(self, info: dict[str, Any]) -> np.ndarray:
        delta = self.goal - self._position(info)
        distance = float(np.linalg.norm(delta))
        return np.array(
            [delta[0] / self.obs_scale, delta[1] / self.obs_scale, distance / self.obs_scale]
        )

    def _task_reward(self, info: dict[str, Any]) -> float:
        distance = float(np.linalg.norm(self.goal - self._position(info)))
        progress = self._previous_distance - distance
        self._previous_distance = distance
        reward = self.progress_weight * progress
        if distance < self.tolerance:
            reward += self.arrival_bonus
        info["goal_distance"] = distance
        return reward


class NaturalWalk(gym.Wrapper):
    """Same objective as `walk`, but ask for a tidier gait.

    Stock `Humanoid-v5` rewards "upright and moving forward" and says nothing
    about *how*. Reinforcement learning takes that literally, so the policy
    converges on a lurching, arm-flailing shuffle -- it scores well, and looking
    like a person was never part of the deal.

    This adds three penalties on top of the stock reward, targeting exactly what
    is visibly wrong:

    * **Arm thrashing.** The arms carry only 25 units of motor gear against the
      legs' 100-300, so swinging them is nearly free and the policy uses them as
      flywheels. Penalising arm joint speed removes that shortcut.
    * **Torso tilt.** The healthy bonus only requires the torso between 1.0 and
      2.0 m; leaning hard forward is legal and helps sprint. Penalising tilt
      asks it to stay over its feet.
    * **Lateral drift.** Only forward velocity is rewarded, so weaving costs
      nothing. Penalising sideways speed asks it to walk in a line.

    Nothing here touches the observation, so a policy trained on `walk` can be
    *evaluated* under `natural` to measure how untidy its gait is. The reward
    components are published in `info` for exactly that purpose.

    The weights are deliberately mild. Penalties strong enough to enforce a
    tidy gait immediately also prevent the policy from ever learning to walk --
    it stands still, which scores better than falling.
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        arm_cost_weight: float = 0.05,
        tilt_cost_weight: float = 2.0,
        drift_cost_weight: float = 0.5,
    ):
        super().__init__(env)
        self.arm_cost_weight = arm_cost_weight
        self.tilt_cost_weight = tilt_cost_weight
        self.drift_cost_weight = drift_cost_weight
        self._arm_dofs = self._locate_arm_dofs()
        self._torso_id = self._locate_torso()

    # ------------------------------------------------------------- lookups

    def _locate_arm_dofs(self) -> list[int]:
        """Velocity indices of the shoulder and elbow joints.

        Looked up by name once, rather than hard-coded, so the wrapper survives
        a model whose joint ordering differs.
        """
        import mujoco

        model = self.unwrapped.model
        dofs = []
        for name in ARM_JOINTS:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id >= 0:
                dofs.append(int(model.jnt_dofadr[joint_id]))
        return dofs

    def _locate_torso(self) -> int:
        import mujoco

        return mujoco.mj_name2id(self.unwrapped.model, mujoco.mjtObj.mjOBJ_BODY, "torso")

    # ------------------------------------------------------------- measures

    def _arm_motion(self) -> float:
        """Mean absolute speed of the arm joints, in rad/s."""
        if not self._arm_dofs:
            return 0.0
        qvel = self.unwrapped.data.qvel
        return float(np.mean([abs(qvel[i]) for i in self._arm_dofs]))

    def _uprightness(self) -> float:
        """1.0 when the torso's own up-axis points at the sky, 0.0 on its side.

        `xmat` is the body's rotation matrix flattened; entry 8 is the z
        component of its third column, i.e. how much of its up-axis survives
        into world-up.
        """
        if self._torso_id < 0:
            return 1.0
        return float(self.unwrapped.data.xmat[self._torso_id].reshape(3, 3)[2, 2])

    # ------------------------------------------------------------ gym hooks

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)

        arm_cost = self.arm_cost_weight * self._arm_motion()
        tilt_cost = self.tilt_cost_weight * max(0.0, 1.0 - self._uprightness())
        drift_cost = self.drift_cost_weight * abs(float(info.get("y_velocity", 0.0)))

        info["cost_arm"] = arm_cost
        info["cost_tilt"] = tilt_cost
        info["cost_drift"] = drift_cost
        info["gait_penalty"] = arm_cost + tilt_cost + drift_cost
        info["uprightness"] = self._uprightness()
        info["arm_motion"] = self._arm_motion()

        return obs, float(reward) - info["gait_penalty"], terminated, truncated, info


WRAPPERS: dict[str, type[gym.Wrapper]] = {
    "natural": NaturalWalk,
    "velocity": TargetVelocity,
    "goal": GoalReaching,
}


# Tasks that replace the environment's forward-velocity reward with a command.
# Two of them would subtract that reward twice, so at most one may be used.
COMMANDED = ("velocity", "goal")


def parse_tasks(spec: str | None) -> list[str]:
    """Split and validate a task spec like `velocity,natural`.

    Composition is the point: the diagnosis for this humanoid's gait is that it
    sprints at 5.4 m/s with a 31-degree forward lean, because forward speed is
    rewarded without limit. Fixing that wants a commanded speed *and* a posture
    penalty, which are two different wrappers.
    """
    names = [part.strip().lower() for part in (spec or DEFAULT_TASK).split(",")]
    names = [name for name in names if name and name != DEFAULT_TASK]

    for name in names:
        if name not in WRAPPERS:
            raise ValueError(
                f"unknown task {name!r}, expected one or more of {', '.join(TASKS)}"
            )
    if len(set(names)) != len(names):
        raise ValueError(f"repeated task in {spec!r}")

    commanded = [name for name in names if name in COMMANDED]
    if len(commanded) > 1:
        raise ValueError(
            f"{' and '.join(commanded)} both replace the forward reward; pick one"
        )
    return names


def wrap_task(
    env: gym.Env, task: str, task_kwargs: dict[str, Any] | None = None
) -> gym.Env:
    """Apply the named task(s) to `env`. `walk` is the unmodified environment.

    Several may be combined with commas -- `velocity,natural` asks for a
    commanded speed *and* a tidy posture. Each wraps the previous, so the last
    named sees the reward the earlier ones produced.

    `task_kwargs` may be flat (applied to a single task) or keyed by task name
    when combining, e.g. `{"velocity": {...}, "natural": {...}}`.
    """
    names = parse_tasks(task)
    if not names:
        return env

    kwargs = task_kwargs or {}
    keyed = all(isinstance(value, dict) for value in kwargs.values()) and bool(kwargs)

    for name in names:
        options = kwargs.get(name, {}) if keyed else (kwargs if len(names) == 1 else {})
        env = WRAPPERS[name](env, **options)
    return env
