"""Scenes must change how the humanoid looks and nothing else.

The whole premise of `--scene` is that a policy trained on the stock model can
be viewed in any scene without retraining. That only holds if the scene files
are physically identical to Gymnasium's `humanoid.xml`. A stray geom without
`density="0"` would shift a body's inertia and quietly degrade the gait --
no error, just a worse-looking walk. These tests make that impossible to miss.
"""

from __future__ import annotations

import numpy as np
import pytest

from humanoid_rl.envs import SCENES, scene_kwargs

pytestmark = pytest.mark.slow

CUSTOM_SCENES = [name for name, scene in SCENES.items() if scene.path is not None]


@pytest.fixture
def stock_model():
    import os

    import mujoco
    from gymnasium.envs.mujoco import humanoid_v5

    path = os.path.join(os.path.dirname(humanoid_v5.__file__), "assets", "humanoid.xml")
    return mujoco.MjModel.from_xml_path(path)


@pytest.mark.parametrize("scene", CUSTOM_SCENES)
def test_scene_file_loads(scene):
    import mujoco

    assert mujoco.MjModel.from_xml_path(str(SCENES[scene].path)) is not None


@pytest.mark.parametrize("scene", CUSTOM_SCENES)
def test_scene_preserves_the_physics_exactly(scene, stock_model):
    import mujoco

    custom = mujoco.MjModel.from_xml_path(str(SCENES[scene].path))

    # Same skeleton.
    for field in ("nq", "nv", "nu", "nbody", "njnt"):
        assert getattr(custom, field) == getattr(stock_model, field), field

    # Same dynamics. Decorative geoms carry density="0", so they must not move
    # a single gram or shift any centre of mass.
    np.testing.assert_allclose(custom.body_mass, stock_model.body_mass)
    np.testing.assert_allclose(custom.body_inertia, stock_model.body_inertia)
    np.testing.assert_allclose(custom.body_ipos, stock_model.body_ipos)
    np.testing.assert_allclose(custom.dof_damping, stock_model.dof_damping)
    np.testing.assert_allclose(custom.dof_armature, stock_model.dof_armature)
    np.testing.assert_allclose(custom.jnt_range, stock_model.jnt_range)
    np.testing.assert_allclose(custom.actuator_gear, stock_model.actuator_gear)


@pytest.mark.parametrize("scene", CUSTOM_SCENES)
def test_scene_keeps_the_observation_space(scene):
    """A changed observation size would make every trained policy unloadable."""
    import gymnasium as gym

    stock = gym.make("Humanoid-v5")
    custom = gym.make("Humanoid-v5", **scene_kwargs(scene, "Humanoid-v5"))
    try:
        assert custom.observation_space.shape == stock.observation_space.shape
        assert custom.action_space.shape == stock.action_space.shape
    finally:
        stock.close()
        custom.close()


@pytest.mark.parametrize("scene", CUSTOM_SCENES)
def test_same_actions_produce_the_same_trajectory(scene):
    """The strongest check: identical inputs, identical simulated motion."""
    import gymnasium as gym

    stock = gym.make("Humanoid-v5")
    custom = gym.make("Humanoid-v5", **scene_kwargs(scene, "Humanoid-v5"))
    try:
        a, _ = stock.reset(seed=7)
        b, _ = custom.reset(seed=7)
        np.testing.assert_allclose(a, b, atol=1e-10)

        rng = np.random.default_rng(0)
        for _ in range(50):
            action = rng.uniform(-0.4, 0.4, size=stock.action_space.shape)
            obs_a, rew_a, term_a, _, _ = stock.step(action)
            obs_b, rew_b, term_b, _, _ = custom.step(action)
            np.testing.assert_allclose(obs_a, obs_b, atol=1e-8)
            assert rew_a == pytest.approx(rew_b)
            assert term_a == term_b
    finally:
        stock.close()
        custom.close()


def test_default_scene_adds_no_arguments():
    assert scene_kwargs("default", "Humanoid-v5") == {}
    assert scene_kwargs(None, "Humanoid-v5") == {}


def test_unknown_scene_lists_the_valid_ones():
    with pytest.raises(SystemExit, match="Unknown scene"):
        scene_kwargs("mars", "Humanoid-v5")


@pytest.mark.parametrize("scene", CUSTOM_SCENES)
def test_a_scene_is_refused_for_an_environment_it_was_not_built_for(scene):
    """A scene replaces the whole model, not just its appearance.

    Handing Humanoid-v5's body to HumanoidStandup-v5 starts it upright at
    z=1.39 instead of lying at z=0.10 -- the stand-up task becomes a
    standing-still task. Both have 348 observations, so nothing would fail.
    """
    with pytest.raises(SystemExit, match="built for"):
        scene_kwargs(scene, "HumanoidStandup-v5")


@pytest.mark.parametrize("scene", CUSTOM_SCENES)
def test_every_scene_declares_the_environments_it_suits(scene):
    assert SCENES[scene].env_ids, f"{scene} would be applied to any environment"


def test_default_scene_is_valid_everywhere():
    """`default` adds no xml_file, so it cannot misapply a model."""
    for env_id in ("Humanoid-v5", "HumanoidStandup-v5", "Walker2d-v5"):
        assert scene_kwargs("default", env_id) == {}
