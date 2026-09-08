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

CUSTOM_SCENES = [name for name, path in SCENES.items() if path is not None]


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

    assert mujoco.MjModel.from_xml_path(str(SCENES[scene])) is not None


@pytest.mark.parametrize("scene", CUSTOM_SCENES)
def test_scene_preserves_the_physics_exactly(scene, stock_model):
    import mujoco

    custom = mujoco.MjModel.from_xml_path(str(SCENES[scene]))

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
    custom = gym.make("Humanoid-v5", **scene_kwargs(scene))
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
    custom = gym.make("Humanoid-v5", **scene_kwargs(scene))
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
    assert scene_kwargs("default") == {}
    assert scene_kwargs(None) == {}


def test_unknown_scene_lists_the_valid_ones():
    with pytest.raises(SystemExit, match="Unknown scene"):
        scene_kwargs("mars")
