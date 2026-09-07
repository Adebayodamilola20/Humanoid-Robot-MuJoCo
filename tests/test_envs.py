"""Environment plumbing, including the viewer's camera-control compatibility."""

from __future__ import annotations

import pytest

from humanoid_rl.envs import patch_viewer_camera_controls

pytestmark = pytest.mark.slow


@pytest.fixture
def camera():
    import mujoco

    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body><geom size="1"/></body></worldbody></mujoco>'
    )
    return mujoco, model, mujoco.MjvCamera(), mujoco.MjvScene(model, 100)


def test_camera_accepts_both_mujoco_signatures(camera):
    """Gymnasium 1.3 passes the scene argument MuJoCo 3.12 removed.

    Without the shim this raises TypeError inside a GLFW callback the moment
    the user drags or scrolls, which reads as a crash in the policy.
    """
    mujoco, model, cam, scn = camera
    patch_viewer_camera_controls()

    start = cam.distance
    # Six arguments: what Gymnasium actually calls.
    mujoco.mjv_moveCamera(model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0, 0.005, scn, cam)
    assert cam.distance != start

    # Five arguments: MuJoCo 3.12's own signature, still honoured.
    middle = cam.distance
    mujoco.mjv_moveCamera(model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0, 0.005, cam)
    assert cam.distance != middle


def test_patching_twice_does_not_nest_wrappers(camera):
    mujoco, *_ = camera
    patch_viewer_camera_controls()
    once = mujoco.mjv_moveCamera
    patch_viewer_camera_controls()
    assert mujoco.mjv_moveCamera is once
