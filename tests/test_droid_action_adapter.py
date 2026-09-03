from __future__ import annotations

import numpy as np
import pytest
from axis_train.action_transform import ActionTransformSpec, make_action_transform_spec
from axis_train.policy import AxisDroidPolicyAdapter


def test_droid_action_round_trip_matches_training_formula() -> None:
    adapter = AxisDroidPolicyAdapter(velocity_scale=1.0 / 15.0)
    current = np.asarray([0.04, 0.04, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    target = np.asarray([0.01, 0.01, -0.18, -0.12, 0.03, 0.08, 0.21, 0.29, 0.43], dtype=np.float32)

    action = adapter.qpos_target_to_droid_action(current, target)
    reconstructed = adapter.droid_action_to_qpos_target(current, action)

    np.testing.assert_allclose(reconstructed, target, atol=1.0e-7)
    assert action.shape == (8,)
    assert 0.0 <= action[7] <= 1.0


def test_action_chunk_is_integrated_sequentially_and_clipped() -> None:
    adapter = AxisDroidPolicyAdapter(velocity_scale=0.1, clip_arm_velocity=True)
    initial = np.asarray([0.04, 0.04, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    actions = np.asarray(
        [
            [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0],
        ],
        dtype=np.float32,
    )

    targets = adapter.droid_action_chunk_to_qpos_targets(initial, actions)

    np.testing.assert_allclose(targets[:, 2], [0.1, 0.2], atol=1.0e-7)
    np.testing.assert_allclose(targets[0, :2], [0.04, 0.04], atol=1.0e-7)
    np.testing.assert_allclose(targets[1, :2], [0.0, 0.0], atol=1.0e-7)


def test_openpi_observation_uses_live_inference_keys_without_action() -> None:
    adapter = AxisDroidPolicyAdapter(velocity_scale=1.0 / 15.0)
    observation = adapter.build_openpi_observation(
        qpos=np.asarray([0.02, 0.02, 1, 2, 3, 4, 5, 6, 7], dtype=np.float32),
        exterior_image=np.zeros((16, 16, 3), dtype=np.uint8),
        wrist_image=np.zeros((16, 16, 3), dtype=np.uint8),
        prompt="pick the object",
    )

    assert set(observation) == {
        "observation.exterior_image_1_left",
        "observation.wrist_image_left",
        "observation.joint_position",
        "observation.gripper_position",
        "prompt",
    }
    np.testing.assert_allclose(observation["observation.gripper_position"], [0.5])


@pytest.mark.parametrize(
    ("method", "value"),
    [
        ("qpos_to_droid_state", np.zeros(8, dtype=np.float32)),
        ("qpos_to_droid_state", np.full(9, np.nan, dtype=np.float32)),
    ],
)
def test_adapter_rejects_invalid_state(method: str, value: np.ndarray) -> None:
    adapter = AxisDroidPolicyAdapter(velocity_scale=1.0 / 15.0)
    with pytest.raises(ValueError):
        getattr(adapter, method)(value)


def test_adapter_is_constructed_from_verified_spec(tmp_path) -> None:
    value = make_action_transform_spec(
        source_fps=5.0,
        source_joint_names=["finger1", "finger2", *[f"joint{i}" for i in range(1, 8)]],
        third_person_source="observation.images.base",
        wrist_source="observation.images.left_wrist",
    )
    path = tmp_path / "action_transform_spec.json"
    import json

    path.write_text(json.dumps(value), encoding="utf-8")
    spec = ActionTransformSpec.load(path)
    adapter = AxisDroidPolicyAdapter.from_action_transform_spec(spec)

    assert adapter.velocity_scale == pytest.approx(0.2)
    assert adapter.clip_arm_velocity is False
    assert adapter.action_transform_spec_sha256 == value["spec_sha256"]


def test_unclipped_adapter_round_trip_preserves_fast_expert_target() -> None:
    adapter = AxisDroidPolicyAdapter(velocity_scale=0.2)
    current = np.asarray([0.04, 0.04, 0.0, -0.7, 0.0, -2.3, 0.0, 1.5, 0.7], dtype=np.float32)
    target = current.copy()
    target[2] += 0.32

    action = adapter.qpos_target_to_droid_action(current, target)
    reconstructed = adapter.droid_action_to_qpos_target(current, action)

    assert action[0] == pytest.approx(1.6)
    np.testing.assert_allclose(reconstructed, target, atol=1.0e-7)
