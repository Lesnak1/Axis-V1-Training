from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from axis_train.action_transform import ActionTransformSpec


def _finite_vector(value: Any, *, size: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


@dataclass(frozen=True)
class AxisDroidPolicyAdapter:
    """Translate between AXIS 9D qpos targets and OpenPI's DROID 8D interface.

    AXIS layout is ``[left_finger, right_finger, arm_joint_1, ..., arm_joint_7]``.
    DROID layout is ``[arm_velocity_1, ..., arm_velocity_7, gripper_closedness]``.
    """

    velocity_scale: float
    gripper_open_q: tuple[float, float] = (0.04, 0.04)
    gripper_close_q: tuple[float, float] = (0.0, 0.0)
    clip_arm_velocity: bool = False
    action_transform_spec_sha256: str | None = None

    @classmethod
    def from_action_transform_spec(
        cls, spec: ActionTransformSpec | str | Path
    ) -> AxisDroidPolicyAdapter:
        resolved = spec if isinstance(spec, ActionTransformSpec) else ActionTransformSpec.load(spec)
        return cls(
            velocity_scale=resolved.velocity_scale,
            gripper_open_q=resolved.gripper_open_q,
            gripper_close_q=resolved.gripper_close_q,
            clip_arm_velocity=resolved.clip_arm_action is not None,
            action_transform_spec_sha256=resolved.spec_sha256,
        )

    def __post_init__(self) -> None:
        if not np.isfinite(self.velocity_scale) or self.velocity_scale <= 0.0:
            raise ValueError(f"velocity_scale must be finite and positive, got {self.velocity_scale}")
        open_q = _finite_vector(self.gripper_open_q, size=2, name="gripper_open_q")
        close_q = _finite_vector(self.gripper_close_q, size=2, name="gripper_close_q")
        if np.any(np.abs(open_q - close_q) <= 1.0e-6):
            raise ValueError("gripper open and close positions must differ for both fingers")

    def gripper_closedness(self, finger_qpos: Any) -> np.float32:
        fingers = _finite_vector(finger_qpos, size=2, name="finger_qpos")
        open_q = np.asarray(self.gripper_open_q, dtype=np.float32)
        close_q = np.asarray(self.gripper_close_q, dtype=np.float32)
        open_ratio = np.clip((fingers - close_q) / (open_q - close_q), 0.0, 1.0).mean()
        return np.float32(1.0 - open_ratio)

    def qpos_to_droid_state(self, qpos: Any) -> np.ndarray:
        axis_qpos = _finite_vector(qpos, size=9, name="qpos")
        return np.concatenate(
            [axis_qpos[2:9], np.asarray([self.gripper_closedness(axis_qpos[:2])], dtype=np.float32)]
        )

    def qpos_target_to_droid_action(self, current_qpos: Any, target_qpos: Any) -> np.ndarray:
        current = _finite_vector(current_qpos, size=9, name="current_qpos")
        target = _finite_vector(target_qpos, size=9, name="target_qpos")
        arm_velocity = (target[2:9] - current[2:9]) / np.float32(self.velocity_scale)
        if self.clip_arm_velocity:
            arm_velocity = np.clip(arm_velocity, -1.0, 1.0)
        return np.concatenate(
            [arm_velocity, np.asarray([self.gripper_closedness(target[:2])], dtype=np.float32)]
        ).astype(np.float32)

    def droid_action_to_qpos_target(self, current_qpos: Any, droid_action: Any) -> np.ndarray:
        current = _finite_vector(current_qpos, size=9, name="current_qpos")
        action = _finite_vector(droid_action, size=8, name="droid_action")
        arm_velocity = action[:7]
        if self.clip_arm_velocity:
            arm_velocity = np.clip(arm_velocity, -1.0, 1.0)
        closedness = np.clip(action[7], 0.0, 1.0)
        open_q = np.asarray(self.gripper_open_q, dtype=np.float32)
        close_q = np.asarray(self.gripper_close_q, dtype=np.float32)
        gripper_target = open_q + closedness * (close_q - open_q)
        arm_target = current[2:9] + arm_velocity * np.float32(self.velocity_scale)
        return np.concatenate([gripper_target, arm_target]).astype(np.float32)

    def droid_action_chunk_to_qpos_targets(self, initial_qpos: Any, droid_actions: Any) -> np.ndarray:
        current = _finite_vector(initial_qpos, size=9, name="initial_qpos")
        actions = np.asarray(droid_actions, dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != 8:
            raise ValueError(f"droid_actions must have shape (N, 8), got {actions.shape}")
        if not np.isfinite(actions).all():
            raise ValueError("droid_actions must contain only finite values")

        targets = np.empty((len(actions), 9), dtype=np.float32)
        for index, action in enumerate(actions):
            current = self.droid_action_to_qpos_target(current, action)
            targets[index] = current
        return targets

    def build_openpi_observation(
        self,
        *,
        qpos: Any,
        exterior_image: Any,
        wrist_image: Any,
        prompt: str,
    ) -> dict[str, Any]:
        state = self.qpos_to_droid_state(qpos)
        exterior = np.asarray(exterior_image)
        wrist = np.asarray(wrist_image)
        for name, image in (("exterior_image", exterior), ("wrist_image", wrist)):
            if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
                raise ValueError(f"{name} must be an HWC uint8 RGB image, got {image.shape} {image.dtype}")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        return {
            "observation.exterior_image_1_left": exterior,
            "observation.wrist_image_left": wrist,
            "observation.joint_position": state[:7],
            "observation.gripper_position": state[7:8],
            "prompt": prompt,
        }
