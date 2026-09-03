from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

ACTION_TRANSFORM_SCHEMA = "axis_action_transform_spec_v1"


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def make_action_transform_spec(
    *,
    source_fps: float,
    source_joint_names: list[str],
    third_person_source: str,
    wrist_source: str,
    action_horizon: int = 15,
) -> dict[str, Any]:
    if not math.isfinite(source_fps) or source_fps <= 0:
        raise ValueError(f"source_fps must be finite and positive, got {source_fps}")
    if len(source_joint_names) != 9 or any(not name for name in source_joint_names):
        raise ValueError("source_joint_names must contain nine non-empty names")
    if action_horizon < 1:
        raise ValueError(f"action_horizon must be positive, got {action_horizon}")
    control_period = 1.0 / float(source_fps)
    body = {
        "schema_version": ACTION_TRANSFORM_SCHEMA,
        "source": {
            "state_layout": source_joint_names,
            "action_layout": "same_as_state",
            "action_semantics": "mujoco_joint_position_target",
            "source_fps": float(source_fps),
            "terminal_action_padding": True,
        },
        "sampling": {
            "train_fps": float(source_fps),
            "downsample_ratio": 1,
            "action_horizon": int(action_horizon),
            "terminal_padding_policy": "exclude_any_horizon_touching_padding",
        },
        "model": {
            "family": "pi05_droid",
            "state_layout": ["arm_joint_position_1_to_7", "gripper_closedness"],
            "action_layout": ["arm_joint_velocity_1_to_7", "gripper_closedness"],
            "state_dim": 8,
            "action_dim": 8,
        },
        "forward": {
            "arm_formula": "(target_arm_qpos-current_arm_qpos)/velocity_scale",
            "velocity_scale": control_period,
            "clip_arm_action": None,
            "gripper_open_q": [0.04, 0.04],
            "gripper_close_q": [0.0, 0.0],
            "gripper_closedness": "mean(1-(q-close_q)/(open_q-close_q)), clipped to [0,1]",
        },
        "inverse": {
            "arm_formula": "current_arm_qpos+model_arm_action*velocity_scale",
            "gripper_formula": "open_q+clip(model_gripper_closedness,0,1)*(close_q-open_q)",
            "rebase_each_executed_action_on_measured_qpos": True,
        },
        "observations": {
            "third_person_source": third_person_source,
            "wrist_source": wrist_source,
            "model_third_person_key": "observation.exterior_image_1_left",
            "model_wrist_key": "observation.wrist_image_left",
        },
    }
    return {**body, "spec_sha256": json_sha256(body)}


def verify_action_transform_spec(spec: Mapping[str, Any]) -> str:
    if spec.get("schema_version") != ACTION_TRANSFORM_SCHEMA:
        raise ValueError(f"Unsupported action transform schema: {spec.get('schema_version')!r}")
    expected = str(spec.get("spec_sha256") or "")
    body = {key: value for key, value in spec.items() if key != "spec_sha256"}
    actual = json_sha256(body)
    if expected != actual:
        raise ValueError(f"Action transform spec hash mismatch: expected={expected}, actual={actual}")
    return actual


def _two_finite_values(value: Any, *, name: str) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{name} must contain exactly two values")
    output = (float(value[0]), float(value[1]))
    if not all(math.isfinite(item) for item in output):
        raise ValueError(f"{name} must contain only finite values")
    return output


@dataclass(frozen=True)
class ActionTransformSpec:
    """Verified, executable binding for AXIS source, OpenPI labels, and control output."""

    path: Path
    spec_sha256: str
    velocity_scale: float
    downsample_ratio: int
    action_horizon: int
    gripper_open_q: tuple[float, float]
    gripper_close_q: tuple[float, float]
    clip_arm_action: tuple[float, float] | None
    source_joint_names: tuple[str, ...]
    third_person_source: str
    wrist_source: str

    @classmethod
    def load(cls, path: str | Path) -> ActionTransformSpec:
        resolved = Path(path).expanduser().resolve(strict=True)
        value = json.loads(resolved.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise TypeError(f"Expected a JSON object: {resolved}")
        spec_sha256 = verify_action_transform_spec(value)

        source = value.get("source")
        sampling = value.get("sampling")
        model = value.get("model")
        forward = value.get("forward")
        inverse = value.get("inverse")
        observations = value.get("observations")
        if not all(isinstance(section, dict) for section in (source, sampling, model, forward, inverse, observations)):
            raise ValueError("Action transform spec is missing a required object section")

        assert isinstance(source, dict)
        assert isinstance(sampling, dict)
        assert isinstance(model, dict)
        assert isinstance(forward, dict)
        assert isinstance(inverse, dict)
        assert isinstance(observations, dict)
        if source.get("action_semantics") != "mujoco_joint_position_target":
            raise ValueError("AXIS source actions must be absolute MuJoCo joint-position targets")
        joint_names = source.get("state_layout")
        if not isinstance(joint_names, list) or len(joint_names) != 9 or any(not str(name) for name in joint_names):
            raise ValueError("source.state_layout must contain nine joint names")
        if model.get("family") != "pi05_droid" or model.get("state_dim") != 8 or model.get("action_dim") != 8:
            raise ValueError("Only the Pi0.5-DROID 8D state/action contract is supported")
        if model.get("action_layout") != ["arm_joint_velocity_1_to_7", "gripper_closedness"]:
            raise ValueError("Unexpected model action layout")
        if forward.get("arm_formula") != "(target_arm_qpos-current_arm_qpos)/velocity_scale":
            raise ValueError("Unexpected forward arm transform")
        raw_clip = forward.get("clip_arm_action")
        if raw_clip is None:
            clip = None
            expected_inverse_formula = "current_arm_qpos+model_arm_action*velocity_scale"
        else:
            clip = _two_finite_values(raw_clip, name="forward.clip_arm_action")
            if clip != (-1.0, 1.0):
                raise ValueError("The only supported Pi0.5-DROID arm clip is [-1, 1]")
            expected_inverse_formula = "current_arm_qpos+clip(model_arm_action,-1,1)*velocity_scale"
        if inverse.get("arm_formula") != expected_inverse_formula:
            raise ValueError("Inverse arm transform does not match forward.clip_arm_action")
        if inverse.get("rebase_each_executed_action_on_measured_qpos") is not True:
            raise ValueError("Inference must rebase every executed action on measured qpos")

        velocity_scale = float(forward.get("velocity_scale", 0.0))
        downsample_ratio = int(sampling.get("downsample_ratio", 0))
        action_horizon = int(sampling.get("action_horizon", 0))
        source_fps = float(source.get("source_fps", 0.0))
        train_fps = float(sampling.get("train_fps", 0.0))
        if not math.isfinite(velocity_scale) or velocity_scale <= 0:
            raise ValueError("forward.velocity_scale must be finite and positive")
        if downsample_ratio < 1 or action_horizon < 1:
            raise ValueError("sampling downsample_ratio and action_horizon must be positive")
        expected_scale = downsample_ratio / source_fps if source_fps > 0 else math.nan
        if not math.isclose(train_fps, source_fps / downsample_ratio, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError("sampling train_fps does not match source_fps/downsample_ratio")
        if not math.isclose(velocity_scale, expected_scale, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError("forward.velocity_scale does not match the executed control period")
        return cls(
            path=resolved,
            spec_sha256=spec_sha256,
            velocity_scale=velocity_scale,
            downsample_ratio=downsample_ratio,
            action_horizon=action_horizon,
            gripper_open_q=_two_finite_values(forward.get("gripper_open_q"), name="forward.gripper_open_q"),
            gripper_close_q=_two_finite_values(
                forward.get("gripper_close_q"), name="forward.gripper_close_q"
            ),
            clip_arm_action=clip,
            source_joint_names=tuple(str(name) for name in joint_names),
            third_person_source=str(observations.get("third_person_source") or ""),
            wrist_source=str(observations.get("wrist_source") or ""),
        )

    def openpi_environment(self) -> dict[str, str]:
        return {
            "AXIS_ACTION_TRANSFORM_SPEC_PATH": str(self.path),
            "AXIS_ACTION_TRANSFORM_SPEC_SHA256": self.spec_sha256,
            "AXIS_CLEAN_DROID_VELOCITY_SCALE": format(self.velocity_scale, ".17g"),
            "AXIS_CLEAN_DROID_DOWNSAMPLE_RATIO": str(self.downsample_ratio),
            "AXIS_CLEAN_DROID_ACTION_HORIZON": str(self.action_horizon),
            "AXIS_CLEAN_DROID_CLIP_VELOCITY_ACTIONS": "true" if self.clip_arm_action is not None else "false",
        }

    def require_environment_matches(self, environ: Mapping[str, str] | None = None) -> None:
        values = os.environ if environ is None else environ
        for name, expected in self.openpi_environment().items():
            actual = values.get(name)
            if actual is None:
                continue
            if name == "AXIS_CLEAN_DROID_VELOCITY_SCALE":
                matches = math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1.0e-12)
            elif name == "AXIS_ACTION_TRANSFORM_SPEC_PATH":
                matches = Path(actual).expanduser().resolve() == Path(expected)
            else:
                matches = actual == expected
            if not matches:
                raise ValueError(f"{name}={actual!r} conflicts with ActionTransformSpec value {expected!r}")


def _task_id(path: Path) -> int | None:
    match = re.match(r"task_(\d+)_", path.name)
    return int(match.group(1)) if match else None


def discover_action_transform_spec(
    clean_root: str | Path,
    *,
    task_ids: set[int] | None = None,
    max_tasks: int | None = None,
) -> ActionTransformSpec:
    root = Path(clean_root).expanduser().resolve(strict=True)
    task_dirs = sorted(path for path in root.glob("task_*_state_train") if path.is_dir())
    if task_ids is not None:
        task_dirs = [path for path in task_dirs if _task_id(path) in task_ids]
    if max_tasks is not None:
        if max_tasks < 1:
            raise ValueError("max_tasks must be positive")
        task_dirs = task_dirs[:max_tasks]
    if not task_dirs:
        raise FileNotFoundError(f"No selected task_*_state_train datasets under {root}")

    specs = [ActionTransformSpec.load(path / "meta" / "action_transform_spec.json") for path in task_dirs]
    hashes = {spec.spec_sha256 for spec in specs}
    if len(hashes) != 1:
        details = ", ".join(f"{spec.path.parent.parent.name}:{spec.spec_sha256}" for spec in specs)
        raise ValueError(f"Selected datasets have different ActionTransformSpec hashes: {details}")
    return specs[0]


def discover_from_openpi_environment(environ: Mapping[str, str] | None = None) -> ActionTransformSpec:
    values = os.environ if environ is None else environ
    clean_root = values.get("AXIS_CLEAN_ROOT")
    if not clean_root:
        raise ValueError("AXIS_CLEAN_ROOT is required")
    raw_task_ids = values.get("OPENPI_AXIS_CLEAN_TASK_IDS")
    task_ids = (
        {int(item.strip()) for item in raw_task_ids.replace(";", ",").split(",") if item.strip()}
        if raw_task_ids
        else None
    )
    raw_max_tasks = values.get("OPENPI_AXIS_CLEAN_MAX_TASKS")
    max_tasks = int(raw_max_tasks) if raw_max_tasks else None
    spec = discover_action_transform_spec(clean_root, task_ids=task_ids, max_tasks=max_tasks)
    spec.require_environment_matches(values)
    return spec


__all__ = [
    "ACTION_TRANSFORM_SCHEMA",
    "ActionTransformSpec",
    "canonical_json_bytes",
    "discover_action_transform_spec",
    "discover_from_openpi_environment",
    "json_sha256",
    "make_action_transform_spec",
    "verify_action_transform_spec",
]
