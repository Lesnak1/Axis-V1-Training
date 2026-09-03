from __future__ import annotations

import json
from pathlib import Path

import pytest
from axis_train.action_transform import (
    ActionTransformSpec,
    discover_action_transform_spec,
    make_action_transform_spec,
)


def _write_spec(root: Path, task_id: int, *, source_fps: float = 5.0) -> Path:
    value = make_action_transform_spec(
        source_fps=source_fps,
        source_joint_names=["finger1", "finger2", *[f"joint{i}" for i in range(1, 8)]],
        third_person_source="observation.images.base",
        wrist_source="observation.images.left_wrist",
    )
    path = root / f"task_{task_id:04d}_mujoco_state_train/meta/action_transform_spec.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")
    return path


def test_discovery_returns_the_only_selected_hashed_spec(tmp_path: Path) -> None:
    path = _write_spec(tmp_path, 3801)

    spec = discover_action_transform_spec(tmp_path, task_ids={3801})

    assert spec.path == path.resolve()
    assert spec.velocity_scale == pytest.approx(0.2)
    assert spec.downsample_ratio == 1
    assert spec.action_horizon == 15
    assert spec.clip_arm_action is None
    assert spec.openpi_environment()["AXIS_CLEAN_DROID_CLIP_VELOCITY_ACTIONS"] == "false"
    assert spec.openpi_environment()["AXIS_CLEAN_DROID_ACTION_HORIZON"] == "15"


def test_action_horizon_is_hash_bound_and_exported(tmp_path: Path) -> None:
    value = make_action_transform_spec(
        source_fps=5.0,
        source_joint_names=["finger1", "finger2", *[f"joint{i}" for i in range(1, 8)]],
        third_person_source="observation.images.base",
        wrist_source="observation.images.left_wrist",
        action_horizon=1,
    )
    path = tmp_path / "action_transform_spec.json"
    path.write_text(json.dumps(value), encoding="utf-8")

    spec = ActionTransformSpec.load(path)

    assert spec.action_horizon == 1
    assert spec.openpi_environment()["AXIS_CLEAN_DROID_ACTION_HORIZON"] == "1"
    default_value = json.loads(_write_spec(tmp_path / "default", 3801).read_text(encoding="utf-8"))
    assert spec.spec_sha256 != default_value["spec_sha256"]


def test_legacy_clipped_spec_remains_verifiable_and_hash_bound(tmp_path: Path) -> None:
    path = _write_spec(tmp_path, 3801)
    value = json.loads(path.read_text(encoding="utf-8"))
    value["forward"]["clip_arm_action"] = [-1.0, 1.0]
    value["inverse"]["arm_formula"] = "current_arm_qpos+clip(model_arm_action,-1,1)*velocity_scale"
    from axis_train.action_transform import json_sha256

    value["spec_sha256"] = json_sha256({key: item for key, item in value.items() if key != "spec_sha256"})
    path.write_text(json.dumps(value), encoding="utf-8")

    spec = ActionTransformSpec.load(path)

    assert spec.clip_arm_action == (-1.0, 1.0)
    assert spec.openpi_environment()["AXIS_CLEAN_DROID_CLIP_VELOCITY_ACTIONS"] == "true"


def test_discovery_rejects_mixed_transform_hashes(tmp_path: Path) -> None:
    _write_spec(tmp_path, 3801, source_fps=5.0)
    _write_spec(tmp_path, 3802, source_fps=15.0)

    with pytest.raises(ValueError, match="different ActionTransformSpec hashes"):
        discover_action_transform_spec(tmp_path)


def test_spec_rejects_a_valid_hash_with_inconsistent_timing(tmp_path: Path) -> None:
    path = _write_spec(tmp_path, 3801)
    value = json.loads(path.read_text(encoding="utf-8"))
    value["forward"]["velocity_scale"] = 1.0 / 15.0
    from axis_train.action_transform import json_sha256

    value["spec_sha256"] = json_sha256({key: item for key, item in value.items() if key != "spec_sha256"})
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="executed control period"):
        ActionTransformSpec.load(path)


def test_environment_override_must_match_spec(tmp_path: Path) -> None:
    path = _write_spec(tmp_path, 3801)
    spec = ActionTransformSpec.load(path)

    with pytest.raises(ValueError, match="conflicts with ActionTransformSpec"):
        spec.require_environment_matches({"AXIS_CLEAN_DROID_VELOCITY_SCALE": "0.0666666667"})

    with pytest.raises(ValueError, match="conflicts with ActionTransformSpec"):
        spec.require_environment_matches({"AXIS_CLEAN_DROID_ACTION_HORIZON": "1"})
