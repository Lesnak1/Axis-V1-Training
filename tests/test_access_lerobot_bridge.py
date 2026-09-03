from __future__ import annotations

import copy
import json
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
import pytest
from axis_replay_native.packed_state_codec import encode_packed_state_v1
from axis_replay_native.replay_qualified_lerobot import (
    CONTROL_SEMANTICS,
    CTRL_REPLAY_RECEIPT_SCHEMA,
    OBSERVATION_REPLAY_RECEIPT_SCHEMA,
    QUALIFIED_MANIFEST_SCHEMA,
    SCENE_SPLIT_SCHEMA,
    export_replay_qualified_lerobot,
    file_sha256,
    json_sha256,
    seal_json,
)
from axis_train.data.validate import validate_lerobot_dataset

JOINT_ORDER = [
    "finger_left",
    "finger_right",
    "arm_joint_1",
    "arm_joint_2",
    "arm_joint_3",
    "arm_joint_4",
    "arm_joint_5",
    "arm_joint_6",
    "arm_joint_7",
]
QPOS_INDICES = [8, 3, 11, 1, 6, 0, 10, 4, 7]
CTRL_INDICES = [9, 2, 7, 0, 10, 4, 1, 8, 5]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _ref(path: Path, *, relative_to: Path) -> dict[str, str]:
    return {"path": str(path.relative_to(relative_to)), "sha256": file_sha256(path)}


def _write_video(path: Path, *, frames: int, fps: float, offset: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (64, 48),
    )
    assert writer.isOpened()
    try:
        for index in range(frames):
            image = np.zeros((48, 64, 3), dtype=np.uint8)
            image[:, :, 0] = (index * 5 + offset) % 255
            image[:, :, 1] = (index * 3 + 2 * offset) % 255
            image[:, :, 2] = (index * 7 + 3 * offset) % 255
            writer.write(image)
    finally:
        writer.release()


def _build_evidence(root: Path, *, checker_passed: bool = True) -> dict[str, object]:
    root.mkdir(parents=True)
    task_id = 17
    attempt_id = 901
    scene_variant_index = 2
    frame_count = 40
    fps = 30.0

    qpos = np.zeros((frame_count, 12), dtype=np.float64)
    qvel = np.zeros((frame_count, 12), dtype=np.float64)
    ctrl = np.zeros((frame_count, 11), dtype=np.float64)
    frame = np.arange(frame_count, dtype=np.float64)
    qpos[:, QPOS_INDICES[0]] = np.linspace(0.04, 0.0, frame_count)
    qpos[:, QPOS_INDICES[1]] = np.linspace(0.038, 0.002, frame_count)
    ctrl[:, CTRL_INDICES[0]] = np.clip(qpos[:, QPOS_INDICES[0]] - 0.001, 0.0, 0.04)
    ctrl[:, CTRL_INDICES[1]] = np.clip(qpos[:, QPOS_INDICES[1]] - 0.001, 0.0, 0.04)
    for arm_index in range(7):
        output_index = arm_index + 2
        qpos[:, QPOS_INDICES[output_index]] = -0.6 + arm_index * 0.2 + frame * 0.001
        # Stay inside the OpenPI velocity action range so arm targets can be
        # reconstructed exactly during the action round-trip gate.
        ctrl[:, CTRL_INDICES[output_index]] = (
            qpos[:, QPOS_INDICES[output_index]]
            + 0.02 * np.sin(frame * 0.2 + arm_index)
        )

    packed = encode_packed_state_v1(
        qpos=qpos,
        qvel=qvel,
        ctrl=ctrl,
        timestep=0.002,
        steps_per_ctrl=5,
        task_id=task_id,
    )
    packed_path = root / "packed_state.json"
    _write_json(packed_path, packed)
    packed_sha = file_sha256(packed_path)

    joints = []
    # Deliberately write layout rows in actuator order, not training order.
    for actuator_id in range(ctrl.shape[1]):
        if actuator_id not in CTRL_INDICES:
            continue
        output_index = CTRL_INDICES.index(actuator_id)
        joints.append(
            {
                "name": JOINT_ORDER[output_index],
                "model_joint_name": f"robot/{JOINT_ORDER[output_index]}",
                "qpos_addr": QPOS_INDICES[output_index],
                "qvel_addr": QPOS_INDICES[output_index],
                "actuator_id": actuator_id,
                "actuator_name": f"actuator_{actuator_id}",
            }
        )
    layout_body = {
        "builder_version": "packed_state_layout_builder_v3",
        "render_scene_sha256": "a" * 64,
        "scene_xml_path": "/immutable/archive/scene.xml",
        "robot_name": "franka",
        "model_dims": {"nq": 12, "nv": 12, "nu": 11, "nbody": 1, "njnt": 9},
        "scene_entity_bindings": {"entities": {}},
        "address_map": {
            "source": "mujoco_model_addresses",
            "robot_joints": joints,
            "free_joints": [],
        },
        "robot_root_pose": {
            "source": "scene_mjcf",
            "body_id": 1,
            "body_name": "franka",
            "kinematics": None,
            "pos": [0.0, 0.0, 0.0],
            "rot": [1.0, 0.0, 0.0, 0.0],
        },
    }
    layout = {
        "schema_version": "packed_state_layout_v1",
        **layout_body,
        "layout_sha256": json_sha256(layout_body),
    }
    layout_path = root / "packed_layout.json"
    _write_json(layout_path, layout)

    selection = {
        "task_id": task_id,
        "selected_attempt_ids": [attempt_id],
        "selected": [
            {
                "attempt_id": attempt_id,
                "scene_variant_index": scene_variant_index,
                "selection_reason": "selected",
            }
        ],
    }
    selection_path = root / "candidate_attempts.json"
    _write_json(selection_path, selection)

    split = seal_json(
        {
            "schema_version": SCENE_SPLIT_SCHEMA,
            "task_id": task_id,
            "scene_assignments": [
                {"scene_variant_index": 0, "split": "val"},
                {"scene_variant_index": 1, "split": "test"},
                {"scene_variant_index": scene_variant_index, "split": "train"},
            ],
        },
        "split_sha256",
    )
    split_path = root / "scene_split.json"
    _write_json(split_path, split)

    ctrl_receipt = seal_json(
        {
            "schema_version": CTRL_REPLAY_RECEIPT_SCHEMA,
            "task_id": task_id,
            "attempt_id": attempt_id,
            "scene_variant_index": scene_variant_index,
            "packed_state_sha256": packed_sha,
            "packed_layout_sha256": layout["layout_sha256"],
            "replay_mode": "ctrl_only_mujoco",
            "control_semantics": CONTROL_SEMANTICS,
            "replay_success": True,
            "checker_passed": checker_passed,
            "qualified_boundary_count": frame_count,
            "executed_ctrl_count": frame_count - 1,
            "source_ctrl_start": 0,
            "source_ctrl_stop_exclusive": frame_count - 1,
            "checker_boundary_index": frame_count - 1,
            "terminal_packed_ctrl_row_consumed": False,
            "checker_details": [{"checker_type": "SyntheticGoalChecker", "passed": checker_passed}],
        },
        "receipt_sha256",
    )
    ctrl_receipt_path = root / "ctrl_replay_receipt.json"
    _write_json(ctrl_receipt_path, ctrl_receipt)

    front_path = root / "observations" / "front.mp4"
    wrist_path = root / "observations" / "wrist.mp4"
    _write_video(front_path, frames=frame_count, fps=fps, offset=11)
    _write_video(wrist_path, frames=frame_count, fps=fps, offset=29)
    observation_receipt = seal_json(
        {
            "schema_version": OBSERVATION_REPLAY_RECEIPT_SCHEMA,
            "task_id": task_id,
            "attempt_id": attempt_id,
            "scene_variant_index": scene_variant_index,
            "packed_state_sha256": packed_sha,
            "packed_layout_sha256": layout["layout_sha256"],
            "replay_mode": "recorded_state_boundary_restore",
            "replay_success": True,
            "source_boundary_start": 0,
            "boundary_count": frame_count,
            "fps": fps,
            "cameras": {
                "observation.images.third_person": {
                    "video": _ref(front_path, relative_to=root),
                    "frame_count": frame_count,
                    "height": 48,
                    "width": 64,
                },
                "observation.images.wrist": {
                    "video": _ref(wrist_path, relative_to=root),
                    "frame_count": frame_count,
                    "height": 48,
                    "width": 64,
                },
            },
        },
        "receipt_sha256",
    )
    observation_receipt_path = root / "observation_replay_receipt.json"
    _write_json(observation_receipt_path, observation_receipt)

    manifest = seal_json(
        {
            "schema_version": QUALIFIED_MANIFEST_SCHEMA,
            "task_id": task_id,
            "task": "place the synthetic object into the synthetic target",
            "simulator": "mujoco",
            "fps": fps,
            "split": "train",
            "source_selection": _ref(selection_path, relative_to=root),
            "split_manifest": _ref(split_path, relative_to=root),
            "model_order": {
                "gripper_joint_names": JOINT_ORDER[:2],
                "arm_joint_names": JOINT_ORDER[2:],
                "control_semantics": CONTROL_SEMANTICS,
            },
            "episodes": [
                {
                    "episode_index": 0,
                    "attempt_id": attempt_id,
                    "scene_variant_index": scene_variant_index,
                    "render_scene_sha256": layout["render_scene_sha256"],
                    "packed_state": _ref(packed_path, relative_to=root),
                    "packed_layout": _ref(layout_path, relative_to=root),
                    "ctrl_replay_receipt": _ref(ctrl_receipt_path, relative_to=root),
                    "observation_replay_receipt": _ref(
                        observation_receipt_path,
                        relative_to=root,
                    ),
                }
            ],
        },
        "manifest_sha256",
    )
    manifest_path = root / "qualification_manifest.json"
    _write_json(manifest_path, manifest)
    return {
        "manifest_path": manifest_path,
        "manifest": manifest,
        "ctrl_receipt_path": ctrl_receipt_path,
        "ctrl_receipt": ctrl_receipt,
        "qpos": qpos,
        "ctrl": ctrl,
    }


def _rebind_ctrl_receipt(evidence: dict[str, object], updated: dict) -> None:
    receipt_path = evidence["ctrl_receipt_path"]
    assert isinstance(receipt_path, Path)
    updated = seal_json(updated, "receipt_sha256")
    _write_json(receipt_path, updated)
    manifest = copy.deepcopy(evidence["manifest"])
    assert isinstance(manifest, dict)
    manifest["episodes"][0]["ctrl_replay_receipt"]["sha256"] = file_sha256(receipt_path)
    manifest = seal_json(manifest, "manifest_sha256")
    manifest_path = evidence["manifest_path"]
    assert isinstance(manifest_path, Path)
    _write_json(manifest_path, manifest)


def test_access_packed_state_exports_replay_qualified_lerobot(tmp_path: Path) -> None:
    evidence = _build_evidence(tmp_path / "evidence")
    result = export_replay_qualified_lerobot(
        evidence["manifest_path"],
        tmp_path / "datasets",
    )
    dataset = Path(result["dataset_path"])

    assert result["episode_count"] == 1
    assert result["frame_count"] == 40
    assert result["real_action_count"] == 39
    assert result["terminal_padding_rows"] == 1

    validation = validate_lerobot_dataset(
        dataset,
        expected_action_dim=9,
        output_dir=tmp_path / "validation",
        decode_video=True,
    )
    assert validation["ok"], validation["checks"]

    table = pq.read_table(dataset / "data" / "chunk-000" / "file-000.parquet")
    actual_states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
    actual_actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    qpos = evidence["qpos"]
    ctrl = evidence["ctrl"]
    assert isinstance(qpos, np.ndarray)
    assert isinstance(ctrl, np.ndarray)
    np.testing.assert_allclose(actual_states, qpos[:, QPOS_INDICES].astype(np.float32))
    np.testing.assert_allclose(actual_actions[:-1], ctrl[:-1, CTRL_INDICES].astype(np.float32))
    np.testing.assert_array_equal(actual_actions[-1], np.zeros(9, dtype=np.float32))
    assert not np.allclose(actual_actions[-2], ctrl[-1, CTRL_INDICES])

    provenance = json.loads((dataset / "meta" / "axis_provenance.json").read_text(encoding="utf-8"))
    assert provenance["transition_contract"]["terminal_packed_ctrl_row_used"] is False
    assert provenance["episodes"][0]["qpos_indices"] == QPOS_INDICES
    assert provenance["episodes"][0]["ctrl_indices"] == CTRL_INDICES


def test_bridge_rejects_checker_failure(tmp_path: Path) -> None:
    evidence = _build_evidence(tmp_path / "evidence", checker_passed=False)
    with pytest.raises(ValueError, match="not ctrl-replay.*Checker qualified"):
        export_replay_qualified_lerobot(evidence["manifest_path"], tmp_path / "datasets")


def test_bridge_rejects_terminal_packed_ctrl_consumption(tmp_path: Path) -> None:
    evidence = _build_evidence(tmp_path / "evidence")
    receipt = copy.deepcopy(evidence["ctrl_receipt"])
    assert isinstance(receipt, dict)
    receipt["executed_ctrl_count"] = 40
    receipt["source_ctrl_stop_exclusive"] = 40
    receipt["terminal_packed_ctrl_row_consumed"] = True
    _rebind_ctrl_receipt(evidence, receipt)

    with pytest.raises(ValueError, match="final packed ctrl row is not a transition action"):
        export_replay_qualified_lerobot(evidence["manifest_path"], tmp_path / "datasets")
