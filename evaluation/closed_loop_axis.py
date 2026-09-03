#!/usr/bin/env python3
"""Evaluate recorded AXIS controls or an OpenPI policy with one frozen controller."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")

INTEGRATION_ROOT = Path(__file__).resolve().parents[1]


def discover_learn_from_failure_root() -> Path:
    """Resolve the exact task runtime shared with Learn from Failure."""
    candidates: list[Path] = []
    configured = os.environ.get("AXIS_LEARN_FROM_FAILURE_ROOT")
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.extend(
        [
            INTEGRATION_ROOT.parent / "learn-from-failure-at-scale",
            INTEGRATION_ROOT.parents[1],
        ]
    )
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        replay_source = resolved / "scripts/advanced/replay-native/src/axis_replay_native"
        if replay_source.is_dir():
            return resolved
    searched = ", ".join(str(path) for path in seen)
    raise RuntimeError(
        "Learn from Failure task runtime was not found. Set "
        f"AXIS_LEARN_FROM_FAILURE_ROOT; searched: {searched}"
    )


LEARN_FROM_FAILURE_ROOT = discover_learn_from_failure_root()
REPLAY_SOURCE = LEARN_FROM_FAILURE_ROOT / "scripts/advanced/replay-native/src"
TASK_RUNTIME_SOURCE_FILES = (
    "axis_replay_native/asset_materializer.py",
    "axis_replay_native/asset_store.py",
    "axis_replay_native/backend_replay.py",
    "axis_replay_native/checker_payload_extractor.py",
    "axis_replay_native/checkers/CheckerManager.py",
    "axis_replay_native/mujoco_model_loader.py",
    "axis_replay_native/state_restore.py",
)
sys.path.insert(0, str(INTEGRATION_ROOT))
sys.path.insert(0, str(REPLAY_SOURCE))
sys.path.insert(0, str(LEARN_FROM_FAILURE_ROOT))

import imageio.v2 as imageio  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from axis_replay_native.asset_materializer import materialize_scene  # noqa: E402
from axis_replay_native.asset_store import LocalAssetStore  # noqa: E402
from axis_replay_native.backend_replay import coerce_initial_state_items  # noqa: E402
from axis_replay_native.checker_payload_extractor import (  # noqa: E402
    build_checker_payload,
    extract_initial_orientations,
    extract_initial_positions,
)
from axis_replay_native.checkers.CheckerManager import CheckerManager  # noqa: E402
from axis_replay_native.mujoco_model_loader import load_model  # noqa: E402
from axis_replay_native.state_restore import restore_scene_state  # noqa: E402
from axis_train.action_transform import ActionTransformSpec  # noqa: E402
from axis_train.policy import AxisDroidPolicyAdapter  # noqa: E402

WIDTH = 320
HEIGHT = 176
ACTION_DIM = 9
EXPECTED_JOINT_ORDER = (
    "franka/panda_finger_joint1",
    "franka/panda_finger_joint2",
    "franka/panda_joint1",
    "franka/panda_joint2",
    "franka/panda_joint3",
    "franka/panda_joint4",
    "franka/panda_joint5",
    "franka/panda_joint6",
    "franka/panda_joint7",
)


class EvaluationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise EvaluationError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def clear_applied_forces(data: mujoco.MjData) -> None:
    if data.qfrc_applied.size:
        data.qfrc_applied[:] = 0.0
    if data.xfrc_applied.size:
        data.xfrc_applied[:] = 0.0


def actuator_joint_mapping(model: mujoco.MjModel) -> dict[str, Any]:
    entries = []
    for actuator_index in range(model.nu):
        joint_id = int(model.actuator_trnid[actuator_index, 0])
        require(joint_id >= 0, f"Actuator {actuator_index} is not joint-backed")
        require(
            int(model.jnt_type[joint_id])
            in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)),
            f"Actuator {actuator_index} is not scalar joint-backed",
        )
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        actuator_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_index)
        require(bool(joint_name and actuator_name), "Actuator and joint names must be present")
        assert joint_name is not None
        assert actuator_name is not None
        entries.append(
            {
                "joint_name": joint_name.replace("__axis_scope__", "/"),
                "compiled_joint_name": joint_name,
                "actuator_name": actuator_name.replace("__axis_scope__", "/"),
                "compiled_actuator_name": actuator_name,
                "actuator_index": actuator_index,
                "qpos_address": int(model.jnt_qposadr[joint_id]),
            }
        )
    entries.sort(key=lambda item: item["joint_name"])
    joint_order = tuple(item["joint_name"] for item in entries)
    require(joint_order == EXPECTED_JOINT_ORDER, f"Unexpected joint order: {joint_order}")
    return {
        "joint_names": list(joint_order),
        "actuator_indices": [item["actuator_index"] for item in entries],
        "qpos_addresses": [item["qpos_address"] for item in entries],
        "entries": entries,
    }


def add_frozen_wrist_cameras(include_path: Path, camera_contract: dict[str, Any]) -> None:
    mount_suffix = str(camera_contract["mount_link_requested"])
    tree = ET.parse(include_path)
    bodies = [
        body
        for body in tree.getroot().iter("body")
        if str(body.attrib.get("name", "")).endswith(mount_suffix)
    ]
    require(len(bodies) == 1, f"Expected one camera mount ending in {mount_suffix!r}")
    body = bodies[0]
    mounts = camera_contract["wrist_mounts_opengl_convention"]
    fovy = float(camera_contract["wrist_intrinsics"]["vertical_fov_degrees"])
    expected_names = {f"{name}_custom" for name in mounts}
    existing = {camera.attrib.get("name") for camera in body.findall("camera")}
    present = expected_names.intersection(existing)
    require(not present or present == expected_names, "Materialized include has only one wrist camera")
    if not present:
        for name, values in mounts.items():
            ET.SubElement(
                body,
                "camera",
                {
                    "name": f"{name}_custom",
                    "mode": "fixed",
                    "pos": " ".join(str(item) for item in values["position"]),
                    "quat": " ".join(str(item) for item in values["quaternion_wxyz"]),
                    "fovy": str(fovy),
                },
            )
        tree.write(include_path, encoding="utf-8", xml_declaration=False)


def apply_and_verify_camera_contract(model: mujoco.MjModel, contract: dict[str, Any]) -> None:
    require(contract.get("render_resolution") == {"width": WIDTH, "height": HEIGHT}, "Bad render resolution")
    compiled_contract = contract["compiled_cameras"]
    base = compiled_contract["camera0"]
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "camera0")
    require(base_id >= 0, "Compiled model has no camera0")
    model.cam_pos[base_id] = np.asarray(base["local_position"], dtype=np.float64)
    model.cam_quat[base_id] = np.asarray(base["local_quaternion_wxyz"], dtype=np.float64)
    model.cam_fovy[base_id] = float(base["vertical_fov_degrees"])
    for contract_name, values in compiled_contract.items():
        internal_name = "camera0" if contract_name == "camera0" else f"{contract_name}_custom"
        camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, internal_name)
        require(camera_id >= 0, f"Compiled model has no camera {internal_name!r}")
        require(
            np.allclose(model.cam_pos[camera_id], values["local_position"], atol=1.0e-7, rtol=0.0),
            f"Camera position mismatch: {internal_name}",
        )
        require(
            np.allclose(model.cam_quat[camera_id], values["local_quaternion_wxyz"], atol=1.0e-7, rtol=0.0),
            f"Camera quaternion mismatch: {internal_name}",
        )


def checker_progress_metrics(details: list[dict[str, Any]]) -> dict[str, Any]:
    leaves: list[dict[str, Any]] = []
    nodes: list[dict[str, Any]] = []

    def visit(node: dict[str, Any]) -> None:
        nodes.append(node)
        children: list[dict[str, Any]] = []
        for key in ("sub_results", "steps"):
            values = node.get(key)
            if isinstance(values, list):
                children.extend(value for value in values if isinstance(value, dict))
        detail = node.get("detail")
        if isinstance(detail, dict):
            children.append(detail)
        if not children:
            leaves.append(node)
        for child in children:
            visit(child)

    for detail in details:
        if isinstance(detail, dict):
            visit(detail)
    xy = [float(node["xy_distance"]) for node in nodes if isinstance(node.get("xy_distance"), (int, float))]
    displacements = []
    for node in nodes:
        delta = node.get("delta")
        if isinstance(delta, list) and len(delta) == 3 and all(isinstance(value, (int, float)) for value in delta):
            displacements.append(float(np.linalg.norm(np.asarray(delta, dtype=np.float64))))
    contact_counts = [
        int(node["contact_count"])
        for node in nodes
        if isinstance(node.get("contact_count"), (int, float))
    ]
    return {
        "checker_leaf_passed": sum(bool(node.get("passed")) for node in leaves),
        "checker_leaf_total": len(leaves),
        "relative_xy_distance_min": min(xy) if xy else None,
        "relative_xy_distance_mean": float(np.mean(xy)) if xy else None,
        "body_displacement_l2_max": max(displacements) if displacements else None,
        "contact_count_max": max(contact_counts) if contact_counts else None,
    }


class FrozenSceneEnvironment:
    def __init__(
        self,
        *,
        scene: dict[str, Any],
        task_contract: Path,
        asset_dir: Path,
        work_root: Path,
        prompt: str,
        checker_config: dict[str, Any],
        camera_contract: dict[str, Any],
        initial_state_index: int,
        initial_state_count: int,
        initial_state_sha256: str,
        video_path: Path | None,
    ) -> None:
        self.scene = scene
        self.prompt = prompt
        self.boundary = 0
        self.total_mj_steps = 0
        self.clipped_by_actuator = np.zeros(ACTION_DIM, dtype=np.int64)
        self.work_dir = Path(tempfile.mkdtemp(prefix=f"scene-{int(scene['index']):03d}-", dir=work_root))
        scene_path = task_contract / "variants" / f"{int(scene['index']):03d}.json"
        require(scene_path.is_file(), f"Missing frozen scene: {scene_path}")
        require(sha256_file(scene_path) == scene["sha256"], f"Scene hash mismatch: {scene_path}")
        self.scene_payload = load_json(scene_path)
        require(self.scene_payload["name"] == scene["name"], "Scene name mismatch")
        initial_state_items = coerce_initial_state_items(self.scene_payload.get("initial_state"))
        require(initial_state_items, "Frozen evaluation scene has no concrete initial-state items")
        require(len(initial_state_items) == initial_state_count, "Frozen initial-state count changed")
        require(0 <= initial_state_index < len(initial_state_items), "Frozen initial-state index is out of range")
        self.initial_state_index = int(initial_state_index)
        self.initial_state_count = int(initial_state_count)
        self.initial_state = initial_state_items[self.initial_state_index]
        self.initial_state_sha256 = sha256_json(self.initial_state)
        require(self.initial_state_sha256 == initial_state_sha256, "Frozen initial-state hash changed")

        resolution = materialize_scene(
            self.scene_payload["mjcf_xml"],
            str(self.work_dir),
            LocalAssetStore(str(asset_dir)),
        )
        require(not resolution.missing_files, f"Missing assets: {resolution.missing_files}")
        add_frozen_wrist_cameras(self.work_dir / "includes/franka_panda.xml", camera_contract)
        self.model, self.data = load_model(str(self.work_dir), resolution.entry_xml_path)
        require((self.model.nq, self.model.nv, self.model.nu) == (37, 33, 9), "Bad model dimensions")
        require(abs(float(self.model.opt.timestep) - 0.001) <= 1.0e-12, "MuJoCo timestep must be 0.001 s")
        apply_and_verify_camera_contract(self.model, camera_contract)
        self.mapping = actuator_joint_mapping(self.model)
        self.actuator_indices = np.asarray(self.mapping["actuator_indices"], dtype=np.int64)
        self.qpos_addresses = np.asarray(self.mapping["qpos_addresses"], dtype=np.int64)
        restore_scene_state(self.model, self.data, self.initial_state, None, None)
        self.initial_positions = extract_initial_positions(self.model, self.data, None, self.initial_state)
        self.initial_orientations = extract_initial_orientations(self.model, self.data, None, self.initial_state)
        self.checker = CheckerManager(checker_config)
        require(self.checker.checker is not None, "Checker tree is absent")
        self.model.vis.global_.offwidth = max(self.model.vis.global_.offwidth, WIDTH)
        self.model.vis.global_.offheight = max(self.model.vis.global_.offheight, HEIGHT)
        self.renderer = mujoco.Renderer(self.model, height=HEIGHT, width=WIDTH)
        self.writer = None
        if video_path is not None:
            video_path.parent.mkdir(parents=True, exist_ok=True)
            self.writer = imageio.get_writer(
                video_path,
                fps=5,
                codec="libx264",
                pixelformat="yuv420p",
                macro_block_size=None,
                ffmpeg_log_level="error",
                quality=7,
            )

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
        self.renderer.close()
        shutil.rmtree(self.work_dir, ignore_errors=True)

    def qpos(self) -> np.ndarray:
        return np.asarray(self.data.qpos[self.qpos_addresses], dtype=np.float32)

    def render(self) -> dict[str, np.ndarray]:
        frames = {}
        for external_name, internal_name in (
            ("base", "camera0"),
            ("left_wrist", "axis_collect_left_wrist_camera_custom"),
        ):
            self.renderer.update_scene(self.data, camera=internal_name)
            frame = self.renderer.render().copy()
            require(frame.shape == (HEIGHT, WIDTH, 3) and frame.dtype == np.uint8, "Bad rendered frame")
            frames[external_name] = frame
        return frames

    def append_video(self) -> None:
        if self.writer is not None:
            frames = self.render()
            self.writer.append_data(np.concatenate((frames["base"], frames["left_wrist"]), axis=1))

    def openpi_observation(self, adapter: AxisDroidPolicyAdapter) -> dict[str, Any]:
        frames = self.render()
        return adapter.build_openpi_observation(
            qpos=self.qpos(),
            exterior_image=frames["base"],
            wrist_image=frames["left_wrist"],
            prompt=self.prompt,
        )

    def step(self, action: np.ndarray) -> None:
        action = np.asarray(action, dtype=np.float64)
        require(action.shape == (ACTION_DIM,), f"Bad action shape: {action.shape}")
        require(np.isfinite(action).all(), "Action contains non-finite values")
        lower = self.model.actuator_ctrlrange[self.actuator_indices, 0]
        upper = self.model.actuator_ctrlrange[self.actuator_indices, 1]
        limited = self.model.actuator_ctrllimited[self.actuator_indices].astype(bool)
        clipped = action.copy()
        clipped[limited] = np.clip(action[limited], lower[limited], upper[limited])
        self.clipped_by_actuator += clipped != action
        self.data.ctrl[self.actuator_indices] = clipped
        for _ in range(200):
            clear_applied_forces(self.data)
            mujoco.mj_step(self.model, self.data)
            self.total_mj_steps += 1
        self.boundary += 1

    def checker_result(self) -> tuple[bool, list[dict[str, Any]]]:
        payload = build_checker_payload(
            self.model,
            self.data,
            None,
            initial_state=self.initial_state,
            initial_positions=self.initial_positions,
            initial_orientations=self.initial_orientations,
            pressed_keys=set(),
        )
        passed, details = self.checker.check_with_details(payload)
        return bool(passed), details


class EvaluationPolicy(Protocol):
    output_semantics: str

    def reset(self) -> None: ...

    def infer(self, environment: FrozenSceneEnvironment) -> np.ndarray: ...


class RecordedActionPolicy:
    def __init__(self, actions: np.ndarray, *, output_semantics: str = "axis_absolute_qpos_target_9d") -> None:
        expected_dim = 9 if output_semantics.startswith("axis_") else 8
        require(
            actions.ndim == 2 and actions.shape[1] == expected_dim,
            f"Bad recorded action shape for {output_semantics}: {actions.shape}",
        )
        require(np.isfinite(actions).all(), "Recorded actions contain non-finite values")
        self.actions = np.asarray(actions, dtype=np.float32)
        self.output_semantics = output_semantics
        self.emitted = False

    def reset(self) -> None:
        self.emitted = False

    def infer(self, environment: FrozenSceneEnvironment) -> np.ndarray:
        del environment
        require(not self.emitted, "Recorded positive control exhausted before Checker success")
        self.emitted = True
        return self.actions.copy()


class WebsocketDroidPolicy:
    output_semantics = "pi05_droid_8d"

    def __init__(self, *, host: str, port: int, openpi_root: Path, adapter: AxisDroidPolicyAdapter) -> None:
        client_source = openpi_root / "packages/openpi-client/src"
        require(client_source.is_dir(), f"OpenPI client source is missing: {client_source}")
        sys.path.insert(0, str(client_source))
        from openpi_client.websocket_client_policy import WebsocketClientPolicy

        self.client = WebsocketClientPolicy(host=host, port=port)
        self.adapter = adapter

    def reset(self) -> None:
        self.client.reset()

    def infer(self, environment: FrozenSceneEnvironment) -> np.ndarray:
        response = self.client.infer(environment.openpi_observation(self.adapter))
        require("actions" in response, "OpenPI response has no actions")
        actions = np.asarray(response["actions"], dtype=np.float32)
        require(actions.ndim == 2 and actions.shape[1] == 8, f"Bad OpenPI action shape: {actions.shape}")
        require(np.isfinite(actions).all(), "OpenPI actions contain non-finite values")
        return actions


def run_episode(
    policy: EvaluationPolicy,
    environment: FrozenSceneEnvironment,
    *,
    adapter: AxisDroidPolicyAdapter | None,
    max_controls: int,
    replan_interval: int,
    samples_per_replan: int,
) -> dict[str, Any]:
    queue = np.empty((0, 9 if policy.output_semantics.startswith("axis_") else 8), dtype=np.float32)
    replan_boundaries: list[int] = []
    inference_seconds: list[float] = []
    raw_policy_actions: list[np.ndarray] = []
    measured_qpos_before_action: list[np.ndarray] = []
    executed_targets: list[np.ndarray] = []
    measured_qpos_after_action: list[np.ndarray] = []
    first_success = None
    last_checker_details: list[dict[str, Any]] = []
    checker_progress_history: list[dict[str, Any]] = []
    policy.reset()
    started = time.perf_counter()
    for boundary in range(max_controls + 1):
        passed, last_checker_details = environment.checker_result()
        checker_progress_history.append(checker_progress_metrics(last_checker_details))
        if passed:
            first_success = boundary
            break
        if boundary == max_controls:
            break
        environment.append_video()
        if len(queue) == 0:
            replan_boundaries.append(boundary)
            inference_started = time.perf_counter()
            predictions = [policy.infer(environment) for _ in range(samples_per_replan)]
            inference_seconds.append(time.perf_counter() - inference_started)
            shapes = {prediction.shape for prediction in predictions}
            require(len(shapes) == 1, f"Policy ensemble returned inconsistent shapes: {shapes}")
            predicted = np.mean(np.stack(predictions), axis=0, dtype=np.float32)
            queue = predicted[:replan_interval]
            require(len(queue) > 0, "Policy returned an empty action chunk")
        raw_action, queue = queue[0], queue[1:]
        qpos_before = environment.qpos()
        if policy.output_semantics == "pi05_droid_8d":
            require(adapter is not None, "DROID output requires an ActionTransformSpec adapter")
            target = adapter.droid_action_to_qpos_target(qpos_before, raw_action)
        else:
            target = raw_action
        raw_policy_actions.append(np.asarray(raw_action, dtype=np.float32).copy())
        measured_qpos_before_action.append(qpos_before.copy())
        executed_targets.append(np.asarray(target, dtype=np.float32).copy())
        environment.step(target)
        measured_qpos_after_action.append(environment.qpos())
    environment.append_video()
    targets = np.asarray(executed_targets, dtype=np.float32)
    target_delta = np.linalg.norm(np.diff(targets, axis=0), axis=1) if len(targets) > 1 else np.asarray([])
    xy_history = [
        value
        for metrics in checker_progress_history
        if (value := metrics["relative_xy_distance_min"]) is not None
    ]
    displacement_history = [
        value
        for metrics in checker_progress_history
        if (value := metrics["body_displacement_l2_max"]) is not None
    ]
    contact_history = [
        value
        for metrics in checker_progress_history
        if (value := metrics["contact_count_max"]) is not None
    ]
    return {
        "status": "checker_success" if first_success is not None else "checker_failed",
        "scene_variant": environment.scene,
        "initial_state_index": environment.initial_state_index,
        "initial_state_count": environment.initial_state_count,
        "initial_state_sha256": environment.initial_state_sha256,
        "controls_executed": environment.boundary,
        "total_mj_steps_executed": environment.total_mj_steps,
        "first_success_boundary": first_success,
        "policy_output_semantics": policy.output_semantics,
        "replan_interval": replan_interval,
        "samples_per_replan": samples_per_replan,
        "replan_boundaries": replan_boundaries,
        "inference_seconds": inference_seconds,
        "clipped_by_model_order_actuator": environment.clipped_by_actuator.tolist(),
        "target_step_l2_mean": float(np.mean(target_delta)) if len(target_delta) else None,
        "target_step_l2_max": float(np.max(target_delta)) if len(target_delta) else None,
        "action_trace": {
            "schema": "axis_closed_loop_action_trace_v1",
            "raw_policy_actions": np.asarray(raw_policy_actions, dtype=np.float32).tolist(),
            "measured_qpos_before_action": np.asarray(measured_qpos_before_action, dtype=np.float32).tolist(),
            "requested_targets": targets.tolist(),
            "measured_qpos_after_action": np.asarray(measured_qpos_after_action, dtype=np.float32).tolist(),
        },
        "checker_progress": checker_progress_metrics(last_checker_details),
        "checker_progress_best": {
            "relative_xy_distance_min": min(xy_history) if xy_history else None,
            "body_displacement_l2_max": max(displacement_history) if displacement_history else None,
            "contact_count_max": max(contact_history) if contact_history else None,
        },
        "final_model_order_state": environment.qpos().tolist(),
        "last_checker_details": last_checker_details,
        "episode_wall_seconds": time.perf_counter() - started,
    }


def resolve_scenes(args: argparse.Namespace, task_contract: Path) -> tuple[str, list[dict[str, Any]]]:
    if args.scene_index is not None:
        path = task_contract / "variants" / f"{args.scene_index:03d}.json"
        payload = load_json(path)
        return "explicit_scene", [{"index": args.scene_index, "name": payload["name"], "sha256": sha256_file(path)}]
    require(args.scene_split is not None and args.role is not None, "Provide --scene-index or --scene-split with --role")
    split = load_json(args.scene_split.resolve(strict=True))
    scenes = split[args.role]
    require(isinstance(scenes, list) and scenes, f"Empty scene role: {args.role}")
    if args.role == "final_test":
        require(args.allow_final_test, "final_test requires --allow-final-test")
    return args.role, scenes


def expand_frozen_episode_specs(
    task_contract: Path,
    scenes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    episodes: list[dict[str, Any]] = []
    for scene in scenes:
        scene_index = int(scene["index"])
        scene_path = task_contract / "variants" / f"{scene_index:03d}.json"
        require(scene_path.is_file(), f"Missing frozen scene: {scene_path}")
        require(sha256_file(scene_path) == scene["sha256"], f"Scene hash mismatch: {scene_path}")
        payload = load_json(scene_path)
        require(payload["name"] == scene["name"], "Scene name mismatch")
        items = coerce_initial_state_items(payload.get("initial_state"))
        require(items, f"Frozen scene {scene_index} has no concrete initial-state items")
        for initial_state_index, item in enumerate(items):
            episodes.append(
                {
                    "scene": scene,
                    "initial_state_index": initial_state_index,
                    "initial_state_count": len(items),
                    "initial_state_sha256": sha256_json(item),
                }
            )
    return episodes


def load_recorded_actions(path: Path) -> np.ndarray:
    table = pq.read_table(path.resolve(strict=True), columns=["action", "action_is_pad"])
    actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    padding = np.asarray(table["action_is_pad"].to_pylist(), dtype=bool)
    require(actions.ndim == 2 and actions.shape[1] == 9, f"Bad action matrix: {actions.shape}")
    require(np.flatnonzero(padding).tolist() == [len(actions) - 1], "Expected one terminal padding row")
    return actions[~padding]


def load_recorded_droid_actions(path: Path, adapter: AxisDroidPolicyAdapter) -> np.ndarray:
    table = pq.read_table(path.resolve(strict=True), columns=["observation.state", "action", "action_is_pad"])
    states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
    targets = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    padding = np.asarray(table["action_is_pad"].to_pylist(), dtype=bool)
    require(states.shape == targets.shape, f"Recorded state/action shape mismatch: {states.shape} != {targets.shape}")
    require(states.ndim == 2 and states.shape[1] == 9, f"Bad recorded state/action matrix: {states.shape}")
    require(np.flatnonzero(padding).tolist() == [len(states) - 1], "Expected one terminal padding row")
    return np.stack(
        [
            adapter.qpos_target_to_droid_action(state, target)
            for state, target in zip(states[~padding], targets[~padding], strict=True)
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=("recorded", "recorded_droid", "websocket"), required=True)
    parser.add_argument("--task-contract", type=Path, required=True)
    parser.add_argument("--asset-dir", type=Path, required=True)
    parser.add_argument("--camera-contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scene-index", type=int)
    parser.add_argument("--scene-split", type=Path)
    parser.add_argument("--role", choices=("development_evaluation", "final_test"))
    parser.add_argument("--allow-final-test", action="store_true")
    parser.add_argument("--max-controls", type=int, default=250)
    parser.add_argument("--replan-interval", type=int, default=15)
    parser.add_argument("--samples-per-replan", type=int, default=1)
    parser.add_argument("--recorded-actions", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--action-transform-spec", type=Path)
    parser.add_argument("--openpi-root", type=Path, default=Path("third_party/openpi"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-video", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        require(args.max_controls > 0, "max-controls must be positive")
        require(args.samples_per_replan > 0, "samples-per-replan must be positive")
        task_contract = args.task_contract.resolve(strict=True)
        asset_dir = args.asset_dir.resolve(strict=True)
        camera_contract_path = args.camera_contract.resolve(strict=True)
        camera_contract = load_json(camera_contract_path)
        role, scenes = resolve_scenes(args, task_contract)
        episode_specs = expand_frozen_episode_specs(task_contract, scenes)
        output = args.output_dir.resolve()
        require(not output.exists(), f"Refusing to overwrite {output}")
        output.mkdir(parents=True)
        work_root = output / ".scene_work"
        work_root.mkdir()
        task_fields = load_json(task_contract / "task_fields.json")
        checker_config = load_json(task_contract / "checker_config.json")
        prompt = str(task_fields["description"])

        adapter = None
        checkpoint = None
        if args.policy in ("recorded", "recorded_droid"):
            require(args.recorded_actions is not None, "recorded policy requires --recorded-actions")
            require(len(episode_specs) == 1, "recorded policy requires one explicit initial-state episode")
            recorded_path = args.recorded_actions.resolve(strict=True)
            if args.policy == "recorded_droid":
                require(args.action_transform_spec is not None, "recorded_droid requires --action-transform-spec")
                action_transform_spec = ActionTransformSpec.load(args.action_transform_spec)
                adapter = AxisDroidPolicyAdapter.from_action_transform_spec(action_transform_spec)
                policy = RecordedActionPolicy(
                    load_recorded_droid_actions(recorded_path, adapter),
                    output_semantics="pi05_droid_8d",
                )
            else:
                action_transform_spec = None
                policy = RecordedActionPolicy(load_recorded_actions(recorded_path))
            policy_evidence = {"recorded_actions": str(recorded_path), "sha256": sha256_file(recorded_path)}
            if action_transform_spec is not None:
                policy_evidence.update(
                    {
                        "action_transform_spec": str(action_transform_spec.path),
                        "action_transform_spec_sha256": action_transform_spec.spec_sha256,
                    }
                )
            require(args.replan_interval >= 250, "recorded positive control must queue the full expert trajectory")
            require(args.samples_per_replan == 1, "recorded positive control does not support action ensembles")
        else:
            require(args.checkpoint is not None, "websocket policy requires --checkpoint")
            checkpoint = args.checkpoint.resolve(strict=True)
            checkpoint_spec_path = checkpoint / "assets/local/axis_clean/action_transform_spec.json"
            checkpoint_spec = ActionTransformSpec.load(checkpoint_spec_path)
            if args.action_transform_spec is not None:
                external_spec = ActionTransformSpec.load(args.action_transform_spec)
                require(
                    external_spec.spec_sha256 == checkpoint_spec.spec_sha256,
                    "Checkpoint and requested ActionTransformSpec hashes differ",
                )
            adapter = AxisDroidPolicyAdapter.from_action_transform_spec(checkpoint_spec)
            require(1 <= args.replan_interval <= checkpoint_spec.action_horizon, "Bad Pi0.5 replan interval")
            policy = WebsocketDroidPolicy(
                host=args.host,
                port=args.port,
                openpi_root=args.openpi_root.resolve(strict=True),
                adapter=adapter,
            )
            policy_evidence = {
                "checkpoint": str(checkpoint),
                "action_transform_spec": str(checkpoint_spec.path),
                "action_transform_spec_sha256": checkpoint_spec.spec_sha256,
                "server_metadata": policy.client.get_server_metadata(),
            }

        episodes = []
        try:
            for episode_spec in episode_specs:
                scene = episode_spec["scene"]
                scene_index = int(scene["index"])
                initial_state_index = int(episode_spec["initial_state_index"])
                initial_state_count = int(episode_spec["initial_state_count"])
                episode_name = f"scene_{scene_index:03d}"
                if initial_state_count > 1:
                    episode_name += f"_initial_state_{initial_state_index:03d}"
                video = None if args.no_video else output / f"{episode_name}.mp4"
                environment = FrozenSceneEnvironment(
                    scene=scene,
                    task_contract=task_contract,
                    asset_dir=asset_dir,
                    work_root=work_root,
                    prompt=prompt,
                    checker_config=checker_config,
                    camera_contract=camera_contract,
                    initial_state_index=initial_state_index,
                    initial_state_count=initial_state_count,
                    initial_state_sha256=str(episode_spec["initial_state_sha256"]),
                    video_path=video,
                )
                try:
                    result = run_episode(
                        policy,
                        environment,
                        adapter=adapter,
                        max_controls=args.max_controls,
                        replan_interval=args.replan_interval,
                        samples_per_replan=args.samples_per_replan,
                    )
                finally:
                    environment.close()
                result["video"] = str(video) if video is not None else None
                (output / f"{episode_name}.json").write_text(
                    json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8",
                )
                episodes.append(result)
                print(
                    json.dumps(
                        {
                            "scene": scene_index,
                            "initial_state_index": initial_state_index,
                            "initial_state_count": initial_state_count,
                            "status": result["status"],
                            "controls": result["controls_executed"],
                        }
                    ),
                    flush=True,
                )
        finally:
            shutil.rmtree(work_root, ignore_errors=True)

        successes = sum(item["status"] == "checker_success" for item in episodes)
        total_controls = sum(int(item["controls_executed"]) for item in episodes)
        summary = {
            "schema": "axis_pi05_closed_loop_panel_v2",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "role": role,
            "task_runtime": {
                "learn_from_failure_root": str(LEARN_FROM_FAILURE_ROOT),
                "replay_source": str(REPLAY_SOURCE),
                "backend_replay_sha256": sha256_file(
                    REPLAY_SOURCE / "axis_replay_native/backend_replay.py"
                ),
                "checker_manager_sha256": sha256_file(
                    REPLAY_SOURCE / "axis_replay_native/checkers/CheckerManager.py"
                ),
                "source_sha256": {
                    relative: sha256_file(REPLAY_SOURCE / relative)
                    for relative in TASK_RUNTIME_SOURCE_FILES
                },
            },
            "panel_contract": {
                "scene_count": len(scenes),
                "episode_count": len(episode_specs),
                "initial_state_selection": "all_concrete_items_in_each_frozen_scene",
            },
            "policy_backend": args.policy,
            "policy_evidence": policy_evidence,
            "task_contract": str(task_contract),
            "checker_sha256": sha256_file(task_contract / "checker_config.json"),
            "camera_contract": {"path": str(camera_contract_path), "sha256": sha256_file(camera_contract_path)},
            "controller": {
                "mujoco_version": mujoco.__version__,
                "timestep_seconds": 0.001,
                "mj_steps_per_control": 200,
                "control_frequency_hz": 5.0,
                "max_controls": args.max_controls,
                "replan_interval": args.replan_interval,
                "samples_per_replan": args.samples_per_replan,
                "checker_every_boundary": True,
                "stop_on_first_success": True,
            },
            "successes": successes,
            "attempts": len(episodes),
            "success_rate": successes / len(episodes),
            "total_controls": total_controls,
            "episodes": episodes,
        }
        (output / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"successes": successes, "attempts": len(episodes), "output": str(output)}, indent=2))
        return 0
    except (EvaluationError, FileNotFoundError, KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
