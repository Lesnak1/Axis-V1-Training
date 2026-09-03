from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from axis_train.action_transform import (
    ACTION_TRANSFORM_SCHEMA,
    make_action_transform_spec,
    verify_action_transform_spec,
)
from axis_train.action_transform import (
    json_sha256 as _json_sha256,
)
from axis_train.data.lerobot_v3 import compute_array_stats

LEGACY_SCHEMA = "axis_dreamzero_multi_episode_lerobot_export_v1"
COMPATIBILITY_SCHEMA = "axis_lerobot_v21_v1_compatibility_view_v1"
def _file_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"Expected a JSON object at {path}:{line_number}")
            rows.append(value)
    return rows


def _link_file(source: Path, target: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(source.resolve())


def _fixed_width(table: pa.Table, column: str, width: int) -> np.ndarray:
    if column not in table.column_names:
        raise KeyError(f"Missing column {column!r}")
    values = np.asarray(table[column].to_pylist(), dtype=np.float32)
    if values.shape != (table.num_rows, width):
        raise ValueError(f"Expected {column} shape ({table.num_rows}, {width}), got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError(f"Column {column!r} contains non-finite values")
    return values


def import_legacy_axis_lerobot_v21(
    source_dataset: str | Path,
    output_root: str | Path,
    *,
    task_id: int,
    simulator: str = "mujoco",
    third_person_source: str = "observation.images.base",
    wrist_source: str = "observation.images.left_wrist",
    action_horizon: int = 15,
) -> dict[str, Any]:
    """Create an immutable, symlink-backed V1 loader view of an audited LeRobot-v2.1 dataset."""

    source = Path(source_dataset).expanduser().resolve(strict=True)
    output_root = Path(output_root).expanduser().resolve()
    if task_id < 1:
        raise ValueError("task_id must be positive")
    if simulator != "mujoco":
        raise ValueError("Only simulator='mujoco' is supported")
    if action_horizon < 1:
        raise ValueError("action_horizon must be positive")

    info = _load_json(source / "meta" / "info.json")
    if info.get("codebase_version") != "v2.1":
        raise ValueError(f"Expected codebase_version='v2.1', got {info.get('codebase_version')!r}")
    fps = float(info.get("fps", 0.0))
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"Invalid source fps: {fps}")
    features = info.get("features", {})
    for key in ("observation.state", "action"):
        if features.get(key, {}).get("shape") != [9]:
            raise ValueError(f"Expected source feature {key!r} to have shape [9]")
    for key in (third_person_source, wrist_source):
        if features.get(key, {}).get("dtype") != "video":
            raise ValueError(f"Expected source video feature {key!r}")

    legacy_episodes = sorted(_load_jsonl(source / "meta" / "episodes.jsonl"), key=lambda row: int(row["episode_index"]))
    legacy_tasks = sorted(_load_jsonl(source / "meta" / "tasks.jsonl"), key=lambda row: int(row["task_index"]))
    if not legacy_episodes or not legacy_tasks:
        raise ValueError("Legacy episode/task metadata is empty")
    episode_indices = [int(row["episode_index"]) for row in legacy_episodes]
    if episode_indices != list(range(len(legacy_episodes))):
        raise ValueError("Legacy episode indices must be contiguous and start at zero")

    output_path = output_root / f"task_{task_id:04d}_{simulator}_state_train"
    output_root.mkdir(parents=True, exist_ok=True)
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(f"Immutable output already exists: {output_path}")

    source_manifest_path = source / "provenance" / "dataset_manifest.json"
    source_manifest = _load_json(source_manifest_path) if source_manifest_path.is_file() else None
    if source_manifest is not None and source_manifest.get("schema") != LEGACY_SCHEMA:
        raise ValueError(f"Unexpected source provenance schema: {source_manifest.get('schema')!r}")
    source_files_by_path = {
        str(item["path"]): item
        for episode in (source_manifest or {}).get("episodes", [])
        for item in episode.get("files", [])
    }

    staging = Path(tempfile.mkdtemp(prefix=f".{output_path.name}.tmp-", dir=output_root))
    all_states: list[np.ndarray] = []
    all_real_actions: list[np.ndarray] = []
    episode_lengths: list[int] = []
    episode_from: list[int] = []
    episode_to: list[int] = []
    source_refs: list[dict[str, Any]] = []
    global_index = 0
    try:
        for episode_index, episode_meta in enumerate(legacy_episodes):
            length = int(episode_meta["length"])
            source_data = source / f"data/chunk-{episode_index // 1000:03d}/episode_{episode_index:06d}.parquet"
            table = pq.read_table(source_data)
            if table.num_rows != length:
                raise ValueError(
                    f"Episode {episode_index} length mismatch: metadata={length}, parquet={table.num_rows}"
                )
            required = {
                "index",
                "observation.state",
                "action",
                "action_is_pad",
                "timestamp",
                "frame_index",
                "episode_index",
                "task_index",
            }
            missing = sorted(required - set(table.column_names))
            if missing:
                raise KeyError(f"Episode {episode_index} is missing columns: {missing}")
            states = _fixed_width(table, "observation.state", 9)
            actions = _fixed_width(table, "action", 9)
            padding = np.asarray(table["action_is_pad"].to_pylist(), dtype=bool)
            if np.flatnonzero(padding).tolist() != [length - 1]:
                raise ValueError(f"Episode {episode_index} must have exactly one terminal action padding row")
            if length < 2 or not np.array_equal(actions[-1], actions[-2]):
                raise ValueError(f"Episode {episode_index} terminal padding must repeat its last real action")
            indices = np.asarray(table["index"].to_pylist(), dtype=np.int64)
            frames = np.asarray(table["frame_index"].to_pylist(), dtype=np.int64)
            episodes = np.asarray(table["episode_index"].to_pylist(), dtype=np.int64)
            timestamps = np.asarray(table["timestamp"].to_pylist(), dtype=np.float64)
            if not np.array_equal(indices, np.arange(global_index, global_index + length)):
                raise ValueError(f"Episode {episode_index} has a non-contiguous global index")
            if not np.array_equal(frames, np.arange(length)) or not np.all(episodes == episode_index):
                raise ValueError(f"Episode {episode_index} frame/episode indices are inconsistent")
            if not np.allclose(timestamps, np.arange(length) / fps, atol=1.0e-5, rtol=0.0):
                raise ValueError(f"Episode {episode_index} timestamps do not match fps={fps}")

            target_data = staging / f"data/chunk-000/file-{episode_index:03d}.parquet"
            _link_file(source_data, target_data)
            episode_source_refs: dict[str, Any] = {
                "episode_index": episode_index,
                "data": {"path": str(source_data), "sha256": None},
                "videos": {},
            }
            relative_data = str(source_data.relative_to(source))
            if relative_data in source_files_by_path:
                episode_source_refs["data"]["sha256"] = source_files_by_path[relative_data].get("sha256")

            for model_key, source_key in (
                ("observation.images.third_person", third_person_source),
                ("observation.images.wrist", wrist_source),
            ):
                source_video = source / (
                    f"videos/chunk-{episode_index // 1000:03d}/{source_key}/episode_{episode_index:06d}.mp4"
                )
                target_video = staging / f"videos/{model_key}/chunk-000/file-{episode_index:03d}.mp4"
                _link_file(source_video, target_video)
                relative_video = str(source_video.relative_to(source))
                episode_source_refs["videos"][model_key] = {
                    "source_key": source_key,
                    "path": str(source_video),
                    "sha256": source_files_by_path.get(relative_video, {}).get("sha256"),
                }

            episode_from.append(global_index)
            global_index += length
            episode_to.append(global_index)
            episode_lengths.append(length)
            all_states.append(states)
            all_real_actions.append(actions[:-1])
            source_refs.append(episode_source_refs)

        prompt_by_index = {int(row["task_index"]): str(row["task"]) for row in legacy_tasks}
        tasks_table = pa.table(
            {
                "task_index": pa.array(list(prompt_by_index), type=pa.int64()),
                "task": pa.array(list(prompt_by_index.values()), type=pa.string()),
            }
        )
        (staging / "meta").mkdir(parents=True, exist_ok=True)
        pq.write_table(tasks_table, staging / "meta" / "tasks.parquet", compression="zstd")

        episode_columns: dict[str, pa.Array] = {
            "episode_index": pa.array(episode_indices, type=pa.int64()),
            "length": pa.array(episode_lengths, type=pa.int64()),
            "dataset_from_index": pa.array(episode_from, type=pa.int64()),
            "dataset_to_index": pa.array(episode_to, type=pa.int64()),
            "tasks": pa.array([row.get("tasks", []) for row in legacy_episodes], type=pa.list_(pa.string())),
        }
        for model_key in ("observation.images.third_person", "observation.images.wrist"):
            prefix = f"videos/{model_key}"
            episode_columns[f"{prefix}/chunk_index"] = pa.array(np.zeros(len(episode_indices)), type=pa.int64())
            episode_columns[f"{prefix}/file_index"] = pa.array(episode_indices, type=pa.int64())
            episode_columns[f"{prefix}/from_timestamp"] = pa.array(np.zeros(len(episode_indices)), type=pa.float32())
        episodes_path = staging / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
        episodes_path.parent.mkdir(parents=True)
        pq.write_table(pa.table(episode_columns), episodes_path, compression="zstd")

        joint_names = list(features["observation.state"].get("names") or [])
        if len(joint_names) != 9:
            raise ValueError("Source observation.state must declare nine ordered joint names")
        video_shape = list(features[third_person_source]["shape"])
        wrist_shape = list(features[wrist_source]["shape"])
        output_features = {
            "observation.state": {**features["observation.state"], "shape": [9]},
            "action": {**features["action"], "shape": [9]},
            "timestamp": {"dtype": "float32", "shape": [1]},
            "frame_index": {"dtype": "int64", "shape": [1]},
            "episode_index": {"dtype": "int64", "shape": [1]},
            "task_index": {"dtype": "int64", "shape": [1]},
            "observation.images.third_person": {
                **features[third_person_source],
                "shape": video_shape,
            },
            "observation.images.wrist": {**features[wrist_source], "shape": wrist_shape},
        }
        action_transform_spec = make_action_transform_spec(
            source_fps=fps,
            source_joint_names=joint_names,
            third_person_source=third_person_source,
            wrist_source=wrist_source,
            action_horizon=action_horizon,
        )
        verify_action_transform_spec(action_transform_spec)
        _write_json(staging / "meta" / "action_transform_spec.json", action_transform_spec)

        transition_contract = {
            "observation_rows": "recorded_state_boundaries",
            "action_rows": "outgoing_ctrl_with_terminal_padding",
            "terminal_action_padding": True,
            "terminal_padding_value": "repeat_last_real_action",
            "terminal_packed_ctrl_row_used": False,
        }
        output_info = {
            "codebase_version": "v3.0-compat",
            "robot_type": "axis_franka_9d",
            "fps": fps,
            "total_episodes": len(episode_indices),
            "total_frames": global_index,
            "total_tasks": len(prompt_by_index),
            "chunks_size": 1000,
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
            "features": output_features,
            "axis_transition_contract": transition_contract,
            "axis_compatibility_view": {
                "schema_version": COMPATIBILITY_SCHEMA,
                "source_dataset": str(source),
                "action_transform_spec_sha256": action_transform_spec["spec_sha256"],
            },
        }
        _write_json(staging / "meta" / "info.json", output_info)
        _write_json(
            staging / "meta" / "stats.json",
            {
                "observation.state": compute_array_stats(np.concatenate(all_states)),
                "action": compute_array_stats(np.concatenate(all_real_actions)),
            },
        )

        source_info_path = source / "meta" / "info.json"
        provenance_body = {
            "schema_version": COMPATIBILITY_SCHEMA,
            "task_id": task_id,
            "simulator": simulator,
            "source_dataset": str(source),
            "source_info": {"path": str(source_info_path), "sha256": _file_sha256(source_info_path)},
            "source_dataset_manifest": (
                {"path": str(source_manifest_path), "sha256": _file_sha256(source_manifest_path)}
                if source_manifest_path.is_file()
                else None
            ),
            "model_order": {
                "joint_names": joint_names,
                "gripper_joint_names": joint_names[:2],
                "arm_joint_names": joint_names[2:],
            },
            "camera_mapping": {
                "observation.images.third_person": third_person_source,
                "observation.images.wrist": wrist_source,
            },
            "transition_contract": transition_contract,
            "action_transform_spec_sha256": action_transform_spec["spec_sha256"],
            "episodes": source_refs,
        }
        provenance = {**provenance_body, "provenance_sha256": _json_sha256(provenance_body)}
        _write_json(staging / "meta" / "axis_provenance.json", provenance)

        staging.rename(output_path)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise

    return {
        "ok": True,
        "dataset_path": str(output_path),
        "task_id": task_id,
        "episode_count": len(episode_indices),
        "frame_count": global_index,
        "real_action_count": int(sum(length - 1 for length in episode_lengths)),
        "terminal_padding_rows": len(episode_indices),
        "source_fps": fps,
        "train_fps": fps,
        "downsample_ratio": 1,
        "velocity_scale": 1.0 / fps,
        "action_horizon": action_horizon,
        "action_transform_spec_sha256": action_transform_spec["spec_sha256"],
    }


__all__ = [
    "ACTION_TRANSFORM_SCHEMA",
    "COMPATIBILITY_SCHEMA",
    "import_legacy_axis_lerobot_v21",
    "make_action_transform_spec",
    "verify_action_transform_spec",
]
