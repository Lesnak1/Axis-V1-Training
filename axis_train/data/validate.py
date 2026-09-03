from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from axis_train.config import ensure_dir, write_json
from axis_train.data.lerobot_v3 import LeRobotV3Dataset, column_to_numpy, compute_array_stats


def _check(name: str, ok: bool, details: Any = None) -> dict[str, Any]:
    return {"name": name, "ok": bool(ok), "details": details}


def _decode_video_sample(path: Path) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(path))
    try:
        ok, frame = cap.read()
        if not ok or frame is None:
            return {"ok": False, "path": str(path), "reason": "could not read first frame"}
        return {
            "ok": True,
            "path": str(path),
            "shape": list(frame.shape),
            "dtype": str(frame.dtype),
            "mean": float(np.mean(frame)),
            "std": float(np.std(frame)),
        }
    finally:
        cap.release()


def validate_lerobot_dataset(
    dataset_path: str | Path,
    *,
    expected_action_dim: int,
    output_dir: str | Path,
    decode_video: bool = False,
) -> dict[str, Any]:
    dataset = LeRobotV3Dataset.from_path(dataset_path)
    output_dir = ensure_dir(output_dir)
    checks: list[dict[str, Any]] = []

    checks.append(_check("dataset_path_exists", dataset.path.exists(), str(dataset.path)))
    checks.append(_check("meta_info_exists", dataset.info_path.exists(), str(dataset.info_path)))
    info = dataset.load_info()
    features = info.get("features", {})
    required = ["observation.state", "action", "timestamp", "frame_index", "episode_index", "task_index"]
    for key in required:
        checks.append(_check(f"feature_or_column_present:{key}", key in features, features.get(key)))

    table = dataset.read_data_table()
    episodes = dataset.read_episodes_table()
    tasks = dataset.read_tasks()
    for key in required:
        checks.append(_check(f"data_column_present:{key}", key in table.column_names))

    actions = column_to_numpy(table, "action")
    states = column_to_numpy(table, "observation.state")
    episode_index = np.asarray(table["episode_index"].to_pylist(), dtype=np.int64)
    frame_index = np.asarray(table["frame_index"].to_pylist(), dtype=np.int64)
    timestamp = np.asarray(table["timestamp"].to_pylist(), dtype=np.float64)

    checks.extend(
        [
            _check("metadata_action_dim", features.get("action", {}).get("shape") == [expected_action_dim]),
            _check("data_action_dim", actions.ndim == 2 and actions.shape[1] == expected_action_dim, list(actions.shape)),
            _check("state_dim_fixed", states.ndim == 2, list(states.shape)),
            _check("action_finite", bool(np.isfinite(actions).all())),
            _check("state_finite", bool(np.isfinite(states).all())),
        ]
    )

    episode_lengths = episodes["length"].to_pylist() if "length" in episodes.column_names else []
    checks.append(_check("episode_lengths_sum_to_frames", int(sum(episode_lengths)) == int(table.num_rows)))
    checks.append(_check("total_frames_matches_info", int(info.get("total_frames", table.num_rows)) == int(table.num_rows)))
    checks.append(_check("total_episodes_matches_info", int(info.get("total_episodes", len(episode_lengths))) == len(episode_lengths)))
    checks.append(_check("task_metadata_non_empty", bool(tasks), tasks))

    monotonic_failures: list[int] = []
    frame_failures: list[int] = []
    for ep in sorted(set(episode_index.tolist())):
        mask = episode_index == ep
        frames = frame_index[mask]
        times = timestamp[mask]
        if len(frames) and not np.array_equal(frames, np.arange(len(frames))):
            frame_failures.append(int(ep))
        if len(times) and np.any(np.diff(times) < -1.0e-6):
            monotonic_failures.append(int(ep))
    checks.append(_check("frame_index_continuous_per_episode", not frame_failures, frame_failures[:20]))
    checks.append(_check("timestamp_monotonic_per_episode", not monotonic_failures, monotonic_failures[:20]))

    if expected_action_dim == 9:
        gripper = actions[:, :2]
        gripper_range = np.ptp(gripper, axis=0).tolist()
        checks.append(_check("raw_gripper_dynamic_range", bool(np.min(np.ptp(gripper, axis=0)) > 1.0e-3), gripper_range))
    elif expected_action_dim == 8:
        gripper = actions[:, 7]
        checks.append(_check("converted_gripper_in_unit_range", bool(gripper.min() >= -1.0e-6 and gripper.max() <= 1.0 + 1.0e-6), [float(gripper.min()), float(gripper.max())]))
        checks.append(_check("converted_gripper_dynamic_range", bool(float(np.ptp(gripper)) > 1.0e-3), float(np.ptp(gripper))))

    video_reports: list[dict[str, Any]] = []
    for video_key in dataset.video_keys():
        video_dir = dataset.videos_dir / video_key
        video_files = sorted(video_dir.glob("chunk-*/file-*.mp4")) if video_dir.exists() else []
        checks.append(_check(f"video_files_exist:{video_key}", bool(video_files), str(video_dir)))
        if decode_video and video_files:
            report = _decode_video_sample(video_files[0])
            video_reports.append(report)
            checks.append(_check(f"video_decodable:{video_key}", bool(report.get("ok")), report))

    summary = {
        "dataset_path": str(dataset.path.resolve()),
        "expected_action_dim": expected_action_dim,
        "num_frames": int(table.num_rows),
        "num_episodes": int(len(episode_lengths)),
        "num_tasks": int(len(tasks)),
        "action_stats": compute_array_stats(actions),
        "state_dim": int(states.shape[1]) if states.ndim == 2 else None,
        "video_samples": video_reports,
        "checks": checks,
        "ok": all(item["ok"] for item in checks),
    }
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "checks.json", checks)
    return summary
