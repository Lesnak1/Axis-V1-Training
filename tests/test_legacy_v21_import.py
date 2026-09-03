from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from axis_train.data.legacy_v21 import (
    import_legacy_axis_lerobot_v21,
    verify_action_transform_spec,
)
from axis_train.data.lerobot_v3 import LeRobotV3Dataset


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _write_legacy_fixture(root: Path, *, bad_padding: bool = False) -> None:
    joint_names = ["finger1", "finger2", *[f"joint{i}" for i in range(1, 8)]]
    features = {
        "observation.state": {"dtype": "float32", "shape": [9], "names": joint_names},
        "action": {"dtype": "float32", "shape": [9], "names": joint_names},
        "action_is_pad": {"dtype": "bool", "shape": [1]},
        "timestamp": {"dtype": "float32", "shape": [1]},
        "frame_index": {"dtype": "int64", "shape": [1]},
        "episode_index": {"dtype": "int64", "shape": [1]},
        "task_index": {"dtype": "int64", "shape": [1]},
        "index": {"dtype": "int64", "shape": [1]},
        "observation.images.base": {"dtype": "video", "shape": [8, 8, 3]},
        "observation.images.left_wrist": {"dtype": "video", "shape": [8, 8, 3]},
    }
    info = {
        "codebase_version": "v2.1",
        "fps": 5.0,
        "total_episodes": 2,
        "total_frames": 9,
        "features": features,
    }
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(json.dumps(info) + "\n", encoding="utf-8")
    _write_jsonl(
        root / "meta" / "episodes.jsonl",
        [
            {"episode_index": 0, "tasks": ["stack tofu"], "length": 5},
            {"episode_index": 1, "tasks": ["stack tofu"], "length": 4},
        ],
    )
    _write_jsonl(root / "meta" / "tasks.jsonl", [{"task_index": 0, "task": "stack tofu"}])

    global_index = 0
    for episode_index, length in enumerate((5, 4)):
        state = np.zeros((length, 9), dtype=np.float32)
        state[:, :2] = 0.04
        state[:, 2:] = np.arange(length, dtype=np.float32)[:, None] * 0.01
        action = state.copy()
        action[:-1, 2:] += 0.02
        action[-1] = action[-2]
        padding = np.zeros(length, dtype=bool)
        padding[-2 if bad_padding and episode_index == 0 else -1] = True
        table = pa.table(
            {
                "index": pa.array(np.arange(global_index, global_index + length), type=pa.int64()),
                "observation.state": pa.array(state.tolist(), type=pa.list_(pa.float32(), 9)),
                "action": pa.array(action.tolist(), type=pa.list_(pa.float32(), 9)),
                "action_is_pad": pa.array(padding),
                "timestamp": pa.array(np.arange(length, dtype=np.float32) / 5.0, type=pa.float32()),
                "frame_index": pa.array(np.arange(length), type=pa.int64()),
                "episode_index": pa.array(np.full(length, episode_index), type=pa.int64()),
                "task_index": pa.array(np.zeros(length), type=pa.int64()),
            }
        )
        data_path = root / f"data/chunk-000/episode_{episode_index:06d}.parquet"
        data_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, data_path)
        for key in ("observation.images.base", "observation.images.left_wrist"):
            video = root / f"videos/chunk-000/{key}/episode_{episode_index:06d}.mp4"
            video.parent.mkdir(parents=True, exist_ok=True)
            video.write_bytes(b"fixture")
        global_index += length


def test_import_legacy_v21_creates_v1_loader_view_and_hashed_spec(tmp_path: Path) -> None:
    source = tmp_path / "legacy"
    _write_legacy_fixture(source)

    result = import_legacy_axis_lerobot_v21(source, tmp_path / "output", task_id=3801)
    output = Path(result["dataset_path"])
    dataset = LeRobotV3Dataset.from_path(output)

    assert result["frame_count"] == 9
    assert result["real_action_count"] == 7
    assert result["velocity_scale"] == pytest.approx(0.2)
    assert dataset.read_data_table().num_rows == 9
    assert dataset.read_episodes_table()["dataset_to_index"].to_pylist() == [5, 9]
    assert dataset.read_tasks() == {0: "stack tofu"}
    assert (output / "data/chunk-000/file-000.parquet").is_symlink()
    assert (output / "videos/observation.images.wrist/chunk-000/file-001.mp4").is_symlink()

    spec = json.loads((output / "meta" / "action_transform_spec.json").read_text(encoding="utf-8"))
    assert verify_action_transform_spec(spec) == result["action_transform_spec_sha256"]
    assert spec["sampling"]["train_fps"] == 5.0
    assert spec["sampling"]["downsample_ratio"] == 1
    assert spec["sampling"]["action_horizon"] == 15
    assert spec["source"]["state_layout"] == ["finger1", "finger2", *[f"joint{i}" for i in range(1, 8)]]


def test_import_legacy_v21_rejects_nonterminal_padding(tmp_path: Path) -> None:
    source = tmp_path / "legacy"
    _write_legacy_fixture(source, bad_padding=True)

    with pytest.raises(ValueError, match="terminal action padding"):
        import_legacy_axis_lerobot_v21(source, tmp_path / "output", task_id=3801)


def test_import_legacy_v21_hash_binds_single_step_horizon(tmp_path: Path) -> None:
    source = tmp_path / "legacy"
    _write_legacy_fixture(source)

    result = import_legacy_axis_lerobot_v21(
        source,
        tmp_path / "output",
        task_id=3801,
        action_horizon=1,
    )
    output = Path(result["dataset_path"])
    spec = json.loads((output / "meta" / "action_transform_spec.json").read_text(encoding="utf-8"))

    assert result["action_horizon"] == 1
    assert spec["sampling"]["action_horizon"] == 1
    assert verify_action_transform_spec(spec) == result["action_transform_spec_sha256"]
