#!/usr/bin/env python3
"""Compare a frozen task reset/render with the corresponding training observation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import av
import numpy as np
import pyarrow.parquet as pq

from evaluation.closed_loop_axis import (
    LEARN_FROM_FAILURE_ROOT,
    REPLAY_SOURCE,
    TASK_RUNTIME_SOURCE_FILES,
    FrozenSceneEnvironment,
    load_json,
    sha256_file,
    sha256_json,
)
from axis_replay_native.backend_replay import coerce_initial_state_items


class ParityError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ParityError(message)


def decode_first_frame(path: Path) -> np.ndarray:
    with av.open(str(path.resolve(strict=True))) as container:
        iterator = container.decode(video=0)
        try:
            return next(iterator).to_ndarray(format="rgb24")
        except StopIteration as error:
            raise ParityError(f"Video contains no frames: {path}") from error


def image_metrics(rendered: np.ndarray, recorded: np.ndarray) -> dict[str, Any]:
    require(rendered.shape == recorded.shape, f"Image shape mismatch: {rendered.shape} != {recorded.shape}")
    require(rendered.dtype == recorded.dtype == np.uint8, "Parity images must be uint8 RGB")
    difference = np.abs(rendered.astype(np.int16) - recorded.astype(np.int16))
    return {
        "shape": list(rendered.shape),
        "mae_u8": float(np.mean(difference)),
        "rmse_u8": float(np.sqrt(np.mean(np.square(difference.astype(np.float32))))),
        "p99_abs_u8": float(np.quantile(difference, 0.99)),
        "max_abs_u8": int(np.max(difference)),
        "exact_pixel_channel_fraction": float(np.mean(difference == 0)),
        "rendered_rgb_sha256": hashlib.sha256(rendered.tobytes()).hexdigest(),
        "recorded_rgb_sha256": hashlib.sha256(recorded.tobytes()).hexdigest(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--task-contract", type=Path, required=True)
    parser.add_argument("--asset-dir", type=Path, required=True)
    parser.add_argument("--camera-contract", type=Path, required=True)
    parser.add_argument("--scene-index", type=int, required=True)
    parser.add_argument("--initial-state-index", type=int, default=0)
    parser.add_argument("--max-state-abs-error", type=float, default=1.0e-6)
    parser.add_argument("--max-camera-mae-u8", type=float, default=2.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    require(args.episode_index >= 0, "episode-index must be nonnegative")
    require(args.initial_state_index >= 0, "initial-state-index must be nonnegative")
    require(args.max_state_abs_error >= 0, "max-state-abs-error must be nonnegative")
    require(args.max_camera_mae_u8 >= 0, "max-camera-mae-u8 must be nonnegative")

    dataset = args.dataset.resolve(strict=True)
    task_contract = args.task_contract.resolve(strict=True)
    asset_dir = args.asset_dir.resolve(strict=True)
    camera_contract_path = args.camera_contract.resolve(strict=True)
    output = args.output.resolve()
    require(not output.exists(), f"Refusing to overwrite {output}")

    scene_path = task_contract / "variants" / f"{args.scene_index:03d}.json"
    scene_payload = load_json(scene_path.resolve(strict=True))
    initial_states = coerce_initial_state_items(scene_payload.get("initial_state"))
    require(initial_states, "Frozen scene contains no concrete initial state")
    require(args.initial_state_index < len(initial_states), "initial-state-index is out of range")
    scene = {
        "index": args.scene_index,
        "name": scene_payload["name"],
        "sha256": sha256_file(scene_path),
    }

    info = load_json(dataset / "meta/info.json")
    chunks_size = int(info["chunks_size"])
    data_path = dataset / str(info["data_path"]).format(
        chunk_index=args.episode_index // chunks_size,
        file_index=args.episode_index,
    )
    table = pq.read_table(data_path.resolve(strict=True), columns=["observation.state"])
    require(len(table) > 0, f"Episode contains no observations: {data_path}")
    recorded_state = np.asarray(table["observation.state"][0].as_py(), dtype=np.float32)
    require(recorded_state.shape == (9,), f"Expected a 9D recorded state, got {recorded_state.shape}")

    video_template = str(info["video_path"])
    video_paths = {
        "third_person": dataset
        / video_template.format(
            video_key="observation.images.third_person",
            chunk_index=args.episode_index // chunks_size,
            file_index=args.episode_index,
        ),
        "wrist": dataset
        / video_template.format(
            video_key="observation.images.wrist",
            chunk_index=args.episode_index // chunks_size,
            file_index=args.episode_index,
        ),
    }
    recorded_images = {name: decode_first_frame(path) for name, path in video_paths.items()}

    temporary_parent = Path(os.environ.get("AXIS_EVAL_TMPDIR", "/dev/shm"))
    require(temporary_parent.is_dir(), f"AXIS_EVAL_TMPDIR is not a directory: {temporary_parent}")
    with tempfile.TemporaryDirectory(prefix="axis-observation-parity-", dir=temporary_parent) as temporary:
        environment = FrozenSceneEnvironment(
            scene=scene,
            task_contract=task_contract,
            asset_dir=asset_dir,
            work_root=Path(temporary),
            prompt=str(load_json(task_contract / "task_fields.json")["description"]),
            checker_config=load_json(task_contract / "checker_config.json"),
            camera_contract=load_json(camera_contract_path),
            initial_state_index=args.initial_state_index,
            initial_state_count=len(initial_states),
            initial_state_sha256=sha256_json(initial_states[args.initial_state_index]),
            video_path=None,
        )
        try:
            rendered_state = environment.qpos()
            rendered_images = environment.render()
        finally:
            environment.close()

    state_difference = np.abs(rendered_state - recorded_state)
    camera_metrics = {
        "third_person": image_metrics(rendered_images["base"], recorded_images["third_person"]),
        "wrist": image_metrics(rendered_images["left_wrist"], recorded_images["wrist"]),
    }
    state_passed = bool(float(np.max(state_difference)) <= args.max_state_abs_error)
    cameras_passed = all(
        float(metrics["mae_u8"]) <= args.max_camera_mae_u8 for metrics in camera_metrics.values()
    )
    passed = state_passed and cameras_passed
    receipt = {
        "schema": "axis_training_evaluation_observation_parity_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "thresholds": {
            "max_state_abs_error": args.max_state_abs_error,
            "max_camera_mae_u8": args.max_camera_mae_u8,
        },
        "task_runtime": {
            "learn_from_failure_root": str(LEARN_FROM_FAILURE_ROOT),
            "replay_source": str(REPLAY_SOURCE),
            "source_sha256": {
                relative: sha256_file(REPLAY_SOURCE / relative)
                for relative in TASK_RUNTIME_SOURCE_FILES
            },
        },
        "task_contract": str(task_contract),
        "scene": scene,
        "initial_state_index": args.initial_state_index,
        "camera_contract": {
            "path": str(camera_contract_path),
            "sha256": sha256_file(camera_contract_path),
        },
        "dataset": str(dataset),
        "episode_index": args.episode_index,
        "data": {"path": str(data_path.resolve()), "sha256": sha256_file(data_path.resolve())},
        "state": {
            "passed": state_passed,
            "rendered": rendered_state.tolist(),
            "recorded": recorded_state.tolist(),
            "max_abs_error": float(np.max(state_difference)),
            "mean_abs_error": float(np.mean(state_difference)),
        },
        "cameras": {
            name: {
                "passed": float(camera_metrics[name]["mae_u8"]) <= args.max_camera_mae_u8,
                "recorded_video": {
                    "path": str(video_paths[name].resolve(strict=True)),
                    "sha256": sha256_file(video_paths[name].resolve(strict=True)),
                },
                **camera_metrics[name],
            }
            for name in camera_metrics
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"passed": passed, "output": str(output), "state": receipt["state"], "cameras": camera_metrics}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
