#!/usr/bin/env python3
"""Measure OpenPI action error on exact, already-seen AXIS training observations."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import av
import numpy as np
import pyarrow.parquet as pq

INTEGRATION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(INTEGRATION_ROOT))

from axis_train.action_transform import ActionTransformSpec  # noqa: E402
from axis_train.policy import AxisDroidPolicyAdapter  # noqa: E402


class DiagnosticError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise DiagnosticError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def decode_video(path: Path) -> list[np.ndarray]:
    with av.open(str(path.resolve(strict=True))) as container:
        frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
    require(bool(frames), f"Video contains no frames: {path}")
    return frames


def quantiles(values: np.ndarray) -> dict[str, float]:
    points = np.quantile(values, [0.0, 0.25, 0.5, 0.75, 0.9, 1.0])
    return dict(zip(("min", "q25", "median", "q75", "q90", "max"), (float(value) for value in points), strict=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--openpi-root", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--samples-per-observation", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        require(args.episode_index >= 0, "episode-index must be nonnegative")
        require(args.samples_per_observation > 0, "samples-per-observation must be positive")
        dataset = args.dataset.resolve(strict=True)
        checkpoint = args.checkpoint.resolve(strict=True)
        output = args.output.resolve()
        require(not output.exists(), f"Refusing to overwrite {output}")

        dataset_spec = ActionTransformSpec.load(dataset / "meta/action_transform_spec.json")
        checkpoint_spec = ActionTransformSpec.load(
            checkpoint / "assets/local/axis_clean/action_transform_spec.json"
        )
        require(
            checkpoint_spec.spec_sha256 == dataset_spec.spec_sha256,
            "Checkpoint and dataset ActionTransformSpec hashes differ",
        )
        adapter = AxisDroidPolicyAdapter.from_action_transform_spec(checkpoint_spec)

        info = load_json(dataset / "meta/info.json")
        data_path = dataset / str(info["data_path"]).format(
            chunk_index=args.episode_index // int(info["chunks_size"]),
            file_index=args.episode_index,
        )
        table = pq.read_table(
            data_path.resolve(strict=True),
            columns=["observation.state", "action", "action_is_pad", "task_index"],
        )
        states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
        targets = np.asarray(table["action"].to_pylist(), dtype=np.float32)
        padding = np.asarray(table["action_is_pad"].to_pylist(), dtype=bool)
        task_indices = np.asarray(table["task_index"].to_pylist(), dtype=np.int64)
        require(states.shape == targets.shape and states.ndim == 2 and states.shape[1] == 9, "Bad state/action")
        require(np.flatnonzero(padding).tolist() == [len(states) - 1], "Expected one terminal padding row")
        require(np.all(task_indices == task_indices[0]), "One episode unexpectedly contains multiple task indices")

        tasks = pq.read_table(dataset / "meta/tasks.parquet").to_pylist()
        prompts = {int(row["task_index"]): str(row["task"]) for row in tasks}
        prompt = prompts[int(task_indices[0])]
        video_template = str(info["video_path"])

        def video_path(key: str) -> Path:
            return dataset / video_template.format(
                video_key=key,
                chunk_index=args.episode_index // int(info["chunks_size"]),
                file_index=args.episode_index,
            )

        base_path = video_path("observation.images.third_person")
        wrist_path = video_path("observation.images.wrist")
        base_frames = decode_video(base_path)
        wrist_frames = decode_video(wrist_path)
        require(len(base_frames) == len(states), "Third-person video/frame count mismatch")
        require(len(wrist_frames) == len(states), "Wrist video/frame count mismatch")

        labels = np.stack(
            [
                adapter.qpos_target_to_droid_action(state, target)
                for state, target in zip(states[~padding], targets[~padding], strict=True)
            ]
        )
        horizon = dataset_spec.action_horizon
        valid_bases = len(labels) - horizon + 1
        require(valid_bases > 0, "Episode is shorter than the action horizon")

        client_source = args.openpi_root.resolve(strict=True) / "packages/openpi-client/src"
        require(client_source.is_dir(), f"Missing OpenPI client source: {client_source}")
        sys.path.insert(0, str(client_source))
        from openpi_client.websocket_client_policy import WebsocketClientPolicy

        client = WebsocketClientPolicy(host=args.host, port=args.port)
        predicted_chunks = []
        for base in range(valid_bases):
            observation = adapter.build_openpi_observation(
                qpos=states[base],
                exterior_image=base_frames[base],
                wrist_image=wrist_frames[base],
                prompt=prompt,
            )
            samples = []
            for _ in range(args.samples_per_observation):
                response = client.infer(observation)
                actions = np.asarray(response["actions"], dtype=np.float32)
                require(actions.shape == (horizon, 8), f"Bad policy action shape: {actions.shape}")
                require(np.isfinite(actions).all(), "Policy actions contain non-finite values")
                samples.append(actions)
            predicted_chunks.append(np.mean(np.stack(samples), axis=0, dtype=np.float32))

        predictions = np.stack(predicted_chunks)
        label_chunks = np.stack([labels[base : base + horizon] for base in range(valid_bases)])
        chunk_abs_error = np.abs(predictions - label_chunks)
        first_abs_error = chunk_abs_error[:, 0]
        first_target_errors = []
        for base, predicted in enumerate(predictions[:, 0]):
            predicted_target = adapter.droid_action_to_qpos_target(states[base], predicted)
            first_target_errors.append(np.abs(predicted_target - targets[base]))
        first_target_errors_array = np.stack(first_target_errors)
        per_observation = []
        for base in range(valid_bases):
            predicted_target = adapter.droid_action_to_qpos_target(states[base], predictions[base, 0])
            per_observation.append(
                {
                    "index": base,
                    "predicted_first_action": predictions[base, 0].tolist(),
                    "label_first_action": labels[base].tolist(),
                    "first_action_mae": float(np.mean(first_abs_error[base])),
                    "first_action_mae_arm": float(np.mean(first_abs_error[base, :7])),
                    "first_action_abs_error_gripper": float(first_abs_error[base, 7]),
                    "full_chunk_mae": float(np.mean(chunk_abs_error[base])),
                    "predicted_absolute_target": predicted_target.tolist(),
                    "label_absolute_target": targets[base].tolist(),
                    "absolute_target_mae_9d": float(np.mean(first_target_errors_array[base])),
                }
            )

        receipt = {
            "schema": "axis_pi05_teacher_forced_actions_v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "dataset": str(dataset),
            "episode_index": args.episode_index,
            "data": {"path": str(data_path), "sha256": sha256_file(data_path.resolve(strict=True))},
            "videos": {
                "third_person": {"path": str(base_path), "sha256": sha256_file(base_path.resolve(strict=True))},
                "wrist": {"path": str(wrist_path), "sha256": sha256_file(wrist_path.resolve(strict=True))},
            },
            "checkpoint": str(checkpoint),
            "action_transform_spec_sha256": dataset_spec.spec_sha256,
            "server_metadata": client.get_server_metadata(),
            "prompt": prompt,
            "action_horizon": horizon,
            "valid_observations": valid_bases,
            "samples_per_observation": args.samples_per_observation,
            "action_trace": {
                "schema": "axis_teacher_forced_action_trace_v1",
                "per_observation": per_observation,
            },
            "metrics": {
                "first_action_mae": float(np.mean(first_abs_error)),
                "first_action_rmse": float(np.sqrt(np.mean(np.square(predictions[:, 0] - labels[:valid_bases])))),
                "first_action_mae_arm": float(np.mean(first_abs_error[:, :7])),
                "first_action_mae_gripper": float(np.mean(first_abs_error[:, 7])),
                "first_action_mae_per_dim": np.mean(first_abs_error, axis=0).tolist(),
                "first_action_per_observation_mae_quantiles": quantiles(np.mean(first_abs_error, axis=1)),
                "full_chunk_mae": float(np.mean(chunk_abs_error)),
                "full_chunk_rmse": float(np.sqrt(np.mean(np.square(predictions - label_chunks)))),
                "first_absolute_target_mae_9d": float(np.mean(first_target_errors_array)),
                "first_absolute_target_mae_per_dim": np.mean(first_target_errors_array, axis=0).tolist(),
                "predicted_action_min_per_dim": np.min(predictions, axis=(0, 1)).tolist(),
                "predicted_action_max_per_dim": np.max(predictions, axis=(0, 1)).tolist(),
                "label_action_min_per_dim": np.min(label_chunks, axis=(0, 1)).tolist(),
                "label_action_max_per_dim": np.max(label_chunks, axis=(0, 1)).tolist(),
            },
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"ok": True, "output": str(output), **receipt["metrics"]}, indent=2))
        return 0
    except (DiagnosticError, FileNotFoundError, KeyError, OSError, TypeError, ValueError) as error:
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
