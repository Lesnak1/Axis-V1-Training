#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from axis_train.data.validate import validate_lerobot_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a rendered LeRobot v3 dataset.")
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--expected-action-dim", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--decode-video", action="store_true")
    parser.add_argument("--no-strict", action="store_true", help="Write reports but do not fail on validation errors.")
    args = parser.parse_args()

    summary = validate_lerobot_dataset(
        args.dataset_path,
        expected_action_dim=args.expected_action_dim,
        output_dir=args.output,
        decode_video=args.decode_video,
    )
    print(f"ok={summary['ok']} frames={summary['num_frames']} episodes={summary['num_episodes']} report={args.output}/summary.json")
    if not summary["ok"] and not args.no_strict:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
