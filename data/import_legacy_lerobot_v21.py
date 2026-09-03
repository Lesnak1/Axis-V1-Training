#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from axis_train.data.legacy_v21 import import_legacy_axis_lerobot_v21


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create an immutable V1-loader compatibility view of an audited Axis LeRobot-v2.1 dataset."
    )
    parser.add_argument("--source-dataset", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--simulator", default="mujoco", choices=("mujoco",))
    parser.add_argument("--third-person-source", default="observation.images.base")
    parser.add_argument("--wrist-source", default="observation.images.left_wrist")
    parser.add_argument("--action-horizon", type=int, default=15)
    args = parser.parse_args()
    result = import_legacy_axis_lerobot_v21(
        args.source_dataset,
        args.output_root,
        task_id=args.task_id,
        simulator=args.simulator,
        third_person_source=args.third_person_source,
        wrist_source=args.wrist_source,
        action_horizon=args.action_horizon,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
