#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from axis_train.data.lerobot_v3 import LeRobotV3Dataset, column_to_numpy


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect the canonical batch shape before wiring OpenPI.")
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--num-frames", type=int, default=4)
    args = parser.parse_args()

    dataset = LeRobotV3Dataset.from_path(args.dataset_path)
    info = dataset.load_info()
    table = dataset.read_data_table()
    states = column_to_numpy(table, "observation.state")
    actions = column_to_numpy(table, "action")
    video_keys = dataset.video_keys()
    n = min(args.num_frames, table.num_rows)
    print(f"dataset={dataset.path}")
    print(f"fps={info.get('fps')} total_frames={table.num_rows} sample_frames={n}")
    print(f"state_shape={states[:n].shape} action_shape={actions[:n].shape}")
    print(f"video_keys={video_keys}")
    if actions.shape[1] != 8:
        print("ERROR: OpenPI canonical training batch expects action dim 8", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
