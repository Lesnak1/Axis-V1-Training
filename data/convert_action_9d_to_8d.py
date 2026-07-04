#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from axis_train.data.action_conversion import convert_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert rendered LeRobot raw 9D actions to canonical Pi0.5 8D.")
    parser.add_argument("--input-dataset", required=True)
    parser.add_argument("--output-dataset", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--copy-videos", action="store_true", help="Copy video files instead of symlinking them.")
    args = parser.parse_args()

    report = convert_dataset(
        args.input_dataset,
        args.output_dataset,
        args.config,
        overwrite=args.overwrite,
        symlink_videos=not args.copy_videos,
    )
    print(
        "converted "
        f"{report['num_frames']} frames: {report['input_action_dim']}D -> {report['output_action_dim']}D; "
        f"report={args.output_dataset}/meta/axis_train_action_conversion_report.json"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
