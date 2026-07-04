#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from axis_train.data.validate import validate_lerobot_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute a dataset summary report.")
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--expected-action-dim", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    validate_lerobot_dataset(
        args.dataset_path,
        expected_action_dim=args.expected_action_dim,
        output_dir=args.output,
        decode_video=False,
    )
    print(f"summary written to {args.output}/summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
