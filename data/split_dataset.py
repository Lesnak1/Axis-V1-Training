#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from axis_train.data.split import make_episode_split


def main() -> int:
    parser = argparse.ArgumentParser(description="Create an episode-level split manifest for LeRobot data.")
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--id-format", choices=["int", "episode"], default="int")
    args = parser.parse_args()

    split = make_episode_split(
        args.dataset_path,
        output_path=args.output,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        id_format=args.id_format,
    )
    print(f"wrote split manifest: {args.output}")
    print({k: len(v) for k, v in split.items() if isinstance(v, list)})
    return 0


if __name__ == "__main__":
    sys.exit(main())
