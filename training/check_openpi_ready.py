#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path


def resolve_openpi_root(raw: str | None) -> Path:
    candidates = []
    if raw:
        candidates.append(Path(raw))
    candidates.append(Path("third_party/openpi"))
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def find_config_names(openpi_root: Path) -> list[str]:
    config_py = openpi_root / "src" / "openpi" / "training" / "config.py"
    if not config_py.exists():
        return []
    text = config_py.read_text(encoding="utf-8")
    return sorted(set(re.findall(r'name="([^"]+)"', text)))


def main() -> int:
    parser = argparse.ArgumentParser(description="Check whether an OpenPI checkout is ready for Axis training hooks.")
    parser.add_argument("--openpi-root", default=None)
    parser.add_argument("--output", default="reports/openpi_readiness.json")
    args = parser.parse_args()

    root = resolve_openpi_root(args.openpi_root)
    config_names = find_config_names(root)
    report = {
        "openpi_root": str(root),
        "exists": root.exists(),
        "pyproject": (root / "pyproject.toml").exists(),
        "train_script": (root / "scripts" / "train.py").exists(),
        "compute_norm_stats_script": (root / "scripts" / "compute_norm_stats.py").exists(),
        "serve_policy_script": (root / "scripts" / "serve_policy.py").exists(),
        "uv_available": shutil.which("uv") is not None,
        "axis_related_configs": [name for name in config_names if "axis" in name or "roboverse" in name],
        "libero_related_configs": [name for name in config_names if "libero" in name],
        "notes": [
            "This check does not run Pi0.5 training.",
            "Apply openpi_patches/axis_train_openpi.patch before checking for AXIS-specific configs.",
            "AXIS LeRobot training uses pi05_axis_clean_local_droid_lora or pi05_axis_clean_local_droid_full after the patch is applied.",
            "LIBERO training/evaluation uses OpenPI's examples/libero workflow plus pi05_libero or pi05_libero_lora.",
        ],
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["exists"] and report["train_script"] else 1


if __name__ == "__main__":
    sys.exit(main())
