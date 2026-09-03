from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

from axis_train.action_transform import make_action_transform_spec
from axis_train.axis_environment import AxisDatasetEnvironment
from axis_train.cli import build_parser


ROOT = Path(__file__).resolve().parents[1]


def _write_spec(dataset: Path, *, task_horizon: int = 1) -> dict[str, object]:
    meta = dataset / "meta"
    meta.mkdir(parents=True)
    payload = make_action_transform_spec(
        source_fps=5.0,
        source_joint_names=[
            "left_finger",
            "right_finger",
            *[f"arm_joint_{index}" for index in range(1, 8)],
        ],
        third_person_source="observation.images.base",
        wrist_source="observation.images.left_wrist",
        action_horizon=task_horizon,
    )
    (meta / "action_transform_spec.json").write_text(
        json.dumps(payload) + "\n", encoding="utf-8"
    )
    return payload


def _fake_openpi(tmp_path: Path) -> Path:
    root = tmp_path / "openpi"
    scripts = root / "scripts"
    scripts.mkdir(parents=True)
    for name in ("compute_norm_stats.py", "train.py", "serve_policy.py"):
        (scripts / name).touch()
    return root


def _clean_environment() -> dict[str, str]:
    env = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith("AXIS_") and not name.startswith("OPENPI_")
    }
    env["AXIS_CLI_PYTHON"] = sys.executable
    env["OPENPI_PYTHON"] = sys.executable
    return env


def test_finetune_binds_axis_task_then_runs_the_standard_openpi_stages(
    tmp_path: Path,
) -> None:
    openpi = _fake_openpi(tmp_path)
    clean_root = tmp_path / "axis-clean"
    dataset = clean_root / "task_03801_mujoco_state_train"
    spec = _write_spec(dataset)

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/axis.py"),
            "finetune",
            "3801",
            str(clean_root),
            "task3801_smoke",
            "--config",
            "pi05_axis_clean_local_droid_action_expert",
            "--compute-norm-stats",
            "--openpi-root",
            str(openpi),
            "--dry-run",
            "--num-train-steps",
            "12",
            "--batch-size",
            "1",
        ],
        env=_clean_environment(),
        check=True,
        text=True,
        capture_output=True,
    )
    lines = completed.stdout.splitlines()
    assert lines[0].startswith("[axis-finetune] task=3801 ")
    assert f"spec={spec['spec_sha256']}" in lines[0]
    assert lines[1].startswith("[openpi:norm-stats] ")
    assert lines[2].startswith("[openpi:train] ")

    train_command = shlex.split(lines[2].removeprefix("[openpi:train] "))
    assert train_command[:2] == [sys.executable, "scripts/train.py"]
    assert "pi05_axis_clean_local_droid_action_expert" in train_command
    assert train_command[train_command.index("--exp-name") + 1] == "task3801_smoke"
    assert "--overwrite" in train_command
    assert train_command[train_command.index("--num-train-steps") + 1] == "12"
    assert train_command[train_command.index("--batch-size") + 1] == "1"
    assert train_command[train_command.index("--checkpoint-base-dir") + 1] == str(
        clean_root / "checkpoints"
    )


def test_axis_dataset_adapter_owns_only_environment_binding(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    openpi = _fake_openpi(project)
    clean_root = tmp_path / "axis-clean"
    dataset = clean_root / "task_0007_isaac_state_train"
    spec = _write_spec(dataset, task_horizon=15)

    adapter = AxisDatasetEnvironment.for_task(
        7,
        clean_root,
        openpi_root=openpi,
        project_root=project,
        environ={},
    )
    bound = adapter.bind(
        {},
        config_name="pi05_axis_clean_local_droid_lora",
        train_mode="resume",
    )

    assert adapter.datasets == (dataset,)
    assert adapter.action_transform.spec_sha256 == spec["spec_sha256"]
    assert bound["AXIS_CLEAN_ROOT"] == str(clean_root)
    assert bound["OPENPI_AXIS_CLEAN_TASK_IDS"] == "7"
    assert bound["AXIS_CLEAN_DROID_ACTION_HORIZON"] == "15"
    assert bound["AXIS_PI05_ASSETS_BASE_DIR"] == str(clean_root / "openpi-assets")
    assert bound["OPENPI_DATA_HOME"] == str(clean_root / "openpi-data")
    assert bound["OPENPI_TRAIN_MODE"] == "resume"


def test_standard_openpi_pipeline_does_not_require_axis_environment(
    tmp_path: Path,
) -> None:
    openpi = _fake_openpi(tmp_path)
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/axis.py"),
            "train",
            "pi05_libero",
            "libero_smoke",
            "--openpi-root",
            str(openpi),
            "--dry-run",
            "--num-train-steps",
            "3",
        ],
        env=_clean_environment(),
        check=True,
        text=True,
        capture_output=True,
    )
    command = shlex.split(completed.stdout.removeprefix("[openpi:train] "))
    assert command[:3] == [sys.executable, "scripts/train.py", "pi05_libero"]
    assert command[command.index("--exp-name") + 1] == "libero_smoke"
    assert command[command.index("--num-train-steps") + 1] == "3"


def test_legacy_shell_entrypoints_are_only_compatibility_shims() -> None:
    wrappers = (
        ROOT / "training/run_compute_norm_stats.sh",
        ROOT / "training/run_train_openpi.sh",
        ROOT / "training/run_axis_finetune.sh",
        ROOT / "training/run_serve_openpi_policy.sh",
        ROOT / "training/run_libero_eval.sh",
        ROOT / "evaluation/run_axis_task.sh",
    )
    for wrapper in wrappers:
        lines = wrapper.read_text(encoding="utf-8").splitlines()
        assert len(lines) <= 6
        assert '"${ROOT}/scripts/axis.py"' in lines[-1]


def test_task_no_video_flag_is_not_inverted_by_argparse() -> None:
    args, extra = build_parser().parse_known_args(
        ["task", "recorded", "3801", "receipt", "--no-video", "--dry-run"]
    )
    assert args.no_video is True
    assert extra == []


def test_libero_eval_is_a_separate_environment_adapter(tmp_path: Path) -> None:
    openpi = _fake_openpi(tmp_path)
    libero_python = openpi / "examples/libero/.venv/bin/python"
    libero_python.parent.mkdir(parents=True)
    libero_python.write_text("#!/usr/bin/env sh\nexit 0\n", encoding="utf-8")
    libero_python.chmod(0o755)
    (openpi / "examples/libero/main.py").touch()

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/axis.py"),
            "libero-eval",
            "--openpi-root",
            str(openpi),
            "--dry-run",
            "--task-suite-name",
            "libero_spatial",
        ],
        env=_clean_environment(),
        check=True,
        text=True,
        capture_output=True,
    )
    command = shlex.split(completed.stdout.removeprefix("[libero:eval] "))
    assert command == [
        str(libero_python),
        "examples/libero/main.py",
        "--task-suite-name",
        "libero_spatial",
    ]
