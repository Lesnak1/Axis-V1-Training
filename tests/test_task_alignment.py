from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _task_environment(tmp_path: Path) -> dict[str, str]:
    lff_root = tmp_path / "learn-from-failure-at-scale"
    task_contract = tmp_path / "task-contract"
    assets = tmp_path / "assets"
    camera = tmp_path / "camera.json"
    scene_split = tmp_path / "scene-split.json"
    recorded = tmp_path / "episode.parquet"
    action_spec = tmp_path / "action-transform.json"
    for directory in (lff_root, task_contract, assets):
        directory.mkdir()
    for file in (camera, scene_split, recorded, action_spec):
        file.write_text("{}\n", encoding="utf-8")

    env = os.environ.copy()
    env.update(
        {
            "AXIS_LEARN_FROM_FAILURE_ROOT": str(lff_root),
            "AXIS_TASK_CONTRACT": str(task_contract),
            "AXIS_TASK_ASSET_DIR": str(assets),
            "AXIS_CAMERA_CONTRACT": str(camera),
            "AXIS_SCENE_SPLIT": str(scene_split),
            "AXIS_RECORDED_ACTIONS": str(recorded),
            "AXIS_ACTION_TRANSFORM_SPEC": str(action_spec),
            "AXIS_EVAL_PYTHON": "/usr/bin/python3",
            "AXIS_TASK_DRY_RUN": "1",
        }
    )
    return env


def _dry_run_task(tmp_path: Path, mode: str, env: dict[str, str]) -> list[str]:
    completed = subprocess.run(
        [
            "bash",
            str(ROOT / "evaluation/run_axis_task.sh"),
            mode,
            "3801",
            str(tmp_path / "receipt"),
        ],
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )
    prefix = "[axis-task] "
    assert completed.stdout.startswith(prefix)
    return shlex.split(completed.stdout[len(prefix) :])


def test_recorded_droid_task_call_pins_lff_runtime_and_positive_control(tmp_path: Path) -> None:
    env = _task_environment(tmp_path)
    env.update(
        {
            "AXIS_SCENE_INDEX": "0",
            "AXIS_MAX_CONTROLS": "80",
            "AXIS_REPLAN_INTERVAL": "250",
            "AXIS_SAMPLES_PER_REPLAN": "1",
        }
    )

    command = _dry_run_task(tmp_path, "recorded_droid", env)

    assert command[:2] == ["env", f"AXIS_LEARN_FROM_FAILURE_ROOT={env['AXIS_LEARN_FROM_FAILURE_ROOT']}"]
    assert command[command.index("--policy") + 1] == "recorded_droid"
    assert command[command.index("--task-contract") + 1] == env["AXIS_TASK_CONTRACT"]
    assert command[command.index("--asset-dir") + 1] == env["AXIS_TASK_ASSET_DIR"]
    assert command[command.index("--camera-contract") + 1] == env["AXIS_CAMERA_CONTRACT"]
    assert command[command.index("--max-controls") + 1] == "80"
    assert command[command.index("--replan-interval") + 1] == "250"
    assert command[command.index("--samples-per-replan") + 1] == "1"
    assert command[command.index("--scene-index") + 1] == "0"
    assert command[command.index("--recorded-actions") + 1] == env["AXIS_RECORDED_ACTIONS"]
    assert command[command.index("--action-transform-spec") + 1] == env["AXIS_ACTION_TRANSFORM_SPEC"]


def test_task_call_preserves_virtualenv_python_symlink(tmp_path: Path) -> None:
    env = _task_environment(tmp_path)
    python_link = tmp_path / "lff-venv/bin/python"
    python_link.parent.mkdir(parents=True)
    python_link.symlink_to(Path(sys.executable).resolve())
    env["AXIS_EVAL_PYTHON"] = str(python_link)

    command = _dry_run_task(tmp_path, "recorded_droid", env)

    assert command[4] == str(python_link)


def test_websocket_task_call_defaults_to_development_split(tmp_path: Path) -> None:
    env = _task_environment(tmp_path)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    env.update({"AXIS_CHECKPOINT": str(checkpoint), "AXIS_POLICY_PORT": "18131"})

    command = _dry_run_task(tmp_path, "websocket", env)

    assert command[command.index("--scene-split") + 1] == env["AXIS_SCENE_SPLIT"]
    assert command[command.index("--role") + 1] == "development_evaluation"
    assert "--allow-final-test" not in command
    assert command[command.index("--checkpoint") + 1] == env["AXIS_CHECKPOINT"]
    assert command[command.index("--port") + 1] == "18131"


def test_openpi_patch_exposes_full_action_expert_scope() -> None:
    patch = (ROOT / "openpi_patches/axis_train_openpi.patch").read_text(encoding="utf-8")
    runner = (ROOT / "training/run_axis_finetune.sh").read_text(encoding="utf-8")
    adapter = (ROOT / "axis_train/axis_environment.py").read_text(encoding="utf-8")

    assert "def _freeze_pi05_backbones_for_action_expert" in patch
    assert 'name="pi05_axis_clean_local_droid_action_expert"' in patch
    assert "freeze_filter=_freeze_pi05_backbones_for_action_expert" in patch
    assert 'return "_1" not in param_path' in patch
    assert '"pi05_axis_clean_local_droid_action_expert": "AXIS_PI05_ACTION_EXPERT_INIT_PARAMS"' in adapter
    assert '"${ROOT}/scripts/axis.py" finetune' in runner


def test_closed_loop_receipt_hashes_the_shared_task_runtime() -> None:
    evaluator = (ROOT / "evaluation/closed_loop_axis.py").read_text(encoding="utf-8")

    for source in (
        "asset_materializer.py",
        "asset_store.py",
        "backend_replay.py",
        "checker_payload_extractor.py",
        "CheckerManager.py",
        "mujoco_model_loader.py",
        "state_restore.py",
    ):
        assert source in evaluator
    assert '"source_sha256"' in evaluator
