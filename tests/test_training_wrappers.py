from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from axis_train.action_transform import make_action_transform_spec

ROOT = Path(__file__).resolve().parents[1]


def _write_action_transform_spec(path: Path) -> None:
    payload = make_action_transform_spec(
        source_fps=5.0,
        source_joint_names=["left_finger", "right_finger", *[f"arm_joint_{index}" for index in range(1, 8)]],
        third_person_source="observation.images.base",
        wrist_source="observation.images.left_wrist",
        action_horizon=1,
    )
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def test_serve_wrapper_places_global_options_before_policy_subcommand(tmp_path: Path) -> None:
    openpi_root = tmp_path / "openpi"
    (openpi_root / "scripts").mkdir(parents=True)
    (openpi_root / "scripts/serve_policy.py").touch()
    fake_python = tmp_path / "fake-python"
    captured_args = tmp_path / "args.txt"
    checkpoint = tmp_path / "checkpoint" / "0"
    spec_dir = checkpoint / "assets/local/axis_clean"
    spec_dir.mkdir(parents=True)
    _write_action_transform_spec(spec_dir / "action_transform_spec.json")
    fake_python.write_text(
        "#!/usr/bin/env bash\nprintf '%s\\n' \"$@\" > \"$AXIS_TEST_CAPTURE_ARGS\"\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "AXIS_TEST_CAPTURE_ARGS": str(captured_args),
            "AXIS_CLI_PYTHON": sys.executable,
            "OPENPI_PYTHON": str(fake_python),
            "OPENPI_ROOT": str(openpi_root),
        }
    )
    subprocess.run(
        [
            "bash",
            str(ROOT / "training/run_serve_openpi_policy.sh"),
            "pi05_axis_clean_local_droid_lora",
            str(checkpoint),
            "--port",
            "18017",
            "--default-prompt",
            "pick the cube",
        ],
        env=env,
        check=True,
    )

    assert captured_args.read_text(encoding="utf-8").splitlines() == [
        "scripts/serve_policy.py",
        "--port",
        "18017",
        "--default-prompt",
        "pick the cube",
        "policy:checkpoint",
        "--policy.config=pi05_axis_clean_local_droid_lora",
        f"--policy.dir={checkpoint}",
    ]


def test_serve_wrapper_replaces_shell_so_signals_reach_server(tmp_path: Path) -> None:
    openpi_root = tmp_path / "openpi"
    (openpi_root / "scripts").mkdir(parents=True)
    (openpi_root / "scripts/serve_policy.py").touch()
    checkpoint = tmp_path / "checkpoint" / "0"
    spec_dir = checkpoint / "assets/local/axis_clean"
    spec_dir.mkdir(parents=True)
    _write_action_transform_spec(spec_dir / "action_transform_spec.json")

    fake_python = tmp_path / "fake-python"
    ready_file = tmp_path / "ready.txt"
    stopped_file = tmp_path / "stopped.txt"
    fake_python.write_text(
        """#!/usr/bin/env python3
import os
import signal
import time
from pathlib import Path

ready = Path(os.environ["AXIS_TEST_READY_FILE"])
stopped = Path(os.environ["AXIS_TEST_STOPPED_FILE"])

def stop(_signum, _frame):
    stopped.write_text("stopped\\n", encoding="utf-8")
    raise SystemExit(0)

signal.signal(signal.SIGINT, stop)
ready.write_text(str(os.getpid()), encoding="utf-8")
while True:
    time.sleep(0.1)
""",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "AXIS_TEST_READY_FILE": str(ready_file),
            "AXIS_TEST_STOPPED_FILE": str(stopped_file),
            "AXIS_CLI_PYTHON": sys.executable,
            "OPENPI_PYTHON": str(fake_python),
            "OPENPI_ROOT": str(openpi_root),
        }
    )
    process = subprocess.Popen(
        [
            "bash",
            str(ROOT / "training/run_serve_openpi_policy.sh"),
            "pi05_axis_clean_local_droid_lora",
            str(checkpoint),
        ],
        env=env,
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 5
        while not ready_file.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert ready_file.exists()
        assert int(ready_file.read_text(encoding="utf-8")) == process.pid

        process.send_signal(signal.SIGINT)
        assert process.wait(timeout=5) == 0
        assert stopped_file.read_text(encoding="utf-8") == "stopped\n"
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=5)
