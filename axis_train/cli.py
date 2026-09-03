from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from axis_train.action_transform import (
    ActionTransformSpec,
    discover_from_openpi_environment,
)
from axis_train.axis_environment import (
    AxisDatasetEnvironment,
    AxisTaskEnvironment,
    DEFAULT_AXIS_CONFIG,
    environment_flag,
    is_axis_config,
)
from axis_train.config import write_json
from axis_train.pipeline import (
    OpenPIPipeline,
    PipelineError,
    openpi_readiness_report,
    resolve_openpi_root,
)


def project_root() -> Path:
    configured = os.environ.get("AXIS_TRAIN_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parents[1]


def _add_openpi_root(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--openpi-root", type=Path, help="Pinned, patched OpenPI checkout"
    )


def _add_dry_run(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve and print the command without executing it",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="axis-train",
        description="OpenPI training pipeline with explicit Axis environment adapters.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="Check the pinned OpenPI integration")
    _add_openpi_root(doctor)
    doctor.add_argument(
        "--output", type=Path, default=Path("reports/openpi_readiness.json")
    )

    spec = subparsers.add_parser(
        "spec", help="Resolve the selected ActionTransformSpec"
    )
    spec.add_argument(
        "--path", type=Path, help="Read a checkpoint/dataset spec directly"
    )
    spec.add_argument(
        "--clean-root", type=Path, help="Override AXIS_CLEAN_ROOT for discovery"
    )
    spec.add_argument("--format", choices=("json", "shell"), default="json")

    norm = subparsers.add_parser(
        "norm-stats", help="Run the ordinary OpenPI normalization stage"
    )
    norm.add_argument("config_name", nargs="?", default=DEFAULT_AXIS_CONFIG)
    _add_openpi_root(norm)
    _add_dry_run(norm)

    train = subparsers.add_parser(
        "train", help="Run the ordinary OpenPI training stage"
    )
    train.add_argument("config_name", nargs="?", default=DEFAULT_AXIS_CONFIG)
    train.add_argument("experiment_name", nargs="?", default="axis_pi05_v001")
    train.add_argument("--mode", choices=("overwrite", "resume", "none"))
    _add_openpi_root(train)
    _add_dry_run(train)

    finetune = subparsers.add_parser(
        "finetune",
        help="Bind one Axis dataset/task contract, then run the standard OpenPI pipeline",
    )
    finetune.add_argument("task_id", type=int)
    finetune.add_argument("clean_root", type=Path)
    finetune.add_argument("experiment_name")
    finetune.add_argument("--config", dest="config_name")
    finetune.add_argument(
        "--compute-norm-stats", action=argparse.BooleanOptionalAction, default=None
    )
    finetune.add_argument("--assets-base-dir", type=Path)
    finetune.add_argument("--data-home", type=Path)
    finetune.add_argument("--checkpoint-base-dir", type=Path)
    finetune.add_argument("--init-params", type=Path)
    finetune.add_argument("--mode", choices=("overwrite", "resume", "none"))
    _add_openpi_root(finetune)
    _add_dry_run(finetune)

    serve = subparsers.add_parser("serve", help="Serve one OpenPI checkpoint")
    serve.add_argument("config_name")
    serve.add_argument("checkpoint_dir", type=Path)
    serve.add_argument("--serve-env", help="Optional OpenPI --env value")
    _add_openpi_root(serve)
    _add_dry_run(serve)

    libero = subparsers.add_parser(
        "libero-eval", help="Run OpenPI's isolated LIBERO environment"
    )
    libero.add_argument("--libero-python", type=Path)
    _add_openpi_root(libero)
    _add_dry_run(libero)

    task = subparsers.add_parser(
        "task",
        help="Invoke the shared Learn-from-Failure task environment",
    )
    task.add_argument("mode", choices=("recorded", "recorded_droid", "websocket"))
    task.add_argument("task_id", type=int)
    task.add_argument("output_dir", type=Path)
    task.add_argument("--profile", type=Path)
    task.add_argument("--learn-from-failure-root", type=Path)
    task.add_argument("--task-contract", type=Path)
    task.add_argument("--asset-dir", type=Path)
    task.add_argument("--camera-contract", type=Path)
    task.add_argument("--scene-split", type=Path)
    task.add_argument("--eval-python")
    task.add_argument("--checkpoint", type=Path)
    task.add_argument("--host")
    task.add_argument("--port", type=int)
    task.add_argument("--scene-index", type=int)
    task.add_argument("--role")
    task.add_argument(
        "--allow-final-test", action=argparse.BooleanOptionalAction, default=None
    )
    task.add_argument("--recorded-actions", type=Path)
    task.add_argument("--action-transform-spec", type=Path)
    task.add_argument("--max-controls", type=int)
    task.add_argument("--replan-interval", type=int)
    task.add_argument("--samples-per-replan", type=int)
    video = task.add_mutually_exclusive_group()
    video.add_argument("--no-video", dest="no_video", action="store_true")
    video.add_argument("--video", dest="no_video", action="store_false")
    task.set_defaults(no_video=None)
    _add_openpi_root(task)
    _add_dry_run(task)
    return parser


def _passthrough(values: Sequence[str]) -> list[str]:
    output = list(values)
    if output and output[0] == "--":
        output.pop(0)
    return output


def _axis_bind_if_needed(
    pipeline: OpenPIPipeline,
    *,
    config_name: str,
    train_mode: str | None = None,
) -> tuple[OpenPIPipeline, AxisDatasetEnvironment | None]:
    if not is_axis_config(config_name):
        return pipeline, None
    adapter = AxisDatasetEnvironment.from_environment(
        openpi_root=pipeline.openpi_root,
        project_root=pipeline.project_root,
        environ=pipeline.environ,
    )
    return (
        pipeline.with_environment(
            adapter.bind(
                pipeline.environ,
                config_name=config_name,
                train_mode=train_mode,
            )
        ),
        adapter,
    )


def _run_doctor(args: argparse.Namespace, root: Path) -> int:
    openpi = resolve_openpi_root(
        root,
        raw=args.openpi_root,
        environ=os.environ,
        require=False,
    )
    report = openpi_readiness_report(root, openpi)
    output = args.output.expanduser()
    if not output.is_absolute():
        output = root / output
    write_json(output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    required = (
        report["exists"],
        report["train_script"],
        report["compute_norm_stats_script"],
        report["serve_policy_script"],
    )
    return 0 if all(required) else 1


def _run_spec(args: argparse.Namespace) -> int:
    values = dict(os.environ)
    if args.clean_root is not None:
        values["AXIS_CLEAN_ROOT"] = str(args.clean_root.expanduser().resolve())
    spec = (
        ActionTransformSpec.load(args.path)
        if args.path is not None
        else discover_from_openpi_environment(values)
    )
    environment = spec.openpi_environment()
    if args.format == "shell":
        for name, value in environment.items():
            print(f"export {name}={shlex.quote(value)}")
    else:
        print(
            json.dumps(
                {
                    "ok": True,
                    "spec_path": str(spec.path),
                    "spec_sha256": spec.spec_sha256,
                    "velocity_scale": spec.velocity_scale,
                    "downsample_ratio": spec.downsample_ratio,
                    "action_horizon": spec.action_horizon,
                    "environment": environment,
                },
                indent=2,
                sort_keys=True,
            )
        )
    return 0


def _run_norm_stats(args: argparse.Namespace, extra: Sequence[str], root: Path) -> int:
    pipeline = OpenPIPipeline.create(
        root,
        openpi_root=args.openpi_root,
        dry_run=args.dry_run,
    )
    pipeline, _ = _axis_bind_if_needed(pipeline, config_name=args.config_name)
    command = pipeline.command(
        "norm-stats",
        ["--config-name", args.config_name, *_passthrough(extra)],
    )
    return pipeline.run("norm-stats", command)


def _run_train(args: argparse.Namespace, extra: Sequence[str], root: Path) -> int:
    mode = args.mode or os.environ.get("OPENPI_TRAIN_MODE", "overwrite")
    pipeline = OpenPIPipeline.create(
        root,
        openpi_root=args.openpi_root,
        dry_run=args.dry_run,
    )
    pipeline, adapter = _axis_bind_if_needed(
        pipeline,
        config_name=args.config_name,
        train_mode=mode,
    )
    if adapter is not None:
        adapter.require_norm_stats(args.config_name)
    pipeline = pipeline.with_environment(
        {
            "XLA_PYTHON_CLIENT_MEM_FRACTION": pipeline.environ.get(
                "XLA_PYTHON_CLIENT_MEM_FRACTION", "0.9"
            )
        }
    )
    command = pipeline.command(
        "train",
        [
            args.config_name,
            "--exp-name",
            args.experiment_name,
            *pipeline.train_mode_arguments(mode),
            *_passthrough(extra),
        ],
    )
    return pipeline.replace("train", command)


def _run_finetune(args: argparse.Namespace, extra: Sequence[str], root: Path) -> int:
    config_name = args.config_name or os.environ.get(
        "AXIS_FINETUNE_CONFIG", DEFAULT_AXIS_CONFIG
    )
    mode = args.mode or os.environ.get("OPENPI_TRAIN_MODE", "overwrite")
    pipeline = OpenPIPipeline.create(
        root,
        openpi_root=args.openpi_root,
        dry_run=args.dry_run,
    )
    adapter = AxisDatasetEnvironment.for_task(
        args.task_id,
        args.clean_root,
        openpi_root=pipeline.openpi_root,
        project_root=root,
        environ=pipeline.environ,
        assets_base_dir=args.assets_base_dir,
        openpi_data_home=args.data_home,
        checkpoint_base_dir=args.checkpoint_base_dir,
    )
    pipeline = pipeline.with_environment(
        adapter.bind(
            pipeline.environ,
            config_name=config_name,
            train_mode=mode,
            init_params=args.init_params,
        )
    )
    print(
        "[axis-finetune] "
        f"task={args.task_id} dataset={shlex.quote(str(adapter.datasets[0]))} "
        f"config={config_name} exp={args.experiment_name} "
        f"spec={adapter.action_transform.spec_sha256}",
        flush=True,
    )

    compute_stats = (
        environment_flag("AXIS_COMPUTE_NORM_STATS", environ=pipeline.environ)
        if args.compute_norm_stats is None
        else args.compute_norm_stats
    )
    if compute_stats:
        stats_command = pipeline.command(
            "norm-stats",
            ["--config-name", config_name],
        )
        pipeline.run("norm-stats", stats_command)
    if not (pipeline.dry_run and compute_stats):
        adapter.require_norm_stats(config_name)

    train_arguments = adapter.training_arguments(_passthrough(extra))
    pipeline = pipeline.with_environment(
        {
            "XLA_PYTHON_CLIENT_MEM_FRACTION": pipeline.environ.get(
                "XLA_PYTHON_CLIENT_MEM_FRACTION", "0.9"
            )
        }
    )
    train_command = pipeline.command(
        "train",
        [
            config_name,
            "--exp-name",
            args.experiment_name,
            *pipeline.train_mode_arguments(mode),
            *train_arguments,
        ],
    )
    return pipeline.replace("train", train_command)


def _run_serve(args: argparse.Namespace, extra: Sequence[str], root: Path) -> int:
    pipeline = OpenPIPipeline.create(
        root,
        openpi_root=args.openpi_root,
        dry_run=args.dry_run,
    )
    checkpoint = args.checkpoint_dir.expanduser()
    if not checkpoint.is_absolute():
        checkpoint = root / checkpoint
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_dir():
        raise PipelineError(f"Checkpoint directory is missing: {checkpoint}")
    if is_axis_config(args.config_name):
        spec = ActionTransformSpec.load(
            checkpoint / "assets/local/axis_clean/action_transform_spec.json"
        )
        spec.require_environment_matches(pipeline.environ)
        pipeline = pipeline.with_environment(spec.openpi_environment())

    environment_name = args.serve_env or pipeline.environ.get("OPENPI_SERVE_ENV")
    environment_arguments = ["--env", environment_name] if environment_name else []
    command = pipeline.command(
        "serve",
        [
            *environment_arguments,
            *_passthrough(extra),
            "policy:checkpoint",
            f"--policy.config={args.config_name}",
            f"--policy.dir={checkpoint}",
        ],
    )
    return pipeline.replace("serve", command)


def _run_libero_eval(args: argparse.Namespace, extra: Sequence[str], root: Path) -> int:
    openpi = resolve_openpi_root(
        root,
        raw=args.openpi_root,
        environ=os.environ,
        require=True,
    )
    raw_python = (
        args.libero_python
        or os.environ.get("LIBERO_PYTHON")
        or Path("examples/libero/.venv/bin/python")
    )
    python = Path(raw_python).expanduser()
    if not python.is_absolute():
        python = openpi / python
    python = python.resolve()
    if not python.is_file() or not os.access(python, os.X_OK):
        raise PipelineError(
            f"LIBERO Python is not executable: {python}. Follow docs/LIBERO_FINETUNE_EVAL.md."
        )
    main_script = openpi / "examples/libero/main.py"
    if not main_script.is_file():
        raise PipelineError(f"LIBERO entrypoint is missing: {main_script}")
    command = [str(python), "examples/libero/main.py", *_passthrough(extra)]
    print(f"[libero:eval] {shlex.join(command)}", flush=True)
    if args.dry_run:
        return 0
    values = dict(os.environ)
    libero_source = str(openpi / "third_party/libero")
    existing_pythonpath = values.get("PYTHONPATH")
    values["PYTHONPATH"] = (
        f"{libero_source}{os.pathsep}{existing_pythonpath}"
        if existing_pythonpath
        else libero_source
    )
    os.chdir(openpi)
    os.execvpe(command[0], command, values)
    raise AssertionError("os.execvpe returned unexpectedly")


def _run_task(args: argparse.Namespace, extra: Sequence[str], root: Path) -> int:
    adapter = AxisTaskEnvironment.create(
        args.task_id,
        project_root=root,
        environ=os.environ,
        profile_path=args.profile,
        learn_from_failure_root=args.learn_from_failure_root,
        task_contract=args.task_contract,
        asset_dir=args.asset_dir,
        camera_contract=args.camera_contract,
        scene_split=args.scene_split,
        openpi_root=args.openpi_root,
    )
    command = adapter.command(
        args.mode,
        args.output_dir,
        eval_python=args.eval_python,
        checkpoint=args.checkpoint,
        host=args.host,
        port=args.port,
        scene_index=args.scene_index,
        role=args.role,
        allow_final_test=args.allow_final_test,
        recorded_actions=args.recorded_actions,
        action_transform_spec=args.action_transform_spec,
        max_controls=args.max_controls,
        replan_interval=args.replan_interval,
        samples_per_replan=args.samples_per_replan,
        no_video=args.no_video,
        evaluator_arguments=_passthrough(extra),
    )
    adapter.announce(command)
    dry_run = args.dry_run or environment_flag("AXIS_TASK_DRY_RUN", environ=os.environ)
    if dry_run:
        return 0
    os.execvpe(command[0], command, dict(os.environ))
    raise AssertionError("os.execvpe returned unexpectedly")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args, extra = parser.parse_known_args(argv)
    root = project_root()
    try:
        if args.command == "doctor":
            if extra:
                parser.error(f"unrecognized arguments: {' '.join(extra)}")
            return _run_doctor(args, root)
        if args.command == "spec":
            if extra:
                parser.error(f"unrecognized arguments: {' '.join(extra)}")
            return _run_spec(args)
        if args.command == "norm-stats":
            return _run_norm_stats(args, extra, root)
        if args.command == "train":
            return _run_train(args, extra, root)
        if args.command == "finetune":
            return _run_finetune(args, extra, root)
        if args.command == "serve":
            return _run_serve(args, extra, root)
        if args.command == "libero-eval":
            return _run_libero_eval(args, extra, root)
        if args.command == "task":
            return _run_task(args, extra, root)
        raise AssertionError(f"Unhandled command: {args.command}")
    except subprocess.CalledProcessError as error:
        print(
            f"ERROR: command failed with exit code {error.returncode}: {shlex.join(error.cmd)}",
            file=sys.stderr,
        )
        return int(error.returncode)
    except (
        FileNotFoundError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
        PipelineError,
    ) as error:
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
