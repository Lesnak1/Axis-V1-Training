from __future__ import annotations

import json
import re
import shlex
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from axis_train.action_transform import (
    ActionTransformSpec,
    discover_action_transform_spec,
)
from axis_train.pipeline import PipelineError


TASK_PROFILE_SCHEMA = "axis_task_profile_v1"
DEFAULT_AXIS_CONFIG = "pi05_axis_clean_local_droid_lora"
INIT_PARAMETER_VARIABLES = {
    "pi05_axis_clean_local_droid_lora": "AXIS_PI05_LORA_INIT_PARAMS",
    "pi05_axis_clean_local_droid_full": "AXIS_PI05_FULL_INIT_PARAMS",
    "pi05_axis_clean_local_droid_action_expert": "AXIS_PI05_ACTION_EXPERT_INIT_PARAMS",
}
DEFAULT_PI05_DROID_PARAMS = Path(
    "/data/openpi/openpi-assets/checkpoints/pi05_droid/params"
)


def is_axis_config(config_name: str) -> bool:
    return "axis" in config_name.lower()


def environment_flag(
    name: str,
    *,
    environ: Mapping[str, str],
    default: bool = False,
) -> bool:
    raw = environ.get(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise PipelineError(f"{name} must be a boolean value, got {raw!r}")


def _path(raw: str | Path, *, base_dir: Path, require: str | None = None) -> Path:
    value = Path(raw).expanduser()
    if not value.is_absolute():
        value = base_dir / value
    resolved = value.resolve()
    if require == "dir" and not resolved.is_dir():
        raise PipelineError(f"Required directory is missing: {resolved}")
    if require == "file" and not resolved.is_file():
        raise PipelineError(f"Required file is missing: {resolved}")
    if require == "exists" and not resolved.exists():
        raise PipelineError(f"Required path is missing: {resolved}")
    return resolved


def _executable_path(raw: str | Path, *, base_dir: Path) -> Path:
    value = Path(raw).expanduser()
    if not value.is_absolute():
        value = base_dir / value
    value = value.absolute()
    if not value.is_file():
        raise PipelineError(f"Evaluation Python is not available: {value}")
    # A virtualenv's Python is normally a symlink. Resolving it here would run
    # the base interpreter without the virtualenv's site-packages.
    return value


def _parse_task_ids(raw: str | None) -> tuple[int, ...] | None:
    if not raw:
        return None
    try:
        values = tuple(
            sorted(
                {
                    int(item.strip())
                    for item in raw.replace(";", ",").split(",")
                    if item.strip()
                }
            )
        )
    except ValueError as exc:
        raise PipelineError(f"Invalid OPENPI_AXIS_CLEAN_TASK_IDS={raw!r}") from exc
    if not values or any(value < 0 for value in values):
        raise PipelineError(f"Invalid OPENPI_AXIS_CLEAN_TASK_IDS={raw!r}")
    return values


def _dataset_task_id(path: Path) -> int | None:
    match = re.match(r"task_(\d+)_.*_state_train$", path.name)
    return int(match.group(1)) if match else None


def _selected_datasets(
    clean_root: Path,
    *,
    task_ids: tuple[int, ...] | None,
    max_tasks: int | None,
) -> tuple[Path, ...]:
    datasets = tuple(
        sorted(path for path in clean_root.glob("task_*_state_train") if path.is_dir())
    )
    if task_ids is not None:
        selected = set(task_ids)
        datasets = tuple(
            path for path in datasets if _dataset_task_id(path) in selected
        )
    if max_tasks is not None:
        if max_tasks < 1:
            raise PipelineError("OPENPI_AXIS_CLEAN_MAX_TASKS must be positive")
        datasets = datasets[:max_tasks]
    if not datasets:
        raise PipelineError(
            f"No selected task_*_state_train dataset under {clean_root}"
        )
    return datasets


@dataclass(frozen=True)
class AxisDatasetEnvironment:
    """Bind replay-qualified Axis data to the ordinary OpenPI pipeline."""

    clean_root: Path
    datasets: tuple[Path, ...]
    task_ids: tuple[int, ...] | None
    action_transform: ActionTransformSpec
    assets_base_dir: Path
    openpi_data_home: Path | None = None
    checkpoint_base_dir: Path | None = None

    @classmethod
    def from_environment(
        cls,
        *,
        openpi_root: Path,
        project_root: Path,
        environ: Mapping[str, str],
    ) -> AxisDatasetEnvironment:
        raw_root = environ.get("AXIS_CLEAN_ROOT")
        if not raw_root:
            raise PipelineError("AXIS_CLEAN_ROOT is required for Axis OpenPI configs")
        clean_root = _path(raw_root, base_dir=project_root, require="dir")
        task_ids = _parse_task_ids(environ.get("OPENPI_AXIS_CLEAN_TASK_IDS"))
        raw_max_tasks = environ.get("OPENPI_AXIS_CLEAN_MAX_TASKS")
        try:
            max_tasks = int(raw_max_tasks) if raw_max_tasks else None
        except ValueError as exc:
            raise PipelineError(
                f"Invalid OPENPI_AXIS_CLEAN_MAX_TASKS={raw_max_tasks!r}"
            ) from exc
        datasets = _selected_datasets(
            clean_root, task_ids=task_ids, max_tasks=max_tasks
        )
        action_transform = discover_action_transform_spec(
            clean_root,
            task_ids=set(task_ids) if task_ids is not None else None,
            max_tasks=max_tasks,
        )
        action_transform.require_environment_matches(environ)

        raw_assets = environ.get("AXIS_PI05_ASSETS_BASE_DIR")
        assets = _path(
            raw_assets or openpi_root / "assets",
            base_dir=openpi_root,
        )
        return cls(
            clean_root=clean_root,
            datasets=datasets,
            task_ids=task_ids,
            action_transform=action_transform,
            assets_base_dir=assets,
        )

    @classmethod
    def for_task(
        cls,
        task_id: int,
        clean_root: str | Path,
        *,
        openpi_root: Path,
        project_root: Path,
        environ: Mapping[str, str],
        assets_base_dir: str | Path | None = None,
        openpi_data_home: str | Path | None = None,
        checkpoint_base_dir: str | Path | None = None,
    ) -> AxisDatasetEnvironment:
        if task_id < 0:
            raise PipelineError(f"Task ID must be non-negative, got {task_id}")
        root = _path(clean_root, base_dir=project_root, require="dir")
        matches = _selected_datasets(root, task_ids=(task_id,), max_tasks=None)
        if len(matches) != 1:
            details = ", ".join(str(path) for path in matches)
            raise PipelineError(
                f"Expected exactly one task {task_id} dataset under {root}; "
                f"found {len(matches)}: {details}"
            )

        configured_task_ids = _parse_task_ids(environ.get("OPENPI_AXIS_CLEAN_TASK_IDS"))
        if configured_task_ids is not None and configured_task_ids != (task_id,):
            raise PipelineError(
                "OPENPI_AXIS_CLEAN_TASK_IDS conflicts with the explicit fine-tune task: "
                f"{configured_task_ids} != {(task_id,)}"
            )

        spec = discover_action_transform_spec(root, task_ids={task_id})
        spec.require_environment_matches(environ)
        assets = _path(
            assets_base_dir
            or environ.get("AXIS_PI05_ASSETS_BASE_DIR")
            or root / "openpi-assets",
            base_dir=project_root,
        )
        data_home = _path(
            openpi_data_home or environ.get("OPENPI_DATA_HOME") or root / "openpi-data",
            base_dir=project_root,
        )
        checkpoint_root = _path(
            checkpoint_base_dir
            or environ.get("AXIS_CHECKPOINT_BASE_DIR")
            or root / "checkpoints",
            base_dir=project_root,
        )
        return cls(
            clean_root=root,
            datasets=matches,
            task_ids=(task_id,),
            action_transform=spec,
            assets_base_dir=assets,
            openpi_data_home=data_home,
            checkpoint_base_dir=checkpoint_root,
        )

    def bind(
        self,
        environ: Mapping[str, str],
        *,
        config_name: str,
        train_mode: str | None = None,
        init_params: str | Path | None = None,
    ) -> dict[str, str]:
        values = dict(environ)
        values.update(
            {
                "AXIS_CLEAN_ROOT": str(self.clean_root),
                "AXIS_PI05_ASSETS_BASE_DIR": str(self.assets_base_dir),
                **self.action_transform.openpi_environment(),
            }
        )
        if self.task_ids is not None:
            values["OPENPI_AXIS_CLEAN_TASK_IDS"] = ",".join(map(str, self.task_ids))
        if self.openpi_data_home is not None:
            values["OPENPI_DATA_HOME"] = str(self.openpi_data_home)
        if train_mode is not None:
            values["OPENPI_TRAIN_MODE"] = train_mode

        values.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        values.setdefault("WANDB_MODE", "disabled")

        parameter_variable = INIT_PARAMETER_VARIABLES.get(config_name)
        if init_params is not None and parameter_variable is None:
            raise PipelineError(
                f"--init-params is not defined for OpenPI config {config_name!r}"
            )
        if parameter_variable is not None:
            if init_params is not None:
                values[parameter_variable] = str(
                    _path(init_params, base_dir=self.clean_root, require="dir")
                )
            elif (
                parameter_variable not in values and DEFAULT_PI05_DROID_PARAMS.is_dir()
            ):
                values[parameter_variable] = str(DEFAULT_PI05_DROID_PARAMS)
        return values

    def norm_stats_path(self, config_name: str) -> Path:
        return self.assets_base_dir / config_name / "local/axis_clean/norm_stats.json"

    def require_norm_stats(self, config_name: str) -> Path:
        path = self.norm_stats_path(config_name)
        if not path.is_file():
            raise PipelineError(
                f"Axis norm stats not found at {path}. Run the norm-stats stage first."
            )
        return path

    def training_arguments(self, arguments: Sequence[str]) -> list[str]:
        output = list(arguments)
        has_checkpoint_base = any(
            value == "--checkpoint-base-dir"
            or value.startswith("--checkpoint-base-dir=")
            for value in output
        )
        if not has_checkpoint_base and self.checkpoint_base_dir is not None:
            output.extend(["--checkpoint-base-dir", str(self.checkpoint_base_dir)])
        return output


@dataclass(frozen=True)
class AxisTaskProfile:
    path: Path
    task_id: int
    task_root: Path
    task_contract: Path
    asset_dir: Path
    camera_contract: Path | None
    scene_split: Path | None
    runtime: dict[str, Any]

    @classmethod
    def load(
        cls, path: str | Path, *, expected_task_id: int | None = None
    ) -> AxisTaskProfile:
        profile_path = Path(path).expanduser().resolve(strict=True)
        payload = json.loads(profile_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise PipelineError(f"Task profile must be a JSON object: {profile_path}")
        if payload.get("schema_version") != TASK_PROFILE_SCHEMA:
            raise PipelineError(
                f"Unsupported task profile schema in {profile_path}: "
                f"{payload.get('schema_version')!r}"
            )
        task_id = int(payload["task_id"])
        if expected_task_id is not None and task_id != expected_task_id:
            raise PipelineError(
                f"Task profile ID {task_id} does not match requested task {expected_task_id}"
            )
        task_root = _path(payload["task_root"], base_dir=profile_path.parent)

        def profile_path_value(name: str, *, base_dir: Path) -> Path | None:
            raw = payload.get(name)
            return _path(raw, base_dir=base_dir) if raw else None

        task_contract = profile_path_value("task_contract", base_dir=task_root)
        asset_dir = profile_path_value("asset_dir", base_dir=task_root)
        if task_contract is None or asset_dir is None:
            raise PipelineError(
                f"Task profile is missing task_contract/asset_dir: {profile_path}"
            )
        runtime = payload.get("runtime") or {}
        if not isinstance(runtime, dict):
            raise PipelineError(
                f"Task profile runtime must be an object: {profile_path}"
            )
        return cls(
            path=profile_path,
            task_id=task_id,
            task_root=task_root,
            task_contract=task_contract,
            asset_dir=asset_dir,
            camera_contract=profile_path_value(
                "camera_contract", base_dir=profile_path.parent
            ),
            scene_split=profile_path_value("scene_split", base_dir=profile_path.parent),
            runtime=dict(runtime),
        )


@dataclass(frozen=True)
class AxisTaskEnvironment:
    """Bind one Axis task profile to the shared Learn-from-Failure runtime."""

    project_root: Path
    task_id: int
    learn_from_failure_root: Path
    task_contract: Path
    asset_dir: Path
    camera_contract: Path
    scene_split: Path | None
    openpi_root: Path
    runtime_defaults: dict[str, Any]
    environ: dict[str, str]

    @classmethod
    def create(
        cls,
        task_id: int,
        *,
        project_root: Path,
        environ: Mapping[str, str],
        profile_path: str | Path | None = None,
        learn_from_failure_root: str | Path | None = None,
        task_contract: str | Path | None = None,
        asset_dir: str | Path | None = None,
        camera_contract: str | Path | None = None,
        scene_split: str | Path | None = None,
        openpi_root: str | Path | None = None,
    ) -> AxisTaskEnvironment:
        if task_id < 0:
            raise PipelineError(f"Task ID must be non-negative, got {task_id}")
        project = project_root.expanduser().resolve()
        values = dict(environ)
        selected_profile = profile_path or values.get("AXIS_TASK_PROFILE")
        if selected_profile is None:
            default_profile = project / f"configs/tasks/{task_id}.json"
            selected_profile = default_profile if default_profile.is_file() else None
        profile = (
            AxisTaskProfile.load(selected_profile, expected_task_id=task_id)
            if selected_profile is not None
            else None
        )

        task_root = (
            profile.task_root
            if profile is not None
            else _path(
                values.get("AXIS_TASK_ROOT") or f"/data/wls_assets/axis/task_{task_id}",
                base_dir=project,
            )
        )
        lff = _path(
            learn_from_failure_root
            or values.get("AXIS_LEARN_FROM_FAILURE_ROOT")
            or project.parent / "learn-from-failure-at-scale",
            base_dir=project,
            require="dir",
        )
        contract = _path(
            task_contract
            or values.get("AXIS_TASK_CONTRACT")
            or (
                profile.task_contract
                if profile is not None
                else task_root / "task_contract_source_prod_20260826_v1"
            ),
            base_dir=task_root,
            require="exists",
        )
        assets = _path(
            asset_dir
            or values.get("AXIS_TASK_ASSET_DIR")
            or (
                profile.asset_dir
                if profile is not None
                else task_root / "assets/training_scenes_split_v2/assets"
            ),
            base_dir=task_root,
            require="dir",
        )
        raw_camera = camera_contract or values.get("AXIS_CAMERA_CONTRACT")
        if raw_camera is None and profile is not None:
            raw_camera = profile.camera_contract
        if raw_camera is None:
            raise PipelineError(
                f"No camera contract configured for task {task_id}; use --camera-contract"
            )
        camera = _path(raw_camera, base_dir=project, require="file")
        raw_split = scene_split or values.get("AXIS_SCENE_SPLIT")
        if raw_split is None and profile is not None:
            raw_split = profile.scene_split
        split = _path(raw_split, base_dir=project) if raw_split is not None else None
        openpi = _path(
            openpi_root or values.get("OPENPI_ROOT") or project / "third_party/openpi",
            base_dir=project,
        )
        return cls(
            project_root=project,
            task_id=task_id,
            learn_from_failure_root=lff,
            task_contract=contract,
            asset_dir=assets,
            camera_contract=camera,
            scene_split=split,
            openpi_root=openpi,
            runtime_defaults=dict(profile.runtime) if profile is not None else {},
            environ=values,
        )

    def _runtime_value(
        self, explicit: Any, env_name: str, key: str, default: Any
    ) -> Any:
        if explicit is not None:
            return explicit
        if env_name in self.environ:
            return self.environ[env_name]
        return self.runtime_defaults.get(key, default)

    def command(
        self,
        mode: str,
        output_dir: str | Path,
        *,
        eval_python: str | None = None,
        checkpoint: str | Path | None = None,
        host: str | None = None,
        port: int | None = None,
        scene_index: int | None = None,
        role: str | None = None,
        allow_final_test: bool | None = None,
        recorded_actions: str | Path | None = None,
        action_transform_spec: str | Path | None = None,
        max_controls: int | None = None,
        replan_interval: int | None = None,
        samples_per_replan: int | None = None,
        no_video: bool | None = None,
        evaluator_arguments: Sequence[str] = (),
    ) -> list[str]:
        if mode not in {"recorded", "recorded_droid", "websocket"}:
            raise PipelineError(
                f"Task mode must be recorded, recorded_droid, or websocket; got {mode!r}"
            )
        python = eval_python or self.environ.get("AXIS_EVAL_PYTHON") or "python3"
        if "/" in python:
            python = str(_executable_path(python, base_dir=self.project_root))
        else:
            resolved_python = shutil.which(python)
            if resolved_python is None:
                raise PipelineError(f"Evaluation Python is not available: {python}")
            python = resolved_python

        output = _path(output_dir, base_dir=self.project_root)
        controls = int(
            self._runtime_value(max_controls, "AXIS_MAX_CONTROLS", "max_controls", 250)
        )
        replan = int(
            self._runtime_value(
                replan_interval, "AXIS_REPLAN_INTERVAL", "replan_interval", 15
            )
        )
        samples = int(
            self._runtime_value(
                samples_per_replan, "AXIS_SAMPLES_PER_REPLAN", "samples_per_replan", 1
            )
        )
        if min(controls, replan, samples) < 1:
            raise PipelineError(
                "max-controls, replan-interval, and samples-per-replan must be positive"
            )

        command = [
            "env",
            f"AXIS_LEARN_FROM_FAILURE_ROOT={self.learn_from_failure_root}",
            f"MUJOCO_GL={self.environ.get('MUJOCO_GL', 'egl')}",
            f"MUJOCO_EGL_DEVICE_ID={self.environ.get('MUJOCO_EGL_DEVICE_ID', '0')}",
            python,
            str(self.project_root / "evaluation/closed_loop_axis.py"),
            "--policy",
            mode,
            "--task-contract",
            str(self.task_contract),
            "--asset-dir",
            str(self.asset_dir),
            "--camera-contract",
            str(self.camera_contract),
            "--output-dir",
            str(output),
            "--max-controls",
            str(controls),
            "--replan-interval",
            str(replan),
            "--samples-per-replan",
            str(samples),
        ]

        selected_scene_index = self._runtime_value(
            scene_index,
            "AXIS_SCENE_INDEX",
            "scene_index",
            None,
        )
        if selected_scene_index is not None and str(selected_scene_index) != "":
            selected_scene_index = int(selected_scene_index)

        if mode in {"recorded", "recorded_droid"}:
            raw_actions = recorded_actions or self.environ.get("AXIS_RECORDED_ACTIONS")
            if raw_actions is None:
                raise PipelineError(f"--recorded-actions is required for {mode}")
            actions = _path(raw_actions, base_dir=self.project_root, require="file")
            command.extend(
                [
                    "--scene-index",
                    str(
                        selected_scene_index if selected_scene_index is not None else 0
                    ),
                    "--recorded-actions",
                    str(actions),
                ]
            )
            if mode == "recorded_droid":
                raw_spec = action_transform_spec or self.environ.get(
                    "AXIS_ACTION_TRANSFORM_SPEC"
                )
                if raw_spec is None:
                    raise PipelineError(
                        "--action-transform-spec is required for recorded_droid"
                    )
                spec = _path(raw_spec, base_dir=self.project_root, require="file")
                command.extend(["--action-transform-spec", str(spec)])
        else:
            raw_checkpoint = checkpoint or self.environ.get("AXIS_CHECKPOINT")
            if raw_checkpoint is None:
                raise PipelineError(
                    "--checkpoint is required for websocket task evaluation"
                )
            checkpoint_path = _path(
                raw_checkpoint, base_dir=self.project_root, require="exists"
            )
            command.extend(
                [
                    "--checkpoint",
                    str(checkpoint_path),
                    "--openpi-root",
                    str(self.openpi_root),
                    "--host",
                    host or self.environ.get("AXIS_POLICY_HOST", "127.0.0.1"),
                    "--port",
                    str(port or int(self.environ.get("AXIS_POLICY_PORT", "8000"))),
                ]
            )
            if selected_scene_index is not None:
                command.extend(["--scene-index", str(selected_scene_index)])
            else:
                if self.scene_split is None or not self.scene_split.is_file():
                    raise PipelineError(f"Scene split is missing: {self.scene_split}")
                selected_role = role or self.environ.get(
                    "AXIS_EVAL_ROLE",
                    str(self.runtime_defaults.get("role", "development_evaluation")),
                )
                command.extend(
                    ["--scene-split", str(self.scene_split), "--role", selected_role]
                )
                final_access = (
                    environment_flag("AXIS_ALLOW_FINAL_TEST", environ=self.environ)
                    if allow_final_test is None
                    else allow_final_test
                )
                if final_access:
                    command.append("--allow-final-test")

        disable_video = (
            environment_flag("AXIS_NO_VIDEO", environ=self.environ)
            if no_video is None
            else no_video
        )
        if disable_video:
            command.append("--no-video")
        command.extend(map(str, evaluator_arguments))
        return command

    def announce(self, command: Sequence[str]) -> None:
        print(f"[axis-task] {shlex.join(list(command))}", flush=True)


__all__ = [
    "AxisDatasetEnvironment",
    "AxisTaskEnvironment",
    "AxisTaskProfile",
    "DEFAULT_AXIS_CONFIG",
    "INIT_PARAMETER_VARIABLES",
    "environment_flag",
    "is_axis_config",
]
