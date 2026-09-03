from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping, Sequence


OPENPI_PROGRAMS = {
    "norm-stats": Path("scripts/compute_norm_stats.py"),
    "train": Path("scripts/train.py"),
    "serve": Path("scripts/serve_policy.py"),
}
TRAIN_MODES = frozenset({"overwrite", "resume", "none"})


class PipelineError(RuntimeError):
    """A user-facing pipeline configuration error."""


def _absolute_path(raw: str | Path, *, base_dir: Path) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def resolve_openpi_root(
    project_root: Path,
    *,
    raw: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
    require: bool = True,
) -> Path:
    values = os.environ if environ is None else environ
    selected = raw or values.get("OPENPI_ROOT") or project_root / "third_party/openpi"
    root = _absolute_path(selected, base_dir=project_root)
    if require and not root.is_dir():
        raise PipelineError(
            f"OpenPI checkout not found at {root}. Set --openpi-root/OPENPI_ROOT "
            "and apply openpi_patches/axis_train_openpi.patch."
        )
    return root


def _resolve_executable(raw: str, *, base_dir: Path) -> str:
    candidate = Path(raw).expanduser()
    if candidate.is_absolute():
        resolved = candidate
    elif "/" in raw:
        resolved = (base_dir / candidate).resolve()
    else:
        found = shutil.which(raw)
        if found is None:
            raise PipelineError(f"Executable not found: {raw}")
        return found
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise PipelineError(f"Executable is not runnable: {resolved}")
    return str(resolved)


@dataclass(frozen=True)
class OpenPIPipeline:
    """The ordinary OpenPI stats/train/serve pipeline.

    It intentionally knows nothing about Axis tasks. Environment adapters bind
    dataset and simulator contracts before commands reach this class.
    """

    project_root: Path
    openpi_root: Path
    environ: dict[str, str]
    dry_run: bool = False

    @classmethod
    def create(
        cls,
        project_root: str | Path,
        *,
        openpi_root: str | Path | None = None,
        environ: Mapping[str, str] | None = None,
        dry_run: bool = False,
    ) -> OpenPIPipeline:
        project = Path(project_root).expanduser().resolve()
        values = dict(os.environ if environ is None else environ)
        resolved_openpi = resolve_openpi_root(
            project,
            raw=openpi_root,
            environ=values,
            require=True,
        )
        values["OPENPI_ROOT"] = str(resolved_openpi)
        return cls(
            project_root=project,
            openpi_root=resolved_openpi,
            environ=values,
            dry_run=dry_run,
        )

    def with_environment(self, updates: Mapping[str, str]) -> OpenPIPipeline:
        values = dict(self.environ)
        values.update({name: str(value) for name, value in updates.items()})
        return replace(self, environ=values)

    def _python_prefix(self) -> list[str]:
        configured = self.environ.get("OPENPI_PYTHON")
        if configured:
            return [_resolve_executable(configured, base_dir=self.openpi_root)]

        local_python = self.openpi_root / ".venv/bin/python"
        if local_python.is_file() and os.access(local_python, os.X_OK):
            return [str(local_python)]

        uv = shutil.which("uv")
        if uv is not None:
            return [uv, "run"]

        python = shutil.which("python3") or shutil.which("python")
        if python is not None:
            return [python]
        raise PipelineError(
            "Could not find OpenPI Python. Set OPENPI_PYTHON or install uv/python."
        )

    def command(self, program: str, arguments: Sequence[str]) -> list[str]:
        try:
            relative_script = OPENPI_PROGRAMS[program]
        except KeyError as exc:
            raise PipelineError(f"Unknown OpenPI program: {program}") from exc
        script = self.openpi_root / relative_script
        if not script.is_file():
            raise PipelineError(f"OpenPI entrypoint is missing: {script}")
        return [*self._python_prefix(), str(relative_script), *map(str, arguments)]

    @staticmethod
    def train_mode_arguments(mode: str) -> list[str]:
        if mode not in TRAIN_MODES:
            raise PipelineError(
                f"Training mode must be one of {sorted(TRAIN_MODES)}, got {mode!r}"
            )
        return [] if mode == "none" else [f"--{mode}"]

    def _announce(self, label: str, command: Sequence[str]) -> None:
        print(f"[openpi:{label}] {shlex.join(list(command))}", flush=True)

    def run(self, label: str, command: Sequence[str]) -> int:
        self._announce(label, command)
        if self.dry_run:
            return 0
        completed = subprocess.run(
            list(command),
            cwd=self.openpi_root,
            env=self.environ,
            check=False,
        )
        if completed.returncode != 0:
            raise subprocess.CalledProcessError(completed.returncode, list(command))
        return 0

    def replace(self, label: str, command: Sequence[str]) -> int:
        """Replace the CLI process so signals reach training/serving directly."""

        self._announce(label, command)
        if self.dry_run:
            return 0
        os.chdir(self.openpi_root)
        os.execvpe(command[0], list(command), self.environ)
        raise AssertionError("os.execvpe returned unexpectedly")


def find_openpi_config_names(openpi_root: Path) -> list[str]:
    config_py = openpi_root / "src/openpi/training/config.py"
    if not config_py.is_file():
        return []
    text = config_py.read_text(encoding="utf-8")
    return sorted(set(re.findall(r'name="([^"]+)"', text)))


def openpi_readiness_report(project_root: Path, openpi_root: Path) -> dict[str, object]:
    config_names = find_openpi_config_names(openpi_root)
    return {
        "openpi_root": str(openpi_root),
        "exists": openpi_root.is_dir(),
        "pyproject": (openpi_root / "pyproject.toml").is_file(),
        "train_script": (openpi_root / OPENPI_PROGRAMS["train"]).is_file(),
        "compute_norm_stats_script": (
            openpi_root / OPENPI_PROGRAMS["norm-stats"]
        ).is_file(),
        "serve_policy_script": (openpi_root / OPENPI_PROGRAMS["serve"]).is_file(),
        "uv_available": shutil.which("uv") is not None,
        "axis_related_configs": [name for name in config_names if "axis" in name],
        "libero_related_configs": [name for name in config_names if "libero" in name],
        "patch": str(project_root / "openpi_patches/axis_train_openpi.patch"),
    }


__all__ = [
    "OPENPI_PROGRAMS",
    "OpenPIPipeline",
    "PipelineError",
    "find_openpi_config_names",
    "openpi_readiness_report",
    "resolve_openpi_root",
]
