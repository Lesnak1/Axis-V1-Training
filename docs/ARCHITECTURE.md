# Architecture

This repository is deliberately a thin integration around OpenPI. It does not
own trajectory acquisition, scene reconstruction, or a second copy of the Axis
task runtime.

```text
replay-qualified LeRobot dataset
              │
              ▼
   AxisDatasetEnvironment
   task selection + hashed action spec + paths
              │
              ▼
       OpenPIPipeline
       norm-stats → train → serve
              │
              ▼
    AxisTaskEnvironment
    shared Learn-from-Failure scene/reset/Checker runtime
```

The two environment adapters only resolve and validate contracts. The ordinary
OpenPI pipeline owns process selection, command ordering, train mode, signal
forwarding, and execution. This keeps LIBERO or another non-Axis OpenPI config
usable without an `AXIS_CLEAN_ROOT`, while an Axis config fails closed unless
its dataset and action contract are complete.

## Authoritative entrypoint

`axis-train` (or `python scripts/axis.py` from a source checkout) is the only
authoritative command surface:

| Command | Owner | Purpose |
| --- | --- | --- |
| `doctor` | pipeline | Check the pinned OpenPI checkout and entrypoints |
| `spec` | Axis data adapter | Resolve and verify one hashed action contract |
| `norm-stats` | pipeline | Run OpenPI normalization |
| `train` | pipeline | Run any OpenPI config |
| `finetune` | Axis data adapter + pipeline | Bind one task dataset, optionally normalize, then train |
| `serve` | pipeline + checkpoint adapter | Serve a checkpoint with its bundled action contract |
| `task` | Axis task adapter | Invoke the shared Learn-from-Failure task runtime |
| `libero-eval` | LIBERO adapter | Invoke OpenPI's isolated LIBERO runtime |

The old files under `training/run_*.sh` and `evaluation/run_axis_task.sh` are
compatibility shims only. They contain no path, config, or task logic and can
be removed after external callers migrate.

## Source boundaries

```text
axis_train/
  pipeline.py             generic OpenPI process pipeline
  axis_environment.py     Axis dataset and task adapters
  cli.py                  command composition only
  action_transform.py     hashed 9D Axis ↔ 8D DROID contract
  policy/                 live policy-side action/observation adapter
  data/                   offline LeRobot validation/import libraries

scripts/axis.py            source-checkout entrypoint
configs/tasks/             frozen task profiles; environment/CLI can override paths
evaluation/*.py            heavy closed-loop/diagnostic executors
data/*.py                  one-shot dataset utility entrypoints
openpi_patches/            pinned OpenPI implementation delta
training/*.sh              legacy compatibility shims
```

Task profiles contain task assets, camera, scene split, and safe evaluation
defaults. They do not contain model hyperparameters. OpenPI config names and
model defaults remain canonical in the pinned OpenPI patch, so there is no
second YAML training config to drift.

## Rendering boundary

The reference repository cloned at
`AxisAIOrg/axis-training-infra@e137743670825f8822ff869ff9983f33ab349382`
has three commits. Commit `07378fe` imports the complete Learn-from-Failure
low-dimensional training/evaluation system. The latest commit `e137743` adds a
separate reproducible rendering subsystem: MuJoCo and Isaac Sim backends,
content-addressed scenes, MJCF-to-USD conversion, domain randomization,
multi-GPU scheduling, manifests, and checked videos.

That renderer is not part of a policy train step. It restores recorded
`packed_state_v1` states without policy inference or physics stepping, and its
Isaac Sim dependencies live in a dedicated `.venv-render` because they conflict
with the training environment. This repository therefore does not vendor that
implementation. Rendered/replay-qualified LeRobot data may cross the input
boundary; simulator packages and scene-generation code do not.

The reference clone is also not a byte-identical replacement for the currently
verified `learn-from-failure-at-scale@be28d12` task runtime. Five of the seven
runtime files hashed by `closed_loop_axis.py` match; `backend_replay.py` and
`mujoco_model_loader.py` differ. Those two files were unchanged by the latest
rendering commit, so the divergence belongs to the earlier extraction/import.
`axis-train task` therefore keeps the runtime root explicit and receipts its
actual source hashes instead of switching repositories by name.

If Isaac Sim becomes a required data producer here, add it as another upstream
renderer adapter with its own environment and receipt. Do not add Isaac imports
or renderer flags to `OpenPIPipeline`.

## Invariants

- `finetune` selects exactly one `task_<id>_*_state_train` directory, including
  zero-padded task names.
- Norm stats, training, checkpoint assets, serving, and evaluation use the same
  verified `ActionTransformSpec` hash.
- Generic `train` does not know about task assets or MuJoCo.
- `task` uses the exact Learn-from-Failure scene materializer, state restore,
  controller, and Checker; their source hashes remain in the evaluation receipt.
- Final-test access stays explicit and is never a checkpoint-selection default.
