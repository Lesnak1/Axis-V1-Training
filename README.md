# AXIS Train Infra

OpenPI Pi0.5 fine-tuning on replay-qualified AXIS LeRobot data.

The repository has one core design: the normal OpenPI
`norm-stats → train → serve` pipeline is unchanged, and thin Axis environment
adapters bind a task dataset, hashed action contract, and the shared
Learn-from-Failure evaluation runtime. See
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the ownership boundaries.

PostgreSQL/S3 acquisition, trajectory qualification, MuJoCo/Isaac Sim scene
rendering, and raw `packed_state_v1` replay remain upstream. The only supported
training input is a local, replay-qualified LeRobot export.

## Setup

```bash
conda env create -f environment.yml
conda activate axis-train-infra
pip install -e .
pytest -q
```

Install the pinned OpenPI checkout and apply the release patch:

```bash
mkdir -p third_party
git clone https://github.com/physical-intelligence/openpi.git third_party/openpi
git -C third_party/openpi checkout c23745b5ad24e98f66967ea795a07b2588ed6c79
git -C third_party/openpi apply --check "$PWD/openpi_patches/axis_train_openpi.patch"
git -C third_party/openpi apply "$PWD/openpi_patches/axis_train_openpi.patch"
cd third_party/openpi && uv sync && cd ../..

axis-train doctor --openpi-root third_party/openpi
```

`axis-train` is installed by `pip install -e .`. From an uninstalled source
checkout, replace it with `python scripts/axis.py` in every example.

## Fine-tune one Axis task

The clean root must contain exactly one selected task directory such as
`task_3801_mujoco_state_train` (zero-padded IDs are also accepted). One command
binds the task, verifies `meta/action_transform_spec.json`, computes norm stats
when requested, and launches the ordinary OpenPI trainer:

```bash
export OPENPI_PYTHON="$PWD/third_party/openpi/.venv/bin/python"

axis-train finetune \
  3801 /path/to/axis-clean task3801_seed42 \
  --openpi-root third_party/openpi \
  --config pi05_axis_clean_local_droid_action_expert \
  --init-params /data/openpi/openpi-assets/checkpoints/pi05_droid/params \
  --compute-norm-stats \
  --num-train-steps 5000 --batch-size 1 --fsdp-devices 1 \
  --save-interval 5000 --keep-period 5000 --seed 42
```

Default output roots are colocated with the clean dataset:

```text
<axis-clean>/openpi-assets   normalization assets
<axis-clean>/openpi-data     OpenPI data/cache home
<axis-clean>/checkpoints     checkpoints
```

Override them with `--assets-base-dir`, `--data-home`, and
`--checkpoint-base-dir`. `--mode overwrite|resume|none` controls OpenPI's run
mode. All remaining arguments are passed unchanged to `scripts/train.py`.

The available Axis configs in the patch are:

- `pi05_axis_clean_local_droid_lora`: Pi0.5 LoRA fine-tuning.
- `pi05_axis_clean_local_droid_action_expert`: train the action expert and
  action/time projections while freezing vision/language backbones; this is the
  verified one-A100 task 3801 path.
- `pi05_axis_clean_local_droid_full`: full/CPT-style fine-tuning; the verified
  single-A100 discriminator OOMed and this mode needs more memory/sharding.

Training completion is not task success. The executed task 3801 run completed
5,000 steps and reloaded its checkpoint, but failed its one-scene closed-loop
Checker gate. The exact identities and results are in
[`docs/TASK3801_LFF_ALIGNMENT.md`](docs/TASK3801_LFF_ALIGNMENT.md).

## Standard OpenPI stages

The underlying stages remain independently callable. Axis configs resolve the
dataset adapter from environment variables; non-Axis configs do not require any
Axis state.

```bash
export OPENPI_ROOT="$PWD/third_party/openpi"
export AXIS_CLEAN_ROOT=/path/to/axis-clean
export OPENPI_AXIS_CLEAN_TASK_IDS=3801
export AXIS_PI05_ASSETS_BASE_DIR=/path/to/openpi-assets

axis-train norm-stats pi05_axis_clean_local_droid_lora
axis-train train pi05_axis_clean_local_droid_lora axis_lora_smoke \
  --num-train-steps 100
```

For example, the generic pipeline can launch a non-Axis config without an
Axis environment:

```bash
axis-train train pi05_libero libero_smoke --num-train-steps 100
```

## Serve and evaluate

Serving loads the action contract bundled in the selected checkpoint and
places OpenPI global options before Tyro's policy subcommand:

```bash
axis-train serve \
  pi05_axis_clean_local_droid_action_expert \
  /path/to/checkpoint/4999 \
  --port 8000
```

Task profiles under `configs/tasks/` freeze task assets, camera, scene split,
and safe development defaults. The task adapter invokes the exact
Learn-from-Failure materializer/state-restore/Checker modules and records their
source hashes in the receipt:

```bash
axis-train task websocket 3801 /path/to/new/eval-receipt \
  --learn-from-failure-root /path/to/learn-from-failure-at-scale \
  --eval-python /path/to/learn-from-failure/.venv/bin/python \
  --checkpoint /path/to/checkpoint/4999 \
  --port 8000 --replan-interval 1
```

WebSocket evaluation defaults to the frozen `development_evaluation` role.
Final-test access requires `--allow-final-test` and must not be used for model
or checkpoint selection.

The recorded-DROID positive control uses the same task adapter:

```bash
axis-train task recorded_droid 3801 /path/to/positive-control-receipt \
  --learn-from-failure-root /path/to/learn-from-failure-at-scale \
  --eval-python /path/to/learn-from-failure/.venv/bin/python \
  --recorded-actions /path/to/episode.parquet \
  --action-transform-spec /path/to/action_transform_spec.json \
  --scene-index 0 --max-controls 80 --replan-interval 250 --no-video
```

`AxisDroidPolicyAdapter` is the live observation/action boundary. It builds the
dotted DROID observation keys, converts 9D Axis qpos into the model's 8D state,
and rebases each returned 8D action on newly measured qpos:

```python
from axis_train.policy import AxisDroidPolicyAdapter

adapter = AxisDroidPolicyAdapter.from_action_transform_spec(
    "/path/to/checkpoint/assets/local/axis_clean/action_transform_spec.json"
)
observation = adapter.build_openpi_observation(
    qpos=current_qpos_9d,
    exterior_image=front_rgb_uint8,
    wrist_image=wrist_rgb_uint8,
    prompt=task_prompt,
)
target_qpos_9d = adapter.droid_action_to_qpos_target(
    current_qpos_9d,
    returned_actions_horizon_x8[0],
)
```

The evaluator resolves those nine logical joint targets to MuJoCo actuators by
joint name; qpos and ctrl array indexes are never assumed to be identical.

## Data boundary and utilities

The expected LeRobot layout and 9D absolute-qpos to 8D DROID action semantics
are documented in [`docs/AXIS_LEROBOT_FORMAT.md`](docs/AXIS_LEROBOT_FORMAT.md).
An audited LeRobot-v2.1 archive can be exposed without copying it:

```bash
python data/import_legacy_lerobot_v21.py \
  --source-dataset /path/to/audited_lerobot_v21 \
  --output-root /path/to/axis-clean \
  --task-id 3801 --action-horizon 15
```

Other scripts under `data/` are one-shot validation/conversion utilities, not
stages of the train pipeline. The shell files under `training/run_*.sh` and
`evaluation/run_axis_task.sh` are compatibility shims that delegate to
`scripts/axis.py`; new automation should call `axis-train` directly.
