# AXIS LeRobot Format

The AXIS OpenPI patch expects a local root containing task-level LeRobot-v3
datasets. Each task directory should be named:

```text
task_<id>_isaac_state_train
```

## Directory Layout

```text
$AXIS_CLEAN_ROOT/
  task_0001_isaac_state_train/
    meta/info.json
    meta/stats.json
    meta/tasks.parquet
    meta/episodes/chunk-000/file-000.parquet
    data/chunk-000/file-000.parquet
    videos/observation.images.third_person/chunk-000/file-000.mp4
    videos/observation.images.wrist/chunk-000/file-000.mp4
```

## Required Columns

Frame parquet files under `data/` must contain:

- `observation.state`
- `action`
- `timestamp`
- `frame_index`
- `episode_index`
- `task_index`

Episode parquet files under `meta/episodes/` must contain:

- `dataset_to_index`
- `videos/observation.images.third_person/chunk_index`
- `videos/observation.images.third_person/file_index`
- `videos/observation.images.third_person/from_timestamp`
- `videos/observation.images.wrist/chunk_index`
- `videos/observation.images.wrist/file_index`
- `videos/observation.images.wrist/from_timestamp`

## Numeric Layout

Raw AXIS state/action are 9D:

```text
[left_gripper_qpos, right_gripper_qpos, arm_joint_1, ..., arm_joint_7]
```

The OpenPI AXIS loader converts to DROID-style 8D:

```text
[arm_joint_1, ..., arm_joint_7, gripper_closedness]
```

Arm actions are computed as velocity-style deltas:

```text
(target_arm_qpos - current_arm_qpos) / AXIS_CLEAN_DROID_VELOCITY_SCALE
```

The default scale is `1 / 15`. Gripper closedness maps finger width to:

```text
0 = open
1 = closed
```

## Useful Environment Variables

- `AXIS_CLEAN_ROOT`: root containing `task_*_isaac_state_train` directories.
- `OPENPI_AXIS_CLEAN_TASK_IDS`: comma-separated task ids to include.
- `OPENPI_AXIS_CLEAN_MAX_TASKS`: cap number of task directories for smoke tests.
- `OPENPI_AXIS_CLEAN_MAX_SAMPLES_PER_TASK`: cap sampled frames per task.
- `OPENPI_AXIS_CLEAN_VIDEO_CONTAINER_CACHE_SIZE`: PyAV container cache size.
- `AXIS_CLEAN_DROID_VELOCITY_SCALE`: arm target delta scale, default `1/15`.
- `AXIS_CLEAN_DROID_DOWNSAMPLE_RATIO`: frame downsample ratio, default `2`.

## Validation

```bash
python data/validate_lerobot.py \
  --dataset-path "$AXIS_CLEAN_ROOT/task_0001_isaac_state_train" \
  --expected-action-dim 9 \
  --output reports/task_0001_raw_9d \
  --decode-video
```

The validator checks metadata, required columns, episode/frame consistency,
finite numeric arrays, gripper dynamic range, and optional video decodability.

