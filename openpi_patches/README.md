# OpenPI Patch

`axis_train_openpi.patch` is a sanitized patch generated against:

```text
physical-intelligence/openpi@c23745b5ad24e98f66967ea795a07b2588ed6c79
```

It adds only release-scoped AXIS training support:

- `LeRobotAxisCleanDroidDataConfig`
- local AXIS-format LeRobot loader
- 9D AXIS qpos to 8D DROID-style online adaptation
- lossless arm-velocity conversion without an internal `[-1, 1]` clip
- norm-stat support for the local AXIS loader
- a separate inference-only repack transform, so live observations never need
  a ground-truth `actions` key
- `pi05_axis_clean_local_droid_lora`
- `pi05_axis_clean_local_droid_full`
- `pi05_axis_clean_local_droid_action_expert`
- `pi05_libero_lora`
- focused AXIS loader, horizon, assets-path, and policy-repack tests

Apply from inside an OpenPI checkout:

```bash
git checkout c23745b5ad24e98f66967ea795a07b2588ed6c79
git apply --check /path/to/axis-train-infra/openpi_patches/axis_train_openpi.patch
git apply /path/to/axis-train-infra/openpi_patches/axis_train_openpi.patch
```

The patch intentionally excludes private data paths, in-domain experiment
configs, generated artifacts, and unrelated RoboVerse experiment loaders.
