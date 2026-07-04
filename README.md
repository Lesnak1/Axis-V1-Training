# AXIS Train Infra

Training-only release for fine-tuning OpenPI Pi0.5 on AXIS-format LeRobot
datasets.

## Contents

- [1. Quick Start](#1-quick-start)
- [2. Training Parameters](#2-training-parameters)

## 1. Quick Start

Create the infra environment:

```bash
conda env create -f environment.yml
conda activate axis-train-infra
pip install -e .
python -m pytest -q
```

Install OpenPI and apply the AXIS training patch:

```bash
mkdir -p third_party
git clone https://github.com/physical-intelligence/openpi.git third_party/openpi
cd third_party/openpi
git checkout c23745b5ad24e98f66967ea795a07b2588ed6c79
git apply --check ../../openpi_patches/axis_train_openpi.patch
git apply ../../openpi_patches/axis_train_openpi.patch
uv sync
```

Set the dataset path. `AXIS_CLEAN_ROOT` should contain AXIS-format LeRobot
task folders, for example `task_0001_isaac_state_train`.

```bash
cd /path/to/AXIS-Train
export OPENPI_ROOT="$PWD/third_party/openpi"
export AXIS_CLEAN_ROOT="/path/to/axis/lerobot/root"
export OPENPI_AXIS_CLEAN_MAX_TASKS=1
export OPENPI_AXIS_CLEAN_MAX_SAMPLES_PER_TASK=64
```

Start one LoRA training run on Pi0.5:

```bash
OPENPI_TRAIN_MODE=overwrite \
bash training/run_train_openpi.sh \
  pi05_axis_clean_local_droid_lora \
  axis_lora_smoke \
  --num-train-steps 100 \
  --checkpoint-base-dir "$PWD/checkpoints"
```

For full/CPT-style fine-tuning, use `pi05_axis_clean_local_droid_full`.

## 2. Training Parameters

Detailed training parameters are set in the patched OpenPI config:

```text
third_party/openpi/src/openpi/training/config.py
```

The patch that adds the AXIS training configs is:

```text
openpi_patches/axis_train_openpi.patch
```

Main config names:

```text
pi05_axis_clean_local_droid_lora   # Pi0.5 LoRA fine-tuning
pi05_axis_clean_local_droid_full   # Pi0.5 full / CPT-style fine-tuning
pi05_libero_lora                   # Pi0.5 LoRA fine-tuning on LIBERO
pi05_libero                        # Pi0.5 full fine-tuning on LIBERO
```

Common fields to edit in `config.py`: `num_train_steps`, `batch_size`,
`lr_schedule`, `optimizer`, `save_interval`, `keep_period`, `fsdp_devices`,
and `weight_loader`.

Runtime overrides go through `training/run_train_openpi.sh`, for example:

```bash
bash training/run_train_openpi.sh CONFIG_NAME EXP_NAME --num-train-steps 1000
```

Dataset/debug controls are environment variables: `AXIS_CLEAN_ROOT`,
`OPENPI_AXIS_CLEAN_TASK_IDS`, `OPENPI_AXIS_CLEAN_MAX_TASKS`, and
`OPENPI_AXIS_CLEAN_MAX_SAMPLES_PER_TASK`.
