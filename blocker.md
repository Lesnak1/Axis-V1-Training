# Blockers

## B001: Fresh Environment Missing `uv`

- Status: fixed
- Evidence: in a fresh clone on `Axis-Main`, `conda run -n axis-train-infra-repro uv sync` failed with `uv: command not found`.
- Fix: added `uv>=0.7` to `environment.yml`; a second clean env (`axis-train-infra-repro-clean`) installed `uv 0.11.26` from the environment file and passed pytest.

## B002: README Used `mamba` On A Host Without `mamba`

- Status: fixed
- Evidence: `Axis-Main` has `conda` but no `mamba`.
- Fix: changed the Quick Start environment command to `conda env create -f environment.yml`.

## B003: GPU Resources Unavailable For Pi0.5 Training Smoke

- Status: fixed
- Evidence: earlier fresh-clone reproduction on `Axis-Main` was blocked because all eight A100 GPUs were heavily loaded, with about 61-75GB/80GB memory in use and high utilization.
- Fix: reran from a fresh GitHub clone at `3440bf85989194b13842142bac2681c8a1ee4e30` in `/home/shmqd/workspace/dhhuang/axis-train-release-repro-v3-main`, using `CUDA_VISIBLE_DEVICES=0`, `XLA_PYTHON_CLIENT_PREALLOCATE=false`, local Pi0.5-DROID checkpoint/assets, batch size 1, and one training step.
- Verification: `reports/train_smoke_gpu0.log` shows `TRAIN_SMOKE_DONE`, `Step 0: grad_norm=10.3751, loss=1.5749, param_norm=1938.4988`, and a saved checkpoint at `checkpoints_smoke/pi05_axis_clean_local_droid_lora/axis_lora_smoke_gpu0_v3/0`.

## B004: LIBERO Eval Dependencies Missing After Base OpenPI Sync

- Status: fixed
- Evidence: in the fresh OpenPI `.venv`, `python -c "import libero"` and `python examples/libero/main.py --help` failed with `ModuleNotFoundError: No module named 'libero'`.
- Impact: LIBERO evaluation cannot run from the base `uv sync` environment alone.
- Fix: added explicit LIBERO submodule/requirements setup to `docs/LIBERO_FINETUNE_EVAL.md`, added `training/run_libero_eval.sh`, and verified `examples/libero/main.py --help` in the fresh LIBERO environment.
- Verification: in the fresh GitHub clone at `3440bf85989194b13842142bac2681c8a1ee4e30`, `reports/libero_setup.log` shows `LIBERO_SETUP_DONE`. `reports/libero_eval_smoke.log` shows `LIBERO_EVAL_SMOKE_DONE`, `Total success rate: 1.0`, and `Total episodes: 10` for `libero_spatial` with one rollout per task using the local `pi05_libero` checkpoint. Ten success videos were written to `reports/libero_videos_smoke/`.

## B005: LIBERO Pi0.5 Full Fine-Tuning Smoke OOM On Busy `Axis-Main`

- Status: fixed
- Evidence: in the fresh GitHub clone at `3440bf85989194b13842142bac2681c8a1ee4e30`, `physical-intelligence/libero` metadata loaded successfully (`fps=10`, `tasks=40`, features include `image`, `wrist_image`, `state`, and `actions`). `reports/libero_train_smoke_gpu0.log` shows `pi05_libero` loaded LIBERO norm stats, initialized the dataloader with batch size 1, and restored local `pi05_base` params successfully.
- Blocker: the full `pi05_libero` run failed during JAX train-state initialization with `RESOURCE_EXHAUSTED: Out of memory while trying to allocate 143327232 bytes`. XLA reported it could only reduce memory to `37.48GiB`, while GPU0 had only about `4.35GiB` free at launch because unrelated jobs occupied all A100s.
- Fix: added `pi05_libero_lora`, a lower-memory Pi0.5 LoRA fine-tuning config for LIBERO LeRobot data, to `openpi_patches/axis_train_openpi.patch`.
- Verification: `reports/libero_lora_train_smoke_gpu0.log` shows `pi05_libero_lora` loaded LIBERO norm stats, initialized the dataloader, restored local `pi05_base` params, completed one training step with `Step 0: grad_norm=0.4534, loss=0.0291, param_norm=1803.7701`, and saved a checkpoint at `checkpoints_smoke/pi05_libero_lora/libero_pi05_lora_smoke_gpu0_v3/0`.
