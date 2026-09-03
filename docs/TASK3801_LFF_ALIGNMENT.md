# Task 3801 parity with Learn from Failure

Status: executed engineering ledger, 2026-09-01. Real-data fine-tuning and the
shared task invocation passed their engineering gates. The resulting model
did not pass the one-scene Checker gate. Training completion, offline action
error, one-scene closed-loop success, development-panel success, and
generalization remain separate claims.

## Frozen identities

| Component | Frozen value |
| --- | --- |
| Axis-V1-Training upstream | `AxisAIOrg/Axis-V1-Training@4ffc886e35f80c748623e3044e27e0acf258c14e` |
| OpenPI base | `physical-intelligence/openpi@c23745b5ad24e98f66967ea795a07b2588ed6c79` |
| Task | `3801`, “Pick up the tofu block and stack it on the plate.” |
| H1 training dataset | `/dev/shm/axis-task3801-pi05-overfit-rank239-h1-lossless-v1/task_3801_mujoco_state_train` |
| H1 action-spec hash | `d9b62803a56e93f2245961963b145a771346669dbf78b785fcb69d3f42f3abcc` |
| Task contract | `/data/wls_assets/axis/task_3801/task_contract_source_prod_20260826_v1` |
| Task assets | `/data/wls_assets/axis/task_3801/assets/training_scenes_split_v2/assets` |
| Camera contract | `/root/workspace/murphy/wls/.ai/tasks/20260825-0849-DreamZero训练/artifacts/task_3801_camera_contract_level0_single_arm_v2.json` |
| Scene split | `/root/workspace/murphy/wls/.ai/tasks/20260825-0849-DreamZero训练/artifacts/task_3801_scene_split_v2.json` |

The H1 dataset contains 80 real controls and one excluded terminal-padding
row. Its action contract is a lossless 5 Hz mapping from Axis 9D absolute
qpos targets to Pi0.5-DROID 8D actions: seven arm deltas divided by `0.2`, no
arm clipping, plus gripper closedness. Inference rebases every predicted arm
delta on newly measured qpos.

## Train/eval parity matrix

| Dimension | Learn from Failure | This repository | Status |
| --- | --- | --- | --- |
| Scene materialization | `axis_replay_native.asset_materializer` | imports the same module from the configured Learn-from-Failure checkout | invariant |
| Initial-state coercion/restore | `backend_replay.py` and `state_restore.py` | same source files | invariant |
| Checker | task `checker_config.json` through `CheckerManager` | same config and class; source hashes written to every summary | invariant |
| MuJoCo controller | `0.001 s`, 200 `mj_step` calls per control | identical, asserted at runtime | invariant |
| Checker cadence | every control boundary, stop on first success | identical | invariant |
| Task/camera/assets | frozen task 3801 paths above | frozen in `configs/tasks/3801.json` and resolved by `axis-train task` | invariant |
| Development/final split | frozen scene-split roles | same split; final role requires `AXIS_ALLOW_FINAL_TEST=1` | invariant |
| Action execution | name-resolved 9D actuator targets | same; Pi0.5 8D output is inverted using measured qpos first | intentional policy adapter |
| Policy backend | recorded expert or historical policy | recorded expert, recorded-DROID round trip, or OpenPI WebSocket | intentional model boundary |

The first Learn-from-Failure H1 model discriminator used LoRA, global batch
4, four-way FSDP, and 5,000 optimizer steps (about 20,000 sampled examples).
The standalone memory discriminator below uses a fully trainable action
expert, global batch 1, and one GPU. Those are declared training differences;
equal step numbers are not equal sample budgets and their metrics are not
presented as a one-to-one method comparison. Task/controller/Checker parity is
held independently of this model-resource difference.

`axis-train task` delegates to `evaluation/closed_loop_axis.py`, which records the resolved Learn-from-Failure root
and hashes every imported task-runtime source file (asset materialization,
state coercion/restore, model loading, checker payload, and CheckerManager) in
`summary.json`. This makes “same task call” auditable rather than a path-name
assumption.

## Reproduction commands

Apply the patch to the pinned OpenPI checkout, then select exactly one Axis
task with the task-ID runner:

```bash
export OPENPI_ROOT=/dev/shm/axis-openpi-c237/repo
export OPENPI_PYTHON="$OPENPI_ROOT/.venv/bin/python"

axis-train finetune \
  3801 \
  /dev/shm/axis-task3801-pi05-overfit-rank239-h1-lossless-v1 \
  task3801_rank239_h1_action_expert_b1_seed42_5000_v1 \
  --openpi-root "$OPENPI_ROOT" \
  --config pi05_axis_clean_local_droid_action_expert \
  --init-params /data/openpi/openpi-assets/checkpoints/pi05_droid/params \
  --data-home /dev/shm/axis-task3801-pi05-overfit-rank239-h1-lossless-v1/openpi-data \
  --assets-base-dir /dev/shm/axis-task3801-pi05-overfit-rank239-h1-lossless-v1/openpi-assets \
  --checkpoint-base-dir /dev/shm/axis-v1-training-task3801-20260901/checkpoints \
  --compute-norm-stats \
  --num-train-steps 5000 --batch-size 1 --fsdp-devices 1 \
  --save-interval 5000 --keep-period 5000 --seed 42
```

The runner rejects a missing/ambiguous `task_3801_*_state_train` directory,
selects the task explicitly, resolves the hashed action contract, and carries
that contract into norm stats, training, checkpoint assets, serving, and
evaluation.

Run the recorded-DROID positive control through the exact task runtime:

```bash
axis-train task recorded_droid 3801 /path/to/new/positive_control_receipt \
  --learn-from-failure-root /root/workspace/murphy/cotrain/learn-from-failure-at-scale \
  --eval-python /dev/shm/lfas-be28d12-runtime/repo/.venv/bin/python \
  --recorded-actions /data/wls_assets/axis/task_3801/dreamzero/episode_rank239_level0_single_arm_v2/data/chunk-000/episode_000000.parquet \
  --action-transform-spec /dev/shm/axis-task3801-pi05-overfit-rank239-h1-lossless-v1/task_3801_mujoco_state_train/meta/action_transform_spec.json \
  --scene-index 0 --max-controls 80 --replan-interval 250 --no-video
```

For a served checkpoint, use `axis-train task websocket ... --checkpoint ...
--port ...`. Omit `--scene-index` to evaluate the frozen development role; do
not enable the final role during model or hyperparameter selection.

## Executed evidence

| Gate | Result |
| --- | --- |
| Standalone regression | `32 passed`; Ruff and shell syntax passed |
| Patched OpenPI focused regression | `6 passed` including exact action-expert parameter partition |
| Patch identity | forward apply and reverse `git apply --check` passed in a fresh detached worktree at the pinned OpenPI commit; 6 focused tests passed there |
| Recorded-DROID positive control | scene 0, 1/1 Checker success at boundary 80; receipt SHA-256 `1b5b91a1a9a8af8f532512953d9f086f0c62d3c755a7ccb09e95a1e9cef7a54b` |
| Training/evaluation observation parity | exact initial 9D state; third-person/wrist RGB MAE `0.9274`/`1.2482` u8; receipt SHA-256 `83100afbaa345af354adc52edb4af9124f7e8f51a79d18bfc3a3e6d8149e58e6` |
| Full 3.35B single-GPU discriminator | data/init passed; backward OOM after a 64.41 GiB compile estimate, so no checkpoint claim |
| Action-expert parameter partition | 430,098,464 trainable; 2,923,335,408 frozen |
| Action-expert real-data step | loss `1.7023`, grad norm `13.5149`; checkpoint 0 finalized |
| Independent checkpoint reload | params and checkpoint-bundled norm stats restored; WebSocket server accepted real observations |
| Step-0 teacher-forced baseline | 8D first-action MAE `0.69433`; 9D absolute-target MAE `0.12048` over all 80 seen observations |
| Action-expert 5k fine-tune | 5,000 consecutive b1 steps completed in 9m50s; final 500-step mean loss `0.0294`; checkpoint 4999 finalized and independently reloaded |
| Step-4999 ensemble-8 teacher forcing | 8D first-action MAE `0.08045`; arm/gripper MAE `0.08777`/`0.02926`; 9D absolute-target MAE `0.01380`; receipt SHA-256 `095720fa1bdcc29fb10b6a0f3eb54891500a7a42ad47dbb4cc02d277fb4716d4` |
| Step-4999 matched-reset closed loop | scene 0, 0/1 after 100 controls; 3/8 Checker leaves, no contact, no actuator clipping, max object displacement `0.06159 m`; receipt SHA-256 `413162b8df5dd1eaa4fcf966c02bc6269c5808984b392543f9001bf831c3b76e` |

The action-expert result proves a viable real-data fine-tuning path on one
A100, including save/reload and task invocation. It does **not** prove model
task success: the matched H1 reset failed 0/1, so the development and final
panels were not run. The frozen final-test panel remains untouched.

## Artifact identities and durability

The final checkpoint is
`/dev/shm/axis-v1-training-task3801-20260901/checkpoints/pi05_axis_clean_local_droid_action_expert/task3801_rank239_h1_action_expert_b1_seed42_5000_v1/4999`.
Its `_CHECKPOINT_METADATA` SHA-256 is
`1eea4030b729ba749df9c713ae5bb3a831cae0d72d0421e06044b3392ea3a52a`;
the sorted relative-path/content-hash manifest of the `params/` tree hashes to
`a0493146fd4a722c19f6be3690e2bfa4e0cf1497721360e34dda329a74658c52`.
Its bundled action-spec file SHA-256 is
`ceb21fc91c8df181e3db03ba19a26f342754a4bd73c9a2b7ef053d4d702490eb`.
The OpenPI patch SHA-256 is
`8a467bc324d249131ed961f139c518e6bf7a89ef717b9247bdc0298a3be8123a`.

All `/dev/shm` experiment outputs are intentionally ephemeral. Copy the final
checkpoint and receipts to a durable experiment store before machine restart
if they are to be retained; the source code, contracts, and reproduction
commands do not depend on those ephemeral paths.
