# Upstream Provenance

- source: `git@github.com:AxisAIOrg/Axis-V1-Training.git`
- imported commit: `4ffc886e35f80c748623e3044e27e0acf258c14e`
- clone date: `2026-09-01`
- OpenPI base: `physical-intelligence/openpi@c23745b5ad24e98f66967ea795a07b2588ed6c79`

This directory is a direct Git clone of the source repository. The local
hardening below is layered on the pinned upstream commit; OpenPI remains a
separately pinned checkout patched by `openpi_patches/axis_train_openpi.patch`.

Local hardening applied during import:

- one global gripper calibration across every parquet shard;
- explicit exclusion of terminal action padding from calibration/statistics;
- video decode failures participate in the dataset validation result;
- episode split ratios fail closed when invalid;
- observation-renderer provenance is retained in
  `task_<id>_<observation_renderer>_state_train`, independently of MuJoCo
  physics/Checker qualification;
- OpenPI sample bases whose full horizon touches terminal padding are removed;
- generated local norm stats and training resolve to the same assets path;
- live policy observations use an inference-only repack that does not require
  ground-truth actions;
- the serving wrapper preserves Tyro's global-option/subcommand ordering;
- an explicit 8D DROID to 9D AXIS policy adapter mirrors the training action
  formula and integrates action chunks step by step;
- a task-ID training runner that freezes one dataset and its hashed action
  contract across norm stats, training, checkpoint assets, and serving;
- a closed-loop task entrypoint that imports the scene materializer, state
  restore, and Checker directly from Learn from Failure and records their
  source hashes in each receipt;
- one `axis-train` CLI that separates the generic OpenPI process pipeline from
  Axis dataset/task environment adapters; former shell entrypoints are thin
  compatibility shims;
- a frozen task-profile boundary for camera/assets/scene defaults, with task
  3801 represented in `configs/tasks/3801.json`;
- clean OpenPI patch application, lint, loader, norm-stat, transformed-batch,
  checkpoint reload, and WebSocket inference regression tests.

This repository does not query Axis PostgreSQL or download S3 payloads. Those
qualification responsibilities remain in the Access/replay pipeline, and
only a replay-qualified final selection may cross the training boundary. For
evaluation, this repository deliberately reuses the configured
Learn-from-Failure task runtime instead of carrying a second task/Checker
implementation.
