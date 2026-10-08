---
name: prithvi-downscale
description: Train and run Prithvi WxC / Prithvi-UNet downscaling (32 km NARR or MERRA-2 to 800 m PRISM precipitation, Tmax, Tmin) through the prithvi-wxc-downscaling MCP tools — build a localized YAML config, preprocess, compute scalars, fine-tune, run tiled inference, and evaluate against PRISM as tracked background jobs. Use when the user asks to downscale, train, fine-tune, run inference, or change a downscaling config (region, dates, variables, epochs, GPUs).
---

# Downscale with Prithvi WxC

Background: [downscaling overview](../../concepts/downscaling-overview.md), [YAML data model](../../concepts/dataset-yaml-model.md).

1. `list_available_configs`, then `read_yaml_config` on the closest base (`MERRA_PRISM.yaml`, `NARR_PRISM.yaml`, or `NARR_PRISM_subdomain.yaml` when NARR refinement may follow; `MERRA_PRISM_smoke.yaml` for a quick MERRA test). For a quick NARR test, do not shorten the user's case: make a separate `<case>_smoke` case as in the `prithvi-refine` skill's quick check.
2. `create_custom_yaml` with that base plus the user's changes (dates, bbox, `predictor_variables`, `target_variables`, `num_epochs`, `num_gpus`, `case_name`). Always do this, even with no changes: it localizes input paths to this machine. Use a new `output_name`; never overwrite a config.
3. `preflight_check` with `stage: training`. If anything is missing, use the `prithvi-data` skill, then re-check.
4. Confirm `num_epochs`, `num_gpus`, and date ranges with the user, then `run_training_pipeline`. It queues, in the training repo's order: training preprocessing → training-only scalars → validation/inference preprocessing → fine-tuning → tiled inference → evaluation (`evaluate_prism_inference.py`), skipping stages whose outputs exist (`queue_inference`, `evaluate` turn the last two off). Inference-only on an existing checkpoint: `start_inference_job` with `checkpoint`; evaluation-only: `start_evaluation_job`.
   For NARR the user can add diffusion / flow-matching residual refinement on top: follow the `prithvi-refine` skill.
5. Monitor with `get_job_status` / `list_jobs` / `get_gpu_status`. Jobs wait for free GPUs automatically. Report job ids and log paths; `cancel_job` only when the user asks.
6. When evaluation finishes, report its metrics (CSV, JSON and PNG under `path_experiment/comparison_plots/<case_name>/`), hand off to `prithvi-analyze`, and point the user at the run manifests (`get_run_manifest`) for provenance.

## Rules

- If scalars are recomputed, downstream training and inference must be re-run; say so.
- Scalars are computed from training products only, so they always run after training preprocessing.
- Stochastic residual refinement (diffusion / flow matching on `y_true - y_hat`) is NARR-only and needs the `stochastic_refinement` code variant; see `prithvi-refine`. MERRA-2 runs the deterministic pipeline only.
- Do not chain shell commands to train; use the MCP tools so every run gets a manifest.
