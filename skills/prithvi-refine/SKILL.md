---
name: prithvi-refine
description: Add stochastic residual refinement to NARR Prithvi-UNet downscaling — a diffusion or flow-matching model of the residual y_true - y_hat trained on top of the frozen deterministic Prithvi model, giving a bias-corrected ensemble (mean and spread) instead of one field. Use when the user asks for diffusion, flow matching, generative or stochastic refinement, residual correction, ensembles, or uncertainty for NARR downscaling. NARR only; MERRA-2 has no refinement.
---

# Stochastic residual refinement (NARR)

Background: [refinement playbook](../../playbooks/refine-narr.md), [downscaling overview](../../concepts/downscaling-overview.md).

Phase 1 is the deterministic Prithvi-UNet fine-tune (`prithvi-downscale`), giving the baseline y_hat. Phase 2 freezes it and learns the residual epsilon = y_true - y_hat with a conditional generative model; at inference every sampled residual r_hat gives one ensemble member y_hat + r_hat.

1. `check_environment`. Refinement needs the `stochastic_refinement` code variant; if the checkout is not at that pin, `setup_code` with `variant: stochastic_refinement` (it also contains everything the deterministic NARR/MERRA pipelines use).
2. Pick a deterministic NARR base config that defines `dates.validation` (`NARR_PRISM_subdomain.yaml`, or `create_custom_yaml` with `validation_start` / `validation_end`). Training, validation and inference ranges must not overlap; `create_custom_yaml` and the pipeline reject configs that break this. Never pass a `NARR_PRISM_<type>.yaml` template as the base.
3. Ask which refiner head, if the user did not say:
   - `diffusion_unet` / `flow_matching_unet`: convolutional UNet refiner with a bottleneck attention block; `refiner_attention: false` gives the variant without attention blocks (own config, checkpoints and outputs)
   - `diffusion_transformer` / `flow_matching_transformer`: attention-based transformer refiner
4. `preflight_check` with `stage: training`, then confirm with the user: base config, head, ensemble size (default 10), refiner epochs, GPUs, and whether Phase 1 must be trained. If `checkpoint_dir/<case_name>/last.ckpt` already exists and they only want refinement, use `train_phase1: false`.
5. `run_training_pipeline` with `config_path`, `refinement_type`, and optionally `refiner_attention`, `ensemble_size`, `refinement_epochs`, `num_gpus`, `train_phase1`. After the deterministic stages (preprocessing → scalars → training → tiled inference → evaluation) it queues:
   1. `phase1_cache`: frozen Phase 1 run once over training and validation dates; caches y_hat and training-only residual statistics
   2. `refinement_training`: learns the residual; Prithvi stays frozen; resumes from `last.ckpt` if present
   3. `refinement_inference`: daily deterministic, residual, members, ensemble mean and spread
   4. `refinement_evaluation`: deterministic vs refined on the same dates and grid cells
   It writes `custom_<base>_<type>.yaml` next to the base config and returns its path.
6. Monitor with `get_job_status` / `list_jobs`. To re-run only the ensemble (e.g. a different `ensemble_size`) or only the comparison, use `start_refinement_inference_job` / `start_refinement_evaluation_job` with that refinement YAML.
7. Report the `refinement_evaluation` metrics (RMSE, bias, correlation, spread and extremes) for deterministic vs refined, then hand off to `prithvi-analyze`.

## Rules

- MERRA-2 has no refinement. Offer the deterministic pipeline instead.
- Do not say refinement improved the result until `refinement_evaluation` has finished, and report where the refined ensemble is worse as well as better.
- Residual statistics come from training dates only; never refit them on validation or inference dates.
- Preprocessed data a case directory links to from another checkout is read-only: the job manager refuses preprocessing and scalars jobs that would write through it. Use a new `case_name` to preprocess new data.
