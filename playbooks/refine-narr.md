---
type: Playbook
title: Refine NARR downscaling with diffusion / flow matching
description: Phase-2 stochastic residual refinement on the frozen Prithvi-UNet — residual cache, refiner training, ensemble inference, deterministic-vs-refined evaluation via MCP.
tags: [playbook, downscaling, narr, refinement, diffusion, flow-matching, ensemble, mcp]
timestamp: 2026-10-08T00:00:00Z
---

Use when the user asks for diffusion, flow matching, generative or stochastic refinement, residual correction, ensembles, or uncertainty for **NARR** downscaling. MERRA-2 has no refinement; use the [downscale playbook](/playbooks/downscale-wxc.md).

# Idea

The deterministic Prithvi-UNet (Phase 1, fine-tuned by the downscale playbook) gives one field y_hat per day. Phase 2 freezes it and trains a conditional generative model of the residual

    epsilon = y_true - y_hat

with diffusion or flow matching. At inference a residual r_hat is sampled from noise for each member, and y_hat + r_hat is one member of a bias-corrected ensemble. The ensemble mean corrects systematic residual errors; the spread expresses uncertainty.

# Prerequisites

1. `check_environment` reports the `stochastic_refinement` code variant (`setup_code` with `variant: stochastic_refinement` otherwise). It contains `narr_prism_refinement.py`, `narr_prism_phase1_cache.py`, `evaluate_refinement.py` and the four `NARR_PRISM_<type>.yaml` templates.
2. A deterministic NARR base config with `dates.validation` (`NARR_PRISM_subdomain.yaml` uses training 1996–2013, validation 2014–2015, inference 2016–2025), routed through `create_custom_yaml` and passing `preflight_check`.

# Refiner heads

| `refinement_type` | Generative objective | Refiner network |
|---|---|---|
| `diffusion_unet` | diffusion | UNet with a bottleneck attention block |
| `diffusion_transformer` | diffusion | transformer (attention throughout) |
| `flow_matching_unet` | flow matching | UNet with a bottleneck attention block |
| `flow_matching_transformer` | flow matching | transformer (attention throughout) |

For a refiner **without attention blocks**, use a UNet head with `refiner_attention: false`: the bottleneck attention is dropped and the run gets its own `custom_<base>_<type>_no_attention.yaml`, checkpoint directory (`..._no_attention`) and outputs (`<inference.output_dir>/no_attention/...`), so it can be compared with the attention variant. Transformer heads are always attention-based.

# Steps

## 1. Run the pipeline

`run_training_pipeline` with `config_path` (the deterministic base), `refinement_type`, and optionally `refiner_attention` (UNet heads), `refiner_clip_sample_range` (diffusion heads), `ensemble_size` (default 10), `refinement_epochs`, `num_gpus`. Fix `refinement_epochs` up front: a refiner checkpoint cannot be resumed with a different epoch count. If `checkpoint_dir/<case_name>/last.ckpt` already exists and only refinement is wanted, pass `train_phase1: false`.

After the deterministic stages (preprocessing → scalars → fine-tuning → tiled inference → evaluation, each skipped when complete) it queues:

| Stage | Script | Output |
|---|---|---|
| `phase1_cache` | `narr_prism_phase1_cache.py build` | y_hat for training/validation days and training-only residual statistics, under `performance.phase1_cache.path` (shared by all heads, keyed by fingerprint) |
| `refinement_training` | `narr_prism_refinement.py train` | `best.ckpt` / `last.ckpt` in the refinement `checkpoint_dir`; Prithvi frozen; resumes from `last.ckpt` only with the same epoch count |
| `refinement_inference` | `narr_prism_refinement.py infer --split inference --refinement-checkpoint <checkpoint_dir>/best.ckpt` | daily `<case>_<type>_refined_YYYYMMDD.nc` under `<inference.output_dir>/refinement_<type>/<case_name>/`: deterministic, residual, members, ensemble mean, spread |
| `refinement_evaluation` | `evaluate_refinement.py` | deterministic vs refined metrics, distributions, extremes and maps under `path_experiment/refinement_evaluation/<case_name>/<variant>/`; Phase 1 is scored from the `*_phase1` fields embedded in the refined files, and the gap to the independent deterministic product (bf16) is reported under `phase1_baseline.independent_product_parity` |

## 2. The refinement config

The pipeline writes `custom_<base>_<variant>.yaml` next to the base config (variant = head, plus `_no_attention` and/or `_clip<range>`, each with its own checkpoints and outputs), keeping the base config's key order — `predictor_variables` order is the preprocessed channel order and the Phase-2 dataset rejects a reordered copy. It holds the base config's data, case, scalars and model, plus the template's `model.phase1`, `model.refinement`, `performance` and refiner training settings. `model.phase1.checkpoint` points at the base case's `checkpoint_dir/<case_name>/last.ckpt`. `create_refinement_config` writes the same file without queuing jobs, for inspection or hand edits.

## 3. Re-run single stages

- Ensemble only (e.g. 50 members): `start_refinement_inference_job` (`config_path` = the refinement YAML, `ensemble_size`, `num_gpus`, `split`).
- Comparison only: `start_refinement_evaluation_job` (`config_path`, `split`). Needs both deterministic and refined daily outputs for the split.

## 4. Quick end-to-end check (smoke test)

When the user wants to confirm everything works, or asks for fewer epochs or a faster test, do not shorten their real case. Create a separate one:

1. `start_download_job` for `narr` and `prism` over one training month and one validation and one inference week (e.g. 2014-01-01..2014-01-31, 2015-01-01..2015-01-07, 2016-01-01..2016-01-07), plus `elevation`.
2. `create_custom_yaml` from `NARR_PRISM_subdomain.yaml`: new `case_name` (`<case>_smoke`), those dates via `training_start/end`, `validation_start/end`, `inference_start/end`, `num_epochs: 2`, `num_gpus: 1`, `extra_overrides: {"limit_steps_train": 300, "limit_steps_valid": 20}`. `preflight_check` with `stage: training`.
3. `run_training_pipeline`: that config, `refinement_type`, `refinement_epochs: 20`, `ensemble_size: 4`, `save_every: 1`, `refiner_clip_sample_range: 3` for diffusion heads.

`tests/e2e_narr_refine_smoke.py` runs exactly this through the MCP tools.

Measured on one A100-class GPU (California subdomain, 22 tiles/day): preprocessing and scalars ~2 min, Phase-1 training ~15 min, deterministic inference ~2 min, Phase-1 cache ~40 min, refiner training ~15 min, ensemble inference ~12 min, refinement evaluation ~1 min — about 1.5 h after downloads. Its metrics are meaningless; it only proves every stage runs.

A diffusion refiner trained this briefly without clipping predicts near-zero noise; DDIM then amplifies it into non-physical fields (tmax ±150 °C) and refinement inference stops on its physical-range check. That is undertraining, not a units bug. `clip_sample` is part of the refiner checkpoint contract, so it must be set before training (`refiner_clip_sample_range`).

## 5. Cost of a full run

| Stage | Measured rate (one GPU) | One year |
|---|---|---|
| Phase-1 training | ~1.4 s/step, 22 tiles/day | ~3 h/epoch |
| Phase-1 cache | ~64 s/day (strict FP32, deterministic by design) | ~13 h for 365 + 365 days |
| Refiner training (diffusion UNet) | ~40 s/epoch on 31 days | ~8 min/epoch |
| Deterministic inference | ~13 s/day bf16 | ~1.3 h |
| Refinement evaluation | ~7 s/day | ~45 min |

Tell the user these before queuing a full run, and suggest the smoke test first.

## 6. Report

Quote the `refinement_evaluation` numbers for deterministic and refined side by side (RMSE, bias, correlation, spread, extremes), per variable. Say where refinement did not help.

# Guardrails

- NARR only. Never pass a `NARR_PRISM_<type>.yaml` template to `run_training_pipeline`; pass the deterministic base.
- Residual statistics come from training dates only.
- Changing the Phase-1 checkpoint invalidates the residual cache and every refiner trained on it.
- A new `refinement_epochs` or clipping setting needs a fresh refinement `checkpoint_dir`: the run refuses an inexact resume. Ask before moving the old directory aside; never delete it.

# Related

- [Downscale playbook](/playbooks/downscale-wxc.md)
- [MCP pipeline tools](/tools/mcp-pipeline-tools.md)
- [Reproducibility rules](/constraints/reproducibility.md)

# Citations

[1] Training repo `examples/NARR_PRISM/narr_prism_refinement.py`, `docs/STOCHASTIC_REFINEMENT.md` at the `stochastic_refinement` pin
