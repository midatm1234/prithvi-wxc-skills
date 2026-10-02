---
name: prithvi-reproduce
description: Reproduce or share a Prithvi WxC downscaling run using run manifests — inspect what code commit, config, pins, interpreter, and inputs a job used; check whether it is reproducible; replay a manifest received from another machine or institution. Use when the user asks to reproduce, replay, re-run, compare, audit, or share a run, or asks "which version / config produced this".
---

# Reproduce and share runs

Every MCP job writes a manifest under `<PIPELINE_DATA_ROOT>/.mcp-state/runs/<job_id>.json`: command, full config text + sha256, code commit, pins (code, weights, data sources), interpreter package versions, input inventory, and `reproducible` with `non_reproducible_reasons`.

## Inspect

1. `list_run_manifests` → pick the run. `get_run_manifest` with `job_id` (add `include_config_text` for the YAML).
2. Report `reproducible`. If false, explain each reason (off-pin or dirty code, non-reference elevation, xesmf present, weights size mismatch) and how to fix it.

## Share

Send the manifest JSON file. It is self-contained; the recipient needs this plugin, the same pins, and the inputs (the `prithvi-data` skill fetches them).

## Replay a received manifest

1. `check_environment`; `setup_code` with the variant whose commit equals the manifest's `code.commit`.
2. `replay_run` with `manifest_path` and `dry_run: true`. Show the plan: localized config changes, stage, arguments.
3. `preflight_check` on the planned config; fetch missing inputs with `prithvi-data`.
4. `replay_run` without `dry_run`. Monitor with `get_job_status`.
5. Compare outputs by values, not file bytes: providers stamp metadata (PRISM writes a fresh `history` attribute on every download) while the data are identical.
