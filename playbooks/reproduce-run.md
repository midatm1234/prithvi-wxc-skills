---
type: Playbook
title: Reproduce a run
description: Inspect, share, and replay Prithvi WxC downscaling runs across machines and institutions using MCP run manifests.
tags: [playbook, reproducibility, provenance, replay]
timestamp: 2026-10-01T09:00:00Z
---

Every MCP job writes `<PIPELINE_DATA_ROOT>/.mcp-state/runs/<job_id>.json` (schema `prithvi-wxc-run-manifest/1`): resolved command, full config text and sha256, code commit and dirty files, pins, interpreter package versions, input inventory, and `reproducible` with `non_reproducible_reasons`.

# Inspect

`list_run_manifests` → `get_run_manifest` (`include_config_text: true` for the YAML). Explain each non-reproducible reason and its fix.

# Share

Send the manifest JSON. It is self-contained; the recipient needs this plugin and the inputs.

# Replay

1. [Set up the machine](/playbooks/setup-machine.md); `setup_code` with the variant whose commit equals the manifest's `code.commit`.
2. `replay_run` with `manifest_path` and `dry_run: true` — review localized config changes and the command.
3. `preflight_check` on the planned config; download what is missing.
4. `replay_run` without `dry_run`; monitor with `get_job_status`.
5. Compare outputs by values (e.g. `xarray` `array_equal`), not file bytes.

# Related

- [Reproducibility rules](/constraints/reproducibility.md)
- [Downscale playbook](/playbooks/downscale-wxc.md)
