# PrithviWxC Downscaling — OKF Knowledge Bundle

Portable [Open Knowledge Format (OKF) v0.1](https://okf.md/) bundle for NASA PrithviWxC / granite-wxc weather and climate downscaling.

Not Cursor-specific. Any agent or tool that can read markdown + YAML frontmatter can load this: clone the repo and point the agent at `index.md`.

## Layout

```
.
├── index.md                 # start here
├── log.md                   # changelog
├── concepts/                # what the pipeline is
├── playbooks/               # how to downscale / analyze
├── tools/                   # MCP tool references
└── constraints/             # reproducibility rules
```

## Use with any tool

1. Clone this repo.
2. Give the agent / LLM the bundle root (or paste / attach `index.md` and follow links).
3. Playbooks describe workflows; concepts and tools provide the graph.

```bash
git clone https://github.com/midatm1234/prithvi-wxc-skills.git
# Entry point:
#   prithvi-wxc-skills/index.md
```

## Conformance

OKF v0.1 three rules:

1. Every concept `.md` (not `index.md` / `log.md`) has YAML frontmatter.
2. Every frontmatter has a non-empty `type`.
3. `index.md` and `log.md` follow the OKF listing / changelog structure.

## Related

Pipeline execution still needs an MCP/tool server that implements the tools named in `tools/`. This bundle is the knowledge layer you append to agents — not the runtime.
