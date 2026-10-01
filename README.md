# PrithviWxC Downscaling Skills

Cursor Agent Skills for NASA PrithviWxC / granite-wxc weather and climate downscaling.

## Skills

| Skill | Path | Use when |
|-------|------|----------|
| `downscale-wxc` | [`skills/downscale-wxc/`](./skills/downscale-wxc/) | Downscale gridded data (MERRA-2, NARR, ERA5, CORDEX, …), train, or run inference |
| `analyze-wxc` | [`skills/analyze-wxc/`](./skills/analyze-wxc/) | Plot, summarize, compare dates, or compute climatologies from NetCDF outputs |

## Install in Cursor

Copy into your personal or project skills folder:

```bash
git clone https://github.com/midatm1234/prithvi-wxc-skills.git
cp -a prithvi-wxc-skills/skills/* ~/.cursor/skills/
# or: cp -a prithvi-wxc-skills/skills/* .cursor/skills/
```

Reload the window. Skills are standard `SKILL.md` Agent Skills (YAML frontmatter + markdown).

## Note

These skills describe MCP tool workflows (`load_by_date`, `create_custom_yaml`, `run_training_pipeline`, …). To actually run the pipeline you also need the companion MCP/plugin package — this repo is the skills layer only.
