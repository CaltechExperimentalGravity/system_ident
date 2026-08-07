---
name: use-conda-run-sysid-env
description: "Run Python/quarto/tools in this project via `conda run -n <env>`, not the env's bare binary"
metadata: 
  node_type: memory
  type: feedback
---

For the `system_ident` project, invoke env tools with `conda run -n sysid python ...`
(and `conda run -n sysid quarto ...`, `... pytest ...`), NOT by calling the env's binary
directly (e.g. `$CONDA_PREFIX/bin/python` or a hard-coded miniconda path). The relevant env is the
`sysid` conda env (has system_ident, plotly, kaleido, jupyter/nbclient, quarto 1.9.38).

**Why:** `conda run` executes the env's activation hooks (env vars, library paths, tool
shims), so dependencies resolve the way the user expects. Calling the bare binary skips
activation — which is what forced manual quarto aarch64 symlink/`QUARTO_*` workarounds when
rendering the docs. `conda run -n sysid quarto render ...` picks up the right toolchain.

**How to apply:** prefix every Python/pytest/quarto invocation (and any subagent dispatch
that runs them) with `conda run -n sysid`. Tell dispatched subagents to do the same.
