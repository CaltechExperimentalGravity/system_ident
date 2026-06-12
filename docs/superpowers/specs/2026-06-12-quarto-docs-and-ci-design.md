# Design: Quarto documentation + GitHub CI for ligo-sysid

Date: 2026-06-12
Status: Approved (pending spec review)

## Goal

Publish full HTML documentation for `ligo-sysid` — **pedagogy + API reference +
worked examples** — built and deployed automatically by GitHub Actions, with the
test suite run as a gate before anything publishes.

## Toolchain

- **Quarto** (website project) as the doc engine.
- **quartodoc** to auto-generate the API reference from the package's existing
  (numpydoc-style, fully typed) docstrings.
- **Jupyter** (`ipykernel`) as the execution engine so worked-example `.qmd`
  pages run at build time and embed real figures (matplotlib). Executed examples
  double as an integration smoke test.
- **control** (already a core dependency) supplies the series/feedback transfer-
  function algebra used by the composite examples (#5).

### Dependency wiring

- Add a `docs` extra to `pyproject.toml`:
  `docs = ["quartodoc", "jupyter", "ipykernel"]`.
  `pip install -e ".[docs]"` then sets up a doc build (core deps, incl. matplotlib
  and control, come along).
- Add `quarto` (conda-forge) to `environment.yml` for local rendering. Quarto is a
  standalone CLI, not a Python package.

## Site structure (`docs/`)

```
docs/
  _quarto.yml          # website project; navbar; quartodoc config; render list
  index.qmd            # what it is, install, 60-second twin-demo quickstart
  tutorial/            # PEDAGOGY
    overview.qmd       # the sysID problem + the iterate-loop
    model.qmd          # TFModel representation (num/den, resonances, eval/jacobian)
    fisher.qmd         # Fisher info -> dispersion -> Pintelon-Schoukens excitation
                       #   (math + refs: LIGO-G2101503, LIGO-G1400084); plots a PSD
    closing-the-loop.qmd  # invfreqs re-fit + cross-pass accumulation; convergence
    safety-and-ops.qmd    # limits, watchdog, ramp-down, operator STOP, dashboard
  examples/            # WORKED EXAMPLES (executed at build)
    01-single-resonance.qmd
    02-double-pendulum.qmd
    03-fabry-perot-cavity.qmd
    04-suspension-multidof.qmd
    05-closed-loop-arm.qmd
  reference/           # quartodoc-generated API pages (build artifact, gitignored)
  superpowers/         # design specs — EXCLUDED from the Quarto render
```

`_quarto.yml` `project.render` lists only `index.qmd`, `tutorial/`, `examples/`,
and `reference/`, and explicitly excludes `superpowers/`. `output-dir: _site`
(gitignored).

### Worked-examples ladder

Each builds a plant, wraps it in a `SuspensionPlant` (a generic named collection
of `TFModel`s), drives it through the real `TwinBackend` + `SysIDLoop` /
`InvfreqsEstimator`, and shows designed-excitation + recovered-model figures.

1. **Single resonance (SHO)** — `TFModel.from_resonances([(f0, Q)], gain)`.
   Model → Fisher → optimal excitation; power concentrates at the resonance.
2. **Double pendulum** — `plant.double_pendulum()` (0.6 Hz Q20, 1.5 Hz Q30).
   Multi-mode excitation; recover both modes; optimal vs. flat excitation.
3. **Fabry–Pérot cavity** — `TFModel.from_zpk([], [cavity_pole], gain)`.
   Single-pole optical low-pass; SNR-limited ID of the cavity pole.
4. **LIGO suspension (multi-DoF)** — `SuspensionPlant.from_resonance_spec` over a
   POS/PIT/YAW resonance stack. Full twin campaign.
5. **Closed-loop LIGO arm (capstone)** — P = suspension ⊗ cavity (series),
   controller C, loop closed via `control.feedback`. Inject with the loop on,
   measure the closed-loop T = P/(1+PC), recover P̂ = T̂/(1 − T̂·C) knowing C;
   demonstrate that loop-ignorant fitting is biased. Controller-division done in
   the notebook (example level), clearly explained.

## CI & publishing — `.github/workflows/ci.yml` (single workflow)

Triggers: `push` to `main`, `pull_request`, `workflow_dispatch`.

Permissions (workflow level): `contents: read`, `pages: write`, `id-token: write`.
`concurrency` group on `pages` to avoid overlapping deploys.

- **`test`** — `actions/setup-python@v5` (3.12) → `pip install -e ".[dev]"` →
  `pytest`. Runs on push + PR. (pip pulls everything incl. control from PyPI; no
  conda needed in CI — fast.)
- **`build-docs`** (`needs: test`) — `pip install -e ".[docs]"` →
  `quarto-dev/quarto-actions/setup@v2` → `quartodoc build` (cwd `docs/`) →
  `quarto render docs/` → `actions/upload-pages-artifact@v3` (path `docs/_site`).
  Runs on push + PR (PRs verify the site renders but do not deploy).
- **`deploy-docs`** (`needs: build-docs`, `if: github.ref == 'refs/heads/main'`) —
  `environment: github-pages` → `actions/deploy-pages@v4`.

Net: PRs run tests + render the docs; only `main` publishes, and only when tests
pass.

## Verification

- Install Quarto (conda-forge) into the `sysid` env and **render the full site
  locally** before committing, confirming quartodoc resolves the API and all five
  examples execute. Do not rely on CI for first-pass validation.
- Add a status badge for the `test` workflow to the README.

## Manual step the user must do (cannot be automated)

Once the repo is on GitHub: **Settings → Pages → Source = "GitHub Actions."**
Until the repo has a remote and that toggle is set, `deploy-docs` is inert, but
`test` and `build-docs` run and prove the site renders. There is currently no git
remote.

## Out of scope / future work

- Closed-loop sysID as a first-class package feature (a controller-aware estimator
  or backend) — flagged in the docs as future work, not built here.
- MIMO / cross-coupling identification.
- Versioned docs / multiple published versions.

## Success criteria

- `quarto render docs/` succeeds locally with all examples executed and the
  quartodoc API reference populated.
- `pytest` stays green (55 passed, 1 skipped).
- The workflow validates on a PR (test + build-docs green) and publishes from
  `main` once Pages is enabled.
