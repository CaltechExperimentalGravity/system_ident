# Quarto Documentation + GitHub CI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish full HTML docs for `ligo-sysid` (pedagogy + API reference + executed worked examples), built and deployed by GitHub Actions with pytest as a gate.

**Architecture:** A Quarto website project under `docs/`. quartodoc auto-generates the API reference from the package's docstrings; Jupyter executes the worked-example pages at build time so they embed real figures (and act as integration smoke tests). One GitHub Actions workflow runs `test` → `build-docs` → `deploy-docs` (deploy is main-only and gated on tests), publishing to GitHub Pages via the official Pages actions.

**Tech Stack:** Quarto, quartodoc, Jupyter/ipykernel, matplotlib, python-control (already a dep), GitHub Actions, GitHub Pages.

**Conventions for this plan:**
- All commands run in the `sysid` conda env: prefix with `conda run --no-capture-output -n sysid`.
- Repo root is `/Users/rana/Desktop/Dropbox/GIT/system_ident`; paths below are relative to it.
- Commit author: `git -c user.name="RXA" -c user.email="rana@caltech.edu" commit`.

---

### Task 1: Docs dependencies, ignore rules, and toolchain

**Files:**
- Modify: `pyproject.toml` (optional-dependencies)
- Modify: `environment.yml` (add quarto)
- Modify: `.gitignore` (Quarto build artifacts)

- [ ] **Step 1: Add the `docs` extra to `pyproject.toml`**

In `pyproject.toml`, under `[project.optional-dependencies]`, add the `docs` line (keep the existing `dashboard`, `cds`, `dev` lines):

```toml
docs = ["quartodoc", "jupyter", "ipykernel"]
```

- [ ] **Step 2: Add Quarto to `environment.yml`**

In `environment.yml`, add `quarto` to the conda `dependencies:` list (it is a standalone CLI from conda-forge, not a Python package), right after `control`:

```yaml
  - control
  - quarto
```

- [ ] **Step 3: Ignore Quarto build artifacts**

Append to `.gitignore`:

```gitignore

# Quarto / docs build artifacts
docs/_site/
docs/.quarto/
docs/reference/
docs/objects.json
.jupyter_cache/
```

- [ ] **Step 4: Install the toolchain into the `sysid` env**

Run:
```bash
conda install -n sysid -c conda-forge quarto -y
conda run --no-capture-output -n sysid python -m pip install -e ".[docs]"
conda run --no-capture-output -n sysid python -m ipykernel install --user --name python3 --display-name python3
```
Expected: quarto installs; pip reports `quartodoc`, `jupyter`, `ipykernel` installed; ipykernel prints `Installed kernelspec python3 in ...`.

- [ ] **Step 5: Verify the tools resolve**

Run:
```bash
conda run --no-capture-output -n sysid quarto --version
conda run --no-capture-output -n sysid quartodoc --help
```
Expected: a Quarto version (e.g. `1.x`) prints, and quartodoc shows its CLI help (subcommands include `build`).

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml environment.yml .gitignore
git -c user.name="RXA" -c user.email="rana@caltech.edu" commit -m "docs: add docs extra, quarto to env, ignore build artifacts"
```

---

### Task 2: Minimal Quarto site (skeleton + landing page)

Get a site rendering before adding quartodoc, so any failure is isolated.

**Files:**
- Create: `docs/_quarto.yml`
- Create: `docs/index.qmd`

- [ ] **Step 1: Create `docs/_quarto.yml` (minimal, no quartodoc yet)**

```yaml
project:
  type: website
  output-dir: _site
  render:
    - index.qmd

website:
  title: "ligo-sysid"
  description: "Optimal-excitation system identification for LIGO suspensions."
  navbar:
    left:
      - href: index.qmd
        text: Home

format:
  html:
    theme: cosmo
    toc: true
    code-copy: true
    code-overflow: wrap

jupyter: python3
```

- [ ] **Step 2: Create `docs/index.qmd`**

````markdown
---
title: "ligo-sysid"
---

Real-time, optimal-excitation **system identification for LIGO suspensions** —
a prior model of a transfer function, an excitation designed to extract the most
Fisher information per unit drive power and measurement time, a re-fit from the
measured response, and repeat. The same channel API drives a digital twin or
(later) CDS hardware, with a safety watchdog and an operator STOP.

## Install

```bash
pip install -e ".[dev]"            # core + tests
pip install -e ".[dev,dashboard]"  # also the live dashboard
pip install -e ".[docs]"           # to build these docs
```

## 60-second quickstart (digital twin)

```bash
ligo-sysid run src/ligo_sysid/configs/twin_demo.yml --twin --yes
# -> DONE (target reached); per-DoF fractional uncertainty ~1e-9
```

Or from Python:

```python
from ligo_sysid.config import RunConfig
from ligo_sysid.loop import SysIDLoop

rc = RunConfig.load("src/ligo_sysid/configs/twin_demo.yml")
backend = rc.build_twin_backend(seed=0)
loop = SysIDLoop(backend, rc.build_estimator(), rc.build_designer(),
                 rc.build_watchdog(backend))
result = loop.run(rc.raw, rc.build_priors(), seed=0)
print("done:", result.done)
```

## Where to go next

- **Tutorial** — the method, end to end: the model, Fisher information and
  optimal excitation, closing the loop, and safety/operations.
- **Examples** — worked, executed runs on a ladder of plants (single resonance →
  double pendulum → Fabry–Pérot cavity → multi-DoF suspension → closed-loop arm).
- **API reference** — every public class and function.
````

- [ ] **Step 3: Render and verify**

Run:
```bash
conda run --no-capture-output -n sysid quarto render docs
```
Expected: render succeeds; `docs/_site/index.html` exists. Verify:
```bash
test -f docs/_site/index.html && echo OK
```
Expected: `OK`.

- [ ] **Step 4: Commit**

```bash
git add docs/_quarto.yml docs/index.qmd
git -c user.name="RXA" -c user.email="rana@caltech.edu" commit -m "docs: minimal Quarto website + landing page"
```

---

### Task 3: API reference via quartodoc

**Files:**
- Modify: `docs/_quarto.yml` (add quartodoc config, sidebar, navbar link)

- [ ] **Step 1: Replace `docs/_quarto.yml` with the quartodoc-enabled version**

```yaml
project:
  type: website
  output-dir: _site
  render:
    - index.qmd
    - tutorial/*.qmd
    - examples/*.qmd
    - reference/*.qmd

website:
  title: "ligo-sysid"
  description: "Optimal-excitation system identification for LIGO suspensions."
  navbar:
    left:
      - href: index.qmd
        text: Home
      - href: reference/index.qmd
        text: API Reference
  sidebar:
    - id: reference
      contents: reference/_sidebar.yml

metadata-files:
  - reference/_sidebar.yml

format:
  html:
    theme: cosmo
    toc: true
    code-copy: true
    code-overflow: wrap

jupyter: python3

quartodoc:
  package: ligo_sysid
  dir: reference
  title: API Reference
  style: pkgdown
  sidebar: reference/_sidebar.yml
  sections:
    - title: Core
      desc: "Model, Fisher information, excitation synthesis, and the loop."
      contents:
        - model.TFModel
        - fisher.fisher_matrix
        - fisher.parameter_covariance
        - fisher.dispersion
        - excitation.timeseries_from_asd
        - loop.SysIDLoop
        - loop.LoopResult
    - title: Estimators
      desc: "Fit a transfer function to measured data."
      contents:
        - estimators.base.Estimator
        - estimators.invfreqs.InvfreqsEstimator
        - estimators.invfreqs.invfreqs
    - title: Input design
      desc: "Design the excitation spectrum."
      contents:
        - design.base.InputDesigner
        - design.pintelon.PintelonSchoukensDesigner
        - design.pintelon.optimal_excitation
    - title: Plant & backends
      desc: "Plant models and the channel API (twin / hardware)."
      contents:
        - plant.SuspensionPlant
        - plant.double_pendulum
        - backends.base.ChannelBackend
        - backends.twin.TwinBackend
    - title: Safety & configuration
      desc: "Watchdog, limits, and run configuration."
      contents:
        - safety.SafetyLimits
        - safety.Watchdog
        - config.RunConfig
```

- [ ] **Step 2: Build the API pages**

Run (quartodoc reads `docs/_quarto.yml`, so run it from `docs/`):
```bash
conda run --no-capture-output -n sysid bash -c "cd docs && quartodoc build"
```
Expected: prints the objects it inspected; creates `docs/reference/index.qmd`,
`docs/reference/_sidebar.yml`, `docs/objects.json`, and one `.qmd` per listed
object (e.g. `docs/reference/model.TFModel.qmd`).

- [ ] **Step 3: Verify the reference generated**

```bash
test -f docs/reference/_sidebar.yml && test -f docs/reference/model.TFModel.qmd && echo OK
```
Expected: `OK`.

- [ ] **Step 4: Render the full site and verify the API pages**

```bash
conda run --no-capture-output -n sysid quarto render docs
test -f docs/_site/reference/index.html && echo OK
```
Expected: render succeeds; `OK`.

- [ ] **Step 5: Commit** (generated `reference/` is gitignored — only the config is committed)

```bash
git add docs/_quarto.yml
git -c user.name="RXA" -c user.email="rana@caltech.edu" commit -m "docs: auto-generate API reference with quartodoc"
```

---

### Task 4: Pedagogy tutorials

Five narrative pages. Only `fisher.qmd` executes code (a small designed-excitation
figure); the rest are prose.

**Files:**
- Create: `docs/tutorial/overview.qmd`
- Create: `docs/tutorial/model.qmd`
- Create: `docs/tutorial/fisher.qmd`
- Create: `docs/tutorial/closing-the-loop.qmd`
- Create: `docs/tutorial/safety-and-ops.qmd`
- Modify: `docs/_quarto.yml` (navbar Tutorial menu)

- [ ] **Step 1: Create `docs/tutorial/overview.qmd`**

````markdown
---
title: "The method"
---

System identification here is an iterative, information-optimal loop. We start
from a **prior** transfer-function model of one degree of freedom, design an
**excitation** that maximises the Fisher information about the model parameters
for a fixed drive-power budget and measurement time, **inject** it, **measure**
the response, **re-fit** the model, and repeat.

```
   prior model ─▶ design optimal excitation ─▶ inject ─▶ measure
        ▲                                                   │
        └──────────────── re-fit model ◀────────────────────┘
```

Why optimal excitation? On real hardware, drive amplitude is bounded by actuator
saturation and the plant's linear range, and measurement time is expensive.
Spending that limited budget where it tightens the parameters most — typically
near resonances — recovers the model far faster than a flat sweep. The rest of
this tutorial develops each block; the [examples](../examples/01-single-resonance.qmd)
run them end to end.

References: Pintelon & Schoukens, *System Identification* (§5.4.2);
[LIGO-G1400084](https://dcc.ligo.org/LIGO-G1400084) (Larry Price);
[LIGO-G2101503](https://dcc.ligo.org/LIGO-G2101503) (Fisher expressions).
````

- [ ] **Step 2: Create `docs/tutorial/model.qmd`**

````markdown
---
title: "The model"
---

Every part of the package speaks one representation: `TFModel`, a transfer
function `G(s) = num(s) / den(s)` stored as polynomial coefficients (descending
powers of `s`), with the leading denominator coefficient held at 1 (the standard
gauge). This mirrors the `{num, den}` convention of the original `sysIDlib`
engine, so the validated math is reused without translation.

Construct one however is natural:

```python
from ligo_sysid.model import TFModel

# from (f0 [Hz], Q) resonances + a gain (optionally with zeros)
g1 = TFModel.from_resonances([(1.0, 20.0)], gain=1.0)

# from zeros / poles / gain
import numpy as np
g2 = TFModel.from_zpk(zeros=[], poles=[-2*np.pi*100.0], gain=2*np.pi*100.0)

# from a {"num":..., "den":...} dict
g3 = TFModel.from_dict({"num": [1.0], "den": [1.0, 6.28, 39.5]})
```

`g.eval(freq)` returns the complex response on a frequency grid [Hz];
`g.jacobian(freq)` gives the parameter sensitivities that feed the Fisher
calculation. See the [API reference](../reference/model.TFModel.qmd).
````

- [ ] **Step 3: Create `docs/tutorial/fisher.qmd`** (executes one figure)

````markdown
---
title: "Fisher information & optimal excitation"
---

The Fisher matrix turns *(model, excitation PSD `Pxx`, quiet readout PSD `Pyy`,
measurement time)* into the information the experiment yields about the model
parameters; its inverse is the parameter covariance. The **dispersion function**
`ν(f)` measures how much each frequency bin contributes to that information given
the current excitation. Reallocating drive power toward high-`ν` bins — under a
fixed total-power budget — is a fixed-point iteration that converges to the
optimal excitation (Pintelon & Schoukens §5.4.2).

```{python}
#| label: fig-optimal-excitation
#| fig-cap: "Optimal excitation vs a flat drive of equal power, for a 1 Hz resonance."
import numpy as np
import matplotlib.pyplot as plt
from ligo_sysid.model import TFModel
from ligo_sysid.design.pintelon import optimal_excitation

freq = np.linspace(0.2, 3.0, 400)
model = TFModel.from_resonances([(1.0, 20.0)], gain=1.0)
Pyy = np.ones_like(freq)                       # white readout noise
Px_tot = 1.0                                   # fixed drive-power budget

Pxx_opt = optimal_excitation(freq, model, Pyy, Px_tot, n_iter=3)
Pxx_flat = np.full_like(freq, Px_tot / (freq[-1] - freq[0]))

fig, ax = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
ax[0].semilogy(freq, np.abs(model.eval(freq)))
ax[0].set_ylabel("|G(f)|")
ax[1].plot(freq, np.sqrt(Pxx_opt), label="optimal")
ax[1].plot(freq, np.sqrt(Pxx_flat), "--", label="flat (equal power)")
ax[1].set_xlabel("Frequency [Hz]"); ax[1].set_ylabel("drive ASD"); ax[1].legend()
plt.show()
```

The optimal drive piles power onto the resonance, where the model parameters are
most observable. The [examples](../examples/01-single-resonance.qmd) quantify the
resulting covariance reduction.
````

- [ ] **Step 4: Create `docs/tutorial/closing-the-loop.qmd`**

````markdown
---
title: "Closing the loop"
---

With a measured transfer function in hand, the **estimator** re-fits the model.
The default `InvfreqsEstimator` is a weighted least-squares fit (a modernised
`invfreqs`): it minimises `|num/den − H|²` weighted by the inverse measurement
variance, holding the model order fixed at the prior's.

`SysIDLoop` runs the full campaign per degree of freedom. The first pass uses a
flat, broadband excitation so the global fit is well conditioned; later passes
use the optimal designer to sharpen the now-trusted parameters. Crucially, each
pass is an independent measurement of the same LTI system on a fixed Welch grid,
so passes are combined by **inverse-variance weighting** per frequency bin and
the model is re-fit on the accumulated estimate — broadband coverage from pass
one is retained while resonance information from the optimal passes is folded in.
The loop stops when every DoF reaches the fractional-uncertainty target, the
iteration budget is spent, or the watchdog aborts.

```python
from ligo_sysid.loop import SysIDLoop
result = SysIDLoop(backend, estimator, designer, watchdog).run(config, priors, seed=0)
print(result.done, {d: round(m.den[1], 3) for d, m in result.models.items()})
```

See the [multi-DoF suspension example](../examples/04-suspension-multidof.qmd)
for a full run with a convergence plot.
````

- [ ] **Step 5: Create `docs/tutorial/safety-and-ops.qmd`**

````markdown
---
title: "Safety & operations"
---

On a real interferometer, only **physical** limits auto-abort: actuator drive vs
saturation, and per-DoF output RMS vs a safe envelope. Coherence and fit-health
are surfaced as status but never trigger an abort. The `Watchdog` checks every
read segment and, on a breach — or an operator STOP, or normal teardown — runs
one shared, idempotent routine: ramp every excitation channel down, restore the
pre-run channel/filter state captured at run start (handing control back to the
existing damping loops), and record the aborted state and its reason.

Limits come from the config's `safety` section:

```yaml
safety:
  actuator_sat: 1.0e-4        # max |drive| before saturation
  rms_ceiling: {POS: 1.0e-6, PIT: 1.0e-7, YAW: 1.0e-7}
  ramp_down_secs: 2.0
```

The CLI (`ligo-sysid run <config> --twin`) requires a confirm-before-inject step
for hardware, and serves a live dashboard (transfer function, coherence, designed
excitation, convergence) with a **STOP** button that calls the same safe handoff
when the `[dashboard]` extra is installed.
````

- [ ] **Step 6: Add the Tutorial menu to `docs/_quarto.yml`**

In `docs/_quarto.yml`, under `website.navbar.left`, insert a Tutorial dropdown
between the Home and API Reference entries:

```yaml
      - text: Tutorial
        menu:
          - href: tutorial/overview.qmd
            text: The method
          - href: tutorial/model.qmd
            text: The model
          - href: tutorial/fisher.qmd
            text: Fisher & excitation
          - href: tutorial/closing-the-loop.qmd
            text: Closing the loop
          - href: tutorial/safety-and-ops.qmd
            text: Safety & operations
```

- [ ] **Step 7: Render and verify (incl. the executed figure)**

```bash
conda run --no-capture-output -n sysid quarto render docs
test -f docs/_site/tutorial/fisher.html && echo OK
```
Expected: render succeeds (the `fisher.qmd` Python cell executes without error);
`OK`.

- [ ] **Step 8: Commit**

```bash
git add docs/_quarto.yml docs/tutorial
git -c user.name="RXA" -c user.email="rana@caltech.edu" commit -m "docs: pedagogy tutorials"
```

---

### Task 5: Example 1 — single resonance (SHO)

**Files:**
- Create: `docs/examples/01-single-resonance.qmd`

- [ ] **Step 1: Create the example**

````markdown
---
title: "Single resonance (SHO)"
---

The smallest complete run: design an optimal excitation for one resonance, show
it tightens the parameters versus a flat drive of equal power, and recover the
model from a noisy measurement.

```{python}
import numpy as np
import matplotlib.pyplot as plt
from ligo_sysid.model import TFModel
from ligo_sysid.fisher import parameter_covariance
from ligo_sysid.design.pintelon import optimal_excitation
from ligo_sysid.estimators.invfreqs import InvfreqsEstimator

true = TFModel.from_resonances([(1.0, 20.0)], gain=1.0)   # 1 Hz, Q=20
prior = TFModel.from_resonances([(0.95, 18.0)], gain=1.1)  # slightly wrong

freq = np.linspace(0.2, 3.0, 400)
Pyy = np.ones_like(freq)
Px_tot = 1.0

Pxx_opt = optimal_excitation(freq, prior, Pyy, Px_tot, n_iter=3)
Pxx_flat = np.full_like(freq, Px_tot / (freq[-1] - freq[0]))
```

Designed excitation vs the plant:

```{python}
#| fig-cap: "The optimal drive concentrates power at the resonance."
fig, ax = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
ax[0].semilogy(freq, np.abs(true.eval(freq))); ax[0].set_ylabel("|G(f)|")
ax[1].plot(freq, np.sqrt(Pxx_opt), label="optimal")
ax[1].plot(freq, np.sqrt(Pxx_flat), "--", label="flat (equal power)")
ax[1].set_xlabel("Frequency [Hz]"); ax[1].set_ylabel("drive ASD"); ax[1].legend()
plt.show()
```

Same power, lower uncertainty — compare the worst-parameter variance:

```{python}
T_tot = 256.0
cov_opt = parameter_covariance(freq, true, Pxx_opt, Pyy, T_tot)
cov_flat = parameter_covariance(freq, true, Pxx_flat, Pyy, T_tot)
print(f"max param variance  flat: {np.max(np.diag(cov_flat)):.3e}")
print(f"max param variance  opt : {np.max(np.diag(cov_opt)):.3e}")
```

Recover the model from a 2%-noise measurement:

```{python}
rng = np.random.default_rng(0)
H = true.eval(freq)
H_err = np.abs(H) * 0.02
H_meas = H + (rng.standard_normal(freq.size) + 1j*rng.standard_normal(freq.size)) * H_err/np.sqrt(2)

fit = InvfreqsEstimator().fit(freq, H_meas, H_err, prior)
print("true den:", np.round(true.den, 3))
print("fit  den:", np.round(fit.den, 3))
```
````

- [ ] **Step 2: Render and verify it executes**

```bash
conda run --no-capture-output -n sysid quarto render docs/examples/01-single-resonance.qmd
test -f docs/_site/examples/01-single-resonance.html && echo OK
```
Expected: render succeeds; `OK`.

- [ ] **Step 3: Commit**

```bash
git add docs/examples/01-single-resonance.qmd
git -c user.name="RXA" -c user.email="rana@caltech.edu" commit -m "docs: example 1 — single resonance"
```

---

### Task 6: Example 2 — double pendulum

**Files:**
- Create: `docs/examples/02-double-pendulum.qmd`

- [ ] **Step 1: Create the example**

````markdown
---
title: "Double pendulum"
---

A two-mode plant (the package's canonical `double_pendulum`: 0.6 Hz Q20 and
1.5 Hz Q30). The optimal excitation now splits power across both resonances.

```{python}
import numpy as np
import matplotlib.pyplot as plt
from ligo_sysid.plant import double_pendulum
from ligo_sysid.model import TFModel
from ligo_sysid.fisher import parameter_covariance
from ligo_sysid.design.pintelon import optimal_excitation
from ligo_sysid.estimators.invfreqs import InvfreqsEstimator

true = double_pendulum()                                          # two modes
prior = TFModel.from_resonances([(0.55, 18.0), (1.6, 28.0)], gain=250.0)

freq = np.linspace(0.2, 5.0, 600)
Pyy = np.ones_like(freq)
Px_tot = 1.0

Pxx_opt = optimal_excitation(freq, prior, Pyy, Px_tot, n_iter=3)
Pxx_flat = np.full_like(freq, Px_tot / (freq[-1] - freq[0]))
```

```{python}
#| fig-cap: "Optimal drive power splits between the two modes."
fig, ax = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
ax[0].semilogy(freq, np.abs(true.eval(freq))); ax[0].set_ylabel("|G(f)|")
ax[1].plot(freq, np.sqrt(Pxx_opt), label="optimal")
ax[1].plot(freq, np.sqrt(Pxx_flat), "--", label="flat (equal power)")
ax[1].set_xlabel("Frequency [Hz]"); ax[1].set_ylabel("drive ASD"); ax[1].legend()
plt.show()
```

```{python}
T_tot = 256.0
cov_opt = parameter_covariance(freq, true, Pxx_opt, Pyy, T_tot)
cov_flat = parameter_covariance(freq, true, Pxx_flat, Pyy, T_tot)
print(f"max param variance  flat: {np.max(np.diag(cov_flat)):.3e}")
print(f"max param variance  opt : {np.max(np.diag(cov_opt)):.3e}")

rng = np.random.default_rng(0)
H = true.eval(freq)
H_err = np.abs(H) * 0.02
H_meas = H + (rng.standard_normal(freq.size) + 1j*rng.standard_normal(freq.size)) * H_err/np.sqrt(2)
fit = InvfreqsEstimator().fit(freq, H_meas, H_err, prior)
print("true den:", np.round(true.den, 3))
print("fit  den:", np.round(fit.den, 3))
```
````

- [ ] **Step 2: Render and verify**

```bash
conda run --no-capture-output -n sysid quarto render docs/examples/02-double-pendulum.qmd
test -f docs/_site/examples/02-double-pendulum.html && echo OK
```
Expected: render succeeds; `OK`.

- [ ] **Step 3: Commit**

```bash
git add docs/examples/02-double-pendulum.qmd
git -c user.name="RXA" -c user.email="rana@caltech.edu" commit -m "docs: example 2 — double pendulum"
```

---

### Task 7: Example 3 — Fabry–Pérot cavity

**Files:**
- Create: `docs/examples/03-fabry-perot-cavity.qmd`

- [ ] **Step 1: Create the example**

````markdown
---
title: "Fabry–Pérot cavity"
---

Different physics: an optical cavity is a single-pole low-pass, `G(s) =
ωc/(s+ωc)`, with the cavity pole `fc` setting the bandwidth. Here the
identification is signal-to-noise limited rather than resonance-dominated.

```{python}
import numpy as np
import matplotlib.pyplot as plt
from ligo_sysid.model import TFModel
from ligo_sysid.estimators.invfreqs import InvfreqsEstimator

fc = 100.0                                  # cavity pole [Hz]
wc = 2 * np.pi * fc
true = TFModel.from_zpk(zeros=[], poles=[-wc], gain=wc)   # DC gain 1

freq = np.logspace(0, 3, 400)               # 1 Hz .. 1 kHz
H = true.eval(freq)
```

```{python}
#| fig-cap: "Cavity response and a 5%-noise measurement."
rng = np.random.default_rng(0)
H_err = np.abs(H) * 0.05
H_meas = H + (rng.standard_normal(freq.size) + 1j*rng.standard_normal(freq.size)) * H_err/np.sqrt(2)

prior = TFModel.from_zpk(zeros=[], poles=[-2*np.pi*70.0], gain=2*np.pi*70.0)
fit = InvfreqsEstimator().fit(freq, H_meas, H_err, prior)

fig, ax = plt.subplots(figsize=(7, 4))
ax.loglog(freq, np.abs(H), label="true")
ax.loglog(freq, np.abs(H_meas), ".", ms=3, alpha=0.4, label="measured")
ax.loglog(freq, np.abs(fit.eval(freq)), label="fit")
ax.set_xlabel("Frequency [Hz]"); ax.set_ylabel("|G(f)|"); ax.legend()
plt.show()

f_pole = abs(fit.den[1]) / (2 * np.pi)       # den = [1, wc] -> pole at wc
print(f"true cavity pole: {fc:.2f} Hz")
print(f"fit  cavity pole: {f_pole:.2f} Hz")
```
````

- [ ] **Step 2: Render and verify**

```bash
conda run --no-capture-output -n sysid quarto render docs/examples/03-fabry-perot-cavity.qmd
test -f docs/_site/examples/03-fabry-perot-cavity.html && echo OK
```
Expected: render succeeds; `OK`.

- [ ] **Step 3: Commit**

```bash
git add docs/examples/03-fabry-perot-cavity.qmd
git -c user.name="RXA" -c user.email="rana@caltech.edu" commit -m "docs: example 3 — Fabry-Perot cavity"
```

---

### Task 8: Example 4 — multi-DoF suspension (full campaign)

**Files:**
- Create: `docs/examples/04-suspension-multidof.qmd`

- [ ] **Step 1: Create the example**

````markdown
---
title: "Multi-DoF suspension (full campaign)"
---

A 3-DoF suspension (POS / PIT / YAW) driven through the real `TwinBackend` and
`SysIDLoop` — the same path the CLI runs. We build a config in Python, run the
campaign, and plot convergence.

```{python}
import numpy as np
import matplotlib.pyplot as plt
from ligo_sysid.config import RunConfig
from ligo_sysid.loop import SysIDLoop

cfg = {
    "run": {"name": "susp3dof", "excitation_mode": "sequential"},
    "channels": {
        "excitation": {"POS": "C1:EXC_POS", "PIT": "C1:EXC_PIT", "YAW": "C1:EXC_YAW"},
        "readback":   {"POS": "C1:RSP_POS", "PIT": "C1:RSP_PIT", "YAW": "C1:RSP_YAW"},
    },
    "measurement": {"fs": 32, "freq_min": 0.1, "freq_max": 5.0,
                    "segment_duration": 64.0, "n_segments": 4, "px_total": 1.0},
    "strategy": {"estimator": "invfreqs", "input_designer": "pintelon_schoukens",
                 "n_design_iter": 3},
    "twin": {"sensor_asd": 1e-9, "plant": {
        "POS": {"resonances": [[0.6, 20], [1.5, 30]], "gain": 300},
        "PIT": {"resonances": [[0.8, 15], [2.2, 25]], "gain": 120},
        "YAW": {"resonances": [[1.2, 18]], "gain": 80},
    }},
    "priors": {
        "POS": {"resonances": [[0.55, 18], [1.6, 28]], "gain": 250},
        "PIT": {"resonances": [[0.75, 13], [2.3, 22]], "gain": 150},
        "YAW": {"resonances": [[1.1, 16]], "gain": 100},
    },
    "safety": {"actuator_sat": 1000.0,
               "rms_ceiling": {"POS": 1000.0, "PIT": 1000.0, "YAW": 1000.0},
               "ramp_down_secs": 2.0},
    "stop_criteria": {"uncertainty_target": 0.05, "max_iter": 2},
}

rc = RunConfig(raw=cfg)
backend = rc.build_twin_backend(seed=0)
priors = rc.build_priors()
watchdog = rc.build_watchdog(backend)
loop = SysIDLoop(backend, rc.build_estimator(), rc.build_designer(), watchdog)
result = loop.run(rc.raw, priors, seed=0)

print("done:", result.done)
for dof, model in result.models.items():
    print(f"  {dof} den: {np.round(model.den, 3)}")
```

Convergence — worst-parameter fractional uncertainty per DoF over the campaign:

```{python}
#| fig-cap: "Fractional parameter uncertainty tightens as passes accumulate."
fig, ax = plt.subplots(figsize=(7, 4))
for dof in result.models:
    pts = [(r.iteration, r.max_frac_uncertainty) for r in result.history if r.dof == dof]
    it, frac = zip(*pts)
    ax.semilogy(it, frac, "o-", label=dof)
ax.set_xlabel("iteration"); ax.set_ylabel("max fractional uncertainty"); ax.legend()
plt.show()
```

Recovered vs true transfer function (POS):

```{python}
#| fig-cap: "Recovered POS model over the true twin plant."
freq = np.linspace(0.2, 5.0, 600)
true_pos = backend.plant["POS"]
fig, ax = plt.subplots(figsize=(7, 4))
ax.semilogy(freq, np.abs(true_pos.eval(freq)), label="true")
ax.semilogy(freq, np.abs(result.models["POS"].eval(freq)), "--", label="recovered")
ax.set_xlabel("Frequency [Hz]"); ax.set_ylabel("|G(f)|"); ax.legend()
plt.show()
```
````

- [ ] **Step 2: Render and verify the campaign runs**

```bash
conda run --no-capture-output -n sysid quarto render docs/examples/04-suspension-multidof.qmd
test -f docs/_site/examples/04-suspension-multidof.html && echo OK
```
Expected: render succeeds (the loop runs to completion inside the build); `OK`.

- [ ] **Step 3: Commit**

```bash
git add docs/examples/04-suspension-multidof.qmd
git -c user.name="RXA" -c user.email="rana@caltech.edu" commit -m "docs: example 4 — multi-DoF suspension campaign"
```

---

### Task 9: Example 5 — closed-loop LIGO arm (capstone)

**Files:**
- Create: `docs/examples/05-closed-loop-arm.qmd`

- [ ] **Step 1: Create the example**

````markdown
---
title: "Closed-loop LIGO arm"
---

The realistic case: the plant is a suspension in series with a Fabry–Pérot
cavity, and it is measured **with a feedback loop closed**. What you measure is
the closed-loop response `T = P/(1+PC)`, not the open-loop plant `P`. Fitting
`T` and calling it the plant is biased; knowing the controller `C`, we recover
`P̂ = T̂/(1 − T̂·C)`.

We use `python-control` (already a dependency) for the series/feedback algebra,
and the package's `TFModel` + `InvfreqsEstimator` for the fit. The closed-loop
`T` is treated as the system the instrument measures (you could equally drive it
through the `TwinBackend`, as in the suspension example); the point here is the
controller-division recovery.

```{python}
import numpy as np
import matplotlib.pyplot as plt
import control as ct
from ligo_sysid.model import TFModel
from ligo_sysid.estimators.invfreqs import InvfreqsEstimator

def to_tfmodel(sys):
    sys = ct.tf(sys)
    num = np.atleast_1d(np.asarray(sys.num[0][0], dtype=float))
    den = np.atleast_1d(np.asarray(sys.den[0][0], dtype=float))
    return TFModel(num=num, den=den)

# Open-loop plant P = suspension (1 Hz, Q=20) x cavity pole (100 Hz)
wn, Q = 2*np.pi*1.0, 20.0
susp = ct.tf([wn**2], [1, wn/Q, wn**2])           # DC gain 1
wc = 2*np.pi*100.0
cav = ct.tf([wc], [1, wc])                        # DC gain 1
P = ct.minreal(susp * cav)

C = ct.tf([2.0], [1.0])                           # proportional controller
T = ct.minreal(ct.feedback(P, C))                 # closed-loop, injection -> readout

P_model = to_tfmodel(P)
T_model = to_tfmodel(T)
```

Measure the closed loop (2% noise) and fit it:

```{python}
freq = np.logspace(-1, np.log10(50.0), 500)
H_T = T_model.eval(freq)
rng = np.random.default_rng(0)
H_err = np.abs(H_T) * 0.02
H_meas = H_T + (rng.standard_normal(freq.size) + 1j*rng.standard_normal(freq.size)) * H_err/np.sqrt(2)

prior = TFModel(num=T_model.num.copy(), den=T_model.den.copy())   # same structure/order
T_hat = InvfreqsEstimator().fit(freq, H_meas, H_err, prior)
```

Recover the open-loop plant by dividing out the controller:

```{python}
#| fig-cap: "Loop-ignorant fit (T) is biased; controller-division recovers P."
T_hat_ct = ct.tf(list(T_hat.num), list(T_hat.den))
P_hat = ct.minreal(T_hat_ct / (1 - T_hat_ct * C))
P_hat_model = to_tfmodel(P_hat)

fig, ax = plt.subplots(figsize=(7, 4))
ax.loglog(freq, np.abs(P_model.eval(freq)), label="true plant P")
ax.loglog(freq, np.abs(T_model.eval(freq)), ":", label="closed loop T (biased)")
ax.loglog(freq, np.abs(P_hat_model.eval(freq)), "--", label="recovered P̂")
ax.set_xlabel("Frequency [Hz]"); ax.set_ylabel("|G(f)|"); ax.legend()
plt.show()
```

The feedback suppresses the resonance in `T`, so a loop-ignorant fit understates
the plant's gain there; dividing out `C` restores the true sharp resonance. A
controller-aware estimator that does this inside the package is a natural
follow-up (see the design spec's "future work").
````

- [ ] **Step 2: Render and verify**

```bash
conda run --no-capture-output -n sysid quarto render docs/examples/05-closed-loop-arm.qmd
test -f docs/_site/examples/05-closed-loop-arm.html && echo OK
```
Expected: render succeeds; `OK`.

- [ ] **Step 3: Add the Examples menu to `docs/_quarto.yml`**

In `docs/_quarto.yml`, under `website.navbar.left`, insert an Examples dropdown
between the Tutorial menu and the API Reference entry:

```yaml
      - text: Examples
        menu:
          - href: examples/01-single-resonance.qmd
            text: Single resonance
          - href: examples/02-double-pendulum.qmd
            text: Double pendulum
          - href: examples/03-fabry-perot-cavity.qmd
            text: Fabry-Perot cavity
          - href: examples/04-suspension-multidof.qmd
            text: Multi-DoF suspension
          - href: examples/05-closed-loop-arm.qmd
            text: Closed-loop arm
```

- [ ] **Step 4: Full render and verify all examples**

```bash
conda run --no-capture-output -n sysid bash -c "cd docs && quartodoc build" && conda run --no-capture-output -n sysid quarto render docs
for n in 01-single-resonance 02-double-pendulum 03-fabry-perot-cavity 04-suspension-multidof 05-closed-loop-arm; do
  test -f "docs/_site/examples/$n.html" || { echo "MISSING $n"; exit 1; }
done && echo OK
```
Expected: render succeeds; `OK`.

- [ ] **Step 5: Commit**

```bash
git add docs/examples/05-closed-loop-arm.qmd docs/_quarto.yml
git -c user.name="RXA" -c user.email="rana@caltech.edu" commit -m "docs: example 5 — closed-loop arm; wire Examples menu"
```

---

### Task 10: GitHub Actions workflow (test → build-docs → deploy-docs)

**Files:**
- Create: `.github/workflows/ci.yml`

- [ ] **Step 1: Create `.github/workflows/ci.yml`**

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
  workflow_dispatch:

permissions:
  contents: read
  pages: write
  id-token: write

concurrency:
  group: pages-${{ github.ref }}
  cancel-in-progress: false

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - name: Install package + test deps
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[dev]"
      - name: Run tests
        run: pytest

  build-docs:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - uses: quarto-dev/quarto-actions/setup@v2
      - name: Install package + docs deps
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[docs]"
          python -m ipykernel install --user --name python3 --display-name python3
      - name: Build API reference
        working-directory: docs
        run: quartodoc build
      - name: Render site
        run: quarto render docs
      - name: Upload Pages artifact
        uses: actions/upload-pages-artifact@v3
        with:
          path: docs/_site

  deploy-docs:
    needs: build-docs
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    steps:
      - id: deployment
        uses: actions/deploy-pages@v4
```

- [ ] **Step 2: Validate the workflow YAML parses**

```bash
conda run --no-capture-output -n sysid python -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml')); print('YAML OK')"
```
Expected: `YAML OK`.

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/ci.yml
git -c user.name="RXA" -c user.email="rana@caltech.edu" commit -m "ci: test + build/deploy docs to GitHub Pages"
```

---

### Task 11: README badge, docs link, and final full verification

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add a CI badge and a docs pointer to `README.md`**

Add as the first line of `README.md` (adjust the `OWNER/REPO` slug once the repo
has a GitHub remote; `CaltechExperimentalGravity/system_ident` is the expected
home):

```markdown
[![CI](https://github.com/CaltechExperimentalGravity/system_ident/actions/workflows/ci.yml/badge.svg)](https://github.com/CaltechExperimentalGravity/system_ident/actions/workflows/ci.yml)
```

And add a short "Documentation" section after the intro paragraph:

```markdown
## Documentation

Full docs (pedagogy, API reference, and executed worked examples) are built with
Quarto and published by CI to GitHub Pages. Build them locally with:

```bash
pip install -e ".[docs]"
(cd docs && quartodoc build) && quarto render docs   # output in docs/_site
```
```

- [ ] **Step 2: Final full local build from a clean state**

```bash
rm -rf docs/_site docs/reference docs/.quarto docs/objects.json
conda run --no-capture-output -n sysid bash -c "cd docs && quartodoc build"
conda run --no-capture-output -n sysid quarto render docs
test -f docs/_site/index.html && test -f docs/_site/reference/index.html && echo "SITE OK"
```
Expected: clean build succeeds; `SITE OK`.

- [ ] **Step 3: Confirm the test suite is still green**

```bash
conda run --no-capture-output -n sysid python -m pytest -q
```
Expected: `55 passed, 1 skipped`.

- [ ] **Step 4: Commit**

```bash
git add README.md
git -c user.name="RXA" -c user.email="rana@caltech.edu" commit -m "docs: README CI badge + documentation section"
```

---

## Post-implementation (manual, by the user)

These cannot be automated from here:

1. Create the GitHub remote (e.g. `CaltechExperimentalGravity/system_ident`) and
   `git push -u origin main`.
2. In the repo: **Settings → Pages → Source = "GitHub Actions."**
3. Push / open a PR to trigger CI. PRs run `test` + `build-docs` (verifying the
   site renders); merges to `main` additionally run `deploy-docs` and publish.

Until step 1–2 are done, `deploy-docs` is inert, but `test` and `build-docs`
still run and prove the pipeline.

## Self-review notes

- **Spec coverage:** toolchain (Task 1), site skeleton + index (Task 2), quartodoc
  API reference (Task 3), pedagogy ×5 (Task 4), examples ×5 incl. closed-loop
  capstone (Tasks 5–9), CI test→build→deploy + Pages (Task 10), local-render
  verification + badge + manual Pages note (Task 11, Post-implementation). All
  spec sections covered.
- **API accuracy:** every call (`optimal_excitation`, `parameter_covariance`,
  `InvfreqsEstimator.fit`, `TFModel.from_resonances/from_zpk/from_dict/eval`,
  `double_pendulum`, `RunConfig(raw=...).build_*`, `SysIDLoop.run`,
  `LoopResult.history/models`, `backend.plant[dof]`) matches the signatures in
  `src/ligo_sysid/`.
- **Build ordering:** `quartodoc build` always precedes `quarto render`, and the
  `metadata-files`/`reference` config is introduced only after the reference is
  generated (Task 3), so no render references a missing file.
