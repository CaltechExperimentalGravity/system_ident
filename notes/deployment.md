# Deploying system_ident to a CDS workstation

How this project reaches the machine that can talk to real LIGO CDS hardware, and the traps that cost
real time the first time round. **Redacted by design:** this repo is public, so concrete hostnames,
accounts, on-machine paths and SSH mechanics are held in a local, gitignored `ssh_deploy.md` instead.
Placeholders below are written `<like-this>`.

If you are picking this up and do not have that local file, ask — do not guess, and do not go hunting
for the values in a public repo.

## Topology

| role | what runs there |
|---|---|
| **Dev machine** | All code edits. Primary conda env `sysid` (py3.12). A second, py3.9 CDS-baseline env also exists here — see [Two environments](#two-environments) — which is what makes the deployment baseline testable *locally*, without the CDS machine. |
| **CDS workstation** | Simulation and `rtsfreerun` dry-runs; and, human-gated, the real hardware. Reached over SSH with key-based auth. Project path `<remote-path>`. |

The `system_ident` checkout on the CDS workstation **exists as of 2026-08-03**, but was delivered by
`rsync` of the working tree, **not** `git clone` — the deploy key below needs repo-admin rights nobody
on the current side has. Consequences: it re-syncs by `rsync` only (`git fetch`/`pull` there will fail,
there is no key), and its provenance is "the commit the tree was copied at", not a verifiable remote
ref. The gitignored local files (this repo's sensitive manifest and access note) are **excluded** from
that copy on purpose — they describe access to the very machine being copied to, which is a shared
account. See issue #22.

## Two environments — and the old one *was* negotiable after all

> **Corrected 2026-08-03.** This section previously said a modern environment "cannot" be solved
> alongside the CDS packages. That was wrong, and the correction is the interesting part: the claim
> described the environment **installed** on that machine (a 2022 `cds-crtools 3.1.2` clone) and
> generalised it to **the channel**. The channel disagrees — CDS publishes a python 3.11 environment
> and every CDS package is on conda-forge with `py311`/`py312`/`py313` builds. The lesson worth keeping:
> *"the box has X" is not evidence for "only X is available".* Original text retained below the new
> target for the fallback rungs that still reference it.

- **Hardware path (current): `sysid_deploy` — python 3.11 with CDS 4.1.4.** A lean environment pinned
  to the CDS-published `cds-py311` set: declarative spec `environment_deploy.yml`, exact export
  `environment_deploy_lock.yml`. Create it, then `pip install -e . --no-deps`. `--no-deps` stays
  **mandatory** — all six core dependencies are pinned in the spec and nothing may bump numpy/scipy
  under the compiled `foton`/`awg`/`nds2` extensions.
  Baseline: **python 3.11.15 / numpy 1.26.4 / scipy 1.13.1 / control 0.10.2 / slycot 0.6.1**.
- **Dev / docs path:** the modern `sysid` environment stays primary. Unchanged.

**The gate is the same ten test files, and it passes identically on the new stack:** `test_step5_safety`,
`test_step7_loop`, `test_step8_cli`, `test_periodic_measurement`, `test_rtsfreerun_backend`,
`test_excitation`, `test_step4_twin`, `test_step6_estimator`, `test_step12_ml_estimator`,
`test_resolution` → **61 passed / 1 skipped** on py3.11, the same as on py3.9. The *full* suite gives
290 passed / 8 failed / 17 skipped, and all 8 failures are `np.trapezoid` (numpy < 2.0) in the
arcade/playground half — issue #23. So **the deployment gate is that named subset**, not "the suite
passes".

**Two caveats that matter more than the version numbers:**

1. **The awg injection path is unverified on 4.1.4.** The site's front ends run advLigoRTS branch-3.4
   (2017) and the only awg client ever proven against them is 3.1.2. The *read* half is verified — the
   `nds2` client versions are identical across both stacks, and a live read-only `getdata` returns data
   at the expected rate. The *injection* half needs an operator, human-chosen channels, and separate
   approval for **every** injection. Fallback ladder in spec §8a; issue #27.
2. **Practical:** that machine's conda is 22.11.1 with no mamba and cannot solve this in reasonable
   time; use a static `micromamba` from a user-owned directory and leave base conda alone.

<details>
<summary>Original (incorrect) text, retained for the fallback rungs</summary>

The CDS control packages — `foton`, `python-foton`, `python-awg`, `libawg`, `dtt-*` (all 3.1.2) and
`python-nds2-client` — are **py3.9-only builds** in the channel available to that machine. A modern
environment therefore **cannot** be solved together with them; this is why the sibling project
`automatic-frf-measurement` had to pin `python=3.9` outright and drop the `anaconda` metapackage. Issue
#22 verifies this on the box rather than taking it on trust, but the expected outcome is:

- **Hardware path:** clone the site `cds` conda environment — it is the ABI reference the compiled
  `foton`/`awg`/`nds2` extension modules were built against — then `pip install -e . --no-deps`.
  All six of this project's core dependencies (numpy, scipy, control, slycot, pyyaml, matplotlib) are
  **already in that lock**, so `--no-deps` is both sufficient and **mandatory**: nothing may bump
  numpy/scipy underneath those extensions.
- **Dev / docs path:** the modern `sysid` environment stays primary. Nothing about this changes it.

The resulting baseline is **python 3.9.13 / numpy 1.22.4 / scipy 1.8.1 / control 0.9.2 /
slycot 0.4.0.0**.

**The important measurement:** the CDS-relevant half of this repo *already passes* on that baseline —
→ **61 passed / 1 skipped**. The *full* suite does not (the DARM / MIMO / SOS / arcade / playground half
needs the fixes in issue #23), so **the deployment gate is that named subset**, not "the suite passes".
Run it on the dev machine's py3.9 env before deploying; that is far cheaper than discovering it
remotely.

</details>

## Three traps

These are not bugs in this code. Each one cost the sibling project time on its first hardware run.

1. **`conda` is not on the non-interactive PATH.** The conda init lives in `~/.bashrc`, which returns
   early for non-interactive shells — so `ssh <host> "conda ..."` fails with `conda: command not
   found`. Source the profile script first:
   `ssh <host> 'source <conda-root>/etc/profile.d/conda.sh && conda env list'`.
   Interactive sessions are fine.

2. **The site `IFO` environment variable is not set by conda**, and `cdsutils/nds.py` reads it at
   *import* time — without it you get `NDSError: IFO environment variable not specified`. Export it
   before running anything on the hardware path.
   **Its value is site configuration, not a constant** (a site-specific prefix; see the local
   `ssh_deploy.md`). In this project it comes from the site profile (issue #26) and is never hardcoded.

3. **Do not source the site workstation rc script.** It prepends a **legacy site CDS python stack** to
   `PYTHONPATH`, after which `import cdsutils` picks up that copy and dies with
   `ModuleNotFoundError: No module named 'matrix'`. Only `PYTHONPATH` is poisoned — `unset PYTHONPATH`
   recovers.

## Git access needs a deploy key for *this* repo

Every git-forge SSH key stored **on** the CDS workstation is passphrase-protected, deliberately. Over
one-shot SSH there is no TTY and no askpass, so remote `git fetch`/`pull` either hangs or fails, and no
agent runs in a non-interactive shell.

The workaround is a **read-only deploy key held on the dev machine** and forwarded over a single,
scoped agent socket — not `ForwardAgent yes`, because the remote account is shared. The sibling
project's key does **not** transfer: it is registered on a different forge and a different project, so
`system_ident` needs its own. Setup steps are in the local `ssh_deploy.md`.

### Rules

- **Never push from the CDS workstation.** Publishing happens on the dev machine, and the deploy key is
  read-only so the server refuses anyway — but do not attempt it.
- **Never run remote git over the plain login alias** — it forwards no agent, so `fetch`/`pull`/`clone`
  will hang on a passphrase prompt. Local-only git on that machine (`status`, `log`, `diff`, `show`,
  `branch`, `add`, `commit`, `switch`, `merge <local>`) is fine.
- Use `GIT_SSH_COMMAND="ssh -o BatchMode=yes"` for remote git, so a credential prompt fails fast
  instead of hanging.
- Never force-push, `reset --hard` or `clean -fdx` on the CDS workstation without explicit
  confirmation.
- Confirm work is committed **and pushed** from the dev machine before deploying.

## Hardware safety

Deploying is not permission to actuate. Before anything can be injected:

- **Only humans authorize the hardware.** Automated tooling — including AI agents driving this
  codebase — must never configure real hardware and must never self-authorize or assume approval.
- **Every individual injection needs separate operator approval. One approval never carries over.**
- Simulation, twin runs, `--help` and import smoke tests need no approval.
- The **read-only** real-transport smoke test — probe the channel rate, read a quiet segment,
  **no injection** — is safe to run without operator approval, and is the only part of the current
  campaign that touches the real CDS libraries at all.

Full rules: `CLAUDE.md` §Hardware safety and `docs/tutorial/safety-and-ops.qmd`.

## See also

- `notes/cds-hardware-bringup-2026-08.md` — the CDS backend campaign handoff.
- `docs/superpowers/specs/2026-08-03-cds-hardware-backend-design.md` §8 — the environment analysis in
  full, with the compat delta.
- Issues #22 (environment) and #23 (py3.9 compatibility + a CI leg).
