---
name: rtsfreerun-env-strategy
description: "How to set up the env for the RTSfreerun twin path — separate repos, build RTS models into sysid env"
metadata: 
  node_type: memory
  type: feedback
---

`system_ident` and `digital_twin`/`rtsfreerun` are kept as **separate repos**. To run the
`--rtsfreerun` path, do **NOT** install `system_ident` into the twin env. Instead build the
compiled RTS model(s) **into this repo's `sysid` conda env** so both import from one interpreter.
(`rtsfreerun` allows only one model per Python process — build whichever a demo needs.)

**Why:** keeps the two codebases independent; explicit project decision (2026-06-18).

**How to apply:** `sysid` env comes from `conda env create -f environment.yml`. Build a model
(one-time toolchain: `conda install -c conda-forge make cmake spdlog rapidjson pybind11`) from the
`digital_twin/rtsfreerun` dir: `PATH=$CONDA_PREFIX/bin:$PATH
RCG_LIB_PATH=…/rtsfreerun-models/<model> model=<model> pip install .`. **`RCG_LIB_PATH` must point at
the model source dir** or the build fails ("Couldn't find model file <model>.mdl"). Verified recipe
lives in the committed README.md §"Run against the RTSfreerun digital twin". The twin box
(macOS/arm64) does have `twin`+`rtsfreerun-dev` envs (an old `.llm` note
claiming otherwise was wrong). Related: [[no-foton-export]].
