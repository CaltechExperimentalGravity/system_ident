/* pyodide-runner — run the REAL system_ident package in the browser (WebAssembly).
 *
 * Boots Pyodide from the CDN once, loads numpy + scipy + matplotlib, installs the freshly-built
 * system_ident wheel and python-control (both deps=False — their deps are the Pyodide builds
 * above), then executes editable code cells. This makes the browser run not just the numpy/scipy
 * core but the time-domain / closed-loop machinery (control.forced_response, CoupledLoop). No
 * server, no install: the same pipeline the docs run, in the visitor's browser. See
 * docs/superpowers/specs/2026-07-05-excitation-arcade-hero-design.md (phase 3).
 */
(function () {
  "use strict";
  const PYODIDE_INDEX = "https://cdn.jsdelivr.net/pyodide/v0.26.2/full/";
  let bootPromise = null;
  let statusEls = [];

  function status(text, kind) {
    statusEls.forEach((e) => { e.textContent = text; e.dataset.k = kind || ""; });
  }

  function boot(wheelUrl) {
    if (bootPromise) return bootPromise;
    bootPromise = (async () => {
      status("loading Python — first run fetches ~30 MB, ~30–60 s…", "load");
      await new Promise((res, rej) => {
        const s = document.createElement("script");
        s.src = PYODIDE_INDEX + "pyodide.js";
        s.onload = res;
        s.onerror = () => rej(new Error("could not load Pyodide from the CDN (offline?)"));
        document.head.appendChild(s);
      });
      const py = await loadPyodide({ indexURL: PYODIDE_INDEX });
      status("loading numpy + scipy + matplotlib…", "load");
      await py.loadPackage(["numpy", "scipy", "matplotlib", "micropip"]);
      status("installing system_ident…", "load");
      await py.runPythonAsync(
        `import micropip\nawait micropip.install(${JSON.stringify(wheelUrl)}, deps=False)`);
      // python-control is pure Python; its deps (numpy/scipy/matplotlib) are the Pyodide builds
      // loaded above, so install it with deps=False. Retry once — the first PyPI fetch can flake
      // on a cold cache. This is what lets the browser drive the time-domain / closed-loop
      // machinery (control.forced_response, CoupledLoop, MIMOTwinBackend).
      status("installing python-control…", "load");
      for (let attempt = 1; ; attempt++) {
        try {
          await py.runPythonAsync(
            "import micropip\nawait micropip.install('control', deps=False)\nimport control");
          break;
        } catch (e) {
          if (attempt >= 2) throw new Error("python-control failed to install: " + e);
          await new Promise((r) => setTimeout(r, 1500));
        }
      }
      await py.runPythonAsync("import sys, io");
      status("Python ready ✓", "ready");
      return py;
    })();
    return bootPromise;
  }

  // Runs share one Python interpreter and swap sys.stdout around each cell's
  // code, so two cells executing at once would interleave stdout and cross
  // their outputs. Serialize every run through one promise chain: a click while
  // another cell is running just queues behind it (boot is still shared).
  let runQueue = Promise.resolve();
  function run(cell, wheelUrl) {
    runQueue = runQueue.then(() => runOne(cell, wheelUrl), () => runOne(cell, wheelUrl));
    return runQueue;
  }

  async function runOne(cell, wheelUrl) {
    const code = cell.querySelector(".pyo-code").value;
    const out = cell.querySelector(".pyo-out");
    const btn = cell.querySelector(".pyo-run");
    btn.disabled = true;
    out.textContent = "";
    try {
      const py = await boot(wheelUrl);
      status("running…", "load");
      py.runPython("_saved_stdout = sys.stdout; sys.stdout = io.StringIO()");
      let val;
      try {
        val = await py.runPythonAsync(code);
      } finally {
        const captured = py.runPython(
          "_c = sys.stdout.getvalue(); sys.stdout = _saved_stdout; _c");
        out.textContent = captured + (typeof val === "string" ? val : "");
        if (val && typeof val.destroy === "function") val.destroy();
      }
      status("Python ready ✓", "ready");
    } catch (e) {
      out.textContent = "⚠ " + (e && e.message ? e.message : e);
      status("error", "err");
    } finally {
      btn.disabled = false;
    }
  }

  function init() {
    const app = document.getElementById("pyodide-app");
    if (!app) return;
    const wheelUrl = app.dataset.wheel;
    statusEls = Array.from(app.querySelectorAll(".pyo-status"));
    app.querySelectorAll(".pyodide-cell").forEach((cell) => {
      const ta = cell.querySelector(".pyo-code");
      // Tab inserts spaces instead of moving focus (it's a code editor).
      ta.addEventListener("keydown", (e) => {
        if (e.key === "Tab") {
          e.preventDefault();
          const s = ta.selectionStart, en = ta.selectionEnd;
          ta.value = ta.value.slice(0, s) + "    " + ta.value.slice(en);
          ta.selectionStart = ta.selectionEnd = s + 4;
        }
      });
      // size the textarea to its content
      ta.style.height = (ta.scrollHeight + 4) + "px";
      cell.querySelector(".pyo-run").addEventListener("click", () => run(cell, wheelUrl));
    });
    // opt-in autorun (?pyorun=1) — for smoke tests; never auto-boots for normal visitors,
    // who click Run so the ~30 MB Pyodide download stays their explicit choice.
    if (/[?&]pyorun=1/.test(location.search)) {
      app.querySelectorAll(".pyodide-cell .pyo-run").forEach((b) => b.click());
    }
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
