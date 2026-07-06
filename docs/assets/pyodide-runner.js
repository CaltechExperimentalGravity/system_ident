/* pyodide-runner — run the REAL system_ident package in the browser (WebAssembly).
 *
 * Boots Pyodide from the CDN once, loads numpy + scipy, installs the freshly-built
 * system_ident wheel (deps=False — numpy/scipy already present, and the package core is
 * numpy/scipy-only), then executes editable code cells. No server, no install: the same
 * pipeline the docs run, in the visitor's browser. See
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
      status("loading numpy + scipy…", "load");
      await py.loadPackage(["numpy", "scipy", "micropip"]);
      status("installing system_ident…", "load");
      await py.runPythonAsync(
        `import micropip\nawait micropip.install(${JSON.stringify(wheelUrl)}, deps=False)`);
      await py.runPythonAsync("import sys, io");
      status("Python ready ✓", "ready");
      return py;
    })();
    return bootPromise;
  }

  async function run(cell, wheelUrl) {
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
      const first = app.querySelector(".pyodide-cell .pyo-run");
      if (first) first.click();
    }
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
