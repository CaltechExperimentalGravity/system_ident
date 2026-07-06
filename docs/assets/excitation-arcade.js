/* Excitation arcade — the landing-page "video game".
 *
 * Drag to redistribute a fixed drive-power budget across frequency; watch the live
 * "time to identify the plant to 5%" fall. Two competing resonances: starving either
 * mode's Q punishes you, and the naive "dump it on the tallest peak" is the WORST move.
 *
 * The math is the Python twin of docs/arcade_reference.py (guarded by tests/test_arcade.py),
 * which is itself faithful to system_ident's TFModel pole convention. Constants below MUST
 * equal that module's. Fisher I = 2*T_REF*df * sum_b Pxx[b] R[b];  frac = sqrt(diag(I^-1))/|θ|;
 * ETA to 5% = C_TIME * max(frac)^2  (Fisher ∝ T, so frac ∝ 1/sqrt(T)).
 */
(function () {
  "use strict";

  // ── locked constants (== docs/arcade_reference.py) ──────────────────────────────
  const N_BINS = 120, F_LO = 0.3, F_HI = 6.0;
  const MODES = [[1.0, 20.0], [2.5, 20.0]];
  const T_REF = 100.0, PX_TOT = 1.0, TARGET = 0.05, C_TIME = 1835.19108;
  const THETA = [MODES[0][0], MODES[0][1], MODES[1][0], MODES[1][1]];
  const AMPS = MODES.map(([f0, Q]) => (2 * Math.PI * f0) ** 2 / Q);

  const FREQ = Array.from({ length: N_BINS }, (_, i) => F_LO + i * (F_HI - F_LO) / (N_BINS - 1));
  const DF = (F_HI - F_LO) / (N_BINS - 1);
  const FLAT_H = PX_TOT / (F_HI - F_LO);

  // ── complex-free FRF: return {re,im} arrays over FREQ for a given theta ──────────
  function frf(theta) {
    const re = new Float64Array(N_BINS), im = new Float64Array(N_BINS);
    for (let k = 0; k < 2; k++) {
      const f0 = theta[2 * k], Q = theta[2 * k + 1], A = AMPS[k];
      const w0 = 2 * Math.PI * f0;
      for (let b = 0; b < N_BINS; b++) {
        const w = 2 * Math.PI * FREQ[b];
        const dr = w0 * w0 - w * w, di = w * w0 / Q;    // denominator
        const den = dr * dr + di * di;
        re[b] += A * dr / den; im[b] += -A * di / den;  // A/(dr+i di)
      }
    }
    return { re, im };
  }

  // Jacobian dG/dθ (central diff) and per-bin real kernels R[b] = Re[dG_i* dG_j].
  // Both are independent of the drive, so precomputed once.
  const D = (function () {
    const cols = [];
    for (let i = 0; i < 4; i++) {
      const h = 1e-6 * Math.max(Math.abs(THETA[i]), 1e-3);
      const hi = THETA.slice(), lo = THETA.slice();
      hi[i] += h; lo[i] -= h;
      const Gh = frf(hi), Gl = frf(lo);
      const re = new Float64Array(N_BINS), im = new Float64Array(N_BINS);
      for (let b = 0; b < N_BINS; b++) {
        re[b] = (Gh.re[b] - Gl.re[b]) / (2 * h);
        im[b] = (Gh.im[b] - Gl.im[b]) / (2 * h);
      }
      cols.push({ re, im });
    }
    return cols;
  })();

  const R = (function () {           // R[b] flattened 4x4 (row-major), symmetric
    const out = [];
    for (let b = 0; b < N_BINS; b++) {
      const m = new Float64Array(16);
      for (let i = 0; i < 4; i++)
        for (let j = 0; j < 4; j++)
          m[4 * i + j] = D[i].re[b] * D[j].re[b] + D[i].im[b] * D[j].im[b]; // Re[conj(Di) Dj]
      out.push(m);
    }
    return out;
  })();

  function fisher(P) {
    const I = new Float64Array(16);
    for (let b = 0; b < N_BINS; b++) {
      const p = P[b], Rb = R[b];
      for (let k = 0; k < 16; k++) I[k] += p * Rb[k];
    }
    for (let k = 0; k < 16; k++) I[k] *= 2 * T_REF * DF;
    return I;
  }

  function inv4(M) {                 // 4x4 inverse via Gauss-Jordan w/ partial pivot
    const a = new Float64Array(32);
    for (let i = 0; i < 4; i++) { for (let j = 0; j < 4; j++) a[8 * i + j] = M[4 * i + j]; a[8 * i + 4 + i] = 1; }
    for (let c = 0; c < 4; c++) {
      let piv = c; for (let r = c + 1; r < 4; r++) if (Math.abs(a[8 * r + c]) > Math.abs(a[8 * piv + c])) piv = r;
      if (piv !== c) for (let k = 0; k < 8; k++) { const t = a[8 * c + k]; a[8 * c + k] = a[8 * piv + k]; a[8 * piv + k] = t; }
      const d = a[8 * c + c] || 1e-30;
      for (let k = 0; k < 8; k++) a[8 * c + k] /= d;
      for (let r = 0; r < 4; r++) if (r !== c) { const f = a[8 * r + c]; for (let k = 0; k < 8; k++) a[8 * r + k] -= f * a[8 * c + k]; }
    }
    const inv = new Float64Array(16);
    for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) inv[4 * i + j] = a[8 * i + 4 + j];
    return inv;
  }

  function fracPerParam(P) {
    const inv = inv4(fisher(P));
    return THETA.map((th, i) => Math.sqrt(Math.max(inv[4 * i + i], 0)) / Math.abs(th));
  }
  function etaPerParam(P) { return fracPerParam(P).map((f) => C_TIME * f * f); }
  function etaOverall(P) { return Math.max.apply(null, etaPerParam(P)); }

  function flatDrive() { return new Float64Array(N_BINS).fill(FLAT_H); }

  function optimalDrive(nIter) {
    nIter = nIter || 16;
    let P = flatDrive();
    for (let it = 0; it < nIter; it++) {
      const inv = inv4(fisher(P));
      const nu = new Float64Array(N_BINS);
      let sum = 0;
      for (let b = 0; b < N_BINS; b++) {
        let tr = 0, Rb = R[b];
        for (let k = 0; k < 16; k++) tr += inv[k] * Rb[k];   // sum_ij inv_ij R_ij (R sym)
        nu[b] = Math.max(PX_TOT * 2 * T_REF * tr, 0);
        P[b] = P[b] * nu[b]; sum += P[b] * DF;
      }
      const s = sum > 0 ? PX_TOT / sum : 1;
      for (let b = 0; b < N_BINS; b++) P[b] *= s;
    }
    return P;
  }

  // Renormalise a drive to the fixed budget with a small floor (every excited line keeps power).
  function normalise(P) {
    const floor = 0.02 * FLAT_H;
    let sum = 0;
    for (let b = 0; b < N_BINS; b++) { if (P[b] < floor) P[b] = floor; sum += P[b] * DF; }
    const s = sum > 0 ? PX_TOT / sum : 1;
    for (let b = 0; b < N_BINS; b++) P[b] *= s;
    return P;
  }

  const ENGINE = { N_BINS, FREQ, DF, THETA, C_TIME, frf, fisher, fracPerParam,
    etaPerParam, etaOverall, flatDrive, optimalDrive, normalise };

  // Node parity harness (tests/); no DOM there.
  if (typeof module !== "undefined" && module.exports) { module.exports = ENGINE; return; }
  if (typeof document === "undefined") return;

  // ── UI ──────────────────────────────────────────────────────────────────────────
  function boot() {
    const root = document.getElementById("excitation-arcade");
    if (!root) return;

    const flatEta = etaOverall(flatDrive());
    const optP = optimalDrive();
    const parEta = etaOverall(optP);

    root.innerHTML = `
      <div class="arc-wrap">
        <canvas class="arc-canvas" aria-label="drag to shape the drive spectrum"></canvas>
        <div class="arc-hud">
          <div class="arc-eta">
            <span class="arc-eta-label">time to identify to 5%</span>
            <span class="arc-eta-val" id="arc-eta">—</span>
            <div class="arc-scale"><div class="arc-scale-fill" id="arc-fill"></div>
              <span class="arc-tick arc-tick-par" id="arc-par"></span>
              <span class="arc-tick arc-tick-flat" id="arc-flat"></span></div>
            <div class="arc-foils"><span id="arc-verdict" class="arc-verdict"></span></div>
          </div>
          <div class="arc-meters" id="arc-meters"></div>
          <div class="arc-controls">
            <button class="arc-btn" id="arc-reset">↺ flat drive</button>
            <button class="arc-btn arc-btn-gold" id="arc-opt">✨ show optimal</button>
          </div>
          <p class="arc-caption">Two resonances share a fixed drive budget. Pile energy where a
          mode's <b>Q</b> is least certain — the tall peak is a trap. Beat the flat sweep
          (${fmt(flatEta)}); the Fisher-optimal “par” is ${fmt(parEta)}.
          <a href="tutorial/fisher.html">Why? →</a></p>
        </div>
      </div>`;

    const cv = root.querySelector(".arc-canvas");
    const ctx = cv.getContext("2d");
    const metersEl = root.querySelector("#arc-meters");
    const PLABEL = ["f₀ #1", "Q #1", "f₀ #2", "Q #2"];
    metersEl.innerHTML = PLABEL.map((l, i) =>
      `<div class="arc-meter"><span class="arc-meter-l">${l}</span>
        <span class="arc-meter-bar"><span class="arc-meter-in" id="arc-m${i}"></span></span></div>`).join("");

    let P = normalise(flatDrive());
    const plant = frf(THETA);
    let plantMax = 0; for (let b = 0; b < N_BINS; b++) plantMax = Math.max(plantMax, Math.hypot(plant.re[b], plant.im[b]));
    const driveDispMax = 6 * FLAT_H;

    // geometry
    let W = 0, H = 0, dpr = 1;
    const PAD = { l: 8, r: 8, t: 26, b: 24 };
    function xPix(f) { return PAD.l + (Math.log(f) - Math.log(F_LO)) / (Math.log(F_HI) - Math.log(F_LO)) * (W - PAD.l - PAD.r); }
    function fAt(px) { return Math.exp((px - PAD.l) / (W - PAD.l - PAD.r) * (Math.log(F_HI) - Math.log(F_LO)) + Math.log(F_LO)); }
    function yDrive(p) { return (H - PAD.b) - Math.min(p / driveDispMax, 1) * (H - PAD.t - PAD.b); }

    function resize() {
      dpr = window.devicePixelRatio || 1;
      W = cv.clientWidth; H = cv.clientHeight;
      cv.width = W * dpr; cv.height = H * dpr; ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      draw();
    }

    function css(v) { return getComputedStyle(root).getPropertyValue(v).trim(); }

    function drawSpectrum(arr, dispMax, stroke, fill) {
      ctx.beginPath();
      for (let b = 0; b < N_BINS; b++) {
        const x = xPix(FREQ[b]), y = (H - PAD.b) - Math.min(arr[b] / dispMax, 1) * (H - PAD.t - PAD.b);
        b ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
      }
      if (fill) {
        ctx.lineTo(xPix(F_HI), H - PAD.b); ctx.lineTo(xPix(F_LO), H - PAD.b); ctx.closePath();
        ctx.fillStyle = fill; ctx.fill();
      }
      if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = 2; ctx.stroke(); }
    }

    function draw() {
      ctx.clearRect(0, 0, W, H);
      const ink = css("--arc-ink") || "#1B2733", gold = css("--arc-gold") || "#C8973A";
      const gray = css("--arc-gray") || "#94A3B8", grid = css("--arc-grid") || "rgba(120,140,170,.18)";
      // frequency gridlines + labels
      ctx.strokeStyle = grid; ctx.fillStyle = gray; ctx.font = "12px 'Outfit',sans-serif"; ctx.textAlign = "center"; ctx.lineWidth = 1;
      [0.5, 1, 2, 5].forEach((f) => { const x = xPix(f); ctx.beginPath(); ctx.moveTo(x, PAD.t); ctx.lineTo(x, H - PAD.b); ctx.stroke(); ctx.fillText(f + " Hz", x, H - 8); });
      // plant landscape (the "terrain": |G(f)|), faint
      const plMag = new Float64Array(N_BINS); for (let b = 0; b < N_BINS; b++) plMag[b] = Math.hypot(plant.re[b], plant.im[b]);
      drawSpectrum(plMag, plantMax * 1.05, null, css("--arc-terrain") || "rgba(120,140,170,.16)");
      // reference drives (flat + optimal), dashed
      ctx.save(); ctx.setLineDash([4, 4]); ctx.globalAlpha = 0.5;
      drawSpectrum(optP, driveDispMax, gold, null);
      ctx.restore();
      // the live drive (gold fill)
      drawSpectrum(P, driveDispMax, gold, hexA(gold, 0.22));
      // peak labels
      ctx.fillStyle = ink; ctx.font = "600 12px 'Outfit',sans-serif";
      ctx.fillText("mode 1", xPix(1.0), PAD.t - 10); ctx.fillText("mode 2", xPix(2.5), PAD.t - 10);
    }

    function fmt(s) {
      if (s < 90) return s.toFixed(0) + " s";
      if (s < 3600) return (s / 60).toFixed(1) + " min";
      return (s / 3600).toFixed(1) + " hr";
    }

    function refreshHUD() {
      const per = etaPerParam(P), overall = Math.max.apply(null, per);
      root.querySelector("#arc-eta").textContent = fmt(overall);
      // you-vs-flat-vs-par scale (log)
      const lg = (x) => Math.log(x), lo = lg(parEta * 0.8), hi = lg(flatEta * 1.15);
      const pos = (x) => Math.max(0, Math.min(1, (lg(x) - lo) / (hi - lo))) * 100;
      root.querySelector("#arc-fill").style.width = (100 - pos(overall)) + "%";
      root.querySelector("#arc-par").style.left = pos(parEta) + "%";
      root.querySelector("#arc-flat").style.left = pos(flatEta) + "%";
      const worst = per.indexOf(overall);
      for (let i = 0; i < 4; i++) {
        const el = document.getElementById("arc-m" + i);
        const frac = Math.max(0, Math.min(1, (lg(Math.max(per[i], 1e-3)) - lg(0.01)) / (lg(flatEta) - lg(0.01))));
        el.style.width = (8 + frac * 92) + "%";
        el.className = "arc-meter-in" + (i === worst ? " worst" : "");
      }
      const v = root.querySelector("#arc-verdict");
      if (overall <= parEta * 1.08) { v.textContent = "🏆 near-optimal!"; v.dataset.k = "win"; }
      else if (overall < flatEta) { v.textContent = "▲ beating the flat sweep"; v.dataset.k = "ok"; }
      else { v.textContent = "flat sweep — now concentrate the energy"; v.dataset.k = "meh"; }
    }

    function sculpt(px, py) {
      const fc = Math.max(F_LO, Math.min(F_HI, fAt(px)));
      const target = Math.max(0, (H - PAD.b - py) / (H - PAD.t - PAD.b)) * driveDispMax;
      const wl = 0.09; // brush width in log10 decades
      for (let b = 0; b < N_BINS; b++) {
        const wgt = Math.exp(-0.5 * ((Math.log10(FREQ[b]) - Math.log10(fc)) / wl) ** 2);
        P[b] += wgt * (target - P[b]) * 0.6;
      }
      P = normalise(P); draw(); refreshHUD();
    }

    // pointer interaction
    let dragging = false;
    function pos(e) { const r = cv.getBoundingClientRect(); const t = e.touches ? e.touches[0] : e; return [t.clientX - r.left, t.clientY - r.top]; }
    cv.addEventListener("pointerdown", (e) => { dragging = true; cv.setPointerCapture(e.pointerId); const [x, y] = pos(e); sculpt(x, y); });
    cv.addEventListener("pointermove", (e) => { if (dragging) { const [x, y] = pos(e); sculpt(x, y); } });
    cv.addEventListener("pointerup", () => { dragging = false; });
    cv.addEventListener("pointercancel", () => { dragging = false; });

    root.querySelector("#arc-reset").addEventListener("click", () => { P = normalise(flatDrive()); draw(); refreshHUD(); });
    root.querySelector("#arc-opt").addEventListener("click", () => {
      // animate flat -> optimal so the shape "forms"
      const start = P.slice(); let t0 = null;
      function step(ts) { if (t0 === null) t0 = ts; const k = Math.min(1, (ts - t0) / 700);
        for (let b = 0; b < N_BINS; b++) P[b] = start[b] + (optP[b] - start[b]) * k;
        draw(); refreshHUD(); if (k < 1) requestAnimationFrame(step); }
      requestAnimationFrame(step);
    });

    function hexA(hex, a) { const h = hex.replace("#", ""); const n = parseInt(h.length === 3 ? h.replace(/(.)/g, "$1$1") : h, 16);
      return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`; }

    new ResizeObserver(resize).observe(cv);
    resize(); refreshHUD();

    // redraw on light/dark toggle (canvas colours are read from CSS vars)
    new MutationObserver(draw).observe(document.body, { attributes: true, attributeFilter: ["class"] });
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
  else boot();
})();
