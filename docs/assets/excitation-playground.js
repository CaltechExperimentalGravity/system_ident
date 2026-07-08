/* Excitation playground — the "try every drive" sandbox.
 *
 * A fixed drive-power budget, ten ways to spend it: optimal / flat / shaped multisines
 * (Schroeder, random, or cophased phase), linear & log swept sines, and white / shaped noise.
 * Each is scored on TWO axes at the same budget:
 *   • time-to-5%  — Fisher/Cramer-Rao, a function of the power spectrum Pxx ALONE.
 *   • crest factor — max|x|/rms of the synthesized DAC waveform, set by the PHASES.
 * The lesson the sandbox makes you feel: estimation speed lives in the spectrum, headroom
 * lives in the phases. The Fisher-optimal spectrum collapses onto ~2 tones, so it is fast AND
 * low-crest for ANY phase (Schroeder == random there); Schroeder's low-crest win is real only
 * for BROADBAND multisines, where the power is spread over many comparable lines.
 *
 * The Fisher/CRB kernel is the arcade's (docs/assets/excitation-arcade.js), and the whole
 * engine is the Python twin of docs/playground_reference.py, guarded by
 * tests/test_playground.py. Constants + formulas MUST stay in lock-step with that module.
 */
(function () {
  "use strict";

  // ── locked constants (== docs/playground_reference.py / arcade_reference.py) ──────
  const N_BINS = 120, F_LO = 0.3, F_HI = 6.0;
  const MODES0 = [[1.0, 20.0], [2.5, 20.0]];
  const T_REF = 100.0, PX_TOT = 1.0, C_TIME = 1835.19108;
  const FS = 40.0, NT = 2048, SEED = 0x51D;        // time-synthesis grid + PRNG seed

  const FREQ = Array.from({ length: N_BINS }, (_, i) => F_LO + i * (F_HI - F_LO) / (N_BINS - 1));
  const DF = (F_HI - F_LO) / (N_BINS - 1);
  const TVEC = Array.from({ length: NT }, (_, j) => j / FS);

  // ── deterministic PRNG shared with the Python reference (mulberry32) ──────────────
  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6D2B79F5) >>> 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  // ── plant FRF (two resonances) + drive-independent information kernels ─────────────
  let THETA = [MODES0[0][0], MODES0[0][1], MODES0[1][0], MODES0[1][1]];
  let AMPS = MODES0.map(([f0, Q]) => (2 * Math.PI * f0) ** 2 / Q);
  let R = null, plantCache = null;

  // NB: the per-mode amplitude AMPS[k] is held FIXED during Fisher differentiation (the
  // parameters are the poles f0,Q only), matching arcade_reference.plant_frf. rebuildPlant
  // sets AMPS from the base plant before finite-differencing, so frf must read AMPS here —
  // recomputing A from the perturbed theta would inject spurious dA/dtheta into the Jacobian.
  function frf(theta) {
    const re = new Float64Array(N_BINS), im = new Float64Array(N_BINS);
    for (let k = 0; k < 2; k++) {
      const f0 = theta[2 * k], Q = theta[2 * k + 1], A = AMPS[k];
      const w0 = 2 * Math.PI * f0;
      for (let b = 0; b < N_BINS; b++) {
        const w = 2 * Math.PI * FREQ[b];
        const dr = w0 * w0 - w * w, di = w * w0 / Q;
        const den = dr * dr + di * di;
        re[b] += A * dr / den; im[b] += -A * di / den;
      }
    }
    return { re, im };
  }

  function rebuildPlant(theta) {          // recompute Jacobian kernels R[b] at this plant
    THETA = theta.slice();
    AMPS = [(2 * Math.PI * theta[0]) ** 2 / theta[1], (2 * Math.PI * theta[2]) ** 2 / theta[3]];
    const D = [];
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
      D.push({ re, im });
    }
    R = [];
    for (let b = 0; b < N_BINS; b++) {
      const m = new Float64Array(16);
      for (let i = 0; i < 4; i++)
        for (let j = 0; j < 4; j++)
          m[4 * i + j] = D[i].re[b] * D[j].re[b] + D[i].im[b] * D[j].im[b];
      R.push(m);
    }
    plantCache = frf(THETA);
  }
  rebuildPlant(THETA);

  function fisher(P) {
    const I = new Float64Array(16);
    for (let b = 0; b < N_BINS; b++) { const p = P[b], Rb = R[b]; for (let k = 0; k < 16; k++) I[k] += p * Rb[k]; }
    for (let k = 0; k < 16; k++) I[k] *= 2 * T_REF * DF;
    return I;
  }
  function inv4(M) {
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

  // ── power spectra Pxx(f) (each normalized to the fixed budget) ─────────────────────
  function normalize(P) {                 // trapezoid area -> PX_TOT
    let area = 0; for (let b = 1; b < N_BINS; b++) area += 0.5 * (P[b] + P[b - 1]) * DF;
    const s = area > 0 ? PX_TOT / area : 1; const out = new Float64Array(N_BINS);
    for (let b = 0; b < N_BINS; b++) out[b] = Math.max(P[b], 0) * s; return out;
  }
  function powerFlat() { return normalize(new Float64Array(N_BINS).fill(1)); }
  function powerShaped(alpha) { const P = new Float64Array(N_BINS); for (let b = 0; b < N_BINS; b++) P[b] = Math.pow(FREQ[b], -alpha); return normalize(P); }
  function powerOptimal(nIter) {
    nIter = nIter || 16; let P = powerFlat();
    for (let it = 0; it < nIter; it++) {
      const inv = inv4(fisher(P)); const nu = new Float64Array(N_BINS);
      for (let b = 0; b < N_BINS; b++) { let tr = 0, Rb = R[b]; for (let k = 0; k < 16; k++) tr += inv[k] * Rb[k]; nu[b] = Math.max(PX_TOT * 2 * T_REF * tr, 0); P[b] = P[b] * nu[b]; }
      P = normalize(P);
    }
    return P;
  }

  // ── time-domain synthesis + crest factor ───────────────────────────────────────────
  function crest(x) { let s = 0, mx = 0; for (let j = 0; j < x.length; j++) { s += x[j] * x[j]; const a = Math.abs(x[j]); if (a > mx) mx = a; } const rms = Math.sqrt(s / x.length); return rms > 0 ? mx / rms : 0; }

  function schroederPhases(P) {           // -2*pi*sum_{l<k}(k-l) q_l, q = P/sum(P)
    let tot = 0; for (let b = 0; b < N_BINS; b++) tot += P[b];
    const phi = new Float64Array(N_BINS); let run = 0, acc = 0;
    for (let k = 1; k < N_BINS; k++) { run += P[k - 1] / tot; acc += run; phi[k] = -2 * Math.PI * acc; }
    return phi;
  }
  function multisine(P, phase, seed) {
    const amp = new Float64Array(N_BINS); for (let b = 0; b < N_BINS; b++) amp[b] = Math.sqrt(2 * Math.max(P[b], 0) * DF);
    let phi;
    if (phase === "schroeder") { const Pf = new Float64Array(N_BINS); for (let b = 0; b < N_BINS; b++) Pf[b] = Math.max(P[b], 1e-30); phi = schroederPhases(Pf); }
    else if (phase === "random") { const rnd = mulberry32(seed || SEED); phi = new Float64Array(N_BINS); for (let b = 0; b < N_BINS; b++) phi[b] = 2 * Math.PI * rnd(); }
    else phi = new Float64Array(N_BINS);
    const x = new Float64Array(NT);
    for (let b = 0; b < N_BINS; b++) { const a = amp[b], w = 2 * Math.PI * FREQ[b], p = phi[b]; if (a === 0) continue; for (let j = 0; j < NT; j++) x[j] += a * Math.cos(w * TVEC[j] + p); }
    return x;
  }
  function sweptSine() {                  // constant-amplitude linear sweep over the band (crest ~ sqrt2)
    const T = NT / FS, mu = (F_HI - F_LO) / T, x = new Float64Array(NT);
    for (let j = 0; j < NT; j++) { const t = TVEC[j]; x[j] = Math.sin(2 * Math.PI * (F_LO * t + 0.5 * mu * t * t)); }
    return x;
  }

  // ── the catalog: everything the scoreboard races ───────────────────────────────────
  const CATALOG = [
    { key: "opt_schroeder", label: "Optimal · Schroeder", group: "Multisine", power: "optimal", champ: true },
    { key: "opt_random", label: "Optimal · random phase", group: "Multisine", power: "optimal" },
    { key: "flat_schroeder", label: "Flat · Schroeder", group: "Multisine", power: "flat" },
    { key: "flat_random", label: "Flat · random phase", group: "Multisine", power: "flat" },
    { key: "cophased", label: "Cophased (impulse)", group: "Multisine", power: "flat" },
    { key: "chirp_lin", label: "Swept sine", group: "Swept sine", power: "flat" },
    { key: "white", label: "Broadband white noise", group: "Noise", power: "flat" },
    { key: "pink", label: "Shaped noise · 1/f", group: "Noise", power: "pink" },
  ];
  function powerOf(e) { return e.power === "optimal" ? powerOptimal() : e.power === "pink" ? powerShaped(1) : powerFlat(); }
  function waveOf(e) {
    const P = powerOf(e);
    switch (e.key) {
      case "opt_schroeder": return multisine(P, "schroeder");
      case "opt_random": return multisine(P, "random");
      case "flat_schroeder": return multisine(P, "schroeder");
      case "flat_random": return multisine(P, "random");
      case "cophased": return multisine(P, "zero");
      case "chirp_lin": return sweptSine();
      case "white": return multisine(P, "random");
      case "pink": return multisine(P, "random");
    }
  }
  function scoreOf(e) { const P = powerOf(e); return { eta: etaOverall(P), crest: crest(waveOf(e)), Pxx: P }; }

  const ENGINE = {
    N_BINS, FREQ, DF, TVEC, THETA: () => THETA, C_TIME, CATALOG,
    frf, fisher, fracPerParam, etaPerParam, etaOverall, rebuildPlant,
    powerFlat, powerShaped, powerOptimal, multisine, sweptSine, crest, schroederPhases,
    powerOf, waveOf, scoreOf, mulberry32,
  };
  if (typeof module !== "undefined" && module.exports) { module.exports = ENGINE; return; }
  if (typeof window !== "undefined") window.EXCITATION_PLAYGROUND = ENGINE;  // parity/test harness
  if (typeof document === "undefined") return;

  // ── UI ──────────────────────────────────────────────────────────────────────────
  function fmtTime(s) { if (s < 90) return s.toFixed(0) + " s"; if (s < 3600) return (s / 60).toFixed(1) + " min"; if (s < 86400) return (s / 3600).toFixed(1) + " hr"; return (s / 86400).toFixed(1) + " d"; }
  function boot() {
    const root = document.getElementById("excitation-playground");
    if (!root) return;

    root.innerHTML = `
      <div class="xpg">
        <div class="xpg-side">
          <div class="xpg-pick" id="xpg-pick"></div>
          <div class="xpg-sliders">
            <label class="xpg-sl">mode 1 &fnof;<sub>0</sub> <b id="xpg-f1v"></b>
              <input type="range" id="xpg-f1" min="0.5" max="1.6" step="0.01" value="1.0"></label>
            <label class="xpg-sl">mode 1 Q <b id="xpg-q1v"></b>
              <input type="range" id="xpg-q1" min="4" max="60" step="1" value="20"></label>
            <label class="xpg-sl">mode 2 &fnof;<sub>0</sub> <b id="xpg-f2v"></b>
              <input type="range" id="xpg-f2" min="1.8" max="5.0" step="0.01" value="2.5"></label>
            <label class="xpg-sl">mode 2 Q <b id="xpg-q2v"></b>
              <input type="range" id="xpg-q2" min="4" max="60" step="1" value="20"></label>
          </div>
          <div class="xpg-btns">
            <button class="xpg-btn xpg-btn-gold" id="xpg-race">▶ Race all drives</button>
            <button class="xpg-btn" id="xpg-plantreset">↺ reset plant</button>
          </div>
        </div>

        <div class="xpg-main">
          <div class="xpg-readout">
            <div class="xpg-big"><span class="xpg-big-l">time to identify to 5%</span>
              <span class="xpg-big-v" id="xpg-eta">—</span></div>
            <div class="xpg-big"><span class="xpg-big-l">drive crest factor (at the DAC)</span>
              <span class="xpg-big-v" id="xpg-crest">—</span></div>
          </div>
          <figure class="xpg-fig"><figcaption>drive spectrum <i>P<sub>xx</sub>(f)</i> over the plant |G(f)|</figcaption>
            <canvas class="xpg-cv" id="xpg-spec" height="200"></canvas></figure>
          <figure class="xpg-fig"><figcaption>the drive the DAC emits — crest = peak / rms</figcaption>
            <canvas class="xpg-cv" id="xpg-wave" height="150"></canvas></figure>
        </div>

        <div class="xpg-board">
          <div class="xpg-board-h">Scoreboard <span>same power budget · fuller bar = better (faster / lower crest)</span></div>
          <div class="xpg-rows" id="xpg-rows"></div>
          <p class="xpg-note">Speed is set by the <b>power spectrum</b>, crest by the <b>phases</b>.
          The optimal drive is nearly <b>two tones</b>, so it is fast <i>and</i> low-crest for any
          phase — <code>Optimal · Schroeder</code> ≈ <code>Optimal · random</code> here. Schroeder
          phase only matters once you go <b>broadband</b>: it pulls a flat multisine's crest from
          ~3.4 down to ~1.9. Swept sines match the crest but waste off-resonance power; broadband
          noise loses on speed. The whole
          <a href="tutorial/why-optimal-excitation.html">why-optimal-excitation</a> story, to play with.
          <a href="examples/interactive.html">▶ run it in the real package →</a></p>
        </div>
      </div>`;

    // build picker
    const pick = root.querySelector("#xpg-pick");
    let html = "", lastGroup = "";
    CATALOG.forEach((e) => {
      if (e.group !== lastGroup) { html += `<div class="xpg-grp">${e.group}</div>`; lastGroup = e.group; }
      html += `<button class="xpg-opt${e.champ ? " champ" : ""}" data-k="${e.key}">${e.label}${e.champ ? " ★" : ""}</button>`;
    });
    pick.innerHTML = html;

    const specCv = root.querySelector("#xpg-spec"), waveCv = root.querySelector("#xpg-wave");
    const sctx = specCv.getContext("2d"), wctx = waveCv.getContext("2d");
    let sel = "opt_schroeder", scores = {}, racing = false;

    function css(v, d) { return getComputedStyle(root).getPropertyValue(v).trim() || d; }
    function recomputeAll() { scores = {}; CATALOG.forEach((e) => { scores[e.key] = scoreOf(e); }); }

    function xPix(cv, f, W) { const L = 34, R = 8; return L + (Math.log(f) - Math.log(F_LO)) / (Math.log(F_HI) - Math.log(F_LO)) * (W - L - R); }

    function drawSpec() {
      const dpr = window.devicePixelRatio || 1, W = specCv.clientWidth, H = specCv.clientHeight;
      specCv.width = W * dpr; specCv.height = H * dpr; sctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      sctx.clearRect(0, 0, W, H);
      const gold = css("--xpg-gold", "#C8973A"), gray = css("--xpg-gray", "#6b7a8d"), grid = css("--xpg-grid", "rgba(120,140,170,.16)");
      const PADt = 10, PADb = 20;
      // gridlines
      sctx.strokeStyle = grid; sctx.fillStyle = gray; sctx.font = "11px 'Outfit',sans-serif"; sctx.textAlign = "center"; sctx.lineWidth = 1;
      [0.5, 1, 2, 5].forEach((f) => { const x = xPix(specCv, f, W); sctx.beginPath(); sctx.moveTo(x, PADt); sctx.lineTo(x, H - PADb); sctx.stroke(); sctx.fillText(f + " Hz", x, H - 6); });
      // plant terrain |G|
      const pl = plantCache; let plMax = 0; const plMag = new Float64Array(N_BINS);
      for (let b = 0; b < N_BINS; b++) { plMag[b] = Math.hypot(pl.re[b], pl.im[b]); if (plMag[b] > plMax) plMax = plMag[b]; }
      sctx.beginPath();
      for (let b = 0; b < N_BINS; b++) { const x = xPix(specCv, FREQ[b], W), y = (H - PADb) - (plMag[b] / (plMax * 1.05)) * (H - PADt - PADb); b ? sctx.lineTo(x, y) : sctx.moveTo(x, y); }
      sctx.lineTo(xPix(specCv, F_HI, W), H - PADb); sctx.lineTo(xPix(specCv, F_LO, W), H - PADb); sctx.closePath();
      sctx.fillStyle = css("--xpg-terrain", "rgba(120,140,170,.16)"); sctx.fill();
      // drive Pxx
      const P = scores[sel].Pxx; let pMax = 0; for (let b = 0; b < N_BINS; b++) if (P[b] > pMax) pMax = P[b];
      const disp = Math.max(pMax, PX_TOT / (F_HI - F_LO) * 1.2);
      sctx.beginPath();
      for (let b = 0; b < N_BINS; b++) { const x = xPix(specCv, FREQ[b], W), y = (H - PADb) - Math.min(P[b] / disp, 1) * (H - PADt - PADb); b ? sctx.lineTo(x, y) : sctx.moveTo(x, y); }
      sctx.lineTo(xPix(specCv, F_HI, W), H - PADb); sctx.lineTo(xPix(specCv, F_LO, W), H - PADb); sctx.closePath();
      sctx.fillStyle = hexA(gold, 0.22); sctx.fill();
      sctx.strokeStyle = gold; sctx.lineWidth = 2;
      sctx.beginPath();
      for (let b = 0; b < N_BINS; b++) { const x = xPix(specCv, FREQ[b], W), y = (H - PADb) - Math.min(P[b] / disp, 1) * (H - PADt - PADb); b ? sctx.lineTo(x, y) : sctx.moveTo(x, y); }
      sctx.stroke();
    }

    function drawWave() {
      const dpr = window.devicePixelRatio || 1, W = waveCv.clientWidth, H = waveCv.clientHeight;
      waveCv.width = W * dpr; waveCv.height = H * dpr; wctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      wctx.clearRect(0, 0, W, H);
      const gold = css("--xpg-gold", "#C8973A"), gray = css("--xpg-gray", "#6b7a8d");
      const x = waveOf(CATALOG.find((e) => e.key === sel));
      let mx = 0; for (let j = 0; j < NT; j++) { const a = Math.abs(x[j]); if (a > mx) mx = a; }
      const mid = H / 2, sc = (H / 2 - 6) / (mx || 1);
      // rms band
      let s = 0; for (let j = 0; j < NT; j++) s += x[j] * x[j]; const rms = Math.sqrt(s / NT);
      wctx.fillStyle = hexA(gray, 0.14); wctx.fillRect(0, mid - rms * sc, W, 2 * rms * sc);
      wctx.strokeStyle = gray; wctx.lineWidth = 1; wctx.beginPath(); wctx.moveTo(0, mid); wctx.lineTo(W, mid); wctx.stroke();
      // waveform (first ~1/4 window, enough to see structure)
      const nShow = NT, step = W / nShow;
      wctx.strokeStyle = gold; wctx.lineWidth = 1.4; wctx.beginPath();
      for (let j = 0; j < nShow; j++) { const px = j * step, py = mid - x[j] * sc; j ? wctx.lineTo(px, py) : wctx.moveTo(px, py); }
      wctx.stroke();
    }

    function drawBoard() {
      const rowsEl = root.querySelector("#xpg-rows");
      const etas = CATALOG.map((e) => scores[e.key].eta), crs = CATALOG.map((e) => scores[e.key].crest);
      const etaLo = Math.min.apply(null, etas), etaHi = Math.max.apply(null, etas);
      const crLo = Math.min.apply(null, crs), crHi = Math.max.apply(null, crs);
      const lg = (x) => Math.log(x);
      rowsEl.innerHTML = CATALOG.map((e) => {
        const sc = scores[e.key];
        // both bars read the same way: fuller = better (lower ETA / lower crest -> longer bar)
        const etaW = 6 + (1 - (lg(sc.eta) - lg(etaLo)) / (lg(etaHi) - lg(etaLo) || 1)) * 94;
        const crW = 6 + (1 - (sc.crest - crLo) / (crHi - crLo || 1)) * 94;
        return `<div class="xpg-row${e.key === sel ? " on" : ""}${e.champ ? " champ" : ""}" data-k="${e.key}">
          <span class="xpg-row-l">${e.label}</span>
          <span class="xpg-row-bars">
            <span class="xpg-bar xpg-bar-eta" title="time to 5%"><i style="width:${etaW}%"></i><em>${fmtTime(sc.eta)}</em></span>
            <span class="xpg-bar xpg-bar-cr" title="crest factor"><i style="width:${crW}%"></i><em>CF ${sc.crest.toFixed(1)}</em></span>
          </span></div>`;
      }).join("");
      rowsEl.querySelectorAll(".xpg-row").forEach((r) => r.addEventListener("click", () => selectDrive(r.dataset.k)));
    }

    function refresh() {
      const sc = scores[sel];
      root.querySelector("#xpg-eta").textContent = fmtTime(sc.eta);
      root.querySelector("#xpg-crest").textContent = sc.crest.toFixed(2);
      pick.querySelectorAll(".xpg-opt").forEach((b) => b.classList.toggle("on", b.dataset.k === sel));
      drawSpec(); drawWave(); drawBoard();
    }
    function selectDrive(k) { sel = k; refresh(); }

    function replant() {
      const f1 = +root.querySelector("#xpg-f1").value, q1 = +root.querySelector("#xpg-q1").value;
      const f2 = +root.querySelector("#xpg-f2").value, q2 = +root.querySelector("#xpg-q2").value;
      root.querySelector("#xpg-f1v").textContent = f1.toFixed(2); root.querySelector("#xpg-q1v").textContent = q1;
      root.querySelector("#xpg-f2v").textContent = f2.toFixed(2); root.querySelector("#xpg-q2v").textContent = q2;
      rebuildPlant([f1, q1, f2, q2]); recomputeAll(); refresh();
    }

    // race animation: bars fill toward final ETA over ~1.4 s, fastest reaching the line first
    function race() {
      if (racing) return; racing = true;
      const rows = root.querySelectorAll(".xpg-row");
      const t0 = performance.now();
      const etas = CATALOG.map((e) => scores[e.key].eta), etaMax = Math.max.apply(null, etas);
      function step(now) {
        const k = Math.min(1, (now - t0) / 1400);
        rows.forEach((r) => {
          const e = scores[r.dataset.k], prog = Math.min(k * (etaMax / e.eta), 1); // faster drive completes sooner
          r.querySelector(".xpg-bar-eta i").style.width = (6 + prog * 94) + "%";
          r.classList.toggle("done", prog >= 1);
        });
        if (k < 1) requestAnimationFrame(step); else { racing = false; drawBoard(); refresh(); }
      }
      requestAnimationFrame(step);
    }

    function hexA(hex, a) { const h = hex.replace("#", ""); const n = parseInt(h.length === 3 ? h.replace(/(.)/g, "$1$1") : h, 16); return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`; }

    pick.querySelectorAll(".xpg-opt").forEach((b) => b.addEventListener("click", () => selectDrive(b.dataset.k)));
    ["xpg-f1", "xpg-q1", "xpg-f2", "xpg-q2"].forEach((id) => root.querySelector("#" + id).addEventListener("input", replant));
    root.querySelector("#xpg-race").addEventListener("click", race);
    root.querySelector("#xpg-plantreset").addEventListener("click", () => {
      root.querySelector("#xpg-f1").value = 1.0; root.querySelector("#xpg-q1").value = 20;
      root.querySelector("#xpg-f2").value = 2.5; root.querySelector("#xpg-q2").value = 20; replant();
    });

    recomputeAll(); replant();
    new ResizeObserver(() => { drawSpec(); drawWave(); }).observe(specCv);
    new MutationObserver(() => refresh()).observe(document.body, { attributes: true, attributeFilter: ["class"] });
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
  else boot();
})();
