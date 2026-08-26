/* Overlay a "Launch in Binder" badge on each runnable example card in the gallery grid.
 *
 * Quarto renders each listing card as an <a class="quarto-grid-link"> wrapping the whole
 * card, so the badge is added as an absolutely-positioned SIBLING (not nested inside that
 * anchor — nested <a> is invalid and would hijack the badge click). The compiled-twin
 * examples (07, 10) are excluded: they need the RTSfreerun model and don't run in Binder.
 */
(function () {
  "use strict";
  const REPO = "CaltechExperimentalGravity/system_ident";
  const RUNNABLE = new Set([
    "01-single-resonance", "02-double-pendulum", "03-fabry-perot-cavity",
    "04-suspension-multidof", "05-closed-loop-arm", "06-two-by-two-coupled",
    "08-darm-calibration", "09-rank1-modal-mimo", "11-reduced-quad", "12-reduced-quad-closed",
    "13-darm-drift-tracking",
  ]);

  function init() {
    document.querySelectorAll("#listing-examples-gallery .g-col-1").forEach((cell) => {
      const link = cell.querySelector("a.quarto-grid-link");
      if (!link || cell.querySelector(".gallery-binder")) return;
      const m = (link.getAttribute("href") || "").match(/([0-9]{2}-[a-z0-9-]+)\.html/);
      if (!m || !RUNNABLE.has(m[1])) return;
      cell.style.position = "relative";
      const badge = document.createElement("a");
      badge.className = "gallery-binder";
      badge.href = `https://mybinder.org/v2/gh/${REPO}/HEAD?urlpath=lab/tree/docs/examples/${m[1]}.ipynb`;
      badge.target = "_blank";
      badge.rel = "noopener";
      badge.title = "Launch this example live in Binder";
      badge.innerHTML = '<img src="https://mybinder.org/badge_logo.svg" alt="Launch in Binder" height="20">';
      // sibling of the card link, so its click is its own — but stop bubbling defensively
      badge.addEventListener("click", (e) => e.stopPropagation());
      cell.appendChild(badge);
    });
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
