---
name: srm-doublet-is-spatial
description: "The 0.672/0.676 Hz SRM/HSTS doublet is REAL and spatial — resolve by plane decoupling, not frequency super-resolution"
metadata: 
  node_type: memory
  type: project
---

The 0.6725/0.6758 Hz "doublet" in the SRM (HSTS) 6-DoF modal ID is **physically real**,
verified by eigendecomposing the continuous-time plant `A` (`lib.load_plant_continuous()`):
two distinct modes whose output mode shapes (`C v`) are **exactly orthogonal** (cos-sim 0):
- **0.67250 Hz** lives in the **{L,P,V}** plane (L,P dominant)
- **0.67583 Hz** lives in the **{T,R,Y}** plane (T,R dominant)

The HSTS plant block-diagonalizes EXACTLY into these two planes (cross-block FRF coupling
~1e-13, machine zero); every in-band mode lives purely in one plane. So the doublet is a
*spatial* doublet — its members are near-coincident in frequency but separated by which
sensors see them, NOT by a resolvable frequency split.

**Resolve it with `mimo_fit.fit_block_decoupled`** (fit each DOF plane independently): each
plane sees only ONE fundamental, so there is no doublet to collapse. On real recovered data
this gave 0.67250/Q49 (plane A) and 0.67591/Q49 (plane B) cleanly. Do NOT try to
frequency-super-resolve it: the rank-1 SHARED-pole 6×6 fit collapses it because it forces
one pole set across two orthogonal modes — that collapse (and the old "unresolvable at any
feasible df" claim) was a parameterization artifact, not a physical limit. The fine-df
doublet campaign + doublet-concentrated drive are unnecessary. Note: every oracle mode has
an imposed uniform Q=50 (a twin modeling assumption); the frequencies/shapes are physical.

See [[compute-the-bound-before-claiming-a-limit]] and [[rank1-modal-mimo-fit]].
