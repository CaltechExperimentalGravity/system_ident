# Pintelon & Schoukens — reference index for `system_ident`

**The book is not in this repository, and never will be.** It is copyrighted:

> R. Pintelon and J. Schoukens, *System Identification: A Frequency Domain Approach*,
> 2nd ed. IEEE Press / Wiley, 2012.

This file is an **index** — chapter/section numbers and a question → chapter map — so an agent
working here knows *where to look* and can cite precisely. It deliberately contains no reproduced
text, equations, figures, or derivations. For any of that, open the book.

## How to get the text

- Buy it, or read it through an institutional Wiley/IEEE Xplore subscription.
- If you keep a personal PDF locally, put it at `docs/SysID-Pintelon.pdf` — that path is
  gitignored precisely so it is never published.
- **Page offset convention** used throughout this repo's notes: citations give the **printed**
  page. In the PDF, **PDF index = printed page + 35**. Equation numbers are quoted as printed.
- Text extraction garbles all the math. If you machine-read the PDF, read the equation pages
  *visually*; do not trust an OCR/`pymupdf4llm` dump for anything symbolic.

## Rules for agents

1. **Never paraphrase an equation from memory.** Cite chapter + section + equation number and
   read the page, or say plainly that you have not read it.
2. The **literal step-2 procedure** this repo implements (joint MIMO parametric fit + CRB) is
   already worked out in [`../pintelon-schoukens-mimo-fit.md`](../pintelon-schoukens-mimo-fit.md).
   Read that first; come here only to find the underlying chapter.
3. Before asserting the method "cannot" do something, check the chapter below **and** apply
   [`../engineering-practices.md`](../engineering-practices.md) — compute the bound with real
   numbers first.

## Question → where to look

| Question that comes up in this repo | Go to |
| --- | --- |
| Why periodic multisine instead of random noise? Leakage-free synchronous DFT | Ch 2 §2.2.3, §2.4, §2.6.3; guideline stated outright in §2.8.1 |
| Per-bin FRF error analysis and noise covariance | Ch 2 §2.4.2, §2.5; Appendix 2.A, 2.C |
| Coherence — what it does and does not tell you | Ch 2 §2.5.4 |
| MIMO FRF from multiple experiments (why one experiment is not enough) | Ch 2 §2.7 |
| **Designing the excitation** — multisine, amplitude spectrum, line placement | **Ch 5**, esp. §5.3 (broadband signals) and §5.4 (optimization for *parametric* measurements) |
| Fisher-optimal drive: optimizing the power spectrum against the parameters | Ch 5 §5.4.2 |
| Crest factor minimization | Ch 5 Appendix 5.A; and Guillaume et al. (1991) below |
| Optimal excitation for MIMO specifically | Dobrowiecki, Schoukens & Guillaume (2006) below |
| Plant model parameterization; identifiability | Ch 6 §6.2, §6.5 |
| MIMO / multivariable model structures | Ch 6 §6.6 |
| Parametric vs nonparametric noise models | Ch 6 §6.7; Ch 13 §13.4–13.5 |
| **The ML estimator with a known noise model** (what `ml_fit` implements) | **Ch 9 §9.11**; approximate/IQML variants §9.12 |
| Uncertainty bounds on the estimated parameters | Ch 9 §9.11.4 |
| Comparison of LS / NLS / TLS / GTLS / ML and their asymptotic properties | Ch 9 §9.8–9.15, summary table §9.15.4 |
| **Conditioning at high model order** — scalar and vector orthogonal polynomials | **Ch 9 §9.16** (the standard fix when a common-denominator fit misbehaves) |
| Transport delay in the model | Ch 9 §9.17 |
| **Identification in feedback / closed loop** | **Ch 9 §9.18**; Pintelon et al. (1992) below |
| Unknown noise model (estimate the noise alongside the plant) | Ch 10; local-polynomial route Ch 12 |
| Local polynomial method for nonparametric FRF | Ch 7; Ch 12 |
| **Model order selection; over- vs under-modeling** | **Ch 11 §11.3, §11.4** |
| Uncertainty bounds on poles/zeros (→ σ on f₀ and Q) | Ch 11 §11.2.3; Guillaume et al. (1989) below |
| Intersample behaviour: ZOH vs band-limited — which applies to a CDS front-end | Ch 13 §13.2–13.3 |
| A step-by-step recipe for a whole identification campaign | **Ch 14** — §14.3 is the checklist |
| **Cramér–Rao lower bound**, efficiency, asymptotic normality | Ch 1 §1.3.2 (intuition); **Ch 16 §16.12** (the theorem); Ch 9 §9.7 |
| Statistics of DFT noise (why the per-bin circular-complex model is right) | Ch 16 §16.16 |
| Properties of LS estimators — deterministic / stochastic weighting | Ch 17 (627) / Ch 18 (651) |
| Semilinear models; separable least squares | Ch 19 |
| **Why over-parameterized fits still give good invariants** (the rank-1 modal lesson: poles are recoverable even when the parameterization is not unique) | **Ch 20**, esp. §20.3 (CRB for invariants) and §20.4 |
| Nonlinear distortions — detecting them, and the best linear approximation | Ch 3, Ch 4 |

## Chapter list (printed start pages)

| Ch | Title | p. |
| --- | --- | --- |
| 1 | An Introduction to Identification | 1 |
| 2 | Measurement of Frequency Response Functions — Standard Solutions | 33 |
| 3 | FRF Measurements in the Presence of Nonlinear Distortions | 73 |
| 4 | Detection, Quantification, and Qualification of Nonlinear Distortions in FRF Measurements | 119 |
| 5 | Design of Excitation Signals | 151 |
| 6 | Models of Linear Time-Invariant Systems | 177 |
| 7 | Measurement of FRFs — The Local Polynomial Approach | 225 |
| 8 | An Intuitive Introduction to Frequency Domain Identification | 279 |
| 9 | Estimation with Known Noise Model | 285 |
| 10 | Estimation with Unknown Noise Model — Standard Solutions | 383 |
| 11 | Model Selection and Validation | 431 |
| 12 | Estimation with Unknown Noise Model — The Local Polynomial Approach | 463 |
| 13 | Basic Choices in System Identification | 497 |
| 14 | Guidelines for the User | 531 |
| 15 | Some Linear Algebra Fundamentals | 545 |
| 16 | Some Probability and Stochastic Convergence Fundamentals | 567 |
| 17 | Properties of Least Squares Estimators with Deterministic Weighting | 627 |
| 18 | Properties of Least Squares Estimators with Stochastic Weighting | 651 |
| 19 | Identification of Semilinear Models | 665 |
| 20 | Identification of Invariants of (Over)Parameterized Models | 699 |
| — | References | 711 |

## Further reading — the primary papers

Taken from the book's own reference list; the book is the place to start, these are where the
individual results were published.

**Surveys**
- R. Pintelon, P. Guillaume, Y. Rolain, J. Schoukens, H. Van hamme (1994). *Parametric
  identification of transfer functions in the frequency domain — a survey.* IEEE Trans. Autom.
  Contr., **39**(11), 2245–2260.
- P. Guillaume, R. Pintelon, J. Schoukens (1996). *Parametric identification of multivariable
  systems in the frequency domain — a survey.* Proc. ISMA21, Leuven, vol. II, 1069–1082.

**Excitation design and crest factor**
- P. Guillaume, J. Schoukens, R. Pintelon, I. Kollar (1991). *Crest-factor minimization using
  nonlinear Chebyshev approximation methods.* IEEE Trans. Instrum. Meas., **40**(6), 982–989.
- T. P. Dobrowiecki, J. Schoukens, P. Guillaume (2006). *Optimized excitation signals for MIMO
  frequency response function measurements.* IEEE Trans. Instrum. Meas., **55**(6), 2072–2079.

**Estimation and numerics**
- P. Guillaume, R. Pintelon (1996). *A Gauss–Newton-like optimization algorithm for "weighted"
  nonlinear least-squares problems.* IEEE Trans. Sign. Proc., **44**(9), 2222–2228.
- A. Bultheel, M. Van Barel, Y. Rolain, R. Pintelon (2005). *Numerically robust transfer function
  modeling from noisy frequency response data.* IEEE Trans. Autom. Contr., **50**(11), 1835–1839.
- P. Guillaume, J. Schoukens, R. Pintelon (1989). *Sensitivity of roots to errors in the
  coefficients of polynomials obtained by frequency domain estimation methods.* IEEE Trans.
  Instrum. Meas., **38**(6), 1050–1056.

**MIMO and noise covariance**
- P. Guillaume, R. Pintelon, J. Schoukens (1996). *Accurate estimation of multivariable frequency
  response functions.* Proc. 13th IFAC Triennial World Congress, San Francisco, 423–428.
- R. Pintelon, P. Guillaume, J. Schoukens (1996). *Measurement of noise (cross-)power spectra for
  frequency-domain system identification purposes: large-sample results.* IEEE Trans. Instrum.
  Meas., **45**(1), 12–21.

**Closed loop**
- R. Pintelon, P. Guillaume, Y. Rolain, F. Verbeyst (1992). *Identification of linear systems
  captured in a feedback loop.* IEEE Trans. Instrum. Meas., **41**(6), 747–754.

**Averaging / windowing (the non-periodic alternative, for contrast)**
- J. Antoni, J. Schoukens (2007). *A comprehensive study of the bias and variance of
  frequency-response-function measurements: optimal window selection and overlapping strategies.*
  Automatica, **43**(10), 1723–1736.
