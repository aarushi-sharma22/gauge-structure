# Gauge Structure of Learned Representations

Research code for the RP-2 paper (calibrated isotypic-projection instrument for
detecting irreducible-representation structure in neural network weights) and a
follow-up thread on the gauge structure of learned representations.

> **Status:** research code, not a polished library. Notebooks are the unit of
> work; `src/` holds the small set of modules that have been unit-tested against
> synthetic ground truth. See the per-notebook ledger below for what has actually
> run and what each result licenses.

## Layout

```
src/                         # tested, reusable modules (see "Tested modules")
notebooks/
  rp2_core/                  # the paper's core experiments (TMS, non-abelian, theorem work)
  gauge/                     # gauge-vs-dimension + seed-gauge (identifiability)
  llm_deviation/             # the paper's third claim: LLM deviation causality
  followup_gauge_llm/        # gauge-in-LLMs — mostly NEGATIVE, future-paper material
data/                        # inputs (gitignored)
outputs/                     # CSVs/figures, regenerated (gitignored)
```

Almost everything runs in Google Colab. CPU-only work (groups, gauge, seed-gauge)
needs numpy/scipy; the LLM notebooks need torch + transformers + a GPU.

## Tested modules (`src/`)

These were built and verified against synthetic ground truth (known-answer checks,
calibration controls). They are the canonical implementations; prefer them over the
inlined copies inside individual notebooks.

| module | what it provides | verified |
|---|---|---|
| `groupcore.py` | `Group`, `cyclic/dihedral/symmetric/alternating/quaternion8`, numeric character tables (Dixon), FS indicators, real isotypic projectors, numeric commutant dimension, `all_eleven_groups()` | character orthogonality, projector completeness (~1e-15), FS indicators (Q8 = −1) all asserted |
| `exp1engine.py` | planted-structure noise model, spectrum-preserving detection null, σ\*, invariance defect, `multiplicity_rep` (fixed-dim / varied-multiplicity construction), copy decomposition, `partial_spearman`, `cluster_bootstrap_ci`, `page_trend_L` | multiplicity→gauge-drift signature reproduced; constructed gauge = m² asserted |
| `exp2core.py` | `CyclicDecomposition` (Fourier modes over Z/n), the full intervention family (circle-only, matched-random, spectrum-preserving rotation, dose), `bootstrap_ci_accuracy`, `did_bootstrap`, `report_null_comparison`, `rotation_vs_matched`, `page_trend_L` | synthetic self-test incl. a calibration check (null model → gap≈0); DiD bootstrap recovers a planted effect |
| `predcore.py` | `train_grokking` (modular-addition quadratic MLP), gauge-invariant/dependent inventory, `procrustes_align` + `spectral_floor`, `commutant_change_decomposition` | gauge-equivalent pairs align to floor; commutant decomposition calibrated |
| `checkpointcore.py` | checkpoint gauge-diffusion analysis: `tangent_fraction`, radial/angular split, `circle_phase`, empirical null + permutation placebo, transfer-efficiency machinery | phase estimator recovers planted rotations to ~1e-4; diffusion signature validated synthetically |

> Note: the notebooks currently **inline** copies of these modules (so each is
> self-contained and runs in Colab without an import path). `src/` is the extracted,
> tested version for reuse. Deduplicating the notebooks to import from `src/` is a
> deliberate *future* refactor, not done here.

## Notebook ledger

Status legend: ✅ ran, clean result · ⚠️ ran, needs scale/follow-up · ❌ ran, negative
· ❓ **status not confirmed by me — please annotate** (earlier notebooks I have not
executed; description inferred from contents).

### `notebooks/rp2_core/` — the paper's core (earlier work, NOT executed by me)

| notebook | appears to cover | status |
|---|---|---|
| `non_abelian_group_ops.ipynb` (~5.3k LOC, 52 cells) | the bulk of RP-2: `GroupMLP`, `TMS`, isotypic analysis, ablation tables, Beta-null moments, relabeling — likely the main instrument + non-abelian corruption experiments | ❓ confirm which results are live |
| `tms_non_abelian.ipynb` | TMS dichotomy (seed-variable vs seed-deterministic), balanced-family search, S5 TMS data, pair sweeps | ❓ confirm |
| `theorm_deciding.ipynb` (~1.9k LOC) | theory/decision work: cyclic recovery quality, deviation selectivity, Fourier projectors, S3/S4/S5/A4/A5 | ❓ confirm |
| `llm_non_abelian_and_abelian.ipynb` | LLM-side non-abelian + abelian diagnostics, commutator defect, conditioned tests, constructed "bite" | ❓ confirm |

**Action needed from you:** for each of these four, mark which experiments are
surviving paper contributions vs superseded/dead ends. I grouped them by name; the
grouping is a guess.

### `notebooks/gauge/` — gauge vs dimension (identifiability)

| notebook | covers | status |
|---|---|---|
| `exp1_gauge_vs_dimension.ipynb` | Step 0/1/2: per-block inventory, σ\* under the calibrated detector, and the decisive multiplicity-at-fixed-dimension decoupling | ✅ Step 2 ran FULL: gauge drift ~0 at m=1, monotone in m (Page z≈10), δ flat in m. Establishes gauge governs *identifiability*, distinct from a dimension effect on invariance. Scaling law `gauge_drift ≈ 0.125·(m−1)·σ²` matched data but is **post-hoc** — run the ambient=48 falsifier to confirm. |
| `pred1_seed_gauge.ipynb` | are seed-to-seed differences in grokked nets gauge? alignment-to-spectral-floor + margin-based intervention transfer | ⚠️ ran at PILOT (p=59, 2 seeds): residual collapses to floor (0.0014 vs 0.0010), matched-null 19× higher, transfer 1.00 aligned vs 0.14 raw. Needs FULL (p=97, 5 seeds). Caveat: dense-symmetric regime only. |

### `notebooks/llm_deviation/` — the paper's third claim

| notebook | covers | status |
|---|---|---|
| `exp2_llm_deviation.ipynb` | is the higher-frequency deviation from the circle causally necessary *specifically* for modular arithmetic? selectivity control (season/ordinal), matched-random null, spectrum-preserving rotation control, dose-response, digits mod-10 | ⚠️ ran on **Qwen only**: selectivity DiD replicated (season ~0.20, ordinal ~0.17), dose-response clean. Needs the FULL grid (4 model families + digits, N_DRAWS=500) to be a claim. **Rotation/orientation control is NOT significant at pilot N (p_vs_matched≈0.05) — do not claim "orientation matters" yet.** |
| `month_irrep_diagnostic.ipynb` | diagnostic for the month-token cyclic structure in an LLM | ❓ confirm role/status |

### `notebooks/followup_gauge_llm/` — future paper, mostly negative

| notebook | covers | status |
|---|---|---|
| `expAC_llm_gauge.ipynb` | Exp A (gauge diffusion across training checkpoints) + Exp C (intervention transfer across checkpoints) | ❌ **NEGATIVE.** A: no rotational gauge drift — the tangent "signal" was 89% off-tangent, angular ≈ 0, phase flat (estimator verified). Circle structure present (30% energy, verified) but does not diffuse — consistent with semantic symmetry-breaking. C: **never ran** (OLMo checkpoint disk limits). Not paper material; one-paragraph negative for a future paper. |

## What the paper can currently claim

1. **Detectability** — the calibrated detector (rp2_core). Solid.
2. **Identifiability** — gauge governs *which* gauge-equivalent embedding is recovered
   under perturbation, distinct from a dimension effect on invariance (`gauge/exp1`,
   FULL). This is a narrower and better claim than "gauge governs recoverability."
3. **Function** — the LLM deviation is causally necessary specifically for modular
   arithmetic (`llm_deviation/exp2`, Qwen only — **pending full grid**).

The gauge-in-LLMs thread (followup) is a *separate future paper* and currently reads
negative. Fine-tuning-delta and checkpoint-diffusion experiments did not show gauge
structure, consistent with cyclic symmetry being only approximate in LLMs.

## Outstanding before submission

- `llm_deviation/exp2`: full grid — `SMOKE=False`, 4 model families, digits task, N_DRAWS=500.
- `gauge/exp1`: ambient=48 falsifier for the scaling law (one config change).
- `gauge/pred1`: FULL run (p=97, 5 seeds) if included.
- Rewrite paper §5 around identifiability-vs-recoverability (writing, not compute).
