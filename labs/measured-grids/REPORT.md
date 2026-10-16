# Discovering stratification in measured 3D grids: pilot report

James B. Wilson's brief of 11 October 2026 (`labs/codex_measured_grid_stratification.txt`), executed 11 October 2026 with Dleto (OpenDleto git `3e27c3c`, Julia 1.12.6).  Code: `labs/measured-grids/`.  Everything below was produced by `run_3dfm.jl` and `run_meshrir.jl`; numbers are copied from `results/3dfm.log` and `results/meshrir.log`.

## 1. Method, as run

**Input.** A real tensor `T[i,j,k]` whose three indices are the measured Cartesian positions, in recorded order.  Only global, invertible normalisations were applied: division by the largest magnitude (audio) and, as a separately reported variant, removal of the global mean (magnetic `Bx`).

**Chisel and operator.** Dleto's universal linear chisel: the all-plus relation `D1 ⊕ D2 ⊕ D3` with `Σ (D1 ⊗ I ⊗ I + I ⊗ D2 ⊗ I + I ⊗ I ⊗ D3) T = 0`.  Two operator families were run and are reported separately:

* `UniversalOp`: each `D_a` any matrix (this is what `stratify` uses by default; unknowns `d1² + d2² + d3²`);
* `SymmetricOp`: each `D_a` symmetric (unknowns `Σ d_a(d_a+1)/2`).  This family was added because the universal family's canonical forms were ill-conditioned (Section 2.3); a symmetric derivation has an orthogonal eigenbasis, so the learned coordinates are an orthonormal change of basis on each axis.

The derivation operator was built with Dleto's `sylvesterLM` and formed densely (`AᵀA`, at most 963 × 963 here) so the entire singular spectrum is available.  Singular values are reported relative to the largest.

**Trivial directions.** The scalar triples `(a I, b I, c I)` with `a+b+c = 0` form the exact two-dimensional null space.  Because `AᵀA` is symmetric, every other eigenvector is orthogonal to this pair, which is the quotient.  A *scalar score* (1 = pure scalar triple) confirms which directions are the trivial ones.

**Criterion for nontrivial directions.** Successive ratios `σ[i+1]/σ[i]` after the scalar pair; a *robust* direction would show a ratio well above 1 (a gap) that survives 1% and 5% perturbation of the measurements.  Two controls that keep the values but destroy any axis relation are run through the same operator: the entries shuffled over positions, and iid Gaussian entries of the same standard deviation.  (Independent axis permutations are not a control, as the brief notes.)

**Transforms.** For a chosen direction, Dleto's `stratify(Γ, δ)` (real canonical form of each `D_a`, universal family) or the orthogonal eigenbasis (symmetric family) gives per-axis transforms `A, B, C`; `S = T ×₁ A ×₂ B ×₃ C`.  Reported: eigenvalue multiplicities and non-normality of each `D_a`, condition numbers, and the relative reconstruction error of `T` from `S` through the inverses.

**Does the representation carry information?** Keep the `k` largest coefficients and reconstruct, in four representations: physical coordinates, learned coordinates (through the inverse transforms), a frozen orthonormal DCT on each axis (fixed after discovery, never used to tune it), and the HOSVD bases (the standard data-adapted orthogonal per-axis basis, added as a comparator once the symmetric family produced orthogonal bases).  Representation size counts the learned basis entries (`Σ d_a²`) in addition to `k`.  A principal-angle test compares the learned vectors that carry the most energy with the leading HOSVD subspace.

## 2. Pilot 1: 3DFM magnetic-field scan

**Data.** `10cmcube_11steps_per_axis_measurement.csv` (SHA-256 verified).  Cumulative sums of `dx, dy, dz` give positions; rows 2–1332 (1-based) cover the `11×11×11` product of levels `-50:10:50` mm exactly once, verified.  Rows 1, 667 and 1333 are centre checks (0,0,0): `Bx` = 428.79, 428.29, 428.05, i.e. a drift of about 0.7 units over the scan, comparable to the whole-cube standard deviation of `Bx` (0.59).  Units are instrument units (Metrolab THM1176 probe; the repository does not state the unit).  Components: `Bx` mean 428.2, std 0.59; `By` 19.7 ± 4.1; `Bz` 11.9 ± 1.0; `Bmod` 428.8 ± 0.59.  All finite.

### 2.1 Spectra

| tensor | family | σ₃..σ₆ (relative) | ratios σ₄/σ₃ … | shuffled control σ₃ | Gaussian control σ₃ |
|---|---|---|---|---|---|
| `Bx` | universal | 3.6e-6, 3.7e-6, 3.8e-6, 4.0e-6 | 1.0, 1.0, 1.0 | 1.1e-4 | 2.5e-1 |
| `By` | universal | 1.1e-4, 1.2e-4, 1.3e-4, 1.4e-4 | 1.1, 1.1, 1.1 | 1.7e-2 | 2.5e-1 |
| `Bz` | universal | 1.1e-4, 1.3e-4, 1.4e-4, 1.5e-4 | 1.2, 1.1, 1.0 | 7.0e-3 | 2.5e-1 |
| `Bx − mean` | universal | 3.0e-3, 3.2e-3, 3.3e-3, 3.5e-3 | 1.0, 1.0, 1.0 | 2.4e-1 | 2.5e-1 |
| `Bx − mean` | symmetric | 4.8e-3, 5.0e-3, 5.2e-3, 5.3e-3 | 1.0, 1.0, 1.0 | 3.0e-1 | 2.5e-1 |

Established: the two scalar directions are exact (σ₁, σ₂ ≤ 1e-9, scalar score 1.000); everything after them is a continuum with successive ratios 1.0–1.2, for every component and both families, and the continuum moves smoothly under 1% and 5% perturbation (e.g. `By`: 1.1e-4 → 1.9e-4 → 8.8e-4).  **No isolated nontrivial direction exists.**

Also established: the continuum sits far below both controls (two to five orders of magnitude for the raw components).  For `Bx` and `Bmod` this is the nearly constant field (the shuffled control is itself low because the values are nearly equal); after removing the mean, the demeaned `Bx` still sits 100× below its controls.  The field is therefore highly structured in the sense that *many* approximate derivations exist, which is what a smooth, low-multilinear-rank tensor produces.

### 2.2 Learned coordinates, universal family

Reconstruction through the inverse transforms is exact (relative error 1e-15 to 1e-14).  Condition numbers of the per-axis transforms range from 2e1 to 4.5e3; most `D_a` have strongly complex eigenvalues (imaginary part up to 98% of the modulus) and 6–10 distinct eigenvalues out of 11, so the real canonical form has 2×2 blocks and little freedom to be a clean diagonalisation.  Keeping the largest learned coefficients is often *worse than keeping none*: at 5% of the coefficients the relative error is 0.27–2.3 (`By`, `Bz`), 1.6–4.3 (`Bx − mean`), against 0.005–0.14 for the frozen DCT.  The sparsity of `S` is an artefact of a non-orthogonal basis, exactly the failure the brief warns about.

### 2.3 Learned coordinates, symmetric family

All `D_a` symmetric by construction, condition numbers 1.0, reconstruction error ≤ 7e-16.  At 5% / 20% of the coefficients:

| tensor | physical | learned (dir. 1) | DCT (frozen) | HOSVD |
|---|---|---|---|---|
| `By` | 0.961 / 0.825 | 0.003 / 0.001 | 0.005 / 0.002 | 0.002 / 0.001 |
| `Bz` | 0.962 / 0.848 | 0.004 / 0.002 | 0.010 / 0.003 | 0.003 / 0.001 |
| `Bx − mean` | 0.760 / 0.467 | 0.059 / 0.025 | 0.141 / 0.036 | 0.052 / 0.024 |

The learned orthogonal coordinates beat the frozen DCT and track the HOSVD curve to within a few percent, for every one of the first three nontrivial directions and for their random combination (`figures/3dfm-Bx-demeaned-sym-coefficients.png`).  The principal angles between the learned vectors carrying the most energy and the leading HOSVD subspaces (ranks 3–4 at 99% energy) are 2–5° on average, up to 17° (direction 1).  Representation size: the learned bases cost `3 × 121 = 363` entries for a tensor of 1,331 entries; adding them, 66 learned coefficients plus the bases (429 numbers, error 0.059) is outperformed by about 300 DCT coefficients alone (error 0.036).

**Interpretation.** The symmetric family does find a useful orthogonal coordinate system, but it is, to within a few degrees, the HOSVD (Tucker) subspace split of a smooth field: near-derivations of a nearly low-rank tensor are operators that vanish on the signal subspace of each axis, and their eigenbases separate that subspace from its complement.  This is a correct unsupervised recovery of known structure (outcome 2 of the brief), not a new pattern; nothing beyond low multilinear rank was found, and no discrete stratification (blocks or strata) exists in these fields for this chisel.

## 3. Pilot 2: MeshRIR S1-M3969

**Data.** `S1-M3969_npy.zip` v1.2 (MD5 verified `2cb598eb44bb9905560c545db7af3432`).  3,969 microphone positions, recorded order `ir_0 … ir_3968`, each with its position and acquisition timestamp in `data.json`; positions are `x, y ∈ {-0.5:0.05:0.5}`, `z ∈ {-0.2:0.05:0.2}` m, every grid point measured exactly once (verified; index map in `results/meshrir-index-map.csv`).  Source at (2.0, 1.5, 0.0) m; room 7.0 × 6.4 × 2.7 m, RT60 0.38 s, 26.3 °C; 32,768 samples at 48 kHz.  The responses are extracted impulse responses, not raw microphone voltages; the dataset page does not document the excitation or deconvolution, so no temporal processing is assumed or undone here.  The full `R[x, y, z, t]` (21×21×9×32768, 496 MB) was loaded; no plane, filter or interpolation.

**Sample rule (fixed before any result was seen).** `t0` = median over microphones of the sample of maximum |response| = 8538 (177.9 ms; arrivals span 8437–8632).  Snapshots at `t0 + {−24, 0, +24, +96, +480, +4800}` samples: direct sound entering, crossing and leaving the grid, early reflections (10 ms), and late reverberation (100 ms).  Each snapshot is divided by its largest magnitude.  Nearby-sample sensitivity at `t0 ± 1`.

### 3.1 Spectra (three-way snapshots)

| instant | family | σ₃ | σ₄/σ₃ | shuffled control σ₃ | learned (sym.) at 5% | DCT | HOSVD | physical |
|---|---|---|---|---|---|---|---|---|
| t0−24 | universal / symmetric | 2.9e-4 / 3.1e-4 | 1.1 / 1.1 | 0.17 / 0.20 | 0.150 | 0.555 | 0.184 | 0.419 |
| t0 | universal / symmetric | 3.8e-4 / 3.3e-4 | 1.0 / 1.0 | 0.20 / 0.22 | 0.229 | 0.506 | 0.220 | 0.520 |
| t0+24 | universal / symmetric | 2.3e-3 / 7.5e-3 | 1.2 / 1.0 | 0.20 / 0.23 | 0.191 | 0.415 | 0.153 | 0.537 |
| t0+96 | universal / symmetric | 1.9e-2 / 4.0e-2 | 1.1 / 1.1 | 0.16 / 0.21 | 0.465 | 0.584 | 0.444 | 0.607 |
| t0+480 | universal / symmetric | 1.0e-1 / 1.2e-1 | 1.0 / 1.1 | 0.21 / 0.24 | 0.686 | 0.708 | 0.640 | 0.758 |
| t0+4800 | universal / symmetric | 1.1e-1 / 1.4e-1 | 1.0 / 1.0 | 0.22 / 0.29 | 0.713 | 0.734 | 0.662 | 0.864 |

Established: the exact scalar pair, then a continuum with successive ratios 1.0–1.2 at every instant and in both families (`figures/meshrir-t8538-sym-spectrum.png`); stable under 1% perturbation (3e-4 → 1.3e-3) and under a one-sample shift (`t0 ± 1`: 2.8e-4, 3.4e-4).  The continuum sits three orders of magnitude below the controls while the direct sound crosses the grid and rises monotonically afterwards, reaching the shuffled control's level in the late reverberation: to this chisel, a diffuse reverberant field is nearly indistinguishable from shuffled values, and the early field is "structured" only in the smooth, low-rank sense already seen in Pilot 1.  **No isolated nontrivial direction at any instant.**

The universal family's canonical forms are again unusable (condition numbers 2e1–3e3, complex eigenvalues; keeping 5% of the learned coefficients gives errors of 2.7–58).  The symmetric family's orthogonal coordinates are comparable to HOSVD (better at t0−24, equal at t0, worse at t0+24) and beat the frozen DCT at every instant; the principal angles to the leading HOSVD subspaces are 0–10° (one axis 25° at t0).  A notable detail: at the direct-sound instants the symmetric near-derivations are nearly scalar on whole axes (1–3 distinct eigenvalues out of 21), so their eigenbases are fixed only by a tiny residual; the learned coordinates are then the eigenvectors of that residual, and they still align with HOSVD.  Representation size: 963 basis entries for 3,969 tensor entries.

### 3.2 Time-valued formulation (four-way tensors)

A separate formulation, as the brief asks: `R[x, y, z, t]` over a 16-sample window (0.33 ms), at the direct sound (`t0−8 … t0+7`) and in the late reverberation (`t0+4800 …`).  With four axes the trivial scalar space is three-dimensional (`Σ c_a = 0`), so the quotient drops three directions; the first version of this run dropped two, which produced a spurious "gap" (also present in the controls) and singular canonical forms, and was corrected (`common.jl`, `ntriv = N − 1`).  Universal family: 1219 unknowns, 4.6 s; symmetric family: 643 unknowns, 1.2 s.

*Late window.* Continuum after the three trivial directions in both families (universal 1.0e-2, 1.0e-2, 1.0e-2, …; symmetric ratios 1.1–1.4), controls at 0.37–0.40.  Learned symmetric coordinates: 0.73 at 5%, worse than DCT (0.62) and HOSVD (0.63).  Nothing.

*Direct window.* Universal family: continuum (4.3e-4, 4.3e-4, 4.4e-4, …, ratios 1.0).  Symmetric family: **three** small values after the trivial ones (3.4e-4, 8.1e-4, 8.7e-4), then a jump by 8× to 7.1e-3, absent from the controls (0.38); learned coordinates 0.20 at 5% against HOSVD 0.30 and DCT 0.39.  This is the only gapped spectrum in the whole pilot, so it was probed (`probe_meshrir_window.jl`, `results/meshrir-window-probe.log`):

* The three directions are, up to scalars, *rescalings of single `y`-slices*: direction 1 has eigenvalue −1 on `y`-index 1 and 0.04 on the other twenty, and is scalar on `x`, `z`, `t`; direction 2 does the same for `y`-index 2; direction 3 acts on `y`-indices 1–2.  Those are precisely the `y`-slices whose energy in the window is 0.00, 0.00 and 0.01 of the maximum: in 0.33 ms the wavefront has not reached them.  An operator that only rescales an empty slice is a near-derivation of the tensor for no reason other than the emptiness.
* Shifting the window by ±8 and ±16 samples moves the gap with the number of empty `y`-slices (4–6 small values when 4–5 slices are empty; no gap at all at `t0+16`, where no slice is empty); halving or doubling the window length does the same.
* Removing the near-empty directions first (`nondeg`, relative tolerance 1e-2: `21×19×16×8` remains) leaves a single small value with ratio 7.9 whose derivation is again a rank-one rescaling of the weakest retained `y`-direction; at tolerance 3e-2 the ratio is 3.3 and the next direction's eigenstructure is scattered over all indices of every axis.

So the gap counts degenerate (unreached) slices.  That is the Tucker-type structure Dleto's `nondeg` is designed to remove before stratification, and after removing it nothing robust remains.  The better-than-HOSVD sparse reconstruction in that window has the same origin: the learned `y`-basis isolates the empty slices, which HOSVD's energy ordering does not single out.

## 4. Pilot 3: Haneda–Ren Part 0 (raw, 48 kHz)

**Data.** `Raw_A_ir_48k_sp1.h5` (MD5 `4a8c756cadac33326b5bcae02135df95` from the Zenodo API, CC BY 4.0).  `/rir` is 4800 × 63648 Float32, `/grid` 63648 × 3; the flattening order (x fastest, then y, then z) was verified from the coordinates themselves, not assumed.  Root attributes: fs 48000, c 340, grid 48 × 51 × 26 at 0.02 m, loudspeaker (5.54, 0.94, 1.42) m, array origin (3.74, 2.94, 1.11) m, room 8.4 × 6.14 × 2.66 m.  This release keeps the microphone-frame reflections; it is distinct from Part I (16 kHz, reflections suppressed), which was used earlier today for the Wavefront Lab and is not mixed in here.

**Sample rule.** Same as Pilot 2: `t0` = median arrival (sampled every 25th position) = 381 (7.94 ms; range 287–473); snapshots at `t0 + {−72, 0, +72, +288, +1440, +3600}` (1.5 ms, 6 ms, 30 ms, 75 ms).  Sweep on the stride-2 subgrid (every second measured position on each axis, 24 × 26 × 13, positions listed in the log); escalation to the full 48 × 51 × 26 grid at `t0` in both families.

### 4.1 Results

| instant | family | σ₃ | σ₄/σ₃ | shuffled σ₃ | learned (sym.) at 5% | DCT | HOSVD | physical | angles to HOSVD (mean, °) |
|---|---|---|---|---|---|---|---|---|---|
| t0−72 | univ. / sym. | 7.4e-5 / 2.5e-4 | 1.2 / 1.1 | 0.12 / 0.14 | 0.011 | 0.460 | 0.011 | 0.017 | 1.9, 0.0, 1.9 |
| t0 | univ. / sym. | 6.6e-4 / 8.6e-4 | 1.0 / 1.2 | 0.24 / 0.23 | 0.149 | 0.468 | 0.156 | 0.282 | 5.3, 2.2, 6.0 |
| t0+72 | univ. / sym. | 1.8e-2 / 2.8e-2 | 1.1 / 1.1 | 0.18 / 0.19 | 0.264 | 0.517 | 0.258 | 0.299 | 7.6, 8.9, 9.0 |
| t0+288 | univ. / sym. | 7.0e-2 / 7.7e-2 | 1.0 / 1.0 | 0.28 / 0.33 | 0.583 | 0.583 | 0.531 | 0.805 | 1.0, 3.1, 0.0 |
| t0+1440 | univ. / sym. | 1.8e-1 / 2.1e-1 | 1.0 / 1.0 | 0.27 / 0.30 | 0.799 | 0.761 | 0.744 | 0.837 | 0, 0, 0 |
| t0+3600 | univ. / sym. | 1.8e-1 / 2.0e-1 | 1.0 / 1.0 | 0.28 / 0.34 | 0.800 | 0.787 | 0.748 | 0.844 | 0, 0, 0 |
| **full grid, t0** | sym. | 6.1e-4 | 1.1 | 0.34 | 0.094 | 0.233 | 0.097 | 0.368 | 7.8, 6.5, 12.0 |
| **full grid, t0** | univ. | 5.9e-4 | 1.0 | 0.34 | 18.4 (cond. 1.5e2–2.1e3) | 0.233 | 0.097 | 0.368 | – |

Timing: 0.3–0.7 s per operator at stride 2; 8.5 s (symmetric, 2853 unknowns) and 21.4 s (universal, 5581 unknowns, dense 250 MB) on the full grid.

Established: the third dataset, denser and raw, gives the same spectrum shape at every instant and both resolutions (continuum, ratios 1.0–1.2, no gap), the same monotone rise toward the control level as the field becomes reverberant, the same unusable universal canonical forms (5% of learned coefficients: errors 6–24), and symmetric learned coordinates equal to HOSVD within noise and aligned with it.  At `t0−72` the wavefront has only entered a corner, most slices are empty, and all data-adapted representations (and the physical one) are trivially sparse, the degenerate case of Section 3.2 again.  Going from a 4 cm to a 2 cm grid does not open a gap: resolution is not what is missing.

### 4.2 The step in the symmetric spectrum at `t0`, probed

`figures/haneda-s2-t381-sym-spectrum.png` shows a step: 21 values between 8.6e-4 and 3.2e-3, then a jump by 3× to 9.5e-3.  Probed (`probe_haneda_step.jl`, `probe_haneda_ordering.jl`; logs in `results/`):

* The `y` axis has 10 slices below 5% of the maximum energy (indices 1–4 ahead of the wavefront, 21–26 behind it).  Directions 1–21 are, to within mixing among themselves, rescalings of those near-empty `y`-slices (direction 1: eigenvalue −1 on `y`-index 23, scalar elsewhere; direction 2: `y`-index 24; …).  The step is where the empty-slice directions run out, as in Section 3.2.
* The ten directions after the step (σ = 1.0e-2 … 1.7e-2, ratios 1.0–1.2, no gap among them) are 94–99% diagonal on the `x` axis in the physical basis, with diagonal entries that are a monotone function of the measured `x` position (|Spearman| = 0.99 on average, 1.00 for most), 65–95% diagonal on `y` with |Spearman| 0.47, and not ordered on `z` (0.28).  Under 1% perturbation: unchanged (0.99 / 0.47 / 0.28).  Shuffled control: 0.15 / 0.19 / 0.20 (chance).  At `t0+36` the ordering is still present on `x` (0.83); at `t0−72`, `t0−36` and `t0+72`, where 13–20 of the 26 `y`-slices are empty, no clean step exists and the ordering is at chance.

So, once the empty-slice directions are passed, the chisel's next near-derivations are diagonal operators that *sort the microphones by position along the propagation direction*, without having been given any coordinates.  A monotone function of `x` over this grid is rank-equivalent to `(x − a)²` for the loudspeaker position, so this is consistent with the spherical-wavefront relation `(x−a)² + (y−b)² + (z−c)² ≈ r²` (the `z` axis is nearly degenerate for this geometry, as the Wavefront Lab found), but the quadratic form itself is not identified, the directions are not separated from the continuum by a gap, and the effect exists only while the wavefront crosses the interior of the grid.  Classification: a numerical pattern consistent with a known physical structure, instant-dependent, not certified robust by the spectrum (outcome 2 at best, not 1).

## 5. Outcome

**Numerical claims (established).**
1. In three independent measured volumes (a magnetic field, two sound fields), for the universal linear chisel in both the universal and the symmetric operator families, the derivation operator has exactly the `N − 1` trivial scalar directions as its null space and, beyond them, a continuum of approximate derivations with successive ratios 1.0–1.2 and no gap, stable under perturbation, nearby instants, and resolution.
2. That continuum lies far below the shuffled and Gaussian controls whenever the field is smooth (magnetic cube; direct sound) and rises to the control level when it is not (late reverberation).  The chisel therefore detects *that* a measured field is structured, but the structure it detects is low multilinear rank, i.e. smoothness.
3. The canonical forms of the universal family are not usable as a representation on these data: non-normal, complex-spectrum derivations give ill-conditioned transforms in which coefficient magnitude does not reflect contribution (errors above 1 after thresholding).
4. The symmetric family yields orthogonal per-axis coordinates that beat a frozen DCT and match HOSVD within a few percent in reconstruction, with leading subspaces 0–10° from HOSVD's.  Counting the basis entries, HOSVD or DCT is as compact or more.
5. Every gap or step found (the four-way direct-sound window in MeshRIR; the `t0` snapshot in Haneda Part 0) counts near-empty slices and disappears after `nondeg`.
6. In Haneda Part 0 at the instant the wavefront crosses the interior of the grid, the symmetric near-derivations immediately after the empty-slice band are diagonal and order the microphones by their measured `x` position (|Spearman| 0.99; chance 0.15; stable under 1% noise), consistent with the known spherical-wavefront relation; not gap-separated, and absent when the front is near the grid's edge.

**Physical interpretation (possible explanations, not established).** The near-derivations are the operators that annihilate the few dominant per-axis modes of a smooth field; their eigenbases separate "signal" from "complement" on each axis, which is why they reproduce the Tucker subspaces.  No evidence of a discrete stratification (blocks, strata, or support on a relation of the form `u_i + v_j + w_k ≈ 0`) was found in the untouched measurements.  The spherical-wavefront relation known to hold in the sound fields (and recovered in the Wavefront Lab by a *supplied* geometric family) is not isolated by the unsupervised universal or symmetric families because measured wavefronts are thick, unevenly weighted shells rather than exact surfaces.

**Scientific novelty.** None claimed.  The result is outcome **4** of the brief for the universal chisel with these two operator families on all three datasets, and outcome **2** (unsupervised recovery of known structure: low multilinear rank everywhere; the position ordering of the direct wavefront in Haneda Part 0) for the symmetric family.

**Unresolved.** (a) Whether a *diagonal* operator family, which turns the search into finding scalar labels `u_i, v_j, w_k`, would isolate the wavefront relation without being told its form; it did so in the Wavefront Lab only partially and with the family supplied.  (b) Whether other chisels of the `null_patterns` paper (face-curve, centroid) see anything in the magnetic field that the universal chisel does not.  (c) Whether a gap appears for *rigid* fields (the magnet at smaller step or a shimmed field) rather than denser audio grids.  (d) A canonical form for non-normal derivations that is numerically meaningful on noisy data.

**Negative result caveat.** A negative result for one chisel and operator family on these slices is not a theorem that the measurements contain no useful structure.

## 6. Figures for a talk

* Measured volumes: `figures/3dfm-measured.png` (four field components, z = 0 slice, instrument units), `figures/meshrir-measured.png`, `figures/haneda-measured.png` (energy / pressure on the physical grid; points above 10% of the maximum).
* Spectra with controls: `figures/3dfm-By-sym-spectrum.png`, `figures/meshrir-t8538-sym-spectrum.png`, `figures/haneda-s2-t381-sym-spectrum.png` (indices after the trivial scalars; relative singular values on a log axis).
* Learned coordinates: `figures/*-representation.png` (left: physical slice, axes labelled "x index / y index"; right: |S| in learned coordinates, axes labelled "learned a / learned b") and `figures/*-basis.png` (learned basis vectors against physical index).
* Validation: `figures/*-coefficients.png` (relative reconstruction error against number of coefficients kept, four representations; learned bases cost `Σ d_a²` extra entries, not shown in the x-axis).

## 7. Speaker explanation

We took three real volumes measured on Cartesian grids, a magnetic field in a 10 cm cube and two sound fields recorded by a robot-moved microphone, and asked the chisel to find, on its own, linear combinations of the grid coordinates along which the data organises itself.  It found the two trivial symmetries every tensor has, and then a continuum: no distinguished direction, in any dataset, at any instant, at any resolution.  Compared with shuffled data the fields are clearly structured, but the structure the chisel can see is smoothness, and when we let it build orthogonal coordinates from that, it rediscovered the standard Tucker (HOSVD) coordinates to within a few degrees.  The one gap we found was the number of microphones the sound had not yet reached, which `nondeg` removes.  Just past that gap, in the densest recording, the chisel's next move is to sort the microphones by their position along the direction the sound travels, with no coordinates supplied: the wavefront's geometry, seen but not isolated.  So: the method behaves correctly and honestly on measured data, it recovers known structure without being told, and it did not discover a new one here.
