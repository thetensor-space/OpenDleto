# Harmonic retrieval with Dleto: non-uniform, clustered frequencies

Direction proposed 11 October 2026: data whose structure lives in a spectral
basis that is *not* the FFT grid, e.g. a few damped exponentials with clustered
frequencies.  A short uniformly sampled record cannot resolve the clusters by
FFT, but the record is a sum of separable exponentials, i.e. a CP model, and
Dleto's derivation eigenbasis is the set of true exponentials.

## Method (what the scripts do)

`T = Σ_m c_m · a_m ⊗ b_m ⊗ c_m` with damped complex exponential factors.
`nondeg` (tolerance at the noise level) reduces to the `M × M × M` core; the
core has exactly `2M` derivations (the two scalars included), so `der`/`stratify`
are called with `nd = 2M` (asking for more than exist ruins the combination);
`stratify` diagonalises the core; the factor of stratified index `p` on axis `a`
is column `p` of `conj(E_a) · X_a⁻ᵀ` (`E` the nondeg basis, `X` the stratify
transform; ITensor contractions are bilinear).  Frequencies are read from the
factors by a weighted log-ratio fit.

## Synthetic check (`synthetic_check.jl n sep`)

Five components, two clustered pairs `sep` cycles/sample apart, damping 0.02,
complex Gaussian noise.  Success means the strata hold 5 nonzeros where the
original is dense, and the fitted frequencies match.

| n | spacing in FFT bins | FFT resolves | Dleto exact down to | first loss |
|---|---|---|---|---|
| 16 | 0.32 | 1–3 of 5 | 30 dB (errors ≤ 6e-4 cycles/sample) | 20 dB: nondeg drops a component |
| 32 | 0.64 | 2–3 of 5 | 30 dB (≤ 1e-4) | 20 dB: one component lost |
| 64 | 1.28 | 4 of 5 | 20 dB (≤ 1e-4) | 10 dB |

Logs in `results/`, figures in `figures/`.  Noise-free recovery is exact in
every case.  Two Dleto defects had to be fixed first (both in `src/`, test suite
passes):

* `nondeg` selected grid positions instead of singular vectors (shapes
  coincided, so it never errored); the basis now enters conjugated, since the
  contraction is bilinear.  Earlier nondeg-based results had the right
  dimensions and a meaningless basis.
* `realCanonicalForm` projected complex eigenvectors to their real parts; on
  complex input it now returns the eigendecomposition.

## Real data: nmrglue's 2D 15N–13C test FID (`nmr/`)

`nmr/prepare_subband.py` reads the NMRPipe file (166 × 1500 complex, F1 15N
5.6 kHz, F2 13C 50 kHz), picks the reference peaks from the full record, and
band-selects a six-peak region by mixing and polyphase decimation (declared
preprocessing).  `nmr/dleto_subband.jl` forms a Hankel third mode and runs the
pipeline.  Outcome: **negative** for this dataset.  The strata are not sparser
than the record (189 of 504 entries above 10% of the maximum versus 101 of
1764) and the fitted components are noise.  Cause: solid-state linewidths
(about 50 Hz) make T2* about 5 ms, so after decimating to the sub-band only two
or three t2 samples carry signal, and the full record is already
linewidth-limited, not record-length-limited; the overall time-domain SNR is
near 0 dB.  The regime the method needs (lines much narrower than their
spacing, record truncated for time) is that of solution-state NMR with
truncated indirect dimensions.

## Real data 2: cyclosporin edited HSQC (Zenodo 13307304, experiment 12, CC BY 4.0)

Solution-state 1H–13C HSQC on a Bruker AV500 (`nmr/read_hsqc.py`: digital-filter
removal, echo/antiecho combined into a complex t1; 128 × 443 complex; 13C
spectral width 20.8 kHz, so one FFT bin of the indirect axis is 163 Hz = 1.3
ppm).  This is the right regime: narrow lines, indirect axis limited by record
length.  Reference peaks come from the full record.

* Cα-H band (5.2–6.05 ppm, 17 cross peaks, `labs/data/nmr/hsqc_ca.h5`,
  `results/hsqc-ca-N128.log`): too many components for the 37-point Hankel
  axes (rank 15), and several cross peaks share a 13C shift (CH2 groups),
  which breaks CP uniqueness on the t1 axis.  Components land on reference
  peaks only sporadically; results vary between runs.
* Three-peak band (1H offsets 305–395 Hz, `hsqc_trio.h5`, `results/hsqc-trio.log`):
  peaks at 13C −1465 and −1546 Hz (81 Hz apart, half an FFT bin at full t1
  length) and −2686 Hz (weak).  Outcome: **partial**.  The 1H frequencies are
  recovered to 1–20 Hz; the two 13C components come out near −1500 ± 50 Hz but
  are not cleanly separated into −1465 and −1546 at any t1 length (errors
  15–100 Hz against an 81 Hz separation), and the weak third peak is not
  found.  The strata are only moderately sparser than the record (44 of 100
  entries above 10% of the maximum versus 2103 of 2560).  The t1 unfolding's
  singular values (1, 0.49, 0.27, 0.11, 0.07) show no rank-3 gap: the
  sub-band is not a clean three-exponential model (neighbouring peaks at the
  filter edges, t1 noise), and with only 8 decimated t2 points the Hankel
  axes carry little averaging.

**Where this leaves the idea.** The mechanism works exactly as proposed on
synthetic data (clusters a third of a bin apart from 16 samples at 40 dB).  On
the one suitable public dataset found, the obstacles are peak density (any
band wide enough to give samples holds too many components) and model
mismatch, not resolution.  Next steps that would settle it: a dataset with
fewer, stronger, well-separated-in-1H peaks (a small molecule rather than an
11-residue peptide) or a 3D experiment, a Hankel-aware chisel that ties the
j and k factors (the same exponential), and an ESPRIT/matrix-pencil baseline
on the same sub-band to separate method limits from data limits.

### ESPRIT baseline on the same three-peak sub-band (`nmr/baseline_esprit.jl`)

Multiple-snapshot Hankel ESPRIT along the 13C axis (t2 columns as snapshots),
model order 3, full t1 length: 13C = −2684, −1843, −1521 Hz against the
reference −2686, (−1872 leaking in at the filter edge), and the pair
−1465 / −1546.  ESPRIT finds the weak third peak, which Dleto misses, but it
does **not** split the 81 Hz pair either, at any t1 length (−1521 at 128,
−1543 at 32, −1554 at 16 points); along 1H it resolves the three protons with
an 8 Hz bias.  So on this record the unresolved pair is a data limit (SNR and
record length: the pair accumulates only half a cycle of phase difference over
the 6.1 ms indirect record), and Dleto's weakness relative to ESPRIT is noise
robustness with only eight decimated t2 points behind the Hankel axes.
Keeping the sparsest of twelve random derivation combinations
(`results/hsqc-trio-best.log`) does not change the outcome.

### Synthetic record with the cyclosporin sub-band parameters (`nmr/synthetic_from_hsqc.jl`)

Three components with the measured frequencies and amplitude ratios (124 : 105 : 10),
13C linewidth 5 Hz, 1H 3 Hz, long t1 axis Hankelized, 8 decimated t2 points.

| t1 points (cycles of relative phase of the 81 Hz pair) | 40 dB | 30 dB | 20 dB | 15 dB |
|---|---|---|---|---|
| 128 (0.5) | pair to 0.3 Hz, third found | 1.1 Hz | pair to 5 Hz, weak third lost by nondeg | pair merged |
| 256 (1.0) | 0.1 Hz | 0.6 Hz | pair resolved, third lost | pair merged |
| 512 (2.0) | 0.0 Hz | 0.1 Hz | pair resolved, third lost | pair resolved (3 Hz), third lost |

So at the real record length the pair is resolvable down to about 20 dB, half an
FFT bin apart; the real record must sit below that per component.  The remedy is
signal-to-noise (more scans, a more concentrated sample) or a longer indirect
acquisition, not a change of method.

## Real data 3: AMP HSQC from BMRB (bmse000005, set01 experiment 6) — the demonstration

Adenosine monophosphate, 100 mM in D2O, Bruker 500 MHz, `hsqcetgp`, 16 scans,
128 complex 13C increments over 170 ppm (one FFT bin = 167 Hz = 1.33 ppm), 443
1H points after digital-filter removal.  Seven CH/CH2 cross peaks at 440–1700×
the noise floor.  Files: `labs/data/nmr/bmse000005/6/` (BMRB metabolomics,
public), reader `nmr/read_hsqc.py`, band `nmr/prepare_subband.py` with centre
−220 Hz, half-width 225 Hz, t2 decimated by 14 (1H 3.8–4.7 ppm: the ribose
carbons C5′ 63.1, C3′ 70.4, C2′ 74.4, C4′ 84.3 ppm; C5′ carries two protons, so
five cross peaks, M = 5).  Hankel third mode on t2 (32 → 17 × 16).  Logs:
`results/amp-ribose.log` (M = 4), `results/amp-ribose-M5.log`; figure
`figures/amp-ribose-fft-vs-dleto.png`.

| 13C increments | FFT bin | strata nonzeros (>10% of max) vs record | 13C errors (Hz) | 1H errors (Hz) | outcome |
|---|---|---|---|---|---|
| 128 | 1.3 ppm | 5 of 256 vs 32,064 of 34,816 | 4–35 | ≤ 6 | all five |
| 32 | 5.3 ppm | 5 of 256 vs 8,024 of 8,704 | 2–38 | ≤ 4 | all five |
| **16** | **10.6 ppm** | 5 of 256 vs 4,007 of 4,352 | 2–33 | ≤ 8 | all five; **C3′/C2′ (4 ppm apart) resolved** |
| 8 | 21.2 ppm | 7 of 192 vs 2,026 of 2,176 | 47–67, pair merged | ≤ 10 | C5′, C4′ right; C3′/C2′ reported at their midpoint |

At 16 increments the zero-filled FFT shows vertical streaks spanning more than
10 ppm, and multiple-snapshot ESPRIT along 13C (`nmr/baseline_esprit.jl`)
merges C3′/C2′ into one component at −1575 Hz; Dleto separates them because
the 1H axis separates the cross peaks and the derivation algebra assigns each
its own 13C factor.  The limit is reached at 8 increments, where the pair has
only 0.19 cycles of relative phase over the record.  Reproduce:

```
python nmr/read_hsqc.py labs/data/nmr/bmse000005/6 labs/data/nmr/amp_hsqc.h5 amp.png
python nmr/prepare_subband.py labs/data/nmr/amp_hsqc.h5 labs/data/nmr/amp_ribose.h5 -220 225 14 0 30000 1
julia -t 4 --project=. labs/harmonic-retrieval/nmr/dleto_subband.jl labs/data/nmr/amp_ribose.h5 16 128,32,16,8 5
julia --project=. labs/harmonic-retrieval/nmr/baseline_esprit.jl labs/data/nmr/amp_ribose.h5 128,32,16,8 4
python nmr/plot_amp.py labs/data/nmr/amp_ribose.h5 results/amp-ribose-M5.log figures/amp-ribose-fft-vs-dleto.png
```
(Python from an environment with nmrglue, numpy, scipy, h5py, matplotlib.)

## Real data 4: a true three-mode record, the 3D HNCO of BMRB 15063

Mouse thiamine triphosphatase, Varian 600 MHz, `timedomain_data/hnco.fid` (40 15N × 32
13C′ × 1024 1H).  `nmr/read_hnco.py` combines the hypercomplex increments (States on
13C′; the 15N echo trace alone, chosen by a scan of the plausible readings for the one
without quadrature images; trace order "regular"), `nmr/prepare_subband3d.py 3.62 0.06
150` cuts the 1H band 3.56–3.68 ppm (two strong peaks with 13C′ 0.93 ppm apart, a weak
third, and a fourth at the band edge with 13C′ −1.24 ppm), decimating only the 1H axis.
The prepared sub-band is versioned in `data/`.  `nmr/dleto_hnco.jl` runs Dleto on the
(15N, 13C′, 1H) tensor as acquired; no mode is constructed.

For a narrower, more reader-friendly walkthrough of just this measured 3D case,
plus a tiny `6×6×6` synthetic warm-up, see `../HNCOFocusedLab.ipynb`.

Reproducible across seeds with M = 3 and nondeg tolerance 0.2 (`results/hnco-band2.log`
holds an earlier M = 4 run that was not stable):

| 13C′ increments | FFT bin | strata nonzeros | recovered 13C′ (ppm) | reference |
|---|---|---|---|---|
| 32 | 0.62 ppm | 3 of 27 (record 4469 of 8960) | 1.00, −0.08, −1.22 | 0.93, 0.00, −1.24 |
| 24, 16 | 0.83, 1.24 ppm | 3 of 27 | 0.99, −0.11, −1.25 | |
| 12 | 1.66 ppm | 4–5 of 27 (record 1728 of 3360) | 1.01, −0.03 to −0.12, −1.15 | |
| 8 | 2.49 ppm | | pair collapses to its average | |

15N within 0.3 ppm and 1H within 0.01 ppm for the two strong peaks at every length down
to 12; the edge peak's 15N is unreliable.  The weak in-band peak (171 × floor) is below
what nondeg keeps.  Notebook: `../ClusteredFrequencyLab.ipynb` (synthetic theory, the AMP
HSQC with the Hankel mode, and this HNCO, with 3D pictures of the tensor before and
after stratification).
