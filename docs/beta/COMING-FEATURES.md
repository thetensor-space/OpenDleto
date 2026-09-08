# Coming in a future `beta`

What is planned, in progress, or measured-and-queued for the next beta releases. Sourced from
`docs/CONTEXT.md` ("Next up", "Next levers", "Order of work"), `docs/review/Refactor-Plan.md`
§7, `docs/design/Deployment-Plan.md` §5, `docs/design/Native-Core-Plan.md`, and the review
findings in [REVIEW.md](REVIEW.md). Written 2026-09-08 against `v1.5-beta-2026-09-04`.

Items are grouped by how certain they are: **committed** (decided with the author, work
scheduled), **planned** (in a design note, not yet scheduled), **gated** (only if a
measurement triggers it).

---

## Committed

### C1. Video regime: the eigensolve is the lever
The `640×480×F×3` movie runs today (Float32, CPU, ~110–170 s for one minute, ~19–26 GB).
Per-stage measurement showed the restricted eigensolve carries ~90 % of the frame-linear
cost because the restriction size grows with `F`; the GPU tensor stages save only ~9 s.
Next: shrink or precondition the restricted solve on video shapes (Kronecker
block-diagonal preconditioner from the whitened Gram structure; a range finder for the
complement of a tiny null space), and a Metal warm-up entry point so a one-shot device
run does not pay ~14 s of pipeline specialisation.

### C2. Float16 as a real storage contract
Today Float16 input is promoted to Float32 before the tensor is touched, so it saves
nothing on the CPU and only `data_floor(Float16)` distinguishes it. The measured contract
(`docs/design/Float16-Metal.md`) is **half-precision storage and operands on the GPU with
fp32 accumulation, a Float32 restricted eigensolve, and a verdict floored by
`data_floor(Float16)`**. Needs a mixed-eltype `_qdn_ttm` and a decision about the host copy
(a precision-policy change). Target: one minute of video under ~15 GB.

### C3. Memory: keep the tensor once
Six copies of the tensor lived at the worst moment of a `d = 500` run; `nondeg` alone
churned 25 GB. The lean sphere harness and the in-place mode products fixed the bench side;
the remaining work is to make the *library* path (`nondeg`, `_qdn_pair_tensor`, the
ITensor boundary) hold one copy, so `d = 1000` at valence 3 stops needing 13.7 GB.

### C4. The lost eigenvalue copy
Determinism (seeded solves) exposed that the restricted solve loses one copy of a multiple
eigenvalue on roughly one `(d, T)` case in eight, set by the start vector. Candidate fixes
already named: the restriction sizes (a 6.8 % overdetermined system at `d = 300` has a
13-dimensional near-null space against 3 true derivations) and the null threshold. This is
the main correctness item on the numerics list.

### C5. `Stratification` return type and the `σ_{e+1}` verdict
Decided 2026-09-02 (Refactor-Plan §5.1): `stratify` returns a `Stratification` struct with
`Σ`, `Xs`, `δ`, `pattern`, `verdict`, `chisel` instead of a tuple, so fields can be added
without breaking callers, and it stops discarding `δ` (the sparsity pattern itself). Paired
with the Algorithm-2 test `σ_{e+1}` that decides "this tensor admits no pattern for this
chisel" — the accuracy oracle `stratify` has never had.

### C6. Stop forming `AᵗA`
`sylvester = ester ∘ sylve` is `AᵗA`, so SylverLining's operator has condition number
`κ(A)²`. `LSMRSolver` already avoids it for `den`; the Z-path should take the SVD of `A` or
run LSQR on it, as Algorithm 2 specifies. Evidence table in `docs/CONTEXT.md`
("Conditioning is the binding constraint").

---

## Planned

### P1. `Chisel` keyed by `Index`, curried builders (Refactor-Plan Phase 1)
The full setting `(𝕋, Ω, P)` as one type; builders return a `ChiselTemplate` that completes
itself against a tensor or frame; engagement is a set of `Index` terms, not positions, so a
chisel survives axis reorders. Transpose convention fixed on the first coordinate and
tracked passively. Decided; not started.

### P2. One contraction kernel with a choice of unknown slot (Phase 2)
T and Z are the same linear system with a different unknown. Extract the chisel-weighted
contraction and the `ester`/`sylve` adjoint pair out of `SylverLining` into a kernel layer
that QuickDer, SylverLining and `den` all consume. This is also the fix for the
restrict/solve/lift pattern being written three times (see REVIEW.md).

### P3. Named closures and wrappers (Phase 6)
`Der`, `Cen`, `Nuc`, `Adj`, `SelfAdjoint` as thin wrappers matching the Magma `Sylver`
surface; Magma's sanity-check harness as an opt-in `verify = true`.

### P4. I-sets and applications (Phase 7)
Ideals; then the der-densor isomorphism method (`der-densor.pdf`). I-sets are entirely
absent today.

### P5. Solver layer clean-up
`LUSolver` is not rank revealing (32 of 38) and should not be offered as a general null
solver; library `println` chatter (490 lines in one test run) becomes `@debug` or the
progress channel; `KrylovKit`/`IterativeSolvers` decide between hard dependency and
extension; `Arpack` gets exercised in CI, not only when a bench environment happens to
carry it.

### P6. Package hygiene for a public beta
Version bump from 0.1.0; `[compat]` for every dependency; drop `IJulia`, `PlotlyJS`,
`PlotlyKaleido`, `CSV`, `DataFrames`, `JSON` from hard deps (plotting is already an
extension); a CHANGELOG at the root; CI; stop versioning `bench/**/*.log` and notebook
outputs. Details and sizes in REVIEW.md.

### P7. Remaining law families as tests
Scalar lower bound `dim Der ≥ dim null(C)`; chisel row-span / torus equivalence; product
closure. The `@test_broken` items in `TestDerivationLaws.jl` (SylverLining on
near-degenerate tensors) become passing or are removed with a stated reason.

### P8. `PrecompileTools` workload
First `stratify` call ≤ 2 s (Native-Core-Plan Phase 1b). Today the first call pays the
full JIT; BOARD.md flagged 2.38 s and the device path far more.

---

## Gated

### G1. CUDA twin of the Metal extension (Deployment-Plan step 3)
`ext/DletoCUDAExt.jl` mirroring `DletoMetalSylver.jl`. Zero dependency on other work;
needs an NVIDIA box to test on. Runs on a server without touching the orchestration layer.

### G2. Trim-safe native library via `juliac` (Deployment-Plan step 2)
A `libdleto_core` with concrete types end to end behind a C ABI, built with `juliac
--trim=safe`, ~1–2 MB, ~30 ms cold start. Triggered when a consumer needs an artifact
without a Julia runtime. The experiment ran; the package as it stands does not trim.

### G3. Mixed-precision eigensolver as the shipped default (Deployment-Plan step 5)
Float32 contraction with a Float64 resolve for Float16/Float32 input; `tol` re-expressed as
`k · eps(T)` throughout.

### G4. Native (Rust/C++) kernels
**Not now.** Measured at `d = 100` the dense branch is at the BLAS floor; the walls are
iteration counts. Gate: a stage measured ≥ 2× off its hardware floor that Julia cannot
close, and a two-week budget after which the branch is deleted if the 2× is not measured.

### G5. `AppleAccelerate` via LBT forwarding
Opt-in, benchmarked on the `syrk`; vendor claims unverified.

---

## Open questions the author has not yet ruled on

1. Keep ITensors as the substrate, or abstract over it? (Beta already runs the hot kernels
   on plain arrays; the ITensor boundary is thin but everywhere.)
2. Which application is next: isomorphism testing, SphereLab-style continuous patterns,
   hypergraphs?
3. Exact arithmetic (rationals / finite fields), or floats only?
4. Chris Liu's thesis: the sufficiency theorems for restriction sizes, and whether valence 3
   with a one-row chisel is essential or incidental — the restriction sizes in QuickDer
   currently rest on the reference implementation's choices, not a stated theorem.
