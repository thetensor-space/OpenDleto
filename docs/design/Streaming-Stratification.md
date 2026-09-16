# Streaming stratification

*Status: design note + phase 1/2 implemented (`src/solvers/StreamingCore.jl`).
Written 2026-09-16 on `and-the-beat-goes-on/streaming-stratification`.*

The question this answers: frames of a video arrive one at a time, so one axis of
the tensor is never complete.  Can stratification still run — building the
restricted core from what has arrived, keeping many candidate solutions alive,
and lifting as the stream fills in?

Yes, under one condition, and most of the machinery is already in the right
shape.  This note says which parts, what the condition is, and what is left.

---

## 1. The condition: the stream axis must be disengaged

Write the tensor as `Γ` with a stream axis `t` whose dimension `d_t` grows.  A
derivation carries one unknown operator per *engaged* axis (`src/Chisels.jl`,
`engaged(P)`), and the unknown on axis `a` is `d_a × d_a`.

If `t` is engaged, the unknown `X_t` is `d_t × d_t` and **grows with the
stream**.  No amount of lifting repairs that: the object being solved for is
itself unbounded.  So the streaming problem is the one where the chisel has a
zero in the `t` column — for a valence-3 video `(x, y, t)` that is the adjoint
chisel `[c₁, c₂, 0]`, and for the `640×480×F×3` movie target it is any chisel
engaging the spatial and colour axes and leaving the frame axis alone.

That is not a limitation invented for convenience.  It is exactly the case
`QuickSylver` already implements: `_qs_system_matrix` (`src/solvers/QuickSylver.jl`)
is a `vcat` of one block per slice of the disengaged third axis, and
`derTrOpsReduced(::QuickSylverMethod, …)` permutes the two engaged axes to the
front and slices along the leftover.  **The slice index is the frame index.**

### What follows from it

Frames enter as *equations*, never as unknowns.  Hence

> **Monotonicity.**  `Z(Γ₁..F₊₁) ⊆ Z(Γ₁..F)` — the derivation space is
> non-increasing along the stream.

Three consequences, and they are the whole reason this is tractable:

1. **Candidates only die.**  A streaming solver never has to discover a
   direction it had already discarded.
2. **A streaming answer is optimistic, never wrong in the dangerous
   direction.**  It reports a space that is too large and refines it.  For a
   codec that is the right error: encode with the current basis, re-key when it
   dies.
3. **Space collapse is a scene cut.**  Over a whole clip the derivation space
   may be trivial while each shot has a rich one.  The frame at which the
   candidate space collapses is a principled cut point, not a heuristic — and
   it gives the MPEG comparison a story about I-frame placement.

## 2. The restriction sizes do not depend on the stream length

`_qdn_restriction_sizes` (`src/solvers/QuickDerN.jl`) enforces two conditions:

* (i) `∏ r_a ≥ Σ_{a engaged} d_a r_a + slack`
* (ii) for each engaged axis needing a lift, `∏_{b≠a} r_b ≥ d_a + slack`

Both sums run over **engaged** axes only.  With `t` disengaged, `d_t` appears in
neither.  The stream axis contributes to `∏ r_a` through `r_t` alone, and a
larger `r_t` only makes both conditions easier.

So `r_t` is a **declared budget**, not a derived quantity, and the solver never
needs to know how long the stream is.  `StreamingCore` computes the remaining
`r_a` by calling `_qdn_restriction_sizes` on the dimension vector with `d_t`
replaced by `r_t` — i.e. the stream axis presented as already saturated at its
budget — and then pins `r[t] = r_t`.

The same arithmetic in QuickSylver's own sizing rule
(`_qs_select_restriction_sizes`) gives the **warm-up length**.  It needs
`a'·b'·c ≥ a'·r + b'·s` with `c` = frames so far; at `a' = b' = m` that is

```
c ≥ (r + s) / m
```

For 640×480 at `m = 32`, about 35 frames — a little over a second at 30 fps.
And `balanced_block_size = ceil(2·max(r,s)² / c)` *shrinks* as `c` grows, so more
frames means a smaller restriction is needed.  The economics run the right way.

## 3. What is a running sum over frames

Everything the solve and the lift consume is a contraction of `Γ` against fixed
matrices on every axis but one or two — and `t` is never one of those two when
it is disengaged.  So each is a sum over frames and accumulates in fixed space.

**The cross sketches.**  `_qdn_cross_sketches` builds `S_a = Γ ×_{b≠a} W_b`.
Since `t ≠ a`, the `t`-mode is contracted with `W_t` and

```
S_a  =  Σ_k  ( F_k ×_{b ∉ {a,t}} W_b )  ⊗  W_t[k, :]
```

one rank-one update per frame into a buffer of `d_a · r^{n-1}` entries.  For the
movie shape that is a few megabytes, independent of `F`.

**The pair tensors.**  The lift's coefficient blocks
`H_{ab} = Γ ×_a W_a⊥ ×_{c ∉ {a,b}} W_c` (`_qdn_pair_tensor`) have the same form
with `t ∉ {a,b}`, so they accumulate identically.  Size
`(d_a − r_a) × d_b × r^{n-2}`; on the movie shape roughly 60 MB per `(a,b)`
pair, again independent of `F`.  Accumulating these in the *same* pass is what
makes the method genuinely single-pass: without them the lift would have to
re-read the stream.

**The sketch on the stream axis itself.**  `_qdn_axis` builds `W_a` from a full
`d_a × d_a` QR, which needs `d_a` up front.  The stream axis cannot have one.
It does not need one either: it is disengaged, so it carries no unknown, no
`W_a⊥`, and no lift — `W_t` is only ever applied as a sketch inside another
axis's contraction.  A plain Gaussian suffices, and its rows are drawn one per
frame from a seeded RNG, so `W_t` is never materialised ahead of the stream.
Its overall scale is global to every `S_a` and `H_{ab}` and cancels out of a
homogeneous null problem.

`restriction = :corner` is **wrong** for the stream axis and must not be used:
`I[:, 1:r_t]` would look only at the first `r_t` frames.

**The Z-law residual.**  Already written as an accumulator.
`der_residual_squares` (`src/Derivations.jl`) blocks along `argmax(dims)` — the
frame axis on a movie — and does `acc[rho] += sum(abs2, Eb)` per block, skipping
disengaged axes through `iszero(c) && continue`.  Because

```
(Γ ×_a M_a)[…, k, …] = F_k ×_a M_a     for every a ≠ t
```

the per-row residual decomposes exactly frame by frame, and the streaming
monitor is the existing kernel called on one frame with the `t` column of the
chisel and the `t` operator dropped.  No new numerics.

## 4. What is *not* a running sum: the lift

This is the actual research problem, and the reason phase 3 is a separate piece
of work.

In QuickSylver's `_qs_solve_and_lift` the per-axis least-squares problem is

```
M_R      = vcat_k  S[Ir, :, k]                     # rows over frames — accumulable
N_R(X_I) = vcat_k (T − X_I·R)[Ir, J_hat, k]        # rows over frames, but depends on X_I
```

`M_Rᵗ M_R` is an `s × s` Gram you accumulate and forget the frame.  But `N_R`
depends on the restricted solution `X_I`, and `X_I` keeps changing as the
candidate set shrinks.

The way through is that **`N_R` is affine in `X_I`**.  Fix a candidate basis
`{X⁽¹⁾ … X⁽ᵖ⁾}` after warm-up and accumulate `M_Rᵗ N_R(X⁽ⁱ⁾)` per candidate:
`p` buffers of `s × |J_hat|`, about 0.9 MB each at 480².  When the candidate
space shrinks to a subspace of the old span, **the accumulators recombine under
the same linear combination**, because everything in sight is linear in `X_I`.
That invariant is what turns "solve for many options and lift as the stream
fills in" into an algorithm.

The price: the warm-up span must be taken generously — large `nd`, loose cut.
Monotonicity protects against the span being too *big*; it does not protect
against a numerically truncated warm-up span being too *small*, and if a later
frame needs a direction that was cut at warm-up, the run restarts.  The relevant
knobs are `QDN_LIFT_GAP_RATIO` / `QDN_LIFT_CEILING`, and the right mode is the
fixed-count policy (`ndreq > 0`), which reports lift residuals instead of
cutting on them.

## 5. If the stream axis must be engaged

Three ways out, in increasing order of interest:

1. **A structured operator class on time.**  The operator set today is
   `UniversalOp / DiagonalOp / SymmetricOp / AntiSymmetricOp / ScalarOp /
   EmptyOp` (`src/ops/OperatorImpls.jl`).  A `ToeplitzOp` or `BandedOp(w)` makes
   `X_t` a filter of `w` taps whatever the stream length.  "A shift-invariant
   derivation on time" is the honest statement of *the same temporal relation
   holds at every offset*, which is what one wants from video, and it is the
   closest structural analogue to motion compensation.  Contained work
   (`coordinates`, `globalDim`, the `embedITensors` path) — but every solver
   that hard-requires `UniversalOp` must be told; `_qs_validate` refuses
   anything else outright.
2. **A sliding window / GOP.**  `X_t` is `W × W` on a window; stratify per GOP
   and lift across GOPs at a second level.  That is MPEG's own hierarchy, which
   makes the comparison structural rather than contrived.
3. **Two passes.**  Stratify space and colour in the streaming regime, then
   stratify the coefficient stream along time offline.

## 6. Known gaps in the streaming path

* **`_qdn_trivial_ders` re-reads `Γ`.**  When whitening truncates a
  rank-deficient mode unfolding, the trivial derivations are read off the full
  tensor.  A streaming run must detect the truncation and decline rather than
  silently drop that part of the space.
* **The change of basis jumps.**  `stratify` draws a random derivation
  combination and runs `realCanonicalForm` per axis (`src/Densors.jl`).  When
  the derivation space shrinks, that basis changes discontinuously and the
  output transform jumps mid-stream.  A compressor wants a *stable* basis, so
  the policy should be: commit after warm-up, report drift, re-key only on
  collapse.  This is a decision to make explicitly, not to inherit from the
  random draw.
* **Host arrays only.**  The accumulator uses `_qdn_mode_order`'s host
  behaviour (natural axis order).  A device path would have to fix the order
  explicitly so that every frame's chain is the same one.

## 7. Plan

| Phase | What | State |
|---|---|---|
| 1 | Streaming core: `push!(core, frame)` accumulating cross sketches and pair tensors | **done** — `src/solvers/StreamingCore.jl` |
| 2 | Streaming Z-law residual, reusing `der_residual_squares` | **done** — same file |
| 3 | Inject the accumulated core into `_qdn_solve_and_lift`; candidate-set lift with per-candidate accumulators | next |
| 4 | Structured time operators (`ToeplitzOp`/`BandedOp`), if `t` must be engaged | later |

Phases 1 and 2 are useful on their own even for batch runs: they make the
`d^n` passes streamable from disk for movies that do not fit in memory.
Phase 3 is the paper.
