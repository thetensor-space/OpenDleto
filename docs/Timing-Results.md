# OpenDleto — measured timing and accuracy

All figures on this page were **measured**, on one machine (Apple silicon, Julia 1.12.3), on the
dates noted. Anything estimated rather than run is labelled so explicitly. Numbers taken while
another heavy job was running are excluded — see *Measurement hygiene* at the end, which is not a
footnote but a correction to figures that appeared earlier in this session's notes.

Reproduce with:

```bash
julia --project=. bench/ChiselOperationBench.jl short   # n = 4..10
julia --project=. bench/ChiselOperationBench.jl long    # der to n = 26, den to n = 15
```

## The test family

Liu's `K_M_field_tensor(n)`: powers of a cyclic shift matrix with random coefficients, then
scrambled by random basis changes on the row and column axes. It is the multiplication tensor of
`K[x]/(xⁿ − c)` — Example 5.2 of `null_patterns.pdf` — so it has a **known** derivation space of
dimension `2n`, present but invisible in the given basis.

Two properties of the family are load-bearing for the measurement:

- A **generic** tensor would be useless here. Its only derivations are the scalars, and its densor
  is then the *whole* tensor space (§5.4, "scalar derivations reveal nothing"), so `den` would be
  measuring nothing.
- The size grid must cross **n = 19**, because the derivation operator is `3n²`-dimensional and
  the dense gate sits at 1 GB; below that every solver variant is the same code path.

## 1. Where the operators actually sit

The two maps have completely different shapes, which is why one grid does not fit both operations.

| n | derivation map (`der`) | densor map (`den`) | dense densor copy |
|---|---|---|---|
| 10 | 300 × 300 | 20 000 × 1 000 | 0.1 GB |
| 15 | 675 × 675 | 101 250 × 3 375 | 2.5 GB |
| 19 | 1 083 × 1 083 | 260 642 × 6 859 | **13.3 GB** |
| 26 | 2 028 × 2 028 | 913 952 × 17 576 | **119.7 GB** |

The derivation map is square and small — 9 MB at n = 19, so densifying it is free and buys the
accuracy of a real SVD. The densor map at the same n is 13.3 GB. This is why the dense gate tests
**bytes**, not dimension: a pure dimension limit of 1000 would refuse the first and accept the
second.

## 2. Null solvers on the derivation map, n = 19

Universal chisel `[1,1,1]`, true dim Der = 38. Measured 2026-09-02.

| solver | dim found | max rel. residual | time | verdict |
|---|---|---|---|---|
| `SVDSolver` | **38** | 4.1e-14 | 4.3 s | correct (densifies; 9 MB) |
| `LUSolver` | 32 | 3.1e-14 | 0.6 s | incomplete — `lu` pivots rows only, so it is **not rank revealing** |
| `KrylovSolver` | 9 | 3.5e-15 | 2.9 s | accurate but partial: Arnoldi stops on an invariant subspace |
| `LanczosSolver` | 0 | — | 22.0 s | wrong end of the spectrum: `svdl` converges to the **largest** singular values |
| `CGSolver` | 19 | 7.1e-07 | 42.6 s | partial and marginal: unpreconditioned LOBPCG on `AᵗA`, i.e. κ(A)² |

Every one of these five was unreachable or crashed before this session's repairs; the table is
what they do once they run. Note this was measured with `nv` set to the **whole space**, which is
also what the old `nd <= 0` handling did — so it doubles as a record of what asking an iterative
solver for the entire spectrum costs.

## 3. Derivation methods, n = 19

| method | chisel | dim | max rel. residual | time |
|---|---|---|---|---|
| `SylverLining` / SVD | `[1,1,1]` | 38 | 4.1e-14 | 4.27 s |
| **`QuickDer`** (Liu's derivation lift, triple restriction) | `[1,1,1]` | 38 | **5.3e-15** | **1.39 s** |
| `SylverLining` / SVD | `[1,-1,0]` | 19 | 4.2e-14 | 0.47 s |
| **`QuickSylver`** (Liu's Sylvester lift, double restriction) | `[1,-1,0]` | 19 | **3.6e-16** | 0.77 s |

`QuickDer` is ~3× faster than the general method at n = 19 and an order of magnitude more
accurate. Both lift solvers restrict to a small corner, solve densely, then lift by least-squares
solves; they differ in how many axes get restricted (three vs two) and hence which chisels they
accept.

## 4. The densor, matrix-free — the main result

`den` used to cost `O(n⁷)` **memory**, because its default solver densified the map. `denLM` was
already abstract with a genuine adjoint; the defaults threw that away. After the repair, on maps
far too large to densify:

| n | `CGSolver` (eigensolver on `AᵗA`) | `LSMRSolver` (projection, unsquared) | true dim |
|---|---|---|---|
| 10 | 10 of 10, 2.0e-12 | **10 of 10, 2.0e-15**, 41 s | 10 |
| 15 | 15 of 15, 1.2e-11, 190 s | **15 of 15, 4.8e-15, 31 s** | 15 |
| 19 | **11 of 19**, 2.1e-05, 166 s | **19 of 19, 1.1e-14, 131 s** | 19 |

`LSMRSolver` is faster *and* four orders more accurate, and it is the only method that gets
n = 19 right at all.

### Why squaring was the problem

Ground truth at n = 10, from a dense SVD of the 20 000 × 1 000 densor map:

```
σ_max                    = 1.451e+03
smallest NONZERO σ       = 2.19e-02
the 10 null directions   ≈ 2.6e-13 .. 4.5e-13
```

So the null space is separated from the rest of the spectrum by **1.5e-5 relative in σ** — an
enormous, comfortable gap — and by **2.3e-10 in σ²**. Still above `eps` at this size, but it sheds
an order of magnitude of headroom per step in n, and it is entirely avoidable.

`LSMRSolver` avoids it by not being an eigensolver at all. It is a projection:

    null(A) = { w − A⁺(A w) : w arbitrary }

since `A⁺A` is the orthogonal projector onto the row space, and `A⁺y` is exactly what LSMR
computes from `A` and `Aᵗ` alone. One LSMR solve per candidate vector; no shift, no eigenvalue
iteration, nothing squared, and the reported value is `σ = ‖Av‖` straight from `A`.

Effect of iterative refinement of the projection, n = 10 (truth ≈ 3e-13):

| passes | σ returned |
|---|---|
| 1 | 2.0e-05 |
| 2 | **2.2e-13** |

One pass leaves an error of about `lsmr_tol · κ_row`. The projector is idempotent in exact
arithmetic, so re-projecting an already-nearly-null vector is standard refinement: `‖Az‖` is now
tiny, so the same *relative* tolerance buys a far smaller absolute correction.

### Verified range, and the remaining wall

`den` is verified correct to **n = 19** (~2 min). At **n = 26** it was measured running at
**7 map applications/s** — each application is `|Δ| = 52` forward plus 52 adjoint contractions on
17 576-element tensors — which **extrapolates to roughly 4 hours**. It was stopped rather than run
to completion, so no n = 26 figure is claimed.

The memory wall is gone; a time wall remains, and it is in the *map application*, not the solver:
cost is `O(k · lsmr_iters)` applications at `O(|Δ| · n³) = O(n⁴)` each. The lever is the LSMR
iteration count, which goes like `√κ_row` and would need a preconditioner. Not attempted.

## 5. Three operations, n = 4..10

Median over 3 trials, universal chisel unless noted. `der` and `stratify` are essentially free at
these sizes; `den` dominates.

| operation | n = 4 | n = 6 | n = 8 | n = 10 |
|---|---|---|---|---|
| `der` (SylverLining/SVD) | 0.002 s | 0.005 s | 0.012 s | 0.026 s |
| `der` (`QuickDer`) | 0.0006 s | 0.003 s | 0.005 s | 0.008 s |
| `den` | 0.43 s | — | — | 1.43 s |
| `stratify` | 0.002 s | 0.005 s | 0.013 s | 0.023 s |

Residuals across all of these stayed at 1e-14 … 1e-16.

## 6. An intended result that looks like a bug

`stratify`'s frame conditioning `κ` comes out **identical across all four (method, chisel)
categories**, to about ten digits, while varying from trial to trial:

| trial | κ, all four categories |
|---|---|
| 1 | 38.09639293138564 / …39932 / …40896 / …39904 |
| 2 | 26.37213820444681 / …4436958 / …454817 / …435377 |

This is correct. `K[x]/(xⁿ − c)` is commutative, so its elements share an eigenbasis, and the
frame that puts a random derivation into real canonical form is the same frame whichever
derivation was drawn and whichever chisel or method found it. **The pattern is a property of the
tensor.** So κ cross-checks that the methods agree rather than discriminating between them, and
the discriminating axis for `stratify` is time.

## Measurement hygiene

Two testset timings recorded earlier in this session were wrong and are corrected here:

| testset | reported under load | **isolated** |
|---|---|---|
| `TransverseOpsIndependant` | 16m39s, 17m18s | **11.4 s** |
| `Tensor Synthesis` | 19m13s | **2m21 s** |

Both were measured while the n = 26 densor run was saturating the CPU; the 5-minute load average
reached 35. They looked exactly like a performance regression introduced by that commit, and were
not one. **Do not run the benchmark and the test suite concurrently on this machine**, and re-time
anything measured alongside another job.
