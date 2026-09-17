# SymmetricGram

`SymmetricGramMethod` is a narrow, opt-in derivation method for dense real
three-way tensors.  It assembles the normal matrix of the universal-chisel
derivation equation after restricting every axis to symmetric operators.
This can be useful when a cubic symmetric model supplies a fixed number of
low-residual directions.  It is not a general replacement for the other
derivation methods and `:Auto` never selects it.

```julia
method = get_derivation_method(:SymmetricGram;
    eigensolver = :inverse, seed = 17,
    iterations = 12, oversample = 24,
    shift = 1e-10, max_bytes = 1 << 30)

Ω = IndTransverseOps(collect(inds(Γ)), SymmetricOp())
rΩ, expand_map, ders, report = derTrOpsReduced(
    method, Ω, UniversalChisel(3), Γ;
    nd = 3, tol = 1e-6, return_diagnostics = true)
```

The accepted input is deliberately specific:

- `Γ` is a real `Float32` or `Float64` ITensor with three equal-size axes.
- `Ω` has independent `SymmetricOp()` operators on all three axes and frames
  matching `Γ` in order.
- The chisel is exactly the one-row, all-ones `UniversalChisel(3)`.
- `nd` is a positive integer.  It asks for exactly that many smallest normal
  modes.  For the sphere model, `nd=3` includes the two scalar directions and
  one geometric direction.
- `tol` is finite and positive.  For `:inverse`, it is the relative normal
  eigen-residual used to stop inverse-subspace iteration; it is neither a
  noise level nor a mode-count rule.  `tol=Inf` is rejected.

`eigensolver=:inverse` is seeded shifted inverse-subspace iteration.  It uses
`seed`, up to `iterations` steps, and `oversample` extra probe columns.
`eigensolver=:eigen` is the dense direct eigensolve path intended for small
validation cases.  `shift` starts the inverse iteration's Cholesky shift as a
relative scale, and `max_bytes` imposes a conservative pre-allocation memory
bound for the dense normal matrix and its construction workspace.

The returned directions use Dleto's raw `SymmetricOp` coordinates and can be
passed through `expand_map` and `embedITensors` like other derivation results.
They are fixed approximate modes, not an inferred exact nullspace.

With `return_diagnostics=true`, the `DerivationReport` records
`policy=:fixed_nd`, the requested and returned counts, the selected solver,
seed, residuals, and normal-solve information.  Its `certified` field is
always `false`: the fixed count is a model assumption.  For `:inverse`, a
step cap reached before the requested residual tolerance produces
`status=:unconverged`; callers should apply their own acceptance policy.

For a notebook with explicitly prepared `Γ_exact` and `Γ_noisy` tensors, use
the same public call for each case; no benchmark file is included:

```julia
exact = derTrOpsReduced(method, Ω, UniversalChisel(3), Γ_exact;
                         nd=3, tol=1e-6, return_diagnostics=true)
noisy = derTrOpsReduced(method, Ω, UniversalChisel(3), Γ_noisy;
                         nd=3, tol=1e-6, return_diagnostics=true)
```

The command-line sphere demo provides matching exact and noisy runs:

```sh
julia --project=. labs/SymmetricSphereDemo.jl \
  --method=SymmetricGram --dim=50 --seed=17 --noise=0 --nd=3 --tol=1e-6

julia --project=. labs/SymmetricSphereDemo.jl \
  --method=SymmetricGram --dim=50 --seed=17 --noise=0.001 --nd=3 --tol=1e-6
```
