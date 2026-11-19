# Approximate derivations

`der` searches for a numerically justified nullspace using a spectral gap.
Its `tol` is a ceiling: raising it need not add directions if the gap above
the exact scalar laws remains dominant. Use `approx_der` when the question is
instead **which directions fall below a chosen relative residual cutoff**.

```julia
using Dleto
approx = Dleto.approx_der(Γ; tol=6e-4)
approx.accepted_indices
approx.approximate_nonscalar_count

view = Dleto.stratify(Γ; mode=:approx, tol=6e-4,
                      descending_index=first(approx.accepted_indices))
view.Σ, view.Xs, view.relative_residual
```

`approx_der` computes the SVD of the **rectangular** universal-chisel
derivation map. It includes every right singular direction satisfying
`σᵢ/σ₁ ≤ tol`. This normally includes the `valence−1` scalar
directions when the cutoff exceeds their numerical residuals. The returned
`laws` follow descending spectral order and match `accepted_indices`.
For any normalized vector in the accepted span, the derivation equation
residual is at most `tol * σ₁` in the map's Euclidean coordinates.

`approx_stratify(Γ; tol, descending_index)` and
`stratify(Γ; mode=:approx, tol, descending_index)` select **one** accepted
nonscalar right singular direction and use the usual coordinate-change
routine. If `descending_index` is omitted, they select the first accepted
nonscalar direction, closest to the cutoff. The return value includes the
transformed tensor `Σ`, frames `Xs`, selected `law`, relative residual,
accepted count, approximate nonscalar count, and threshold gap. The result
is an exploratory coordinate view, not a certificate of exact off-block
zeros. Several approximate directions need not give a stable common
stratification.

Check frame condition numbers and reconstruct after removing the proposed
off-block entries in the **original tensor coordinates**. Report the kept
support alongside reconstruction error and compare the same procedure with
a suitable control tensor. The `max_dense_bytes` argument prevents accidental
dense SVD allocation on large inputs; raise it explicitly when appropriate.
