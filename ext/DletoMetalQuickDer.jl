#
# DletoMetalQuickDer -- Apple-GPU support for the QuickDer kernel.
#
# Included by ext/DletoMetalExt.jl, which owns the device hooks
# (`gpu_available`/`to_gpu`/`to_cpu`/`gpu_sync`).  Scope: whatever
# `src/solvers/QuickDerN.jl` and the `GramSolver` device path
# (`src/solvers/NullSolvers.jl`) need from `MtlArray` beyond what Metal.jl and
# GPUArrays already provide.
#
# THE HEADLINE IS HOW LITTLE IS HERE, and that is a finding, not an omission.
# `QuickDerN.jl` routes every contraction through three functions --
# `_qdn_ttm` (permutedims + reshape + `mul!`), `_qdn_unfold` (permutedims +
# reshape) and `_qdn_slice` (`copy(selectdim(...))`) -- and allocates its
# outputs with `similar(G, ...)` rather than `Matrix{T}(undef, ...)`.  Metal.jl
# 1.10.3 / GPUArrays implement all of that for `MtlArray`, none of it
# scalar-indexes, and none of it silently falls back to the host, so the
# generic code runs on the device unchanged.  Probed on an M4 Max (40 GPU
# cores, 51.8 GB recommended working set); see the `### quickder-metal` entry
# on the night board:
#
#   reshape 3-d -> 2-d ...................... MtlMatrix, no copy
#   permutedims (3-d, 4-d) .................. MtlArray, kernel
#   copy(selectdim(A, a, range)) ............ MtlArray, kernel
#   A[:, 1:3, :] and A[:, [1,3,5], :] ....... MtlArray, kernel
#   mul!(C, transpose(A), B), A' * A ........ MPSMatrixMultiplication
#   broadcast against a Transpose wrapper ... kernel
#   vcat / hcat / kron / diag / norm ........ kernel
#   view(G, diagind(G)) .+= s ............... kernel
#   cholesky(Symmetric(G, :U)) .............. MPSMatrixDecompositionCholesky
#   C \ X, ldiv!, triangular solves ......... MPSMatrixSolveCholesky
#   lu / lu! / \ (square) ................... MPSMatrixDecompositionLU
#
# WHAT IS MISSING FROM Metal.jl 1.10.3, and therefore stays on the host: `qr`
# and `svd`.  `Metal/src/linalg.jl` wraps LU, Cholesky and the triangular
# solves through MPS and stops there.  The generic LinearAlgebra fallback does
# NOT degrade gracefully: it throws `Cannot access the contents of a private
# buffer`, not a scalar-indexing warning, because `MtlArray`'s default storage
# is `PrivateStorage`.  That is the pitfall to remember when reading GPU code
# in this package -- a missing kernel surfaces as a *storage* error, so the fix
# is always "do this piece on the host", never "make the buffer shared".
#
# The three host-resident pieces, and why they are cheap:
#   * the per-axis sketch bases `W_a`, `W_a⊥` -- one `d x d` `qr` per axis,
#     built once in `_qdn_axis` and then uploaded by `_qdn_axes_device`;
#   * `GramSolver`'s subspace re-orthonormalization `qr(X)`, `n x (k+p)`;
#   * `GramSolver`'s Rayleigh--Ritz `svd(M·X)`, `m x (k+p)`.
# With `k + p` a few dozen the last two move well under a megabyte per subspace
# step, against a Gram that is `m·n²` flops.
#

using Metal: MtlArray

# ---------------------------------------------------------------------------
# `to_cpu` of a WRAPPED device array
# ---------------------------------------------------------------------------
#
# DletoMetalExt.jl declares `to_cpu(::MtlArray)`.  Everything that reaches
# `to_cpu` from the QuickDer kernel is a real `MtlArray` today (`reshape` and
# `permutedims` of one both give one back), but a view or a lazy transpose of a
# device array is a `SubArray`/`ReshapedArray`/`Transpose` and would take the
# generic `to_cpu(x) = x` identity -- returning something that is not on the
# host at all, whose next scalar read throws the private-buffer error above
# instead of saying what went wrong.  These four methods close that hole, and
# they are the reason `_qdn_host` and `_gram_host` can be written once against
# `to_cpu` without asking what shape of wrapper they were handed.

Dleto.to_cpu(x::SubArray{<:Any,<:Any,<:MtlArray}) = Array(x)
Dleto.to_cpu(x::Base.ReshapedArray{<:Any,<:Any,<:MtlArray}) = Array(x)
Dleto.to_cpu(x::LinearAlgebra.Transpose{<:Any,<:MtlArray}) = Array(x)
Dleto.to_cpu(x::LinearAlgebra.Adjoint{<:Any,<:MtlArray}) = Array(x)
