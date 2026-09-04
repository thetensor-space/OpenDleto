
#
# Strata Dleto: Derivation Method SylverLining
#   Algorithms for solving Sylvester equations arising in chiseling.
# -----------------------------------------------------------------------------
# Copyright 2022-2026 Peter A. Brooksbank, Martin D. Kassabov, James B. Wilson
# 
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the “Software”), 
# to deal in the Software without restriction, including without limitation the 
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or 
# sell copies of the Software, and to permit persons to whom the Software is 
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in 
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE 
# SOFTWARE.
# -----------------------------------------------------------------------------

import LinearMaps
using ITensors
using SparseArrays



"""
    Sylver Lininig Derivation Method

    Derivation Method implemented via linear maps. 
"""
struct SylverLiningMethod <: DerivationMethod
    solver::Symbol
    # Which `sylvesterLM` kernel the solve applies; see `sylvesterLM`.  It has
    # to live on the method and not only on `derTrOpsReduced`'s keyword,
    # because the `der(:SylverLining, Γ; kwargs...)` family forwards its
    # `kwargs` to the *method constructor*, not to the solve (see the NOTE in
    # src/Derivations.jl) -- so `der(:SylverLining, Γ; backend = :metal)` can
    # only reach the kernel through here.
    backend::Symbol
end;
# One-positional-argument form kept: it predates `backend` and is what
# `SylverLiningMethod(:SVDSolver)` still means.
SylverLiningMethod(solver::Symbol) = SylverLiningMethod(solver, :auto);
# The default solver is `:AutoSolver`, not `:SVDSolver`: it densifies when that
# is cheap (which it is for this map -- 9MB at n = 19) and stays matrix-free
# when it is not, instead of calling `Matrix` unconditionally.
SylverLiningMethod(; solver::Symbol=:AutoSolver, backend::Symbol=:auto) =
    SylverLiningMethod(solver, backend);


function derTrOpsReduced(method::SylverLiningMethod,
    Ω::TransverseOps, 
    P::AbstractMatrix, 
    Γ::ITensor;
    tol::Real=TOL_DEFAULT,
    nd=-1,  # Don't type as integer to allow Inf
    progress=false,
    # Which sylvester kernel to build; see `sylvesterLM`.  Defaults to whatever
    # the method was constructed with (`:auto`, i.e. the CPU array kernel,
    # unless the caller asked for another), and can be overridden per call so a
    # benchmark or a bisecting caller can pin a kernel without editing source.
    backend::Symbol=method.backend,
    # The same opt-in diagnostics QuickDer takes: `true` appends a
    # `DerivationReport`.  SylverLining fills what it has -- the derivation
    # operator's own `NullVerdict`, the two element types, the solver, the
    # Z-law residual of each returned direction -- and leaves the
    # restriction/lift fields `nothing`, because it has no sketch and no lift.
    # See `DerivationReport`; the API is uniform so that switching method does
    # not switch the code that reads the answer.
    return_diagnostics::Bool=false,
    kwargs...,
    )
    # No return-type annotation: the return is a 3-tuple, or a 4-tuple whose
    # last element is a `DerivationReport`, and that type is defined in a file
    # included after this one (annotations are evaluated at definition time,
    # bodies are not).  The 3-tuple is unchanged.
    Γ_frame = inds(Γ)
    val = ndims(Γ)
    @assert Γ_frame == frames(Ω) "Incompatable Indexes"
    @assert val == size(P, 2) "Incompatable Chisel"
    
    # STORAGE type vs COMPUTE type.  `Tc` is what the arithmetic runs in and
    # `T` is what the caller handed in; they differ only for Float16, which has
    # no BLAS or LAPACK anywhere and promotes to Float32 (see
    # `Dleto.compute_eltype`).  Without that promotion every route here is
    # confidently wrong in Float16: measured on the sphere at d = 10, true
    # nullity 3, the dense SVD reported 40 derivations and the Gram solver 41,
    # both at a reconstruction error of 0.9, because the half-precision noise
    # floor `eps16*‖L‖ ~ 0.09` relative is an order of magnitude ABOVE the
    # operator's first nonzero eigenvalue.  The answer is rounded back to `T`
    # at the end, so a Float16 tensor still yields Float16 coordinates -- the
    # data's precision is preserved, not inflated.
    T = eltype(Γ)
    Tc = compute_eltype(T)
    Γc = T === Tc ? Γ :
         ITensor(Array{Tc}(ITensors.array(Γ, Γ_frame...)), Γ_frame...)
    # Compute reduced operators (matching what sylvesterLM does internally)
    eng = engaged(P)
    (Ω_reduced,expand_map) = reduceByEngaged(Ω, eng, Tc)
    # Chisels default to Float64; do not let that promote a Float32 tensor.
    P_eng = Matrix{Tc}(P[:,eng])

    # if globalDim(reducedΩ) < 10000
    # MDK, we need to reduce the chisel and pass the reduced chisel to the helper function
    sylvester, ester_map = sylvesterLM(Ω_reduced, P_eng, Γc; backend=backend)
    
    # All of the null-solver policy -- when densifying is cheap enough, how
    # many vectors to ask an iterative method for, how to grow that request
    # until the null space is bracketed, which tolerance decides "null" --
    # lives in `solve_nullspace` (src/solvers/NullSolvers.jl) and is tuned
    # there once.  This function used to carry its own copy of all of it: a
    # hand-rolled `globalDim < 1000` dense gate, a dense `eigen` branch that
    # bypassed the solver interface entirely, `nv = globalDim(Ω)` asking an
    # iterative solver for the whole spectrum, and its own filter and
    # truncation.  `den` carried a second, slightly different copy.  Tuning
    # either one left the other stale.
    #
    # An empty result is a mathematical fact, not a solver failure: it says the
    # reduced derivation space is trivial, i.e. Γ conforms to no sparsity
    # pattern for this chisel.  A Tucker chisel on a generic tensor is the
    # standard example -- it forces each engaged D_a into the a-th radical,
    # which is zero, and its only scalar derivation lives on the disengaged
    # axis that the engagement reduction drops.  This used to assert
    # "Not enough eigenvalues computed; increase `tol` parameter", which
    # misreported the fact as a convergence problem.  Callers that need a
    # derivation (`stratify`) report it themselves.
    # Hand over the SQUARE map `sylvester = sylve∘ester`, not the rectangular
    # `ester_map`.  Both have the same null space -- a derivation is exactly an
    # `X` with `ester(X) = 0` -- but the rectangular one is `m·d^n × globalDim`
    # (160000 x 840 at d = 20, valence 4), so the dense gate sees a 1 GB matrix
    # and the SVD of it blows the memory budget, while the Gram operator is
    # 840 x 840.  `AutoSolver`'s solver ordering is tuned on the square shape too.
    #
    # `squared=true` is what keeps the accuracy the rectangular form bought:
    # the eigenvalues of `AᵗA` are `σ²`, so `solve_nullspace` squares the
    # relative ceiling and an accepted direction still satisfies `σ/σ_max ≤ tol`
    # rather than `√tol`.
    #
    # `store_eltype = T` is what keeps a promoted run honest: the nullity cut is
    # made at the FLoat32 arithmetic's floor, but certification additionally
    # requires the first value above the cut to clear the resolution of the
    # stored data (`eps(Float16)` = 9.8e-4 relative).  A near-derivation under
    # that is reported as undecidable rather than counted or discarded silently.
    (λ, vecs, verdict) = solve_nullspace(sylvester, method.solver; tol=tol, nd=nd,
                                         squared=true, store_eltype=real(T),
                                         progress=progress, label="der")

    coords = size(vecs, 2) == 0 ? zeros(T, globalDim(Ω_reduced), 0) : Matrix{T}(vecs)
    return_diagnostics || return (Ω_reduced, expand_map, coords)

    # The Z-law residuals are measured on the PROMOTED tensor, in `Tc`: a
    # Float16 answer checked in Float16 measures its own storage rounding
    # rather than the equation.  One pass over the tensor per direction, which
    # is why this is behind the keyword.
    Gc = Array{Tc}(ITensors.array(Γc, Γ_frame...))
    rep = DerivationReport(; method = :SylverLining, store_eltype = T,
                           compute_eltype = Tc, dims = collect(size(Gc)),
                           verdict = verdict,
                           requested_nd = nd isa Integer ? Int(nd) : -1,
                           policy = (nd isa Integer && nd > 0) ? :fixed_nd : :auto,
                           returned = size(coords, 2),
                           scalar_dim = _der_scalar_dim(P),
                           solver = method.solver,
                           residuals = _der_zlaw_residuals(Ω, expand_map, coords, Gc, P))
    return (Ω_reduced, expand_map, coords, rep)
end



"""
    sylvesterLM_itensor(Ω::TransverseOps, P::AbstractMatrix, Γ::ITensor)

    The reference implementation of `sylvesterLM`, written as ITensor
    contractions.  Kept as the oracle the array kernel is checked against (see
    test/TestSylverLining.jl) and as the `backend = :itensor` escape hatch: it
    is the shortest statement of what the maps mean, and it is the thing to
    read if the array kernel ever disagrees with the transpose law.
"""
function sylvesterLM_itensor(Ω::TransverseOps, P::AbstractMatrix, Γ::ITensor) #::Tuple{LinearMaps.LinearMap, LinearMaps.LinearMap}
#temporary the function retunrs also the naked functions in addition to the linear maps
#this is done for only for testing but needs to be fixed during the merge.
#I am gettign stupid error is ch is and empty matrix!!!
#both Ω and Ch are assumed to be reduced!

    Γ_frame = inds(Γ)
    val = ndims(Γ)
    engsize = valency(Ω)
    @assert engsize == size(P, 2) "Incompatable Chisel"
    T = eltype(Γ)
    P_typed = Matrix{T}(P)
    ch_axis = Index(size(P, 1), "chisel")
    eng_axis = Index(engsize, "engaged")
    Cs = [ ITensor(P_typed[:,a], ch_axis) for a in 1:engsize ]

    Γ_frame_ch = (ch_axis, inds(Γ)...)

    Ωframe=frames(Ω)
    ΩframeTemp=framesTemporary(Ω)
    Γs_relabled =  [ replaceind(Γ, Ωframe[a], ΩframeTemp[a]) for a in 1:engsize]

    # Compute sizes for LinearMap
    densor_dim = prod([ITensors.dim(f) for f in Γ_frame_ch ])
    op_dim = globalDim(Ω)

    # Takes a vectorized representation of derivations
    # Returns a vectorized representation of the tensor
    function ester(Xvec)
        Xs = unsafe_embedITensorsSwapped(Ω, Xvec)
        Σ = [ Cs[a]*Xs[a]*Γs_relabled[a] for a in 1:engsize ] |> sum 
        return vec(Array(Σ, Γ_frame_ch...))
    end
    
    # sylv: takes a vectorized tensor, returns a vector of matrices (one per axis)
    function sylve(y)
        Σ = ITensor(y, Γ_frame_ch...)
        Ys = [ Γ * replaceind!(Cs[a]*Σ , Ωframe[a], ΩframeTemp[a]) for a in 1:engsize] 
        return unsafe_transposeEmbed(Ω,Ys)
    end

    # Compose sylv and ester as in sylvester4
    function sylvester(Xvec)
        Xs = unsafe_embedITensorsSwapped(Ω, Xvec)
        Σ = [ Cs[a]*Xs[a]*Γs_relabled[a] for a in 1:engsize ] |> sum 
        Ys = [ Γ * replaceind!(Cs[a]*Σ , Ωframe[a], ΩframeTemp[a]) for a in 1:engsize] 
        return unsafe_transposeEmbed(Ω,Ys)
    end

    # Wrap ester and sylve as LinearMaps
    densor_map = LinearMaps.LinearMap{T}(ester, sylve, densor_dim, op_dim; ismutating=false)
    derdensor_map = LinearMaps.LinearMap{T}(sylvester, sylvester, op_dim, op_dim; ismutating=false, issymmetric=true, isposdef=false)
    return derdensor_map, densor_map
end;


# =============================================================================
# The array kernel
# =============================================================================
#
# WHY a second implementation exists.  `sylvesterLM_itensor` above is the
# algebra written the way it is stated; it is also why the eigensolver is
# slower than the arithmetic says it should be.  ITensor's `*` already reduces
# to permute + GEMM internally, but it offers no in-place entry point: every
# apply allocates a fresh tensor per contraction per engaged axis, plus the
# operator ITensors, plus the `Array`/`vec` at the boundary.  Measured at
# valence 3, universal chisel and operators: 4.4 MB per `ester` at d = 40 and
# 33 MB at d = 80 -- and a solve is several hundred applies.  That is GC
# pressure bolted onto arithmetic already sitting at the BLAS floor.
#
# The kernel below runs the same algebra over plain `Array`s with preallocated
# scratch and `ismutating = true` LinearMaps, so a warmed-up apply allocates
# only the few bytes `mul!` needs for its own wrappers.
#
# THE ALGEBRA, written out once so the code can stay terse.  Let `Γ` have axes
# `1..n` with dims `dims` and `N = prod(dims)`; let transverse operator `a` act
# on tensor axis `p = axpos[a]` -- NOT necessarily `a`, because `Ω` may be the
# engagement-reduced operator set whose frames are a subset of `Γ`'s -- and let
# `M_a` be its matrix.  The orientation is the one thing here that is easy to
# get backwards: `unsafe_embedITensorsSwapped` builds
# `ITensor(M_a, framesTemp[a], frames[a])`, so contracting it against `Γ` sums
# over the FIRST matrix index.  The mode product is therefore `Γ ×_p M_aᵗ`, not
# `Γ ×_p M_a`.  With `P` the chisel matrix (`m` rows),
#
#     ester:  R[c, i_1..i_n] = Σ_a P[c,a] · (Γ ×_p M_aᵗ)[i_1..i_n]
#     sylve:  Y_a[f, t]      = Σ_c P[c,a] Σ_{i_k, k ≠ p} Γ[..f..] R[c, ..t..]
#
# and `sylve` finishes by pushing each `Y_a` through `unsafe_transposeEmbed`,
# which is by definition the adjoint of the embedding -- that is what makes the
# pair a genuine transpose pair, which test/TestSylverLining.jl pins.  In the
# mode-`p` unfolding `U_a` -- the axes other than `p` column-major in the rows,
# `p` in the columns, so `U_a` is `(N/d_p) × d_p` -- each line is one GEMM:
#
#     T_a = U_a · M_a                             ((N/d_p) × d_p)
#     Y_a = U_aᵗ · (Z_a)_(p) ,     Z_a[i] = Σ_c P[c,a] R[c,i]
#
# `R` keeps the chisel axis FIRST and column-major, which is exactly what
# `vec(Array(Σ, (ch_axis, frame...)))` produced, so the two backends agree
# vector-for-vector and not merely up to a reshape.
#
# THREADING.  The scratch is shared by the two maps returned from one call, so
# one apply may be in flight at a time.  Every solver we drive (Arpack,
# KrylovKit, IterativeSolvers) applies the operator serially; this is the
# reason these maps must not be handed to a threaded matrix-vector loop.

"""
    SYLVER_SPARSE_DENSITY

Density at or below which the array kernel unfolds `Γ` into `SparseMatrixCSC`
instead of dense matrices, making the mode product cost `O(nnz · d)` instead of
`O(d^(n+1))`.

The inputs that motivate the branch are sphere octants, whose support
`i_1 + ... + i_n = d - 1` is about `d^(n-1)/(n-1)!` of `d^n` entries: 1.3% at
valence 3, d = 40 and 0.06% at valence 4, d = 24.  So the threshold only has to
sit below "somewhat sparse".  5% is comfortably above those densities and far
enough below 1 that a dense tensor pays nothing but the one `count` scan.
"""
const SYLVER_SPARSE_DENSITY = 0.05

# --- in-place embedding of the operator coordinates --------------------------
#
# `unsafe_embed`/`unsafe_transposeEmbed` in ops/OperatorImpls.jl allocate a
# fresh matrix (or a fresh `vcat` of per-axis blocks) on every call, which is
# precisely what this kernel exists to avoid.  These write into caller-owned
# buffers instead.  Each method must stay bit-identical to its allocating twin
# -- the `backend = :itensor` cross-check in test/TestSylverLining.jl is what
# pins that -- and every unrecognised `Operator` falls back to the allocating
# twin, so adding an operator type stays correct by default and only costs
# speed until someone writes the in-place method.

function _embed_op!(M::AbstractMatrix, op::Operator, d::Integer,
                    x::AbstractVector, off::Integer)
    copyto!(M, unsafe_embed(op, d, view(x, (off + 1):(off + localDim(op, d)))))
    return M
end

function _embed_op!(M::AbstractMatrix, ::UniversalOp, d::Integer,
                    x::AbstractVector, off::Integer)
    @inbounds for j in 1:d, i in 1:d
        M[i, j] = x[off + (j - 1) * d + i]
    end
    return M
end

function _embed_op!(M::AbstractMatrix, ::DiagonalOp, d::Integer,
                    x::AbstractVector, off::Integer)
    fill!(M, zero(eltype(M)))
    @inbounds for i in 1:d
        M[i, i] = x[off + i]
    end
    return M
end

function _embed_op!(M::AbstractMatrix, ::SymmetricOp, d::Integer,
                    x::AbstractVector, off::Integer)
    k = off
    @inbounds for j in 1:d
        for i in 1:j
            v = x[k + i]
            M[i, j] = v
            M[j, i] = v
        end
        k += j
    end
    return M
end

function _embed_op!(M::AbstractMatrix, ::AntiSymmetricOp, d::Integer,
                    x::AbstractVector, off::Integer)
    k = off
    @inbounds for j in 1:d
        for i in 1:(j - 1)
            v = x[k + i]
            M[i, j] = v
            M[j, i] = -v
        end
        M[j, j] = zero(eltype(M))
        k += j - 1
    end
    return M
end

function _embed_op!(M::AbstractMatrix, ::ScalarOp, d::Integer,
                    x::AbstractVector, off::Integer)
    fill!(M, zero(eltype(M)))
    @inbounds for i in 1:d
        M[i, i] = x[off + 1]
    end
    return M
end

_embed_op!(M::AbstractMatrix, ::EmptyOp, d::Integer,
           x::AbstractVector, off::Integer) = fill!(M, zero(eltype(M)))

# The transpose-embeddings ACCUMULATE into `x`.  Independent axes each write a
# disjoint block, where accumulation is the same as assignment, but a symmetry
# block has several axes sharing one block of coordinates and the allocating
# twin sums them (the `... |> sum` in TransverseOpsSymmetries).
# `_transverse_embed_all!` zeroes `x` first, so both cases come out right.

function _tembed_op!(x::AbstractVector, off::Integer, op::Operator,
                     M::AbstractMatrix, d::Integer)
    v = unsafe_transposeEmbed(op, M)
    @inbounds for k in eachindex(v)
        x[off + k] += v[k]
    end
    return x
end

function _tembed_op!(x::AbstractVector, off::Integer, ::UniversalOp,
                     M::AbstractMatrix, d::Integer)
    @inbounds for j in 1:d, i in 1:d
        x[off + (j - 1) * d + i] += M[i, j]
    end
    return x
end

function _tembed_op!(x::AbstractVector, off::Integer, ::DiagonalOp,
                     M::AbstractMatrix, d::Integer)
    @inbounds for i in 1:d
        x[off + i] += M[i, i]
    end
    return x
end

function _tembed_op!(x::AbstractVector, off::Integer, ::SymmetricOp,
                     M::AbstractMatrix, d::Integer)
    k = off
    @inbounds for j in 1:d
        for i in 1:(j - 1)
            x[k + i] += M[i, j] + M[j, i]
        end
        x[k + j] += M[j, j]
        k += j
    end
    return x
end

function _tembed_op!(x::AbstractVector, off::Integer, ::AntiSymmetricOp,
                     M::AbstractMatrix, d::Integer)
    k = off
    @inbounds for j in 1:d
        for i in 1:(j - 1)
            x[k + i] += M[i, j] - M[j, i]
        end
        k += j - 1
    end
    return x
end

function _tembed_op!(x::AbstractVector, off::Integer, ::ScalarOp,
                     M::AbstractMatrix, d::Integer)
    s = zero(eltype(M))
    @inbounds for i in 1:d
        s += M[i, i]
    end
    x[off + 1] += s
    return x
end

_tembed_op!(x::AbstractVector, off::Integer, ::EmptyOp,
            M::AbstractMatrix, d::Integer) = x

# Square in-place transpose, for the dualised axes of a symmetry block.
function _transpose_square!(M::AbstractMatrix)
    n = size(M, 1)
    @inbounds for j in 1:n, i in 1:(j - 1)
        M[i, j], M[j, i] = M[j, i], M[i, j]
    end
    return M
end

# --- the same, one whole transverse operator set at a time -------------------

function _embed_all!(Ms::Vector{<:AbstractMatrix}, Ω::IndTransverseOps,
                     x::AbstractVector)
    @inbounds for a in 1:Ω.val
        _embed_op!(Ms[a], Ω.localOps[a], Ω.axisDims[a], x, Ω.soffsets[a] - 1)
    end
    return Ms
end

function _embed_all!(Ms::Vector{<:AbstractMatrix}, Ω::TransverseOpsSymmetries,
                     x::AbstractVector)
    @inbounds for a in 1:Ω.val
        # `soffsets[a]` already points at the block representative's slot, so
        # the axes of one symmetry block all read the same coordinates.
        _embed_op!(Ms[a], Ω.localOps[a], Ω.axisDims[a], x, Ω.soffsets[a] - 1)
        Ω.duals[a] && _transpose_square!(Ms[a])
    end
    return Ms
end

# Fallback for a TransverseOps someone adds later: correct, and allocating.
function _embed_all!(Ms::Vector{<:AbstractMatrix}, Ω::TransverseOps,
                     x::AbstractVector)
    Mats = unsafe_embedMatrices(Ω, x)
    for a in eachindex(Ms)
        copyto!(Ms[a], Mats[a])
    end
    return Ms
end

function _transverse_embed_all!(x::AbstractVector, Ω::IndTransverseOps,
                                Ys::Vector{<:AbstractMatrix})
    fill!(x, zero(eltype(x)))
    @inbounds for a in 1:Ω.val
        _tembed_op!(x, Ω.soffsets[a] - 1, Ω.localOps[a], Ys[a], Ω.axisDims[a])
    end
    return x
end

function _transverse_embed_all!(x::AbstractVector, Ω::TransverseOpsSymmetries,
                                Ys::Vector{<:AbstractMatrix})
    fill!(x, zero(eltype(x)))
    @inbounds for a in 1:Ω.val
        M = Ω.duals[a] ? transpose(Ys[a]) : Ys[a]
        _tembed_op!(x, Ω.soffsets[a] - 1, Ω.localOps[a], M, Ω.axisDims[a])
    end
    return x
end

function _transverse_embed_all!(x::AbstractVector, Ω::TransverseOps,
                                Ys::Vector{<:AbstractMatrix})
    copyto!(x, unsafe_transposeEmbed(Ω, [Ys[a] for a in 1:valency(Ω)]))
    return x
end

# --- the kernel --------------------------------------------------------------

"""
    _sylver_plan(Ω, P, Γ)

Everything both the CPU and the device kernel need to know about the layout,
computed once on the host: the chisel matrix in `Γ`'s element type, the mode-`p`
unfolding shapes and permutations, which tensor axis each transverse operator
acts on, and `Γ` as a plain array.

Returned as a NamedTuple with fields

    T engsize n m dims N axpos perms iperms da Ka flat Pm GA op_dim densor_dim

so that the layout conventions documented in the block comment above have ONE
statement in the source rather than one per backend.
"""
function _sylver_plan(Ω::TransverseOps, P::AbstractMatrix, Γ::ITensor)
    Γ_frame = inds(Γ)
    engsize = valency(Ω)
    @assert engsize == size(P, 2) "Incompatable Chisel"

    T = eltype(Γ)
    # Chisels default to Float64; do not let that promote a Float32 tensor.
    Pm = Matrix{T}(P)
    m = size(Pm, 1)
    Ωframe = frames(Ω)
    n = length(Γ_frame)
    dims = ntuple(k -> ITensors.dim(Γ_frame[k]), n)
    N = prod(dims)

    # Which tensor axis each transverse operator acts on.  `Ω` is often the
    # engagement-reduced set, so operator `a` is not in general axis `a`; the
    # ITensor backend gets this for free from `replaceind`, we look it up once.
    axpos = [findfirst(isequal(Ωframe[a]), Γ_frame) for a in 1:engsize]
    @assert !any(isnothing, axpos) "Ω has a frame that is not an index of Γ"

    GA = Array(Γ, Γ_frame...)

    # Mode-`p` unfolding permutations: the axes OTHER than `p` keep their order
    # and stay column-major in the ROWS, and `p` becomes the column, giving a
    # `(N/d_p) × d_p` matrix `U_a`.  The other obvious convention -- axis `p`
    # first -- is worse on both counts that matter.  It writes the mode product
    # as `Mᵗ · Γ_(p)`, and for a sparse Γ that is dense × sparse, the one CSC
    # product Julia has no good kernel for: measured at valence 3, d = 80,
    # sphere-octant density, 0.49 ms against 0.14 ms for `Γ_(p) · M` this way
    # round, and 0.47 ms against 0.12 ms on `sylve`'s side.  It also puts the
    # permutation-free axis at `p = 1` rather than `p = n`, and `p = n` is the
    # one that also lets the residual be written in place (see `flat` below).
    # Dense GEMM is indifferent (0.82 ms either way round), and so is Metal's.
    perms = [NTuple{n,Int}(((k for k in 1:n if k != axpos[a])..., axpos[a]))
             for a in 1:engsize]
    iperms = [NTuple{n,Int}(invperm(collect(perms[a]))) for a in 1:engsize]
    da = [dims[axpos[a]] for a in 1:engsize]
    Ka = [N ÷ da[a] for a in 1:engsize]
    # For the LAST axis that permutation is the identity, so the unfolding is a
    # reshape of Γ itself and both closures skip the O(d^n) permute entirely.
    flat = [axpos[a] == n for a in 1:engsize]

    return (T=T, engsize=engsize, n=n, m=m, dims=dims, N=N, axpos=axpos,
            perms=perms, iperms=iperms, da=da, Ka=Ka, flat=flat, Pm=Pm, GA=GA,
            op_dim=globalDim(Ω), densor_dim=m * N)
end

"""
    _sylvesterLM_array(Ω, P, Γ; allow_sparse = true)

`sylvesterLM` over plain arrays; see the block comment above for the algebra
and the layout conventions.  `allow_sparse = false` forces the dense unfoldings
even for a sparse `Γ` -- that is what `backend = :array` selects, and it is how
the two branches get compared against each other.
"""
function _sylvesterLM_array(Ω::TransverseOps, P::AbstractMatrix, Γ::ITensor;
                            allow_sparse::Bool = true)
    pl = _sylver_plan(Ω, P, Γ)
    T, engsize, n, m = pl.T, pl.engsize, pl.n, pl.m
    dims, N, Pm, GA = pl.dims, pl.N, pl.Pm, pl.GA
    perms, iperms, da, Ka, flat = pl.perms, pl.iperms, pl.da, pl.Ka, pl.flat

    sparse_Γ = allow_sparse && N > 0 &&
               count(!iszero, GA) / N <= SYLVER_SPARSE_DENSITY

    # The unfoldings are the only copies of Γ we keep: `engsize - 1` of them
    # dense (the flat axis reshapes `GA` in place), or `engsize · nnz` sparse.
    Ud = Vector{Matrix{T}}(undef, sparse_Γ ? 0 : engsize)
    Us = Vector{SparseMatrixCSC{T,Int}}(undef, sparse_Γ ? engsize : 0)
    for a in 1:engsize
        # `permutedims`, not `permutedims!` into shared scratch: the permuted
        # shape is `dims[perms[a]]`, which differs per axis once the frame is
        # ragged.  This runs once per construction, so the copy is free.
        U = flat[a] ? reshape(GA, Ka[a], da[a]) :
                      reshape(permutedims(GA, perms[a]), Ka[a], da[a])
        if sparse_Γ
            Us[a] = sparse(U)
        else
            Ud[a] = U
        end
    end

    # Scratch.  `Wkd[a]` is one term's mode-`p` unfolding; it stays DENSE even
    # in the sparse branch, because it is the mode product's dense output on one
    # side and the residual on the other, and the residual is O(d^n) whatever Γ
    # looks like -- there is nothing to save by making it sparse.  `Wten[a]` is
    # that same memory seen as an n-array, so folding back to natural axis order
    # is one `permutedims!` and no copy.  A flat axis needs neither.
    empty_mat() = Matrix{T}(undef, 0, 0)
    empty_ten() = Array{T,n}(undef, ntuple(_ -> 0, n))
    Wkd = [flat[a] ? empty_mat() : Matrix{T}(undef, Ka[a], da[a])
           for a in 1:engsize]
    Wten = [flat[a] ? empty_ten() :
            reshape(Wkd[a], ntuple(k -> dims[perms[a][k]], n)) for a in 1:engsize]
    Ms = [Matrix{T}(undef, da[a], da[a]) for a in 1:engsize]
    Ys = [Matrix{T}(undef, da[a], da[a]) for a in 1:engsize]
    Pcol = [Pm[:, a] for a in 1:engsize]
    Fold = Array{T,n}(undef, dims)          # one `ester` term, natural order
    Foldv = reshape(Fold, N)
    Foldkd = [reshape(Fold, Ka[a], da[a]) for a in 1:engsize]
    R = Matrix{T}(undef, m, N)              # the residual, chisel axis first

    # When the chisel has ONE row -- the universal and adjoint chisels, i.e.
    # every derivation solve `der`/`stratify` actually runs -- the residual is
    # its own chisel-collapsed self up to the scalar `P[1,a]`, and that scalar
    # rides along as the GEMM's `α`.  So `sylve` needs no gemv and `ester` on
    # the flat axis writes straight into `R`.  `m > 1` (Tucker, centroid) pays
    # for the collapse into `Zt` and the scatter out of `Fold`.
    Rnat = m == 1 ? reshape(R, dims) : empty_ten()
    Rkd = [m == 1 ? reshape(R, Ka[a], da[a]) : empty_mat() for a in 1:engsize]
    Zt = Array{T,n}(undef, m == 1 ? ntuple(_ -> 0, n) : dims)
    Ztv = reshape(Zt, length(Zt))
    Znat = m == 1 ? Rnat : Zt               # `sylve`'s chisel-collapsed residual
    Znatkd = [reshape(Znat, Ka[a], da[a]) for a in 1:engsize]

    # ester: R <- Σ_a P[:,a] ⊗ vec(Γ ×_p M_aᵗ), the mode product being the one
    # GEMM `U_a · M_a` in the unfolding above.
    function ester_core!(x::AbstractVector)
        _embed_all!(Ms, Ω, x)
        fill!(R, zero(T))
        for a in 1:engsize
            pa = Pcol[a]
            if flat[a] && m == 1
                # No scratch, no permute, no scatter: the term accumulates on
                # top of the residual in place.
                if sparse_Γ
                    LinearAlgebra.mul!(Rkd[a], Us[a], Ms[a], pa[1], one(T))
                else
                    LinearAlgebra.mul!(Rkd[a], Ud[a], Ms[a], pa[1], one(T))
                end
            else
                dest = flat[a] ? Foldkd[a] : Wkd[a]
                if sparse_Γ
                    LinearAlgebra.mul!(dest, Us[a], Ms[a], one(T), zero(T))
                else
                    LinearAlgebra.mul!(dest, Ud[a], Ms[a], one(T), zero(T))
                end
                flat[a] || permutedims!(Fold, Wten[a], iperms[a])
                @inbounds for i in 1:N
                    t = Foldv[i]
                    for c in 1:m
                        R[c, i] += pa[c] * t
                    end
                end
            end
        end
        return R
    end

    # sylve: Y_a <- U_aᵗ · (Σ_c P[c,a] R[c,·])_(p), then back through the
    # adjoint of the embedding.
    function sylve_core!(x::AbstractVector)
        for a in 1:engsize
            pa = Pcol[a]
            α = one(T)
            if m == 1
                α = pa[1]
            else
                LinearAlgebra.mul!(Ztv, transpose(R), pa)
            end
            flat[a] || permutedims!(Wten[a], Znat, perms[a])
            Zu = flat[a] ? Znatkd[a] : Wkd[a]
            if sparse_Γ
                LinearAlgebra.mul!(Ys[a], transpose(Us[a]), Zu, α, zero(T))
            else
                LinearAlgebra.mul!(Ys[a], transpose(Ud[a]), Zu, α, zero(T))
            end
        end
        _transverse_embed_all!(x, Ω, Ys)
        return x
    end

    # The residual is copied in and out of `R` rather than reshaping the
    # caller's vector: LinearMaps hands us whatever the solver holds (often a
    # view), and a reshaped view is not a `StridedMatrix`, so BLAS would fall
    # off its fast path.  The copy is O(m·d^n), the same order as the chisel
    # accumulation it sits beside.
    ester!(y, x) = (ester_core!(x); copyto!(y, R); y)
    sylve!(x, y) = (copyto!(R, y); sylve_core!(x); x)
    sylvester!(z, x) = (ester_core!(x); sylve_core!(z); z)

    densor_dim = m * N
    op_dim = globalDim(Ω)
    densor_map = LinearMaps.LinearMap{T}(ester!, sylve!, densor_dim, op_dim;
                                         ismutating=true)
    derdensor_map = LinearMaps.LinearMap{T}(sylvester!, sylvester!, op_dim, op_dim;
                                            ismutating=true, issymmetric=true,
                                            isposdef=false)
    return derdensor_map, densor_map
end


"""
    sylvesterLM(Ω::TransverseOps, P::AbstractMatrix, Γ::ITensor; backend = :auto)

    Constructs LinearMaps for the derivation and densor maps associated to the given chisel `P` and tensor `Γ`.

    - `Ω`: the transverse operator space the derivation is sought in.
    - `P`: A matrix whose columns define the chisel polynomials.
    - `Γ`: The input tensor.

    Returns a tuple `(derdensor_map, densormap)` where:
    - `derdensor_map`: the composed derivation-densor `LinearMap` (a real symmetric operator).
    - `densormap`: the densor operator with transpose---the derivation operator---included.

    `backend` picks the kernel:
    - `:auto` (default) -- the array kernel, with sparse unfoldings when `Γ` is
      at or below `SYLVER_SPARSE_DENSITY` and dense ones otherwise.  Stays on
      the CPU: a GPU is opt-in, never picked for you.
    - `:array` -- the array kernel with dense unfoldings, whatever `Γ` looks like.
    - `:metal` -- the dense array kernel with the unfoldings and the residual
      resident on an Apple GPU (`ext/DletoMetalSylver.jl`, needs `using Metal`).
      Float32 arithmetic, CPU-facing vectors; dense only.
    - `:itensor` -- the ITensor reference implementation, `sylvesterLM_itensor`.

    All of them produce the same vectors (to fp32 rounding for `:metal`); the
    array kernel exists only because it does so without allocating per apply.
    Both `Ω` and `P` are assumed reduced.
"""
function sylvesterLM(Ω::TransverseOps, P::AbstractMatrix, Γ::ITensor;
                     backend::Symbol = :auto)
    backend === :itensor && return sylvesterLM_itensor(Ω, P, Γ)
    backend === :auto && return _sylvesterLM_array(Ω, P, Γ; allow_sparse=true)
    backend === :array && return _sylvesterLM_array(Ω, P, Γ; allow_sparse=false)
    backend === :metal && return _sylvesterLM_metal(Ω, P, Γ)
    error("sylvesterLM: unknown backend $backend " *
          "(expected :auto, :array, :metal or :itensor)")
end

"""
    _sylvesterLM_device(::Val{device}, Ω, P, Γ)

The device-kernel entry point, keyed by a device token so that a GPU extension
can ADD a method instead of replacing one.

This file defines only the `::Val` fallback, which errors.  The Apple-GPU
kernel is `_sylvesterLM_device(::Val{:metal}, ...)` in
`ext/DletoMetalSylver.jl`, loaded with `Metal`; nothing device-specific lives
here beyond the name of the token.

THE TOKEN IS NOT DECORATION.  An extension may only add methods with new
signatures (or set a `Ref`) -- redefining a signature the core package already
defines is method overwriting, which Julia 1.12 refuses during precompilation,
so the whole extension silently falls back to being recompiled on every
`using Metal`.  `gpu_available`/`gpu_sync` in src/solvers/NullSolvers.jl solve
the same problem with `Ref`s; dispatch does it here, because the kernel needs
`Γ`'s types anyway.

Errors rather than silently falling back to the CPU: `:metal` is an explicit
request, and a caller benchmarking a GPU wants to be told it never ran on one.
`:auto` is the backend that is allowed to choose.
"""
function _sylvesterLM_device(::Val{device}, Ω::TransverseOps, P::AbstractMatrix,
                             Γ::ITensor) where {device}
    error("sylvesterLM: backend = :$device needs its GPU extension loaded. " *
          (gpu_available() ?
           "A GPU is available but no `sylvesterLM` device kernel is " *
           "registered for :$device -- is ext/DletoMetalSylver.jl present?" :
           "Add `using Metal` (Apple Silicon only) before building the map; " *
           "`gpu_available()` is currently false."))
end

_sylvesterLM_metal(Ω::TransverseOps, P::AbstractMatrix, Γ::ITensor) =
    _sylvesterLM_device(Val(:metal), Ω, P, Γ)
