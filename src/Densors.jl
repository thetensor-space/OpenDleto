
# ============================================================================
# den -- the T-set (densor)
# ============================================================================
#
# The derivation equation is bilinear in (tensor, operators):
#
#     R(Γ, X) = Σ_a P[:,a] ⊗ (Γ · X_a)
#
# Fixing Γ makes it linear in X and its nullspace is the Z-set (`der`).
# Fixing X makes it linear in Γ and its nullspace is the T-set (`den`).  Same
# equation, different unknown slot -- so `den` reuses the very operator that
# the Z-law tests, `applyDerivation`, which makes the two sides consistent by
# construction rather than by coincidence.
#
# The map is built column by column against the standard basis of the tensor
# space and the nullspace taken by SVD.  That costs prod(dims) contractions
# per element of Δ, which is fine for the sizes in the labs and wants an
# iterative solver later; the rectangular map is used directly (never NᵗN),
# per docs/review/Refactor-Plan.md Phase 3.

"""
    __transposeOps(D::Vector{ITensor}) :: Vector{ITensor}

    Swap the two indices of each operator.  Used to build the adjoint of the
    densor map: the adjoint of `s -> s ·_a D_a` is `r -> r ·_a D_aᵗ`.
"""
__transposeOps(D::Vector{ITensor}) =
    [ replaceinds(X, (inds(X)[1], inds(X)[2]) => (inds(X)[2], inds(X)[1])) for X in D ]

"""
    __denAdjointApply(R::ITensor, Dt::Vector{ITensor}, C::Chisel, fr) :: ITensor

    One block of the adjoint: contract the chisel axis of `R` against column
    `a` of the chisel, apply the transposed operator on axis `a`, and sum.
"""
function __denAdjointApply(R::ITensor, Dt::Vector{ITensor}, C::Chisel, fr)
    acc = nothing
    for a in eachindex(fr)
        T = R * ITensor(C.ch[:, a], C.ch_axis)
        X = Dt[a]
        i1, i2 = inds(X)
        (ci, oi) = haskey(C.idx, i1) ? (i1, i2) : (i2, i1)
        term = replaceind(T * X, oi, ci)
        acc = acc === nothing ? term : acc + term
    end
    return acc
end

"""
    denLM(Ω::TransverseOps, P::AbstractMatrix, Δ::Vector{Vector{ITensor}})
        -> (A, AtA, fr, dims)

    The densor system as **abstract linear maps**, never as a dense matrix, so
    that any `NullSolver` can be applied to it.

    - `A`: the rectangular map from the tensor space to the stacked residuals,
      one block per element of `Δ`.  Its nullspace is the T-set.  Carries a
      genuine adjoint, so `Matrix(A)`, SVD and LSQR-style solvers all work.
    - `AtA`: the composition `Aᵗ∘A` on the tensor space -- square and
      symmetric, which is what the eigen- and Krylov-based solvers need.  This
      is the densor-side analogue of `sylvesterLM`'s `derdensor_map`, and it
      squares the condition number, so prefer `A` where the solver allows it.
    - `fr`, `dims`: the frame and axis dimensions, to read solution vectors
      back as tensors.
"""
function denLM(Ω::TransverseOps, P::AbstractMatrix, Δ::Vector{Vector{ITensor}})
    isempty(Δ) && error("den: need at least one derivation to solve against.")
    fr = collect(frames(Ω))
    C = Chisel(P, fr)
    dims = [ITensors.dim(i) for i in fr]
    N = prod(dims)
    m = size(P, 1)
    blk = m * N                     # residual coordinates per element of Δ
    ch_and_fr = (C.ch_axis, fr...)
    Δt = [ __transposeOps(D) for D in Δ ]

    function forward(svec)
        s = ITensor(reshape(collect(svec), dims...), fr...)
        out = Vector{Float64}(undef, length(Δ) * blk)
        off = 0
        for D in Δ
            R = applyDerivation(s, D, C)
            out[off+1:off+blk] = vec(Array(R, ch_and_fr...))
            off += blk
        end
        return out
    end

    function adjoint(rvec)
        acc = nothing
        off = 0
        for Dt in Δt
            R = ITensor(reshape(collect(rvec[off+1:off+blk]), m, dims...), ch_and_fr...)
            off += blk
            term = __denAdjointApply(R, Dt, C, fr)
            acc = acc === nothing ? term : acc + term
        end
        return vec(Array(acc, fr...))
    end

    A = LinearMaps.LinearMap(forward, adjoint, length(Δ) * blk, N; ismutating=false)
    AtA = LinearMaps.LinearMap(v -> adjoint(forward(v)), v -> adjoint(forward(v)),
                               N, N; ismutating=false, issymmetric=true, isposdef=false)
    return (A, AtA, fr, dims)
end

# Solvers that need a square symmetric operator rather than the rectangular
# map.  SVD- and LU-based solvers densify and handle rectangular input.
__needsSquare(sym::Symbol) = !(sym in (:SVDSolver, :LUSolver))

"""
    den(Ω::TransverseOps, P::AbstractMatrix, Δ::Vector{Vector{ITensor}};
        tol::Real=1e-6, nd=-1) :: Vector{ITensor}

    The **T-set**, or densor: a basis of the tensors that admit every element
    of `Δ` as a `P`-derivation.  This is `T(P, Δ)` of Densor.pdf; with `Δ` the
    derivation algebra of a tensor it is the densor subspace of that tensor.

    Every returned `s` satisfies the defining equation approximately:

        applyDerivation(s, D, Chisel(P, frame)) ≈ 0   for every D in Δ

    which is the T-law of `test/TestDerivationLaws.jl`.

    - `Ω`: transverse operators, supplying the frame.
    - `P`: a linear chisel.
    - `Δ`: a spanning set of operators, as returned by `der`.
    - `nd`: if positive, return at most this many basis tensors.
    - `tol`: tolerance for the nullspace.
"""
function den(Ω::TransverseOps, P::AbstractMatrix, Δ::Vector{Vector{ITensor}};
             tol::Real=1e-6, nd=10, solver::Symbol=:SVDSolver) :: Vector{ITensor}
    (A, AtA, fr, dims) = denLM(Ω, P, Δ)
    N = prod(dims)

    # nd < 0 asks for a basis, so request every vector; otherwise ask for the
    # batch size wanted.  Either way, only vectors whose singular value is
    # below `tol` are genuine solutions.
    want = nd < 0 ? N : min(floor(Int, nd), N)
    L = __needsSquare(solver) ? AtA : A
    result = solve(L, solver; nv=want)

    keep = findall(v -> abs(v) < tol, result.vals)
    isempty(keep) && return ITensor[]
    if nd > 0 && length(keep) > nd
        keep = keep[1:floor(Int, nd)]
    end
    return [ ITensor(reshape(result.vecs[:, j], dims...), fr...) for j in keep ]
end

# A single derivation is the common case; accept it without wrapping.
den(Ω::TransverseOps, P::AbstractMatrix, D::Vector{ITensor}; kwargs...) =
    den(Ω, P, [D]; kwargs...)

# Method-taking forms, for interface symmetry with `der`.  The dense nullspace
# is method-independent today; when an iterative densor solver exists it
# dispatches here.
den(::DerivationMethod, Ω::TransverseOps, P::AbstractMatrix, Δ::Vector{Vector{ITensor}};
    kwargs...) = den(Ω, P, Δ; kwargs...)

den(::DerivationMethod, Ω::TransverseOps, P::AbstractMatrix, D::Vector{ITensor};
    kwargs...) = den(Ω, P, [D]; kwargs...)

"""
    den(Γ::ITensor; tol, nd, method) :: Vector{ITensor}

    The densor subspace of `Γ` itself: solve for the derivations of `Γ` with
    the universal chisel, then for every tensor admitting them.  `Γ` must lie
    in the result, which is the Galois law.
"""
function den(Γ::ITensor; tol::Real=1e-6, nd=10, solver::Symbol=:SVDSolver,
             method::Union{DerivationMethod,Symbol}=:SylverLining, kwargs...)
    ch, fr, Ω = universalSetup(Γ)
    m = method isa Symbol ? get_derivation_method(method; kwargs...) : method
    # The derivations are the *constraints* here, so always take all of them:
    # a truncated Δ under-constrains the densor and returns too large a space.
    Δ = der(m, Ω, ch, Γ; tol=Float64(tol), nd=-1)
    return den(Ω, ch, Δ; tol=tol, nd=nd, solver=solver)
end

"""
    stratify(Γ::AbstractArray, der::Vector{ITensor})
    :: NamedTuple{(:Σ, :Xs), Tuple{AbstractArray, Vector{ITensor}}}

    Stratify the tensor Γ using the spall and the specified positions.
    Returns a named tuple with the sculpted tensor and the transforms used.

    - `Γ`: The input tensor
    - `der` a derivation to specify strata

    Returns a named tuple with fields:
    - `Σ` The sculpted tensor
    - `Xs` a transverse operator 
"""
function stratify(
        Γ::ITensor, 
        der::Vector{ITensor}
    ) :: NamedTuple{(:Σ, :Xs), Tuple{ITensor, Vector{ITensor}}}
    Xs = [ 
        let X = der[i]
            D, T = realCanonicalForm(Array(X, inds(X)...))
            ITensor(Matrix(T), inds(X)...)
        end for i in 1:length(der) ]
    return (;Σ=Γ*Xs, Xs=Xs)
    # retag the indexes
end


# doing reduced here is dangerous since it changes axis and we need to change Ω 
function stratify(
        Ω::TransverseOps, 
        ch::AbstractMatrix,          
        Γ::ITensor;
        tol::Float64=1e-6,
        nd=-1,
        method::Union{DerivationMethod, Symbol}=:SylverLining,
        method_kwargs...
    )
    selected_method = method isa Symbol ? get_derivation_method(method; method_kwargs...) : method
    (rΩ, expand_map, ders) = derTrOpsReduced(selected_method, Ω, ch, Γ; tol=tol, nd=nd, method_kwargs...)
    if size(ders,2) == 0
        # Not necessarily a solver failure: the derivation space really can be
        # trivial, in which case Γ conforms to no sparsity pattern for this
        # chisel and there is nothing to stratify along.  Happens for a Tucker
        # chisel on a generic tensor, and whenever the operator space excludes
        # the scalars.
        error("No nontrivial derivations for this chisel, so Γ exhibits no " *
              "sparsity pattern to stratify along. Check the chisel and the " *
              "operator space; if a pattern is expected, loosen `tol`.")
    end
    @info "Found $(size(ders,2)) derivations for stratification."
    # Select a random linear combination of derivations
    n_ders = size(ders,2)
    # n_axes = ndims(Γ)
    coefs = [ randn() for _ in 1:n_ders ]
    der = ders*coefs
    # expand into tensors
    # δ = embedITensors(rΩ,der)
    δ = embedITensors(Ω,expand_map(der))
    # Get the stratified nondegenerate tensor
    Δ, Xs = stratify(Γ, δ)
    return (;Σ=Δ, Xs=Xs)
end

# ---- Convenience wrappers for stratify ---
function stratify(
        Γ::ITensor;
        tol::Float64=1e-6,
    nd=-1,
    method::Union{DerivationMethod, Symbol}=:SylverLining,
    reduced=false,
    method_kwargs...
    )
    # Use universal chisel and transverse ops
    ch = UniversalChisel(length(inds(Γ)))
    fr = collect(inds(Γ))
    Ω = IndTransverseOps(fr, UniversalOp())    
    return stratify(Ω, ch, Γ; tol=tol, nd=nd, method=method, method_kwargs...)
end

function stratify(
        Γ::AbstractArray,
        der::Vector{ITensor}
    )
    # Convert array to ITensor and delegate
    return stratify(__ITensor(Γ), der)
end

function stratify(
        Γ::AbstractArray;
        tol::Float64=1e-6,
        nd=-1,
        method::Union{DerivationMethod, Symbol}=:SylverLining,
        method_kwargs...
    )
    return stratify(__ITensor(Γ); tol=tol, nd=nd, method=method, method_kwargs...)
end



