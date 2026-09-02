
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
    denLM(Ω::TransverseOps, P::AbstractMatrix, Δ::Vector{Vector{ITensor}})
        :: Tuple{AbstractMatrix, Vector{<:Index}, Vector{Int}}

    The densor system: a matrix whose nullspace is the T-set of `Δ` for the
    chisel `P`, in the coordinates of the tensor space of `Ω`.  Returns the
    matrix together with the frame and the axis dimensions needed to read
    nullspace vectors back as tensors.
"""
function denLM(Ω::TransverseOps, P::AbstractMatrix, Δ::Vector{Vector{ITensor}})
    fr = collect(frames(Ω))
    C = Chisel(P, fr)
    dims = [ITensors.dim(i) for i in fr]
    N = prod(dims)
    isempty(Δ) && error("den: need at least one derivation to solve against.")

    cols = Vector{Vector{Float64}}()
    for n in 1:N
        e = zeros(Float64, N)
        e[n] = 1.0
        s = ITensor(reshape(e, dims...), fr...)
        col = Float64[]
        for D in Δ
            R = applyDerivation(s, D, C)
            append!(col, vec(Array(R, (C.ch_axis, fr...))))
        end
        push!(cols, col)
    end
    return (hcat(cols...), fr, dims)
end

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
             tol::Real=1e-6, nd=-1) :: Vector{ITensor}
    (M, fr, dims) = denLM(Ω, P, Δ)
    ns = LinearAlgebra.nullspace(M; atol=Float64(tol))
    k = size(ns, 2)
    if nd > 0 && k > nd
        k = floor(Int, nd)
    end
    return [ ITensor(reshape(ns[:, j], dims...), fr...) for j in 1:k ]
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
function den(Γ::ITensor; tol::Real=1e-6, nd=-1,
             method::Union{DerivationMethod,Symbol}=:SylverLining, kwargs...)
    ch, fr, Ω = universalSetup(Γ)
    m = method isa Symbol ? get_derivation_method(method; kwargs...) : method
    Δ = der(m, Ω, ch, Γ; tol=Float64(tol), nd=-1)
    return den(Ω, ch, Δ; tol=tol, nd=nd)
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
    if size(ders,2) ==0
        # should never happen as there are always trivial derivations
        # so this indicates an error
        # it can happen if we use operators which do not contain scalars, or use full Tucker chisel
        error("No derivations found for the given tensor, this indicates failure to converge in solvers, consider adjusting parameters.")
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



