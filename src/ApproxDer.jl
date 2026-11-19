"""
    approx_der(Γ; tol, max_dense_bytes=512*1024^2)

Return the universal-chisel right singular directions of the rectangular
derivation map whose residuals satisfy `σ/σ₁ ≤ tol`. This is a numerical
near-null space, not a basis of exact derivations. The `ndims(Γ)-1` scalar directions are included when their
numerical residuals fall under the cutoff, and counted separately.

The result contains `laws`, `accepted_indices` in descending spectral order,
`relative_singular_values`, `scalar_dimension`,
`approximate_nonscalar_count`, `threshold_gap`, `map_shape`, and `cutoff`.
The map is materialized for a dense SVD; `max_dense_bytes` is a guard.
"""
function approx_der(Γ::ITensor; tol::Real,
                    max_dense_bytes::Integer=512 * 1024^2)
    ndims(Γ) >= 3 || throw(ArgumentError("approx_der requires valence at least 3"))
    isfinite(tol) && 0 < tol < 1 ||
        throw(ArgumentError("tol must be finite and strictly between 0 and 1"))
    max_dense_bytes > 0 || throw(ArgumentError("max_dense_bytes must be positive"))

    Ω = UniversalOps(Γ)
    _, N = sylvesterLM(Ω, UniversalChisel(ndims(Γ)), Γ)
    bytes = Base.checked_mul(Base.checked_mul(size(N, 1), size(N, 2)),
                             sizeof(eltype(N)))
    bytes <= max_dense_bytes ||
        throw(ArgumentError("dense derivation map needs $bytes bytes; raise max_dense_bytes explicitly"))
    F = LinearAlgebra.svd(Matrix(N); full=false)
    isempty(F.S) && throw(ArgumentError("empty derivation map"))
    first(F.S) > 0 || throw(ArgumentError("zero derivation map"))
    relative_singular_values = F.S ./ first(F.S)
    accepted_indices = findall(x -> x <= tol, relative_singular_values)
    laws = [embedITensors(Ω, F.V[:, i]) for i in accepted_indices]
    scalar_dimension = ndims(Γ) - 1
    approximate_nonscalar_count = max(length(accepted_indices) - scalar_dimension, 0)
    threshold_gap = isempty(accepted_indices) || first(accepted_indices) == 1 ?
        nothing : relative_singular_values[first(accepted_indices) - 1] /
                  relative_singular_values[first(accepted_indices)]
    return (; laws, accepted_indices, relative_singular_values,
            scalar_dimension, approximate_nonscalar_count, threshold_gap,
            map_shape=size(N), cutoff=Float64(tol))
end

approx_der(Γ::AbstractArray; kwargs...) = approx_der(__asITensor(Γ); kwargs...)

"""
    approx_stratify(Γ; tol, descending_index=nothing, max_dense_bytes=512*1024^2)

Use one tolerance-selected nonscalar singular direction to change coordinates.
By default choose the highest-residual nonscalar direction under the cutoff;
`descending_index` selects another accepted direction. The returned `Σ` and
`Xs` are an exploratory approximate stratification. They do not imply exact
off-block zeros; inspect reconstruction and frame conditioning separately.
"""
function approx_stratify(Γ::ITensor; tol::Real,
                         descending_index::Union{Nothing,Integer}=nothing,
                         max_dense_bytes::Integer=512 * 1024^2)
    result = approx_der(Γ; tol, max_dense_bytes)
    n = length(result.relative_singular_values)
    candidates = filter(i -> i <= n - result.scalar_dimension,
                        result.accepted_indices)
    isempty(candidates) &&
        throw(ArgumentError("cutoff admits no nonscalar approximate direction"))
    index = isnothing(descending_index) ? first(candidates) : Int(descending_index)
    index in candidates ||
        throw(ArgumentError("descending_index must select a nonscalar direction under tol"))
    location = findfirst(==(index), result.accepted_indices)
    law = result.laws[location]
    adapted = stratify(Γ, law)
    return (; Σ=adapted.Σ, Xs=adapted.Xs, law, descending_index=index,
            relative_residual=result.relative_singular_values[index],
            accepted_count=length(result.accepted_indices),
            approximate_nonscalar_count=result.approximate_nonscalar_count,
            threshold_gap=result.threshold_gap, cutoff=result.cutoff)
end

approx_stratify(Γ::AbstractArray; kwargs...) =
    approx_stratify(__asITensor(Γ); kwargs...)
