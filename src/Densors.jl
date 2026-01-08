
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
        tol::Float64=1e-6
    )
    (rΩ, expand_map, ders) = derTrOpsReduced(Ω,ch,Γ; tol=tol)
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

# will deal with wrapers later
# # ---- Convenience wrappers for stratify ---
# function stratify(
#         Γ::ITensor;
#         tol::Float64=1e-6,
#         reduced=false
#     )
#     # Use universal chisel and transverse ops
#     ch = UniversalChisel(length(inds(Γ)))
#     fr = collect(inds(Γ))
#     Ω = IndTransverseOps(fr, UniversalOp())    
#     return stratify(Ω, ch, Γ; tol=tol, reduced=reduced)
# end

# function stratify(
#         Γ::AbstractArray,
#         der::Vector{ITensor}
#     )
#     # Convert array to ITensor and delegate
#     return stratify(__ITensor(Γ), der)
# end

# function stratify(
#         Γ::AbstractArray
#     )
#     return stratify(__ITensor(Γ))
# end



