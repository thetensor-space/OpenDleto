
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
        (
            X = der[i]; 
            D, T = realCanonicalForm(Array(X, inds(X)...)); 
            ITensor(Matrix(T), inds(X)...) 
        ) for i in 1:length(der) ]
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



# to be moved into Utils.jl
"""
    Real canonical form of a matrix
    Group conjugate pairs of complex eigenvalues and eigenvectors into real blocks.

    Returns a named tuple with:
    - `D`: A block diagonal matrix with real blocks.
    - `T`: A matrix whose columns are the real eigenvectors or the real 
    and imaginary parts of complex conjugate pairs.

    LawA
    ```julia
    res = realCanonicalForm(M); isapprox(M * res.T, res.T * res.D)
    ```
"""
function realCanonicalForm( M ::AbstractMatrix; tol::Float64=1e-10):: NamedTuple{(:D, :T), Tuple{AbstractMatrix,AbstractMatrix}}
    @assert size(M,1)==size(M,2) "Matrix must be square"
    if all( M .|> (x -> abs(x) < tol) )
        # M is zero matrix, return identityh transformation
        return (;D=zeros(eltype(M),size(M)), T = LinearAlgebra.Diagonal([ 1.0 for i = 1:size(M,1)]) )
    end
    eig = LinearAlgebra.eigen(M)
    evalues = eig.values
    evec = eig.vectors
    if isa(M, LinearAlgebra.Symmetric)              # no need to do anything if the matrix is symmetric
        return (; D= LinearAlgebra.Diagonal(evalues), T=evec) 
    end 
    found_complex=false
    n  = real.(evec)
    nn = real.(evec)
    D = zeros(eltype(n),size(M))
    D[1,1] = real(evalues[1])
    for i = 2: size(M,2)
        if ((((n[:,i] - n[:,i-1]) .|> x -> x*x) |> sum) > tol)
            # nn[:,i] =n[:,i]
            D[i,i] = real(evalues[i])
        else 
            nn[:,i] = imag.(evec[:,i])
            D[i,i] = real(evalues[i])
            D[i,i-1] = -imag(evalues[i])
            D[i-1,i] = imag(evalues[i])
            found_complex=true
        end
    end
    return (;D = found_complex ? D : LinearAlgebra.Diaginal(real.(evalues)) , T = nn)
end
