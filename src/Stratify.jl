
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
    )
    # Convert ITensors to Matrices before calling blockdiag
    Xs = Vector{ITensor}(undef, length(der))
    for i in 1:length(der)
        X = der[i]
        D, T = blockdiag(Array(X, inds(X)...))
        Xs[i] = ITensor(T, inds(X)...)
    end
    return (;Σ=Γ*Xs, Xs=Xs)
end

function stratify(
        Γ::ITensor
    )
    ders = der(Γ)
    return stratify(Γ, ders)
end


"""
    Group conjugate pairs of complex eigenvalues and eigenvectors into real blocks.

    Returns a named tuple with:
    - `D`: A block diagonal matrix with real blocks.
    - `T`: A matrix whose columns are the real eigenvectors or the real 
    and imaginary parts of complex conjugate pairs.

    Law
    ```julia
    res = blockdiag(M); isapprox(M * res.T, res.T * res.D)
    ```
"""
function blockdiag(M::Matrix)
    res = LinearAlgebra.eigen(M)
    eigenvals = res.values
    eigenvecs = res.vectors
    n = length(eigenvals)
    real_diag = zeros(real(eltype(eigenvals)), n,n)
    real_blocks = zeros(real(eltype(eigenvecs)), n, n)
    
    processed = falses(n)
    for i in 1:n
        if processed[i]
            continue
        end
        
        λ = eigenvals[i]
        if isreal(λ)
            # Real eigenvalue
            real_diag[i,i] = real(λ)
            real_blocks[:, i] = real(eigenvecs[:, i])
            processed[i] = true
        else
            j = i+findfirst(j -> !processed[j] && isapprox(eigenvals[j], conj(λ)), (i+1):n)
            if j !== nothing
                # Create real 2D subspace from conjugate pair
                v = eigenvecs[:, i]
                # Fill in the real block diagonal entries
                real_diag[i,i] = real(λ);  real_diag[i,j] = imag(λ);
                real_diag[j,i] = -imag(λ);  real_diag[j,j] = real(λ);
                # Fill in the real block eigenvectors
                real_blocks[:, i] = real(v)
                real_blocks[:, j] = imag(v)
                processed[i] = processed[j] = true
            end
        end
    end
    return (;D = real_diag, T = real_blocks)
end
