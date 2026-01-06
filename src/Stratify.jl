
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
        Ω::TransverseOps, 
        ch::AbstractMatrix,          
        Γ::ITensor;
        tol::Float64=1e-6,
        reduced=false
    )
    if reduced
        # @info "Computing nondegenerate basis for stratification."
        Γ, Ps = nondeg(Γ; tol=tol, mode=:trunc)
        @info "Reduced tensor size: ", size(Γ)
    end
    ders = der(Ω,ch,Γ; tol=tol)
    if isempty(ders)
        # should never happen as there are always trivial derivations
        # so this indicates an error
        error("No derivations found for the given tensor, this indicates failure to converge in solvers, consider adjusting parameters.")
    end
    @info "Found $(length(ders)) derivations for stratification."
    # Select a random linear combination of derivations
    # Each derivation is a Vector{ITensor} with one ITensor per axis
    # if length(ders) >= length(inds(Γ))
    #     δ = ders[length(inds(Γ))] 
    # else
        n_ders = length(ders)
        n_axes = ndims(Γ)
        coefs = [ randn() for _ in 1:n_ders ]
        
        # Initialize δ as zeros with same structure as ders[1]
        δ = Vector{ITensor}(undef, n_axes)
        for a in 1:n_axes
            # Sum: coefs[1]*ders[1][a] + coefs[2]*ders[2][a] + ...
            δ[a] = coefs[1] * ders[1][a]
            for d in 2:n_ders
                δ[a] += coefs[d] * ders[d][a]
            end
        end
    # end 

    # Get the stratified nondegenerate tensor
    Δ, Xs = stratify(Γ, δ)
    Zs = similar(Xs)
    if reduced
        # Map back to original space
        # match each Xs[i] with Ps[j]
        count = 1
        for X in Xs
            for (i,P) in enumerate(Ps)
                common = intersect(inds(X),inds(P))
                if length(common) > 0
                    # transpose P.
                    # transpose_idx = findfirst(!=(common[1]), inds(P))
                    # ci = addtag(tanspose_idx, "Transpose")
                    # Pt = ITensor(P, ci, common[1])
                    Zs[count] = P * X
                    count += 1
                    deleteat!(Ps, i)
                    break
                end
            end
        end
        @info "Mapped stratification back to original space."
        Δ = Δ * Zs
    else
        Zs = Xs
    end
    return (;Σ=Δ, Xs=Zs)
end

# ---- Convenience wrappers for stratify ---
function stratify(
        Γ::ITensor;
        tol::Float64=1e-6,
        reduced=false
    )
    # Use universal chisel and transverse ops
    ch = UniversalChisel(length(inds(Γ)))
    fr = collect(inds(Γ))
    Ω = IndTransverseOps(fr, UniversalOp())    
    return stratify(Ω, ch, Γ; tol=tol, reduced=reduced)
end

function stratify(
        Γ::AbstractArray,
        der::Vector{ITensor}
    )
    # Convert array to ITensor and delegate
    return stratify(__ITensor(Γ), der)
end

function stratify(
        Γ::AbstractArray
    )
    return stratify(__ITensor(Γ))
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
function blockdiag(M::Matrix; tol::Float64=1e-10)
    res = LinearAlgebra.eigen(M)
    eigenvals = res.values
    eigenvecs = res.vectors
    n = length(eigenvals)
    real_diag = zeros(real(eltype(eigenvals)), n,n)
    real_blocks = zeros(real(eltype(eigenvecs)), n, n)
    
    processed = falses(n)
    warning = false
    for i in 1:n
        if processed[i]
            continue
        end
        
        λ = eigenvals[i]
        # Check if eigenvalue is effectively real (imaginary part below tolerance)
        if abs(imag(λ)) < tol
            # Real eigenvalue
            real_diag[i,i] = real(λ)
            real_blocks[:, i] = real(eigenvecs[:, i])
            processed[i] = true
        else
            # Find the conjugate pair in remaining eigenvalues
            j = nothing
            for k in (i+1):n
                if !processed[k] && isapprox(eigenvals[k], conj(λ); atol=tol)
                    j = k
                    break
                end
            end
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
            else
                # No conjugate found - treat as real (shouldn't happen for real matrices)
                warning = true
                real_diag[i,i] = real(λ)
                real_blocks[:, i] = real(eigenvecs[:, i])
                processed[i] = true
            end
        end
    end
    if warning
        @warn "No conjugate pair found for some eigenvalues, treating as real"
    end            
    return (;D = real_diag, T = real_blocks)
end
