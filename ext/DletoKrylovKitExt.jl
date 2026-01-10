module DletoKrylovKitExt

using Dleto
using LinearMaps
using LinearAlgebra

import KrylovKit
using KrylovKit: eigsolve

export KrylovSolver

struct KrylovSolver <: Dleto.NullSolver end

function Dleto.solve(::KrylovSolver, L::LinearMap; nd::Integer = 10, tol::Float64 = 1e-8)
    println("Using KrylovSolver...")
    nev = min(nd, size(L, 1))  # Number of eigenvalues to compute
    x0 = randn(size(L, 2))     # Initial guess vector

    # Retry logic for convergence
    max_attempts = 5
    maxiter = 100
    krylovdim = max(10, 2*nev)
    converged = false
    
    # Initialize variables that will be set in the loop
    λ = ComplexF64[]
    vecs = Vector{ComplexF64}[]
    
    for attempt in 1:max_attempts
        λ, vecs, info = eigsolve(L, x0, nev, :SR;
            maxiter=maxiter,
            krylovdim=krylovdim,
            tol=tol
        )
        
        converged = info.converged >= nev
        
        if converged
            # Success! Convert and continue
            λ = real.(λ)
            vecs = [real.(v) for v in vecs]
            break
        else
            # Not enough converged, increase parameters and retry
            @warn "Attempt $attempt: Only $(info.converged) of $nev eigenvalues converged. Retrying with increased parameters..."
            maxiter = Int(round(maxiter * 1.5))
            krylovdim = min(Int(round(krylovdim * 1.5)), size(L, 1))
            x0 = randn(size(L, 2))  # New random start
            
            if attempt == max_attempts
                # Last attempt failed, use what we have
                @warn "Final attempt: Using $(info.converged) converged eigenvalues out of $nev requested."
                λ = real.(λ)
                vecs = [real.(v) for v in vecs]
            end
        end
    end
    return (;vals=λ, vecs=vecs)
end

function __init__()
    println("Loading Dleto KrylovKit Extension")
end

end