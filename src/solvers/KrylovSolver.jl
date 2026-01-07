

import IterativeSolvers
using IterativeSolvers: lobpcg

import Arpack
using Arpack: eigs

import KrylovKit
using KrylovKit: eigsolve

function solve(::LanczosSolver, L::LinearMap; nv::Integer = 10)
    println("Using LanczosSolver...")
    # Use IterativeSolvers to compute the null space of L.
    # We only need the right singular vectors as we want a right null space.
    # svdl does partial SVD via Lanczos bidiagonalization, so we need to 
    # ask for a larger number of singular values `nsv` to reach the smallest ones.
    vals = minimum(size(L))
    S, V = IterativeSolvers.svdl(L; nsv=vals, vecs=:right)
    println(S)
    nvals = min(nv, length(S))
    inds = length(S):-1:(length(S)-nvals+1)
    return (;vals=S[inds], vecs=V[:, inds])
end

function solve(::ArpackSolver, L::LinearMap; nv::Integer = 20)
    println("Using ArpackSolver...")
    # Use Arpack to compute the null space of L.
    vals, vecs = Arpack.eigs(L; nev=nv, which=:SM)
    return (;vals=vals, vecs=vecs)
end

function solve(::ArpackDenseSolver, L::LinearMap; nv::Integer = 20)
    println("Using ArpackDenseSolver...")
    M = Matrix(L) # Convert LinearMap to dense Matrix to allow LU-Factorization
    # Use Arpack to compute the null space of L.
    vals, vecs = Arpack.eigs(M; nev=nv, which=:LM, sigma=0.0)
    return (;vals=vals, vecs=vecs)
end

function solve(::CGSolver, L::LinearMap; nv::Integer = 10)
    println("Using CGSolver...")
    res = lobpcg(L, false, 10 ) #; maxiter=size(L,2) / 2, tol=1e-16)
    return (;vals=res.λ, vecs=res.X)
end

function solve(::KrylovSolver, L::LinearMap; nd::Integer = 10, tol::Float64 = 1e-8)
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
