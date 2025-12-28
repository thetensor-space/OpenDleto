"""
    NullSolvers

    An interface and options for solving the null spaces that arrise in Dleto.

    [TBD: Surely in Julia there is a package or standard assembly of null space solvers?
    Until I find this, here are some basic options implemented directly.]
"""
module NullSolvers


using LinearMaps
using LinearAlgebra
using IterativeSolvers
using Arpack

using LinearAlgebra

export NullSolver, LUSolver, ArpackDenseSolver, SVDSolver, LanczosSolver, ArpackSolver, CGSolver, solve

abstract type NullSolver end 
        
"""
    solve(method::NullSolver, L::LinearMap; nv::Integer=10)

    - method: An instance of a subtype of `NullSolver` defining the solving method.
    - L: A `LinearMap` defining the dual-primal A'A transform
    - nv: Number of approximate null vectors to compute (default: 10).

    Returns a named tuple with the singular-type values and right approximate null vectors
"""
function solve(method::NullSolver, L::LinearMap; nd::Integer=10) end

struct SVDSolver <: NullSolver end
struct LUSolver <: NullSolver end
struct LanczosSolver <: NullSolver end
struct ArpackSolver <: NullSolver end
struct ArpackDenseSolver <: NullSolver end
struct CGSolver <: NullSolver end


# Map Symbol to solver type and call solve, defaulting to SVDSolver
function solve(L, sym::Symbol=:SVDSolver; kwargs...)
    solver =
        sym === :SVDSolver      ? SVDSolver() :
        sym === :LanczosSolver  ? LanczosSolver() :
        sym === :ArpackSolver   ? ArpackSolver() :
        sym === :ArpackDenseSolver ? ArpackDenseSolver() :
        sym === :CGSolver       ? CGSolver() :
        sym === :LUSolver       ? LUSolver() :
        error("Unknown solver symbol: $sym")
    return solve(solver, L; kwargs...)
end

function solve(::LUSolver, L::LinearMap; nv::Integer = 10)
    println("Using SVDSolver...")
    # Use LinearAlgebra to compute the null space of L.
    println("Converting LinearMap to Matrix for SVD...")
    M = Matrix(L)
    svds = LinearAlgebra.svd(M)
    nvals = min(nv, length(svds.S))
    return (;vals=svds.S[end:-1:(end-nvals+1)], vecs=svds.V[:, end:-1:(end-nvals+1)])
end

function solve(::LUSolver, L::LinearMap; nv::Integer = 10, tol = 1e-8)
    println("Using LUSolver on Matrix...")
    M = Matrix(L)
    println("Performing LU Factorization...")
    F = lu(M)
    U = F.U
    L = F.L
    p = F.p  # permutation

    # Determine rank and free variables
    rank = sum(abs.(diag(U)) .> tol)
    n = size(M, 2)
    # free_vars = rank+1:n
    free_vars = (n-nv+1):n
println("Matrix rank: $rank, free variables: ", free_vars, " with tolerance $tol")
    # Build null space basis
    null_basis = []
    for j in free_vars
        v = zeros(n)
        v[j] = 1
        # Solve U * x = -U[:, free_vars]*v_free for pivot variables
        rhs = -U[:, free_vars] * v[free_vars]
        x = U[1:rank, 1:rank] \ rhs[1:rank]
        v[1:rank] = x
        push!(null_basis, v)
    end

    return null_basis #(;vals=svds.S[end:-1:(end-nvals+1)], vecs=svds.V[:, end:-1:(end-nvals+1)])
end

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

end # module