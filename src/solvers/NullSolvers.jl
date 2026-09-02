"""
    NullSolvers

    An interface and options for solving the null spaces that arrise in Dleto.

    [TBD: Surely in Julia there is a package or standard assembly of null space solvers?
    Until I find this, here are some basic options implemented directly.]
"""


using LinearMaps
using LinearAlgebra



export NullSolver, LUSolver, SVDSolver, solve

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


# Map Symbol to solver type and call solve, defaulting to SVDSolver
function solve(L, sym::Symbol=:SVDSolver; kwargs...)
    solver =
        sym === :SVDSolver      ? SVDSolver() :
        sym === :LanczosSolver  ? LanczosSolver() :
        sym === :ArpackSolver   ? ArpackSolver() :
        sym === :ArpackDenseSolver ? ArpackDenseSolver() :
        sym === :CGSolver       ? CGSolver() :
        sym === :LUSolver       ? LUSolver() :
        sym === :KrylovSolver    ? KrylovSolver() :
        error("Unknown solver symbol: $sym")
    return solve(solver, L; kwargs...)
end

    
function solve(::SVDSolver, L::LinearMap; nv::Integer = 10)
    println("Using SVDSolver...")
    # Use LinearAlgebra to compute the null space of L.
    println("Converting LinearMap to Matrix for SVD...")
    M = Matrix(L)
    svds = LinearAlgebra.svd(M)
    nvals = min(nv, length(svds.S))
    return (;vals=svds.S[end:-1:(end-nvals+1)], vecs=svds.V[:, end:-1:(end-nvals+1)])
end

# NOTE: this does not honour the interface documented on `solve` above -- it
# returns a bare Vector of basis vectors rather than a `(;vals, vecs)` named
# tuple, so callers written against the contract (e.g. `den`) fail with
# `FieldError: type Array has no field vals`.  The commented-out return at the
# end of the body shows the intended shape.  Separately, `free_vars` is chosen
# as the last `nv` columns regardless of the computed rank.  Left as-is here:
# fixing it is more than a return-type change.
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

