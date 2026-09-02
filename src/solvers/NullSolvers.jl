"""
    NullSolvers

    An interface and options for solving the null spaces that arrise in Dleto.

    [TBD: Surely in Julia there is a package or standard assembly of null space solvers?
    Until I find this, here are some basic options implemented directly.]
"""


using LinearMaps
using LinearAlgebra



export NullSolver, LUSolver, SVDSolver, solve
export register_solver!, available_solvers

abstract type NullSolver end

"""
    solve(method::NullSolver, L::LinearMap; nv::Integer=10)

    - method: An instance of a subtype of `NullSolver` defining the solving method.
    - L: A `LinearMap`.  Rectangular for SVD/LU; the eigen- and Krylov-based
      solvers want the square symmetric A'A.
    - nv: Number of approximate null vectors to compute (default: 10).

    Returns a named tuple `(;vals, vecs)`: the singular-type values and the
    right approximate null vectors as the columns of `vecs`.

    The keyword is `nv` for every implementation.  It used to be `nd` here and
    in the KrylovKit extension while the others used `nv`, so dispatching
    through the symbol factory raised an unsupported-keyword error.
"""
function solve(method::NullSolver, L::LinearMap; nv::Integer=10) end

struct SVDSolver <: NullSolver end
struct LUSolver <: NullSolver end

"""
    SOLVER_REGISTRY :: Dict{Symbol, NullSolver}

Symbol -> solver instance.  Extensions register themselves here when their
trigger package loads.

This replaces a hard-coded `if/elseif` chain over seven symbol names.  That
chain could not work: the extension solver *types* are defined inside the
extension modules, so `KrylovSolver()` and friends were `UndefVarError:
not defined in Dleto`.  Five of the seven advertised solvers were unreachable
-- including Krylov, Lanczos and CG, whose packages are in [deps].
"""
const SOLVER_REGISTRY = Dict{Symbol, NullSolver}()

"""
    register_solver!(name::Symbol, solver::NullSolver)

Make `solver` reachable as `solve(L, name)`.  Called by extensions at load.
"""
register_solver!(name::Symbol, solver::NullSolver) = (SOLVER_REGISTRY[name] = solver)

"""Symbols currently reachable, i.e. whose packages are loaded."""
available_solvers() = sort(collect(keys(SOLVER_REGISTRY)))

function solve(L, sym::Symbol=:SVDSolver; kwargs...)
    haskey(SOLVER_REGISTRY, sym) || error(
        "Unknown or unavailable solver :$sym. Available now: " *
        string(available_solvers()) *
        ". Extension solvers need their package loaded first " *
        "(KrylovKit for :KrylovSolver, IterativeSolvers for :LanczosSolver " *
        "and :CGSolver, Arpack for :ArpackSolver and :ArpackDenseSolver).")
    return solve(SOLVER_REGISTRY[sym], L; kwargs...)
end

register_solver!(:SVDSolver, SVDSolver())
register_solver!(:LUSolver, LUSolver())

    
function solve(::SVDSolver, L::LinearMap; nv::Integer = 10)
    println("Using SVDSolver...")
    # Use LinearAlgebra to compute the null space of L.
    println("Converting LinearMap to Matrix for SVD...")
    M = Matrix(L)
    svds = LinearAlgebra.svd(M)
    nvals = min(nv, length(svds.S))
    return (;vals=svds.S[end:-1:(end-nvals+1)], vecs=svds.V[:, end:-1:(end-nvals+1)])
end

"""
    solve(::LUSolver, L; nv, tol)

    Null vectors by LU plus back-substitution.

    CAVEAT: `lu` pivots rows only, so it is **not rank revealing** -- this is
    only valid when the leading `rank` columns are independent.  It therefore
    reports an honest residual `‖Lv‖/‖v‖` for each vector it returns, so a
    caller filtering on `vals` discards a bad basis rather than trusting it.
    Prefer `SVDSolver` unless you know the column order is benign.

    Previously this returned a bare `Vector`, violating the `(;vals, vecs)`
    contract above -- callers written against it died with
    `FieldError: type Array has no field vals` -- and chose its free variables
    as the last `nv` columns regardless of the computed rank.
"""
function solve(::LUSolver, L::LinearMap; nv::Integer = 10, tol = 1e-8)
    M = Matrix(L)
    n = size(M, 2)
    F = lu(M; check = false)
    U = F.U
    r = min(sum(abs.(diag(U)) .> tol), size(U, 1), n)

    cols = Vector{Vector{eltype(M)}}()
    for j in (r + 1):n
        v = zeros(eltype(M), n)
        v[j] = 1
        if r > 0
            # U[1:r,1:r] x = -U[1:r,j] makes the pivot variables consistent.
            v[1:r] = U[1:r, 1:r] \ (-U[1:r, j])
        end
        push!(cols, v)
    end

    if isempty(cols)
        return (; vals = eltype(M)[], vecs = zeros(eltype(M), n, 0))
    end

    V = hcat(cols...)
    vals = [ norm(M * V[:, k]) / max(norm(V[:, k]), eps()) for k in 1:size(V, 2) ]
    keep = 1:min(nv, size(V, 2))
    return (; vals = vals[keep], vecs = V[:, keep])
end

