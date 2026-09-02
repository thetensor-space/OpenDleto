module DletoArpackExt
using Dleto
using Arpack
using LinearMaps
using LinearAlgebra

export ArpackDenseSolver, ArpackSolver

struct ArpackSolver <: Dleto.NullSolver end
struct ArpackDenseSolver <: Dleto.NullSolver end

# NOTE: Arpack is a weakdep and is not installed in the current manifest, so
# these two paths are unexercised -- the corrections below (clamping `nev`
# below the dimension, as ARPACK requires, and returning real parts for a
# symmetric map) are by inspection, not by test.

function Dleto.solve(::ArpackSolver, L::LinearMap; nv::Integer = 20)
    println("Using ArpackSolver...")
    nev = clamp(nv, 1, size(L, 2) - 2)   # ARPACK requires nev < n - 1
    vals, vecs = Arpack.eigs(L; nev=nev, which=:SM)
    return (;vals=real.(vals), vecs=real.(vecs))
end

function Dleto.solve(::ArpackDenseSolver, L::LinearMap; nv::Integer = 20)
    println("Using ArpackDenseSolver...")
    M = Matrix(L) # Convert LinearMap to dense Matrix to allow LU-Factorization
    nev = clamp(nv, 1, size(M, 2) - 2)
    vals, vecs = Arpack.eigs(M; nev=nev, which=:LM, sigma=0.0)
    return (;vals=real.(vals), vecs=real.(vecs))
end

function __init__()
    println("Loading Dleto Arpack Extension")
    Dleto.register_solver!(:ArpackSolver, ArpackSolver())
    Dleto.register_solver!(:ArpackDenseSolver, ArpackDenseSolver())
end




end
