module DletoArpackExt
using Dleto
using Arpack
using LinearMaps
using LinearAlgebra

export ArpackDenseSolver, ArpackSolver

struct ArpackSolver <: NullSolver end
struct ArpackDenseSolver <: NullSolver end

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

function __init__()
    println("Loading Dleto Arpack Extension")
end


end