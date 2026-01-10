module DletoIterativeSolversExt

using Dleto
using LinearMaps
using LinearAlgebra

import IterativeSolvers: lobpcg
using IterativeSolvers

export  LanczosSolver,  CGSolver

struct LanczosSolver <: Dleto.NullSolver end
struct CGSolver <: Dleto.NullSolver end



function Dleto.solve(::LanczosSolver, L::LinearMap; nv::Integer = 10)
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


function Dleto.solve(::CGSolver, L::LinearMap; nv::Integer = 10)
    println("Using CGSolver...")
    res = lobpcg(L, false, 10 ) #; maxiter=size(L,2) / 2, tol=1e-16)
    return (;vals=res.λ, vecs=res.X)
end

function __init__()
    println("Loading Dleto IterativeSolvers Extension")
end

end