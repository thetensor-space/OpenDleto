module DletoArpackExt
using Dleto
using Arpack
using LinearMaps
using LinearAlgebra

# can not export from extension package
#export ArpackDenseSolver, ArpackSolver

struct ArpackSolver <: Dleto.NullSolver end
struct ArpackDenseSolver <: Dleto.NullSolver end


function Dleto.solve(::ArpackSolver, L::LinearMaps.LinearMap; nd::Integer = 10, tol::Float64 = 1e-8, kwargs...)
    # println("Using ArpackSolver...")
    # Use Arpack to compute the null space of L.
    vals, vecs = Arpack.eigs(L; nev=nd, ncv = min(5*nd+1,size(L,1)), sigma=-0.1, which=:LR, kwargs...)
    small = [ abs(vals[i]) < tol for i = 1:length(vals)]
    rvals = vals[small]
    rvecs = vecs[:,small]
    if (length(rvals)) < nd || (nd < 0)
        return (; vals=rvals, vecs=rvecs )
    else
        return (; vals=rvals[1:nd], vecs=rvecs[:,1:nd] )
    end
end



function Dleto.solve(::ArpackDenseSolver, L::LinearMaps.LinearMap; nd::Integer = 10, tol::Float64 = 1e-8, kwargs...)
    # println("Using ArpackDenseSolver...")
    M = Matrix(L) # Convert LinearMap to dense Matrix to allow LU-Factorization
    # Use Arpack to compute the null space of L.
    # vals, vecs = Arpack.eigs(M; nev=nd, which=:SR, sigma=0.0,kwargs...)
    vals, vecs = Arpack.eigs(M; nev=nd, ncv = min(5*nd+1,size(M,1)), sigma=-0.1, which=:LR, kwargs...)
    small = [ abs(vals[i]) < tol for i = 1:length(vals)]
    rvals = vals[small]
    rvecs = vecs[:,small]
    if (length(rvals)) < nd || (nd < 0)
        return (; vals=rvals, vecs=rvecs )
    else
        return (; vals=rvals[1:nd], vecs=rvecs[:,1:nd] )
    end
end

function __init__()
    println("Loading Dleto Arpack Extension")
    # println("Loading Dleto Arpack Extension: Registering :ArpackSolver and :ArpackDenseSolver nullsolvers")
    # NullSolversDict[:ArpackSolver] = ArpackSolver()
    # NullSolversDict[:ArpackDenseSolver] = ArpackDenseSolver()
end


end