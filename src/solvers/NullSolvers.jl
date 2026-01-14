"""
    NullSolvers

    An interface and options for solving the null spaces that arrise in Dleto.

    [TBD: Surely in Julia there is a package or standard assembly of null space solvers?
    Until I find this, here are some basic options implemented directly.]
"""


import LinearMaps
import LinearAlgebra



# export NullSolver, LUSolver, SVDSolver, solve
export NullSolversDict

abstract type NullSolver end 
        
"""
    solve(method::NullSolver, L::LinearMap; nv::Integer=10)

    - method: An instance of a subtype of `NullSolver` defining the solving method.
    - L: A `LinearMap` defining the dual-primal A'A transform
    - nv: Number of approximate null vectors to compute (default: 10).

    Returns a named tuple with the singular-type values and right approximate null vectors
"""
function solve(method::NullSolver, L::LinearMaps.LinearMap; nd::Integer=10, kwargs...) end

#generate NullSolver from symbol
function NullSolver(sym::Symbol=:default) 
    if sym in keys(NullSolversDict)
        return NullSolversDict[sym]
    else
        error("Unknown solver symbol: $sym")
    end
end;

#I think we should drop this function
# Map Symbol to solver type and call solve, defaulting to SVDSolver
function solve(L::LinearMaps.LinearMap, sym::Symbol=:default; kwargs...)
    if sym in keys(NullSolversDict)
        return solve(NullSolversDict[sym], L; kwargs...)
    else
        error("Unknown solver symbol: $sym")
    end
end;



"""
    Dictionary of implemented/loaded solvers
"""
NullSolversDict = Dict{Symbol, NullSolver}()







