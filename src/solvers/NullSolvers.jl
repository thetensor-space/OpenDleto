#
# Strata Dleto: Null Solvers
#   Creation and adaptation of null space solvers for tensor decomposition.
#
# -----------------------------------------------------------------------------
# Copyright 2022-2026 Peter A. Brooksbank, Martin D. Kassabov, James B. Wilson
# 
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the “Software”), 
# to deal in the Software without restriction, including without limitation the 
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or 
# sell copies of the Software, and to permit persons to whom the Software is 
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in 
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE 
# SOFTWARE.
#-----------------------------------------------------------------------------


"""
    NullSolvers

    An interface and options for solving the null spaces that arise in Dleto.

    [TBD: Surely in Julia there is a package or standard assembly of null space solvers?
    Until I find this, here are some basic options implemented directly.]
"""


using LinearMaps
using LinearAlgebra

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
struct KrylovSolver <: NullSolver end

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
    
