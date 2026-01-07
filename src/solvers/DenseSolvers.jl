#
# Strata Dleto: Dense Null Solvers
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


import LinearAlgebra
using LinearAlgebra

function solve(::SVDSolver, L::AbstractMatrix; nv::Integer = 10)
    @debug "Using SVD solver..."
    # Use LinearAlgebra to compute the null space of L.
    svds = LinearAlgebra.svd(L)
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
