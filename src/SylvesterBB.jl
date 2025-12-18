
#
# Strata Dleto: Sylvester Solvers
#   Algorithms for solving Sylvester equations arising in chiseling.
# -----------------------------------------------------------------------------
# Copyright 2022-2025 Peter A. Brooksbank, Martin D. Kassabov, James B. Wilson
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
# -----------------------------------------------------------------------------

module SylvesterBB


using LinearMaps
using TensorOperations
using Arpack
using IterativeSolvers
using CUDA
using cuTENSOR


export sylverlin3, sylverlin1, toMat, solveeig, solveit, cusylverlin3

#----------- Black-box Sylvester solvers --------------------------------------
function sylverlin3(s::AbstractArray, chisel::Matrix)
    # Make black-box linear map.
    d1 = size(s,1);
    d2 = size(s,2);
    d3 = size(s,3);
    dims = (size(chisel,1), size(s,1), size(s,2), size(s,3))
    function apply(x) 
        # without copying data, partition the vector and index it as square matrices
        A = reshape(view(x, 1:d1^2),          (d1, d1))
        B = reshape(view(x, (d1^2+1):(d1^2+d2^2)), (d2, d2))
        C = reshape(view(x, (d1^2+d2^2+1):(d1^2+d2^2+d3^2)), (d3, d3))
        # create an empty tensor of the required size
        # t = similar(s)
        t = zeros(eltype(s), dims)
        # call fast tensor contraction algorithms.
        @tensor t[c,i,j,k] += (chisel[:,1])[c] * A[i,i'] * s[i',j,k] + (chisel[:,2])[c] * B[j,j'] * s[i,j',k] + (chisel[:,3])[c] * C[k,k'] * s[i,j,k']
        return vec(t)
    end
    function apply_transpose(y)
        # Reshape vec into a tensor of the same shape as s
        t = reshape(y, dims)
        # Compute the contractions for the transpose
        # (You may need to flatten the result at the end)
        # Example using TensorOperations.jl:
        @tensor A′[ i,i′] := (chisel[:,1])[c]*t[c,i,j,k] * s[i′,j,k]
        @tensor B′[ j,j′] := (chisel[:,2])[c]*t[c,i,j,k] * s[i,j′,k]
        @tensor C′[ k,k′] := (chisel[:,3])[c]*t[c,i,j,k] * s[i,j,k′]
        # Flatten and concatenate A′, B′, C′ into a vector
        return vcat(vec(A′), vec(B′), vec(C′))
    end
    L = LinearMap( apply, apply_transpose, size(chisel,1)*size(s,1)*size(s,2)*size(s,3), size(s,1)^2 + size(s,2)^2 + size(s,3)^2; ismutating=false )
    return L
end


function cusylverlin3(s::CuArray, chisel::Matrix)
    # Make black-box linear map.
    d1 = size(s,1);
    d2 = size(s,2);
    d3 = size(s,3);
    dims = (size(chisel,1), size(s,1), size(s,2), size(s,3))
    function apply(x) 
        # without copying data, partition the vector and index it as square matrices
        A = reshape(view(x, 1:d1^2),          (d1, d1))
        B = reshape(view(x, (d1^2+1):(d1^2+d2^2)), (d2, d2))
        C = reshape(view(x, (d1^2+d2^2+1):(d1^2+d2^2+d3^2)), (d3, d3))
        # create an empty tensor of the required size
        # t = similar(s)
        t = zeros(eltype(s), dims)
        # call fast tensor contraction algorithms.
        @cutensor t[c,i,j,k] += (chisel[:,1])[c] * A[i,i'] * s[i',j,k] + (chisel[:,2])[c] * B[j,j'] * s[i,j',k] + (chisel[:,3])[c] * C[k,k'] * s[i,j,k']
        return vec(t)
    end
    function apply_transpose(y)
        # Reshape vec into a tensor of the same shape as s
        t = reshape(y, dims)
        # Compute the contractions for the transpose
        # (You may need to flatten the result at the end)
        # Example using TensorOperations.jl:
        @cutensor A′[ i,i′] := (chisel[:,1])[c]*t[c,i,j,k] * s[i′,j,k]
        @cutensor B′[ j,j′] := (chisel[:,2])[c]*t[c,i,j,k] * s[i,j′,k]
        @cutensor C′[ k,k′] := (chisel[:,3])[c]*t[c,i,j,k] * s[i,j,k′]
        # Flatten and concatenate A′, B′, C′ into a vector
        return vcat(vec(A′), vec(B′), vec(C′))
    end
    L = LinearMap( apply, apply_transpose, size(chisel,1)*size(s,1)*size(s,2)*size(s,3), size(s,1)^2 + size(s,2)^2 + size(s,3)^2; ismutating=false )
    return L
end


function toMat(L)
    nrows, ncols = size(L)
    M = zeros(Float64, nrows, ncols)
    for j in 1:ncols
        e_j = zeros(Float64, ncols)
        e_j[j] = 1.0
        M[:, j] = L * e_j
    end
    return M
end

function solveeig(t)
    L = sylverlin(t)
    vals, vecs, info = svdl(L) #; nsv=10)
    # vals, vecs, info = eigs(L*L'; nev=10, which=:SM)
    if info != 0
        println("Arpack did not converge, info = $info")
        println("Falling back to IterativeSolvers.svdl")
        vals, vecs = svdl(L; nsv=10, which=:SR)
    end
    offset1 = size(t,1)^2
    offset2 = offset1 + size(t,2)^2
    A = reshape(view(x,                     1:offset1),           (size(t,1), size(t,1)))
    B = reshape(view(x,       (offset1+1):offset2), (size(t,2), sizein(t,2)))
    C = reshape(view(x, (offset2+1):length(x)), (size(t,3), size(t,3)))
    return A, B, C
end

function solveit(t)
    L = sylverlin(t)
    u, s, v = svdl(L; nsv=10, which=:SM)
    offset1 = size(t,1)^2
    offset2 = offset1 + size(t,2)^2
    A = reshape(view(v[:,end],                     1:offset1),           (size(t,1), size(t,1)))
    B = reshape(view(v[:,end],       (offset1+1):offset2), (size(t,2), size(t,2)))      
    C = reshape(view(v[:,end], (offset2+1):length(v[:,end])), (size(t,3), size(t,3)))
    return A, B, C
end


function solvelobpcg(t)
    L = sylverlin3(t,[1 1 1])
    dual_primal = LinearMap((x)->L'* (L * x), (x)->L'*(L'*x), size(L,2), size(L,2); ismutating=false, issymmetric=true, isposdef=true)
    @time res = lobpcg(dual_primal, false, 10; maxiter=size(L,2), tol=1e-16)
    res.λ
    res.X
    for v in eachcol(res.X)
     println(norm(L*v))
    end
    return res
end

end # module SylvesterBB