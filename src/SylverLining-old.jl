
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

"""
    SylverLining

    Module providing black-box Sylvester function for using in black-box Dleto chiselling.
    
    This method constructs a `LinearMap` representing the Sylvester operator defined by the
    transverse operators and the chisel, and the tensor.  It also constructs its 
    dual map and their composition for use in black-box solvers such as SVD, Krylov, and Conjugate Gradient.

    For low-valence tensors (valence 1 to 4) there are optimized implementations that use 
    tensor contraction algorithms for efficiency.  
"""
module SylverLining
using ITensors
# General Sylvester operator using ITensor.jl for arbitrary axes

function asITensor(t::AbstractArray)
    n_axes = ndims(t)
    # Create ITensor Index objects for each axis
    inds = [Index(size(t, i), "a$i") for i in 1:n_axes]
    # Convert t to ITensor
    T = ITensor(inds...)
    T .= t
    return T, inds
end

function sylvester_general_itensor(ch::Matrix, t::ITensor)
    T, inds = asITensor(t)

    # Chisel columns as ITensor vectors
    chisels = [ch[:,i] for i in 1:size(ch,2)]

    # Compute sizes for LinearMap
    ester_out_size = prod(size(t)) * size(ch,1)
    ester_in_size = sum(size(t, i)^2 for i in 1:n_axes)

    # ester: takes a vector of matrices (one per axis), returns a vectorized tensor
    function ester(Xvec)
        Y = ITensor(Index(size(ch,1), "c"), inds...)
        for (i, X) in enumerate(Xvec)
            Xi = ITensor(inds[i]', inds[i])
            Xi .= X
            c = chisels[i]
            C = ITensor(Index(length(c), "c"))
            C .= c
            Y += C * ncon((T, Xi), ([inds...], (inds[i]', inds[i])))
        end
        return vec(Y)
    end

    # sylv: takes a vectorized tensor, returns a vector of matrices (one per axis)
    function sylv(y)
        dims = (size(ch,1), [size(t, i) for i in 1:n_axes]...)
        Y = reshape(y, dims)
        Xs = []
        for i in 1:n_axes
            Yten = ITensor(Index(size(ch,1), "c"), inds...)
            Yten .= Y
            c = chisels[i]
            C = ITensor(Index(length(c), "c"))
            C .= c
            Xi = ncon((T, C, Yten), ([inds...], ("c",), ("c", inds...)))
            push!(Xs, Array(Xi))
        end
        return Xs
    end

    # Compose sylv and ester as in sylvester4
    function sylvester(x)
        return sylv(ester(x))
    end

    # Wrap ester and sylv as LinearMaps
    ester_map = LinearMap(ester, sylv, ester_out_size, ester_in_size; ismutating=false)
    sylv_map = LinearMap(sylvester, ester_in_size, ester_in_size; ismutating=false, issymmetric=true, isposdef=true)
    return sylv_map, ester_map
end
using ITensors

# General Sylvester operator using ITensor.jl for arbitrary axes
function sylvester_general_itensor(ch::Matrix, t::AbstractArray, axes::Vector{Int})
    # Create ITensor Index objects for each axis
    inds = [Index(size(t, ax), "a$ax") for (ax) in axes]
    # Convert t to ITensor
    T = ITensor(inds...)
    T .= t

    # Chisel columns as ITensor vectors
    chisels = [ch[:,i] for i in 1:size(ch,2)]
    
    # Define the Sylvester operator
    function sylvester(xvec)
        # xvec is a vector of matrices, one for each axis
        Xs = xvec
        # Contract each chisel with T along the corresponding axis
        Y = zero(T)
        for (i, X) in enumerate(Xs)
            # Contract along axis i
            # Reshape X to match the index
            Xi = ITensor(inds[i]', inds[i])
            Xi .= X
            # Multiply chisel column
            c = chisels[i]
            C = ITensor(Index(length(c), "c$i"))
            C .= c
            # Contract: C[c] * T[...] * Xi[inds[i]', inds[i]]
            # General contraction: sum over all but one axis
            Y += C * T * Xi
        end
        return vec(Y)
    end

    # Return as a LinearMap (or as needed for your framework)
    return sylvester
end

using LinearMaps
using TensorOperations
using Arpack
using IterativeSolvers
using CUDA
using cuTENSOR

export sylvester
# export sylverlin3, sylverlin1, toMat, solveeig, solveit, cusylverlin3

"""
        sylvester(ops::TransverseOps, C::LinearChisel, Γ::AbstractArray) -> Tuple{LinearMap, LinearMap}
    
        Constructs a LinearMap representing the Sylvester operator defined by the
        transverse operators `ops` and the chisel `chisel`.

The `SylvesterMap` return is the composite of a pair maps 
 - `ester(mats::Vector{AbstractMatrix})` applies the given matirces as `C`-derivations to the given tensor.
 - `Sylv(s::AbstractArray)` is the dual to `ester`


```math
\text{Sylv}:(\\mathbb{K}^{c\times \\prod_a d_a})^*  \\to  \\prod_{a\\in \\mathcal{E}} \\mathbb{K}^{d_a x d_a}  \\to \\Omega 
\text{ester}:\\Omega \\to \\prod_{a\\in \\mathcal{E}} \\mathbb{K}^{d_a x d_a} \\to (\\mathbb{K}^{c\times \\prod_a d_a})^*
````
were ``\\mathcal{E}`` is the set of engaged axes in the chisel, and ``d_a`` is the dimension of axis ``a``.
The composition ``\\text{Sylv}\\text{ester}`` is a symmetric positive definite operator on ``\\Omega`` whose 
null spaces are the `C`-derivations for the chisel `C`.  The function is a black-box `LinearMap` which 
can be used efficiently in several solvers that use iterative methods.  The degree of efficiency depends 
on the valence and sparsity of the original tensor and the size of the chisel and the number of engaged 
columns.  


        This method dispatches to a number of internal implementations based on the
        structure of the transverse operators and the chisel.  It is presently optimized 
        for low-valence tensors.  The default for higher valence tensors is to use the
        direct Sylvester solver.

        returns `Sylvester::LinearMap`, `ester::LinearMap` (with built-in transpose `Sylv`)
"""
function sylvester(Ω::TransverseOps, P::LinearChisel, Γ::AbstractArray) :: Tuple{LinearMap, LinearMap}
    val = ndims(Γ)
    C = P.polynomials
    if val == 1
        return sylverester1(Ω, C, Γ)
    elseif val == 2
        return sylverester2(Ω, C, Γ)
    elseif val == 3
        return sylverester3(Ω, C, Γ)
    elseif val == 4
        return sylverester4(Ω, C, Γ)
    end

    # Fall back to direct Sylvester solver for higher valence tensors.
    return nothing # for now just get things working again.

    # Universal chisel der map.
    function ester(x) 
        X = transverse(ops, x)[1]
        # Overkill to use @tensor here but this is substituted later 
        # with BLAS calls for efficiency.  Also makes it more clear 
        # what is going on in higher valence cases below.
        @tensor y[c,i] = ch[c]*t[i']*X[i,i']
        return vec(y)
    end


    der_map(x) = sum( [ x[a]* Γ for a in length(x)] )
    # Universal chisel dual map.
    densor_map(y) = [ ]
    return 
end

#----------- Sylvester linear map ----------------------------------------

## Valence 1 (vector) case
function sylvester1(ch::Matrix,t::AbstractArray)
    # Domain: Tensor space (K^d)^* 
    # (1) transverse makes flat to matrix
    # (2) then mult with t'. 
    # (3) Kronecker with ch.
    # Codomain: K^{c x d}
    # @tensor cond[c,a] = ch[c]*t[a] # not worth making a big tensor 
    function ester(x) 
        X = transverse(ops, x)[1]
        # Overkill to use @tensor here but this is substituted later 
        # with BLAS calls for efficiency.  Also makes it more clear 
        # what is going on in higher valence cases below.
        @tensor y[c,i] = ch[c]*t[i']*X[i,i']
        return vec(y)
    end

    # Domain: K^{c x d}
    # (1) Contract with ch'
    # (2) mult with t
    # (3) contains flattens matrix to vector
    function sylv(y)
        Y = reshape(y, (size(ch,1), size(t,1)))
        @tensor z[i,i'] = t[i]*ch[c]*Y[c,i']
        return unsafe_contains(ops, collect(z))
    end

    # Compose sylv  ester
    function sylvester(x)
        X = transverse(ops, x)[1]
        @tensor z[a] = t[a']*ch[c]*ch[c]*t[a]*X[a]
        return unsafe_contains(ops, z)
    end
    #TBD: for convergence we should work on conditioning ch' ch,
    # something like a Grassmannian Frames (tight frames)--see Emily King.

    e = LinearMap( ester, sylv, size(t,1)^2, size(ch,1)*size(t,1); ismutating=false )
    se = LinearMap( sylvester,  size(t,1)^2, size(t,1)^2; ismutating=false, issymmetric=true, isposdef=true )
    return se, e
end

## Valence 2 (matrix) case
function sylvester2(ch::Matrix,t::AbstractArray)
    C1 = ch[:,1]
    C2 = ch[:,2]
    # Domain: Tensor space (K^{d1 x d2})^* 
    # (1) transverse makes flat to matrix
    # (2) then mult with t on left and right 
    # (3) Kronecker with ch.
    # Codomain: K^{c x d1 x d2}
    function ester(x) 
        X = transverse(ops, x)
        # Chisels are tiny, no point in using views.
        @tensor Y[c,i,j] = C1[c]*t[i',j]*(X[1])[i,i'] + C2[c]*t[i,j']*(X[2])[j,j']
        return vec(Y)
    end

    # Domain: K^{c x d1 x d2}
    # (1) Contract with ch'
    # (2) mult with t
    # (3) contains flattens matrix to vector
    function sylv(y)
        Y = reshape(y, (size(ch,1), size(t,1), size(t,2)))
        @tensor X1[i,i'] = t[i,j] * C1[c] * Y[c,i',j]
        @tensor X2[j,j'] = t[i,j] * C2[c] * Y[c,i,j'] 
        return unsafe_contains(ops, collect(X1,X2))
    end

    # Compose sylv  ester
    function sylvester(x)
        X = transverse(ops, x)
        @tensor Y[c,i,j] = C1[c]*t[i',j]*(X[1])[i,i'] + C2[c]*t[i,j']*(X[2])[j,j']
        @tensor X1[i,i'] = t[i,j] * C1[c] * Y[c,i',j]
        @tensor X2[j,j'] = t[i,j] * C2[c] * Y[c,i,j'] 
        return unsafe_contains(ops, collect(X1,X2))
    end
    #TBD: for convergence we should work on conditioning ch' ch,
    # something like a Grassmannian Frames (tight frames)--see Emily King.

    e = LinearMap( ester, sylv, size(t,1)^2, size(ch,1)*size(t,1); ismutating=false )
    se = LinearMap( sylvester,  size(t,1)^2, size(t,1)^2; ismutating=false, issymmetric=true, isposdef=true )
    return se, e
end

## Valence 3 tensor
function sylvester3(ch::Matrix,t::AbstractArray)
    C1 = ch[:,1]
    C2 = ch[:,2]
    C3 = ch[:,3]
    function ester(x) 
        # X = transverse(ops, x)
        offset = 0; 
        X1 = reshape(view(x[1:(dims[1]*dims[1])]), (dims[1], dims[1]))
        offset += dims[1]*dims[1]
        X2 = reshape(view(x[(offset+1):(offset + dims[2]*dims[2])]), (dims[2], dims[2]))
        offset += dims[2]*dims[2]
        X3 = reshape(view(x[(offset+1):(offset + dims[3]*dims[3])]), (dims[3], dims[3]))
        # Chisels are tiny, no point in using views.
        @tensor Y[c,i,j,k] = C1[c]*t[i',j,k]*X1[i,i'] + C2[c]*t[i,j',k]*X2[j,j'] + C3[c]*t[i,j,k']*X3[k,k']
        return vec(Y)
    end
    function sylv(y)
        Y = reshape(y, (size(ch,1), size(t,1), size(t,2), size(t,3)))
        @tensor X1[i,i'] = t[i,j,k] * C1[c] * Y[c,i',j,k]
        @tensor X2[j,j'] = t[i,j,k] * C2[c] * Y[c,i,j',k] 
        @tensor X3[k,k'] = t[i,j,k] * C3[c] * Y[c,i,j,k']
        return [vec(M) for M in mats] |> vcat
        # return unsafe_contains(ops, collect(X1,X2,X3))
    end

    # Compose sylv  ester
    function sylvester(x)
        # X = transverse(ops, x)
        offset = 0; 
        X1 = reshape(view(x,1:(dims[1]*dims[1])), (dims[1], dims[1]))
        offset += dims[1]*dims[1]
        X2 = reshape(view(x,(offset+1):(offset + dims[2]*dims[2])), (dims[2], dims[2]))
        offset += dims[2]*dims[2]
        X3 = reshape(view(x,(offset+1):(offset + dims[3]*dims[3])), (dims[3], dims[3]))
        @tensor Z1[i,i'] = t[i,j] * C1[c] * C1[c]*t[i',j]*X1[i,i'] + t[i,j] * C1[c]*C2[c]*t[i,j']*X2[j,j']+ t[i,j] * C1[c]*C3[c]*t[i,j']*X3[j,j']
        @tensor Z2[j,j'] = t[i,j] * C2[c] * C1[c]*t[i',j]*X1[i,i'] + t[i,j] * C2[c]*C2[c]*t[i,j']*X2[j,j']+ t[i,j] * C2[c]*C3[c]*t[i,j']*X3[j,j']
        @tensor Z3[k,k'] = t[i,j] * C3[c] * C1[c]*t[i',j]*X1[i,i'] + t[i,j] * C3[c]*C2[c]*t[i,j']*X2[j,j']+ t[i,j] * C3[c]*C3[c]*t[i,j']*X3[j,j']
        # @tensor Y[c,i,j] = C1[c]*t[i',j]*X1[i,i'] + C2[c]*t[i,j']*X2[j,j']+ C3[c]*t[i,j']*X3[j,j']
        # @tensor X1[i,i'] = t[i,j] * C1[c] * Y[c,i',j]
        # @tensor X2[j,j'] = t[i,j] * C2[c] * Y[c,i,j'] 
        return vcat([vec(Z1), vec(Z2), vec(Z3)])
    end
    #TBD: for convergence we should work on conditioning ch' ch,
    # something like a Grassmannian Frames (tight frames)--see Emily King.

    e = LinearMap( ester, sylv, size(t,1)^2+size(t,2)^2+size(t,3)^2, size(ch,1)*size(t,1); ismutating=false )
    se = LinearMap( sylvester,  size(t,1)^2+size(t,2)^2+size(t,3)^2, size(t,1)^2+size(t,2)^2+size(t,3)^2; ismutating=false, issymmetric=true, isposdef=false )
    return se, e
end


## Valence 4 tensor
function sylvester4(ch::Matrix,t::AbstractArray)
    C1 = ch[:,1]
    C2 = ch[:,2]
    C3 = ch[:,3]
    C4 = ch[:,4]
    function ester(x) 
        X = transverse(ops, x)
        # Chisels are tiny, no point in using views.
        @tensor Y[c,i,j,k,l] = C1[c]*t[i',j,k,l]*(X[1])[i,i'] + C2[c]*t[i,j',k,l]*(X[2])[j,j'] 
                            + C3[c]*t[i,j,k',l]*(X[3])[k,k'] + C4[c]*t[i,j,k,l']*(X[4])[l,l']
        return vec(Y)
    end
    function sylv(y)
        Y = reshape(y, (size(ch,1), size(t,1), size(t,2), size(t,3), size(t,4)))
        @tensor X1[i,i'] = t[i,j,k,l] * C1[c] * Y[c,i',j,k,l]
        @tensor X2[j,j'] = t[i,j,k,l] * C2[c] * Y[c,i,j',k,l] 
        @tensor X3[k,k'] = t[i,j,k,l] * C3[c] * Y[c,i,j,k',l]
        @tensor X4[l,l'] = t[i,j,k,l] * C4[c] * Y[c,i,j,k,l']
        return unsafe_contains(ops, collect(X1,X2,X3,X4))
    end

    # Compose sylv  ester
    function sylvester(x)
        X = transverse(ops, x)
        @tensor Y[c,i,j] = C1[c]*t[i',j]*(X[1])[i,i'] + C2[c]*t[i,j']*(X[2])[j,j']
        @tensor X1[i,i'] = t[i,j] * C1[c] * Y[c,i',j]
        @tensor X2[j,j'] = t[i,j] * C2[c] * Y[c,i,j'] 
        return unsafe_contains(ops, collect(X1,X2))
    end
    #TBD: for convergence we should work on conditioning ch' ch,
    # something like a Grassmannian Frames (tight frames)--see Emily King.

    e = LinearMap( ester, sylv, size(t,1)^2, size(ch,1)*size(t,1); ismutating=false )
    se = LinearMap( sylvester,  size(t,1)^2, size(t,1)^2; ismutating=false, issymmetric=true, isposdef=true )
    return se, e
end


# #----------- Black-box Sylvester solvers --------------------------------------
# function sylverlin3(s::AbstractArray, chisel::Matrix)
#     # Make black-box linear map.
#     d1 = size(s,1);
#     d2 = size(s,2);
#     d3 = size(s,3);
#     dims = (size(chisel,1), size(s,1), size(s,2), size(s,3))
#     function apply(x) 
#         # without copying data, partition the vector and index it as square matrices
#         A = reshape(view(x, 1:d1^2),          (d1, d1))
#         B = reshape(view(x, (d1^2+1):(d1^2+d2^2)), (d2, d2))
#         C = reshape(view(x, (d1^2+d2^2+1):(d1^2+d2^2+d3^2)), (d3, d3))
#         # create an empty tensor of the required size
#         # t = similar(s)
#         t = zeros(eltype(s), dims)
#         # call fast tensor contraction algorithms.
#         @tensor t[c,i,j,k] += (chisel[:,1])[c] * A[i,i'] * s[i',j,k] + (chisel[:,2])[c] * B[j,j'] * s[i,j',k] + (chisel[:,3])[c] * C[k,k'] * s[i,j,k']
#         return vec(t)
#     end
#     function apply_transpose(y)
#         # Reshape vec into a tensor of the same shape as s
#         Y = reshape(y, dims)
#         # Compute the contractions for the transpose
#         # (You may need to flatten the result at the end)
#         # Example using TensorOperations.jl:
#         @tensor A′[ i,i′] := (chisel[:,1])[c]*Y[c,i,j,k] * s[i′,j,k]
#         @tensor B′[ j,j′] := (chisel[:,2])[c]*Y[c,i,j,k] * s[i,j′,k]
#         @tensor C′[ k,k′] := (chisel[:,3])[c]*Y[c,i,j,k] * s[i,j,k′]
#         # Flatten and concatenate A′, B′, C′ into a vector
#         return vcat(vec(A′), vec(B′), vec(C′))
#     end
#     L = LinearMap( apply, apply_transpose, size(chisel,1)*size(s,1)*size(s,2)*size(s,3), size(s,1)^2 + size(s,2)^2 + size(s,3)^2; ismutating=false )
#     return L
# end


# function cusylverlin3(s::CuArray, chisel::Matrix)
#     # Make black-box linear map.
#     d1 = size(s,1);
#     d2 = size(s,2);
#     d3 = size(s,3);
#     dims = (size(chisel,1), size(s,1), size(s,2), size(s,3))
#     function apply(x) 
#         # without copying data, partition the vector and index it as square matrices
#         A = reshape(view(x, 1:d1^2),          (d1, d1))
#         B = reshape(view(x, (d1^2+1):(d1^2+d2^2)), (d2, d2))
#         C = reshape(view(x, (d1^2+d2^2+1):(d1^2+d2^2+d3^2)), (d3, d3))
#         # create an empty tensor of the required size
#         # t = similar(s)
#         t = zeros(eltype(s), dims)
#         # call fast tensor contraction algorithms.
#         @cutensor t[c,i,j,k] += (chisel[:,1])[c] * A[i,i'] * s[i',j,k] + (chisel[:,2])[c] * B[j,j'] * s[i,j',k] + (chisel[:,3])[c] * C[k,k'] * s[i,j,k']
#         return vec(t)
#     end
#     function apply_transpose(y)
#         # Reshape vec into a tensor of the same shape as s
#         t = reshape(y, dims)
#         # Compute the contractions for the transpose
#         # (You may need to flatten the result at the end)
#         # Example using TensorOperations.jl:
#         @cutensor A′[ i,i′] := (chisel[:,1])[c]*t[c,i,j,k] * s[i′,j,k]
#         @cutensor B′[ j,j′] := (chisel[:,2])[c]*t[c,i,j,k] * s[i,j′,k]
#         @cutensor C′[ k,k′] := (chisel[:,3])[c]*t[c,i,j,k] * s[i,j,k′]
#         # Flatten and concatenate A′, B′, C′ into a vector
#         return vcat(vec(A′), vec(B′), vec(C′))
#     end
#     L = LinearMap( apply, apply_transpose, size(chisel,1)*size(s,1)*size(s,2)*size(s,3), size(s,1)^2 + size(s,2)^2 + size(s,3)^2; ismutating=false )
#     return L
# end


# function toMat(L)
#     nrows, ncols = size(L)
#     M = zeros(Float64, nrows, ncols)
#     for j in 1:ncols
#         e_j = zeros(Float64, ncols)
#         e_j[j] = 1.0
#         M[:, j] = L * e_j
#     end
#     return M
# end

# function solveeig(t)
#     L = sylverlin(t)
#     vals, vecs, info = svdl(L) #; nsv=10)
#     # vals, vecs, info = eigs(L*L'; nev=10, which=:SM)
#     if info != 0
#         println("Arpack did not converge, info = $info")
#         println("Falling back to IterativeSolvers.svdl")
#         vals, vecs = svdl(L; nsv=10, which=:SR)
#     end
#     offset1 = size(t,1)^2
#     offset2 = offset1 + size(t,2)^2
#     A = reshape(view(x,                     1:offset1),           (size(t,1), size(t,1)))
#     B = reshape(view(x,       (offset1+1):offset2), (size(t,2), sizein(t,2)))
#     C = reshape(view(x, (offset2+1):length(x)), (size(t,3), size(t,3)))
#     return A, B, C
# end

# function solveit(t)
#     L = sylverlin(t)
#     u, s, v = svdl(L; nsv=10, which=:SM)
#     offset1 = size(t,1)^2
#     offset2 = offset1 + size(t,2)^2
#     A = reshape(view(v[:,end],                     1:offset1),           (size(t,1), size(t,1)))
#     B = reshape(view(v[:,end],       (offset1+1):offset2), (size(t,2), size(t,2)))      
#     C = reshape(view(v[:,end], (offset2+1):length(v[:,end])), (size(t,3), size(t,3)))
#     return A, B, C
# end


end # module SylvesterBB