
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

module SylvesterSolvers

export chisel, sculpt, Spall

using LinearAlgebra
using Arpack
using SparseArrays
using ProgressMeter
# using Plots
using Statistics
using ..TensorSpaces: Engagement
using ..Chisels: LinearChisel, UniversalChisel, constraints

# Chisels.jl and TensorSpaces.jl are included by the main Dleto module

struct Spall{T}
    values :: AbstractVector{T}
    vectors :: AbstractMatrix{T}
    chisel :: LinearChisel
end;

#-------------------------------
# technical functions for performing svd and returing the smallest singular vectors 
# these will throw error if there are less than 10 singular vectors

"""
    LinearAlgebraSVD(M::AbstractMatrix, max::Integer=10)

    Uses LinearAlgebra.svd to compute the smallest singular vectors of M.
    We use at most `max` of the smallest singular values so this method 
    is suboptimal for large systems.  Consider Arpack method for large systems.

    Results returned in reverse order so that smallest singular values are first.
"""
function LinearAlgebraSVD(M::AbstractMatrix, max::Integer=10) 
    svds = LinearAlgebra.svd(M)
    n_vals = min(max, length(svds.S))
    println("Computed $(length(svds.S)) singular values, returning $n_vals smallest.")
    return (;vals=svds.S[end:-1:(end-n_vals+1)], vecs=svds.U[:, end:-1:(end-n_vals+1)])
end;

function LinearAlgebraEigen(M::AbstractMatrix, max::Integer=10) 
    eigens = LinearAlgebra.eigen( LinearAlgebra.Symmetric(M*M') )
    n_vals = min(max, size(eigens)[1])
    println("Computed $(size(eigens)[1]) singular values, returning $n_vals smallest.")
    ## WARNING: This produces possibly complex eigen values but should be 
    ## real valued by positivity of M*M'.  We take real part in chisel function.
    vals = [ sqrt.(real.(v)) for v in eigens.values[end:-1:(end-n_vals+1)] ] 
    vecs = [ real.(v) for v in eigens.vectors[:, end:-1:(end-n_vals+1)] ]
    return (;vals, vecs)
end;

"""
    ArpackEigen(M::AbstractMatrix)

    Uses Arpack to compute the smallest singular vectors of M by computing the
    largest eigenvectors of M*M'.  If Arpack fails, it falls back to LinearAlgebraEigen.
"""
function ArpackEigen(M::AbstractMatrix, max::Integer=20) 
    n_vals = min(max, size(M)[1])
    vals = Float64[]
    vecs = Float64[]
    
    primal_dual =  LinearAlgebra.Symmetric(M*M')
    try 
        # println("Computing SVD via Arpack...", max)
        eigens = Arpack.eigs( primal_dual  ; which =:SR, nev =n_vals )
        vals = eigens[1][:, end:-1:(end-n_vals+1)] 
        vecs = eigens[2][:, end:-1:(end-n_vals+1)] 
    catch e
        #sometimes Arpack fails to converge, so we build fall back to LinearAlgebraEigen
        # println("Arpack failed with error: $e. Falling back to LinearAlgebraEigen.")
        eigens = LinearAlgebra.eigen(primal_dual)
        vals = eigens.values[end:-1:(end-n_vals+1)] 
        vecs = eigens.vectors[:, end:-1:(end-n_vals+1)] 
    end
    return (;vals, vecs)
end;



"""
    chisel(t::AbstractArray{T}, 
        category::Array{Engagement}, 
        maxdim::Integer=10,
        svdfunc::Function=ArpackEigen
        ) :: Spall{T} where T <: AbstractFloat

    Chisels off part of the tensor t called the spall, 
    according to the specified chisel category.
    Uses the specified svdfunc to compute the spall of the constraint matrix.

    - `t`: The input tensor
    - `category`: Array of Engagement specifying the chisel
    - `maxdim`: Maximum number of singular vectors to compute (default: 10)
    - `svdfunc`: Function to compute SVD (default: ArpackEigen)

    Returns a Spall struct containing the singular values and vectors.
"""
function chisel(t::AbstractArray{T}, 
    category::Array{Engagement}, 
    maxdim::Integer=10,
    svdfunc::Function=ArpackEigen
    ) :: Spall{T} where T <: AbstractFloat
    
    # Sanity checks
    if length(category) != ndims(t)
        error("Category length must match tensor valence")
    end

    # Build the constraints.
    ⚒ = UniversalChisel(category)
    M = constraints(⚒, t) 

    # Chisel off some part of the tensor and collect the spall.
    spall = svdfunc(M, maxdim)

    # Enter spall analysis.
    return Spall(spall.vals, spall.vecs, ⚒)
end

"""
    chisel(t::AbstractArray, max::Integer=10) :: Spall 

    Chisels off part of the tensor t called the spall, 
    using a default primal chisel category.
    Uses ArpackEigen or LinearAlgebraSVD depending on tensor size.

    - `t`: The input tensor
    - `max`: Maximum number of singular vectors to compute (default: 10)

    Returns a Spall struct containing the singular values and vectors.
"""
function chisel(t::AbstractArray, max::Integer=10) :: Spall 
    # Convert to floating point to avoid integer conversion issues
    if eltype(t) <: Integer
        println("Converting integer tensor to Float32 for numerical stability.")
        # Float 32 should be sufficient precision for SVD
        # if not user can convert for themselves.
        t = Float32.(t)
    end

    category = fill(Primal, ndims(t))
    mdim = maximum(size(t))
    if mdim <= 10
        svdfunc = LinearAlgebraSVD
    else
        svdfunc = ArpackEigen
    end
    return chisel(t, category, max, svdfunc)
end


function sculpt(t::AbstractArray, 
    spall::Spall,
    pos::Vector{T} where T <: Integer
    )

    chisel = spall.chisel
    category = chisel.category
    engaged = findall( e -> e != Disengaged, category )

    # We use all the positions provided to build a random matrix
    # in that span with invertible transforms.
    matrices = [ zeros(eltype(t), size(t,a), size(t,a)) for a in engaged ]
    for p in pos 
        flatmats = spall.vectors[:, p] 
        offset = 0
        for (i, a) in enumerate(engaged)
            mat_a = chisel.operators.toMatrix(flatmats, size(t,a), offset)
            matrices[i] += mat_a
            offset += chisel.operators.dimFormula(i, t)
        end
  
    end
    for i in 1:length(matrices)
        println("Ranks ", rank(matrices[i]) )
    end
    tensor = act(t, category, matrices)
    return (;tensor, matrices)
end

end # module SylvesterSolvers