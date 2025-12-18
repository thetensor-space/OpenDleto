
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
    SylvesterSolvers

    Module providing algorithms for solving Sylvester equations arising in chiseling.

    This module consists of two approaches to solving Sylvester equations:
    - Black-box Sylvester solvers that use iterative methods to find solutions without forming large matrices explicitly.
    - Direct Sylvester solvers that construct and solve the Sylvester equations directly.

"""
module SylvesterSolvers

export Derivation, der, sculpt, constraints, constraint

using LinearAlgebra
using Arpack
using SparseArrays
using LinearMaps
using IterativeSolvers
using ProgressMeter
# using Plots
using Statistics
using ..TensorSpaces: Engagement, Primal, Dual, Ambidextrous, Disengaged, act

using ..Chisels: LinearChisel
using ..TransverseOperators: TransverseOps, contains, unsafe_contains, transverse


# ------------------------------ Black-box Sylvester solvers -------------------------------

"""
        SylvesterMap(ops::TransverseOps, C::LinearChisel,)
    
        Constructs a LinearMap representing the Sylvester operator defined by the
        transverse operators `ops` and the chisel `chisel`.

The `SylvesterMap` return is the composite of a pair maps 
 - `ester(mats::Vector{AbstractMatrix})` applies the given matirces as `C`-derivations to the given tensor.
 - `Sylv(s::AbstractArray)` is the dual to `ester`


```math
\text{Sylv}:(\\mathbb{K}^{c\times \\prod_a d_a})^*  \\to  \\prod_{a\in \mathcal{E}} \\mathbb{K}^{d_a x d_a}  \\to \\Omega 
\text{ester}:\\Omega \\to \\prod_{a\in \mathcal{E}} \\mathbb{K}^{d_a x d_a} \\to (\\mathbb{K}^{c\times \\prod_a d_a})^*
````
were ``\\mathcal{E}`` is the set of engaged axes in the chisel, and ``d_a`` is the dimension of axis ``a``.
The composition ``\text{Sylv}\text{ester}`` is a symmetric positive definite operator on ``\\Omega`` whose 
null spaces are the `C`-derivations for the chisel `C`.  The function is a black-box `LinearMap` which 
can be used efficiently in several solvers that use iterative methods.  The degree of efficiency depends 
on the valence and sparsity of the original tensor and the size of the chisel and the number of engaged 
columns.  


        This method dispatches to a number of internal implementations based on the
        structure of the transverse operators and the chisel.  It is presently optimized 
        for low-valence tensors.  The default for higher valence tensors is to use the
        direct Sylvester solver.

        returns `Sylvester::LinearMap`, `Sylv`::LinearMap, `ester::LinearMap`
"""
function SylvesterMap(ops::TransverseOps, chisel::LinearChisel, t::AbstractArray) :: Tuple{LinearMap, LinearMap, LinearMap}
    val = ndims(t)
    ch = chisel.polynomials
    if val == 1
        # Domain: Tensor space (K^d)^* 
        # (1) transverse makes flat to matrix
        # (2) then mult with t'. 
        # (3) Kronecker with ch.
        # Codomain: K^{c x d}
        ester(x) = ch*(t'*transverse(ops, x))

        # Domain: K^{c x d}
        # (1) Contract with ch'
        # (2) mult with t
        # (3) contains flattens matrix to vector
        sylv(y) = unsafe_contains(ops, [ (t*ch')*view(y,(offset+1):(offset+size(t,1))) for offset in 0:size(t,1):(length(ch)-1)*size(t,1) ] |> vcat )

        Sylv = LinearMap( x -> ch*(x*t)', y -> (ch' * y') * t , length(ch)*size(t,1), size(t,1)^2; ismutating=false )
        ester = LinearMap( x -> (ch' * x') * t , y -> ch*(y*t)' , size(t,1)^2, length(ch)*size(t,1); ismutating=false )
        Sylvester = LinearMap( x ->
        apply(x) = ch*(x*t)'
        apply_transpose(y) = (ch' * y') * t 
    
        LinearMap( apply, apply_transpose, length(ch)*size(t,1), size(t,1)^2; ismutating=false )
    elseif val == 2
        return sylverester2(ops, chisel, t)
    elseif val == 3
        return sylverester3(ops, chisel, t)
    elseif val == 4
        return sylverester4(ops, chisel, t)
    end

    # Fall back to direct Sylvester solver for higher valence tensors.
end

# 
# Highly specialized black-box Sylvester solvers for valence 1,2,3 tensors.
#
function sylverester13(s::AbstractArray, chisel::Vector)
    d_a = size(s,1);
    dims = (length(chisel), size(s,1), size(s,2), size(s,3))
    function apply(x) 
        # without copying data, partition the vector and index it as square matrices
        A = reshape(view(x, 1:d_a^2), (d_a, d_a))
        # create an empty tensor of the required size
        t = zeros(eltype(s), dims)
        # call fast tensor contraction algorithms: this is what uses explicit valence.
        @tensor t[c,i,j,k] = chisel[c]*A[i,i'] * s[i',j,k]
        return vec(t)
    end
    function apply_transpose(y)
        # Reshape vec into a tensor of the same shape as s
        t = reshape(y, dims)
        # Compute the contractions for the transpose
        # (You may need to flatten the result at the end)
        # Example using TensorOperations.jl:
        @tensor A′[c,i,i′] := chisel[c]*t[i,j,k] * s[i′,j,k]
        # Flatten and concatenate A′, B′, C′ into a vector
        return vec(A′)
    end
    L = LinearMap( apply, apply_transpose, size(chisel,1)*size(s,1)*size(s,2)*size(s,3), size(s,1)^2; ismutating=false )
    return L
end


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
    svds = LinearAlgebra.svd(M')
    n_vals = min(max, length(svds.S))
    return (;vals=svds.S[end:-1:(end-n_vals+1)], vecs=svds.U[:, end:-1:(end-n_vals+1)])
end;

"""
    LinearAlgebraSVD(M::AbstractMatrix, max::Integer=10)

    Uses LinearAlgebra.svd to compute the smallest singular vectors of M.
    We use at most `max` of the smallest singular values so this method 
    is suboptimal for large systems.  Consider Arpack method for large systems.

    Results returned in reverse order so that smallest singular values are first.
"""
function BlackBoxSVD(M::LinearMap, max::Integer=10) 
    # Compute svd via IterativeSolvers.
    # We only need the right singular vectors as we want a right null space.
    # svdl does partial SVD via Lanczos bidiagonalization, so we need to 
    # ask for a larger number of singular values `nsv` to reach the smallest ones.
    vals = min(size(M))
    S, L = IterativeSolvers.svdl(M; nsv=vals, vecs=:right)
    nvals = min(max, length(S))
    return (;vals=S[end:-1:(end-nvals)], vecs=L[:, end:-1:(end-nvals)])
end;

function LinearAlgebraEigen(M::AbstractMatrix, max::Integer=10) 
    eigens = LinearAlgebra.eigen( LinearAlgebra.Symmetric(M'*M) )
    n_vals = min(max, size(eigens)[1])
    println("Computed $(size(eigens)[1]) eigen values, returning $n_vals smallest.")
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
    
    primal_dual =  LinearAlgebra.Symmetric(M'*M)
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


function group_conjugate_pairs(eigenvals, eigenvecs)
    n = length(eigenvals)
    real_blocks = zeros(real(eltype(eigenvecs)), n, n)
    
    processed = falses(n)
    for i in 1:n
        if processed[i]
            continue
        end
        
        λ = eigenvals[i]
        if isreal(λ)
            # Real eigenvalue
            real_blocks[:, i] = real(eigenvecs[:, i])
            processed[i] = true
        else
            # Find conjugate pair
            conj_idx = findfirst(j -> !processed[j] && isapprox(eigenvals[j], conj(λ)), 1:n)
            if conj_idx !== nothing
                # Create real 2D subspace from conjugate pair
                v = eigenvecs[:, i]
                real_blocks[:, i] = real(v)
                real_blocks[:, conj_idx] = imag(v)
                processed[i] = processed[conj_idx] = true
            end
        end
    end
    
    return real_blocks
end


extractAxisTubes(t::AbstractArray, index::Tuple, engaged::Vector{<:Integer}) = 
     [t[ntuple(i -> i == a ? Colon() : index[i], ndims(t))...] for a in engaged]


"""
Constructs a chisel constraint equation as specified by 
the chisel, the tensor, and the indices.
    
    - the `chisel`
    - the tensor `t`
    - `s``: the chisel equation 
    - `the_is`: the indices for each axis

    Returns a (sparse) vector representing the equation:

    0 = Σ_a Σ_{l_a} λ_{sa} t[i_1,... , l_a, ...,t_val] * X[i_a, l_a]

where the sum is over all axes (modes) a and l_a runs over the dimension of the 
a-th axis.  The primal form assume t is given as input and solves for the matrices X_a.
The dual form assumes the matrices X_a are given and solves for the tensor t.
"""
function constraints(chisel::LinearChisel, 
    t::AbstractArray, 
    the_is::AbstractVector{<:Tuple})
    
    cat = chisel.category
    Ω = chisel.operators
    engaged = findall(e -> e != Disengaged, cat)
    
    # find the number of variables from the chisel and tensor size
    nvars = sum(a -> Ω.dimFormula(a, t), engaged)
    neqns = size(chisel.polynomials, 1)
    
    # initialize the equation matrix for all indices
    M = zeros(eltype(t), (neqns * length(the_is), nvars))
    
    # Process each index tuple
    for (eq_idx, is) in enumerate(the_is)
        # Calculate row offset for this set of equations
        row_offset = (eq_idx - 1) * neqns
        
        # Fetch the terms in the tensor we need in the equation
        tubes = extractAxisTubes(t, Tuple(is), engaged)
        
        # Assemble the equation
        col_offset = 0
        for (idx, a) in enumerate(engaged)
            tube = tubes[idx]
            Ω.insert!(M, row_offset, col_offset, chisel.polynomials[:, a], tube, is[a])
            col_offset += Ω.dimFormula(a, t)
        end
    end
    
    return M
end;

"""
    constraint(chisel::LinearChisel, t::AbstractArray, is::Tuple)

    Build the constraint equations for a single index tuple.
        - the `chisel`
        - the tensor `t`
        - `is`: the index for each axis
"""
function constraint(chisel::LinearChisel, t::AbstractArray, is::Tuple) 
    return constraints(chisel, t, [is])
end;

function constraints(chisel::LinearChisel, t::AbstractArray)
    all_is = vec([Tuple(is) for is in CartesianIndices(t)])
    return constraints(chisel, t, all_is)
end;

# t = reshape(collect(1:24), (2,3,4))
# M = reduce(vcat, [ constraints(uc, t, Tuple(is)) for is in CartesianIndices(t) ])
## Hack to top printing message upon loading
;
# println("Chisels.jl loaded.")

struct Derivation 
    chisel :: LinearChisel
    ops :: TransverseOps
    op :: Vector{Number}
    # mats :: Vector{Matrix}
end

"""
    der(t::AbstractArray, 
        category::Array{Engagement}, 
        nsamples::Integer=10,
        svdfunc::Function=ArpackEigen
        ) :: Vector{Derivation} 

    Computes up to `nsamples` many C-derivations of `t` for the to the given chisel C.
    If `nsamples` is negative or exceeds the dimension of the derivation space
    then the a basis for the derivation space is returned.

    - `C`: a linear chisel
    - `t`: The input tensor
    - `nsamples`: Maximum number of singular vectors to compute (default: 10)
    - `svdfunc`: Function to compute SVD (default: ArpackEigen)

    Returns a vector of derivations.
"""
function der(C::LinearChisel,
    t::AbstractArray, 
    nsamples::Integer=10,
    svdfunc::Function=ArpackEigen
    ) :: Vector{Derivation}
    
    # Sanity checks
    if length(C.category) != ndims(t)
        error("Category length must match tensor valence")
    end

    # Build the constraints.
    M = constraints(C, t) 

    # Chisel off some part of the tensor and collect the spall.
    spall = svdfunc(M, nsamples)

    # Break up the matrices
    engaged = findall( e -> e != Disengaged, C.category )
    ders = Vector{Derivation}()
    offset = 0
    for i in 1:nsamples
        der_i = Vector{Matrix{eltype(t)}, length(engaged)}()
        for a in engaged
            if C.category[a] == Dual
                der_i[a] = C.operators.toMatrix(spall.vecs, size(t,a), offset)' 
            else # Primal or Ambidextrous
                der_i[a] = C.operators.toMatrix(spall.vecs, size(t,a), offset)
            end
            offset += C.operators.dimFormula(a, t)
        end
        push!(ders, Derivation(C, der_i))
    end
    return ders
end

"""
    der(t::AbstractArray, max::Integer=10) :: Spall 

    Chisels off part of the tensor t called the spall, 
    using a default primal chisel category.
    Uses ArpackEigen or LinearAlgebraSVD depending on tensor size.

    - `t`: The input tensor
    - `max`: Maximum number of singular vectors to compute (default: 10)

    Returns a Spall struct containing the singular values and vectors.
"""
function der(t::AbstractArray, nsamples::Integer=10) :: Vector{Derivation} 
    # Convert to floating point to avoid integer conversion issues
    if eltype(t) <: Integer
        println("Converting integer tensor to Float32 for numerical stability.")
        # Float 32 should be sufficient precision for SVD
        # if not user can convert for themselves.
        t = Float32.(t)
    end

    mdim = maximum(size(t))
    if mdim <= 10
        svdfunc = LinearAlgebraSVD
    else
        svdfunc = ArpackEigen
    end
    return der(t, UniversalChisel(ndims(t)), nsamples, svdfunc)
end

"""
    stratify(t::AbstractArray, 
    der::Vector{Derivation},
    pos::Vector{T} where T <: Integer
    ) 

    Sculpt the tensor t using the spall and the specified positions.
    Returns a named tuple with the sculpted tensor and the transforms used.

    - `t`: The input tensor
    - `spall`: The Spall struct obtained from chiseling
    - `pos`: Vector of integer positions indicating which singular vectors to use

    Returns a named tuple with fields:
    - `tensor`: The sculpted tensor
    - `transform`: The list of transformation matrices applied to each mode
"""
function sculpt(t::AbstractArray, 
    ders::Vector{Derivation},
    pos::Vector{T} where T <: Integer
    )
    
    chisel = ders[1].chisel
    category = chisel.category
    engaged = findall( e -> e != Disengaged, category )

    # take a random combo of ders given then diagonalize the list.
    der = ders[1]

    temp = [ LinearAlgebra.eigen(matrices[i]) for i in 1:length(der) ]
    transform = map( eigs -> group_conjugate_pairs(eigs.values, eigs.vectors), temp )
    tensor = act(t, category, transform)
    return (;tensor, transform)
end

end # module SylvesterSolvers