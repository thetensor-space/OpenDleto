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

using SparseArrays

include("Chisel.jl")


#-------------------------------
# technical functions for performing svd and returing the smallest singular vectors 
# these will throw error if there are less than 10 singular vectors
function LinearAlgebraSVD(M::AbstractMatrix, max::Integer=10)
    svds = LinearAlgebra.svd(M)
    return svds.U[:,end:-1:end-max]
end;

function LinearAlgebraEigen(M::AbstractMatrix, max::Integer=10)
    eigens = LinearAlgebra.eigen( LinearAlgebra.Symmetric(M*M') )
    return eigens.vectors[:,1:max]
end;

#sometimes this crashes so we build fall back to LinearAlgebraEigen
function ArpackEigen(M::AbstractMatrix)
    try 
        eigens = Arpack.eigs( LinearAlgebra.Symmetric(M*M')  ; which =:SR, nev =20)
        # @show eigens[1]
        return eigens[2]
    catch e
        return LinearAlgebraEigen(M)
    end
end;



"""
The spall are the bits thrown off when chiseling.  Accordingly, 
this function constructs a constraint equation as specified by 
the chisel, the tensor, and the indices.
    
    - the chisel 
    - the tensor t
    - s: the chisel equation 
    - i_a: the indices for each axis

    Returns a (sparse) vector representing the equation:

    0 = Σ_a Σ_{l_a} λ_{sa} t[... , l_a, ...] * X[i_a, l_a]

where the sum is over all axes (modes) a and l_a runs over the dimension of the 
a-th axis.  The primal form assume t is given as input and solves for the matrices X_a.
The dual form assumes the matrices X_a are given and solves for the tensor t.
"""
function spall(chisel::Chisel, t::AbstractArray{T,val}, s::Integer, is::Vector{Integer,val})
    num_vars = 0
    for a in chisel.engaged 
        num_vars += chisel.dim(size(t)[a],a)
    end

    # a sparse vector to hold the equation
    the_spall = spzeros(T, num_vars)
    
    for a in chisel.engaged  # loop over all engaged modes
        for l_a in chisel.var_iterator(a, size(t)[a])  
            # Create index for tensor access
            tensor_idx = copy(is)
            tensor_idx[a] = l_a
            
            # Determine variable index in the_spall
            var_index = chisel.var_index(a, l_a, size(t))
            
            # Add contribution to equation
            the_spall[var_index] += t[tensor_idx...] * chisel.chisel[s, a]
        end
        for l_a in 1:size(t, a)  # loop over dimension of mode a
            # Create index for tensor access
            tensor_idx = copy(i_a)
            tensor_idx[a] = l_a
            
            # Add contribution to equation
            equation_value += t[tensor_idx...] * X[a][i_a[a], l_a]
        end
    end
    
    return equation_value
end

#-------------------------------
# technical function for building the linear system 
# use with caution, there are no checks for consistency
function buildFullLinearSystem(t::AbstractArray{T}, eqMatrix::AbstractMatrix)::Matrix{T} where T
    sizes = [size(t)...]
    Msize = size(eqMatrix)
    blocks = sizes .|> (n -> n*n)
    numvars =  sum(i -> blocks[i], 1: Msize[2])
    M = zeros(T, ( numvars, Msize[1] * length(t) )  )
    k=0
    println("\tSizes: ", size(M))
    R = CartesianIndices(t)
    for ci in R                            #  loop over entries of tensor
        li = LinearIndices(t)[ci]
        for i = 1:Msize[1] 
            s=0
            for j = 1:Msize[2]                
                # extract 1 dimensional slice of the tensor
                first = li - (ci[j] - 1)*stride(t,j)
                last = first + (sizes[j]- 1)*stride(t,j)
                slice = t[first:stride(t,j):last]
                # add it to the condition
                modifyRow!( M, numvars*k + s, sizes[j], ci[j], slice, eqMatrix[i,j] )
                s += blocks[j]
            end
            k += 1
        end
    end
    return M
end


#-------------------------------
# technical function for building the linear system 
# use with caution, there are no checks for consistency
function buildLinearSystem(t::AbstractArray{T}, eqMatrix::AbstractMatrix)::Matrix{T} where T
    sizes = [size(t)...]
    Msize = size(eqMatrix)
    blocks = sizes  .|> (n -> n*(n+1)÷ 2) 
    numvars =  sum(i -> blocks[i], 1: Msize[2])
    M = zeros( T, ( numvars, Msize[1] * length(t) )  )
    # println("\tNumber of blocks: ", Msize)
    # println("\tNumber of variables: ", numvars)
    k=0
    # println("\tSizes: ", size(M))
    R = CartesianIndices(t)
    # println("Number of coordinates: ", length(R))
    # println("loops to do: ", Msize[1] * Msize[2] * length(R))
    for ci in R                            #  loop over entries of tensor
        li = LinearIndices(t)[ci]
        for i = 1:Msize[1] 
            s=0
            for j = 1:Msize[2]                
                # extract 1 dimensional slice of the tensor
                first = li - (ci[j] - 1)*stride(t,j)
                last = first + (sizes[j]- 1)*stride(t,j)
                slice = t[first:stride(t,j):last]
                # add it to the condition
                modifyRow!( M, numvars*k + s, sizes[j], ci[j], slice, eqMatrix[i,j] )
                s += blocks[j]
            end
            k += 1
        end
    end
    return M
end

#-------------------------------
# technical function for building the linear system 
# use with caution, there are no checks for consistency
# t[i,j,k]*(X[i,i] + Y[j,j] + Z[k,k]) =0
function buildDiagonalSystem(t::AbstractArray{T})::Matrix{T} where T
    sizes = [size(t)...]
    println("Sizes: ", sizes)
    numvars = sum(sizes)  # Total number of diagonal entries across X, Y, Z
    
    # Count non-zero entries to determine number of equations
    # non_zero_indices = findall(x -> abs(x) > 1e-12, t)
    num_equations = length(t)
    M = zeros(T, (numvars, num_equations))
    # M = zeros(Float64, (numvars, num_equations))
    println("Numvars: ", numvars)
    println("Matrix size: ", size(M))
    
    eq_idx = 1
    for ci in CartesianIndices(t)
        t_val = t[ci]
        
        # X[i,i] coefficient (first sizes[1] variables)
        M[ci[1], eq_idx] = t_val
        
        # Y[j,j] coefficient (next sizes[2] variables)
        M[sizes[1] + ci[2], eq_idx] = t_val
        
        # Z[k,k] coefficient (last sizes[3] variables)
        M[sizes[1] + sizes[2] + ci[3], eq_idx] = t_val
        
        eq_idx += 1
    end
    
    return M
end

function stratify(t::AbstractArray{T}, svdfunc::Function=ArpackEigen) where T
    # test valancy
    if ndims(t) != 3
        throw(DimensionMismatch("wrong arity of tensor"))
    end
    sizes = [size(t)...]
    blocks = sizes  .|> (n -> n*n ) 

        # Determine the float type from the tensor
    TensorType = eltype(t)
    if !(TensorType <: AbstractFloat)
        TensorType = Float64  # fallback for non-float tensors
    end

    # set up system of lin equation
    # println("\r\n\tBuilding linear system...")
    @time M = buildFullLinearSystem(t, SurfaceMatrix)

    # do SVD and pick the smallest vectors 
    # println("\r\n\tComputing singular vectors for ", size(M), "...\n\t")
        # @time lastsvds = svd(M)
    @time lastsvds = svdfunc(M)

    # println("\r\n\tExtracting matrices...")
    # exctract the correct vector
    maineigenvector = lastsvds[:,3]

    # expand to matrices
    @time XMatrix = expandToMatrix(maineigenvector, sizes[1], 0)
    @time YMatrix = expandToMatrix(maineigenvector, sizes[2], blocks[1])
    @time ZMatrix = expandToMatrix(maineigenvector, sizes[3], blocks[1] + blocks[2])

    return changeTensor(t, XMatrix, YMatrix, ZMatrix)
end;
