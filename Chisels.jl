#
# Strata Dleto: Chisels
#   Creation and adaptation of chisels for tensor decomposition.
#
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
#-----------------------------------------------------------------------------



struct Chisel{T}
    polynomials :: Array{T,2}
    engaged :: Vector{Integer}
    dimFormula :: Function
end

function universalSpall!(M)
    # placeholder
end

function spall(chisel::Chisel, 
    t::AbstractArray{T,v}, 
    engaged::Vector{Integer}
    is::Vector{<:Integer}) where {T,v}

    # find the number of variables from the chisel and tensor size
    nvars = sum(a -> chisel.dimFormula(a, size(t)[a]), engaged)
    neqns = size(chisel.polynomials, 1)
    # initialize the equation matrix
    M = zeros(T, ( nvars, neqns ) )

    # equation polynomial in the chisel makes one equation.
    for s in 1:neqns

                        # extract 1 dimensional slice of the tensor
        first = li - (ci[j] - 1)*stride(t,j)
        last = first + (sizes[j]- 1)*stride(t,j)
                slice = t[first:stride(t,j):last]
                # add it to the condition
                modifyRow!( M, numvars*k + s, sizes[j], ci[j], slice, eqMatrix[i,j] )

        eqMatrix = spall(chisel, t, s, engaged)
        chisel.spall!(M,
    end
    # Linearize the index
    lin_is = LinearIndices(t)[is]
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
    return M
end
"""
    Create a universal chisel with selected engaged axes.
"""
function UniversalChisel(valence::Integer, engaged::Vector{<:Integer})
    mat = zeros(Float16, 1, valence)
    for a in engaged
        mat[1,a] = 1.0
    end
    function if_engaged(a,n)
        if a in engaged
            return n*n
        else
            return 0
        end
    end
    return Chisel{Float16}(mat, engaged, if_engageda)
end

"""
    Create a universal chisel with all axes engaged.
"""
function UniversalChisel(valence::Integer)
    return UniversalChisel(valence, collect(1:valence))
end

# ch = UniversalChisel(3)
# ch = UniversalChisel(5, [1,3])

"""
    Create an orthogonal chisel with selected engaged axes.
"""
function OrthogonalChisel(valence::Integer, engaged:: Vector{<:Integer})
    mat = ones(Float16, 1, valence)
        function if_engaged(a,d)
        if a in engaged
            return d*(d+1) ÷ 2
        else
            return 0
        end
    end
    return Chisel{Float16}(mat, engaged, if_engaged)
end

"""
    Create an orthogonal chisel with selected engaged axes.
"""
function OrthogonalChisel(valence::Integer)
    return OrthogonalChisel(valence, collect(1:valence))
end

"""
    Create a Tucker chisel with selected engaged axes.
"""
function TuckerChisel(valence::Integer, engaged:: Vector{Integer})
    mat = zeros(Float16, length(engaged), valence)
        function if_engaged(n,a)
        if a in engaged
            return n*n
        else
            return 0
        end
    end
    return Chisel(mat, engaged, if_engaged)
end
"""
    Create a Tucker chisel with all axes engaged.  
"""
function TuckerChisel(valence::Integer)
    return TuckerChisel(valence, collect(1:valence))
end

"""
    Create an adjoint chisel with specified primal and dual axes.
"""
function AdjointChisel(valence::Integer, primal::Integer, dual::Integer)
    mat = zeros(Float16, length(engaged), valence)
    function if_engaged(a,d)
        if a == primal || a == dual
            return d*d
        else
            return 0
        end
    end
    return Chisel(mat, engaged, if_engaged)
end

"""
Constructs a chisel constraint equation (the "spall") as specified by 
the chisel, the tensor, and the indices.
    
    - the `chisel`
    - the tensor `t`
    - `s``: the chisel equation 
    - `is`: the indices for each axis

    Returns a (sparse) vector representing the equation:

    0 = Σ_a Σ_{l_a} λ_{sa} t[i_1,... , l_a, ...,t_val] * X[i_a, l_a]

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
