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

include("TensorSpace.jl") # Imports Tensor categories.

#-------------------------------Chisel Type-------------------------------------

"""
    Ops

    An internal representation of the operators on the engaged axes.
    This is a flexible interface to allow different types of chisels
    domains.  For example, if the domain is to be full or symmetric 
    matrices than the space defined modifies the equation constraints
    to model them in that subspace rather than adding them as furhter 
    constraint equations.  This lowers the dimension and improves 
    performance.
"""
struct Ops 
    dimFormula :: Function
    toVar :: Function
    insert :: Function
    description :: String
end;



"""
    LinearChisel{T}  
    A linear chisel structure for tensor decomposition constraints.

    Access chisels through methods below.
"""
struct LinearChisel{T<:Number}
    category :: Array{Engagement}
    polynomials :: Array{T,2}
    operators :: Ops
end;

"""
    Print a linear chisel.
"""
function Base.show(io::IO, ::MIME"text/plain", chisel::LinearChisel)
    print(io, "Linear chisel category: ")
    Base.show(io, MIME("text/plain"), collect(chisel.category))
    println(io)
    println(io, chisel.polynomials)
    print(io, chisel.operators.description)
end;

#---------------Universal Chisel-----------------------------------------------
"""
    Create a universal operators with selected engaged axes.
"""
function UniversalOps(category::Array{Engagement}, poly::Array{T,2}) where T
    engaged = findall(e -> e != Disengaged, category)
    uni_dims(a,t) = a in engaged ? size(t,a)*size(t,a) : 0
    # Flatten and strided indexing
    uni_var(a,t,i) = 1+(i-1) * size(t,a)
    uni_insert(a,tube, i_a) = poly[:, a] .* tube'
    return Ops(uni_dims, uni_var, uni_insert, "Universal Operators")
end;

"""
    Create a universal chisel with selected engaged axes.
"""
function UniversalChisel(category::Array{Engagement})
    mat = zeros(Number, 1, length(category))
    for (idx, a) in enumerate(category)
        if a != Disengaged
            mat[1,idx] = 1
        end
    end
    # Each engaged operator is a square matrix

    return LinearChisel(category, mat, UniversalOps(category, mat))
end;


"""
    Create a universal chisel with all specified primal and dual axes.
    Note that if an axis is specified in both primal and dual lists,
    then it recieves teh primal designation. The primal and dual values 
    must be in the range 1 to valence.
"""
function UniversalChisel(valence::Integer, primals::Vector{<:Integer}, duals::Vector{<:Integer})
    cat = fill(Disengaged, valence)
    for a in duals
        cat[a] = Dual
    end
    for a in primals
        cat[a] = Primal
    end
    return UniversalChisel(cat)
end;

"""
    Create a universal chisel with all axes engaged.
"""
function UniversalChisel(valence::Integer)
    cat = fill(Primal, valence)
    return UniversalChisel(cat)
end;

# ch = UniversalChisel(3)
# ch = UniversalChisel(5, [1,3])


#-------------------------------Tucker Chisel----------------------------------
"""
    Create a Tucker chisel with selected engaged axes.
"""
function TuckerChisel(valence::Integer, engaged::Vector{<:Integer})
    cat = fill(Disengaged, valence)
    for a in engaged
        cat[a] = Primal
    end
    mat = zeros(Number, length(engaged), valence)
    for (idx, a) in enumerate(engaged)
        mat[idx, a] = 1
    end
    return LinearChisel(cat, mat, UniversalOps(cat, mat))
end;

"""
    Create a Tucker chisel with all axes engaged.  
"""
function TuckerChisel(valence::Integer)
    return TuckerChisel(valence, collect(1:valence))
end;

#-------------------------------Centroid Chisel--------------------------------
"""
    Create a centroid chisel with specified engaged axes.
"""
function CentroidChisel(valence::Integer, engaged::Vector{<:Integer})
    cat = fill(Disengaged, valence)
    for a in engaged
        cat[a] = Ambidextrous
    end
    mat = zeros(Number, length(engaged)*(length(engaged)-1) ÷ 2, valence)
    row = 1
    for primal in 1:length(engaged)
        for dual in (primal+1):length(engaged)
            mat[row,engaged[primal]] += 1
            mat[row,engaged[dual]] += -1
            row += 1
        end
    end
    return LinearChisel(cat, mat, UniversalOps(cat, mat))
end;

function CentroidChisel(valence::Integer)
    return CentroidChisel(valence, collect(1:valence))
end;

#-------------------------------Adjoint Chisel---------------------------------
"""
    Create an adjoint chisel with specified primal and dual axes.
"""
function AdjointChisel(valence::Integer, primal::Integer, dual::Integer)
    cat = fill(Disengaged, valence)
    cat[dual] = Dual
    cat[primal] = Primal
    mat = zeros(Number, 1, valence)
    mat[1,primal] += 1
    mat[1,dual] += -1
    return LinearChisel(cat, mat, UniversalOps(cat, mat))
end;


#-------------------------------Orthogonal Chisel------------------------------

# Position of (i,j) in lower triangular d×d matrix (i≥j)
sym_pos(d,i,j) = i >= j ? (i-1)*i ÷ 2 + j : (j-1)*i ÷ 2 + i

# Inverse: given position k in lower triangular storage, return (i,j)
function sym_index(d, k)
    # Row i starts at position (i-1)*i/2 + 1
    # Find i such that (i-1)*i/2 < k ≤ i*(i+1)/2
    i = ceil(Int, (-1 + sqrt(1 + 8*k)) / 2)
    j = k - (i-1)*i ÷ 2
    return (i, j)
end

# Position of (i,j) in upper triangular d×d matrix (i≤j)
upper_tri_pos(d,i,j) = (i-1)*(2*d-i) ÷ 2 + (j-i+1)

"""
    Create symmetric operators with selected engaged axes.
"""
function SymmetricOps(category::Array{Engagement}, poly::Array{T,2}) where T
    engaged = findall(e -> e != Disengaged, category)
    sym_dims(a,t) = a in engaged ? size(t,a)*(size(t,a)+1) ÷ 2 : 0
    # Flatten and strided indexing - for row i, start position (1-indexed) is 1+(i-1)*i/2
    sym_var(a,t,i) = 1+((i-1)*i) ÷ 2
    # Take tube and put in tube[x] in position stride(i_a,x) for x 1 to i_a
    # and then put in tube[x] in position stride(x,i_a) for x > i_a
    pos = [ sym_pos(size(t,a), x, i_a) for x in 1:size(t,a) ]
    vec(tube)[pos]
    sym_insert(a,tube,i_a) = poly[:, a] .* vec(tube)[1:i_a]' 
    return Ops(sym_dims, sym_var, sym_insert, "Symmetric Operators")
end;

"""
    Create a symmetric chisel with selected engaged axes.
"""
function SymmetricChisel(category::Array{Engagement})
    mat = zeros(Number, 1, length(category))
    for (idx, a) in enumerate(category)
        if a != Disengaged
            mat[1,idx] = 1
        end
    end
    # Each engaged operator is a square matrix

    return LinearChisel(category, mat, SymmetricOps(category, mat))
end;

"""
    Create a universal chisel with all specified primal and dual axes.
    Note that if an axis is specified in both primal and dual lists,
    then it recieves teh primal designation. The primal and dual values 
    must be in the range 1 to valence.
"""
function SymmetricChisel(valence::Integer, primals::Vector{<:Integer}, duals::Vector{<:Integer})
    cat = fill(Disengaged, valence)
    for a in duals
        cat[a] = Dual
    end
    for a in primals
        cat[a] = Primal
    end
    return SymmetricChisel(cat)
end;

"""
    Create a universal chisel with all axes engaged.
"""
function SymmetricChisel(valence::Integer)
    cat = fill(Primal, valence)
    return SymmetricChisel(cat)
end;

#-------------------------------Spall Construction-----------------------------

# Extract the tubes along engaged axes
extractAxisTubes(t::AbstractArray, index::Tuple, engaged::Vector{<:Integer}) = 
        [t[ntuple(i -> i == a ? Colon() : index[i], ndims(t))...] for a in engaged]


        """
Constructs a chisel constraint equation (the "spall") as specified by 
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
function spall(chisel::LinearChisel, 
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
        offset = 0
        for (idx, a) in enumerate(engaged)
            # Extract the tube for this axis
            tube = tubes[idx]
            # Determine the variable position
            inset = Ω.toVar(a, t, is[a])
            insert = Ω.insert(a, tube, is[a])
            col_start = offset + inset
            col_end = offset + inset + size(insert,2) - 1
            # if col_end > nvars
            #     println("ERROR: axis=$a, is[a]=$(is[a]), offset=$offset, inset=$inset, size(insert,2)=$(size(insert,2)), nvars=$nvars")
            #     println("Trying to access columns $col_start:$col_end")
            #     error("Column index out of bounds")
            # end
            view(M, (row_offset+1):(row_offset+neqns), col_start:col_end) .= insert
            # advance the offset
            offset += Ω.dimFormula(a, t)
        end
    end
    
    return M
end;

function spall(chisel::LinearChisel, t::AbstractArray, is::Tuple) 
    return spall(chisel, t, [is])
end;

function spall(chisel::LinearChisel, t::AbstractArray)
    all_is = vec([Tuple(is) for is in CartesianIndices(t)])
    return spall(chisel, t, all_is)
end;

# t = reshapce(collect(1:24), (2,3,4))
# M = reduce(vcat, [ spall(uc, t, Tuple(is)) for is in CartesianIndices(t) ])

## Hack to top printing message upon loading
;
# println("Chisels.jl loaded.")