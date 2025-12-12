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
    Engagement

An enum-like type representing the role of an axis in a chisel:
- `Primal`: axis is engaged in primal mode
- `Dual`: axis is engaged in dual mode, an approprriate transpose is applied 
- `Ambidextrous`: axis is engaged in both primal and dual modes
- `Disengaged`: axis is not engaged so constraints are dropped or applied as identity on these axes
"""
@enum Engagement begin
    Primal
    Dual
    Ambidextrous
    Disengaged
end

function Base.show(io::IO, ::MIME"text/plain", engagement::Array{Engagement})
    for e in engagement
        if e == Primal
            print(io, "→")
        elseif e == Dual
            print(io, "←")
        elseif e == Ambidextrous
            print(io, "↔")
        else
            print(io, "·" )
        end
    end
end
Base.show(io::IO, e::Engagement) = Base.show(io, MIME("text/plain"), [e])


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
    dimFormula(a,t) = a in engaged ? size(t,a)*size(t,a) : 0
    # Flatten and strided indexing
    toVar(a,t,i) = 1+(i-1) * size(t,a)
    insert(a,tube) = poly[:, a] .* tube'
    return Ops(dimFormula, toVar, insert, "Universal Operators")
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
"""
    Create a universal operators with selected engaged axes.
"""
function SymmetricOps(engaged::Vector{<:Integer}, poly::Array{T,2}) where T
    dimFormula(a,t) = a in engaged ? size(t,a)*(size(t,a)+1) ÷ 2 : 0
    # Flatten and strided indexing

    toVar(a,t,i) = 1+(i-1) * size(t,a)
    insert(a,tube) = poly[:, a] .* tube'
    return Ops(dimFormula, toVar, insert)
end;

"""
    Create an orthogonal chisel with selected engaged axes.
"""
function OrthogonalChisel(valence::Integer, engaged:: Vector{<:Integer})
    mat = ones(Float16, 1, valence)
    # Each engaged operator is a square matrix
    dimFormula(a,t) = a in engaged ? size(t,a)*(size(t,a)+1) ÷ 2 : 0
    # Flatten and strided indexing
    toVar(a,t,i) = size(t,a)*(size(t,a)+1) ÷ 2 - (size(t,a)-i+1)*(size(t,a)-i+2) ÷ 2 + 1
    insert(a,tube) = mat[:, a] .* tube'
    return LinearChisel{Float16}(mat, engaged, dimFormula, toVar, insert)
end;

"""
    Create an orthogonal chisel with selected engaged axes.
"""
function OrthogonalChisel(valence::Integer)
    return OrthogonalChisel(valence, collect(1:valence))
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
    - `is`: the indices for each axis

    Returns a (sparse) vector representing the equation:

    0 = Σ_a Σ_{l_a} λ_{sa} t[i_1,... , l_a, ...,t_val] * X[i_a, l_a]

where the sum is over all axes (modes) a and l_a runs over the dimension of the 
a-th axis.  The primal form assume t is given as input and solves for the matrices X_a.
The dual form assumes the matrices X_a are given and solves for the tensor t.
"""
function spall(chisel::LinearChisel, 
    t::AbstractArray, 
    is::Tuple) 

    cat = chisel.category
    Ω = chisel.operators
    engaged = findall(e -> e != Disengaged, cat)
    # find the number of variables from the chisel and tensor size
    nvars = sum(a -> Ω.dimFormula(a, t), engaged)
    neqns = size(chisel.polynomials, 1)

    # initialize the equation matrix
    M = zeros(eltype(t), ( neqns, nvars ) )

    # Fetch the terms in the tensor we need in the equation.
    tubes = extractAxisTubes(t, Tuple(is), engaged)
    # Assemble the equation 
    offset = 0
    for (idx, a) in enumerate(engaged)
        # Extract the tube for this axis
        tube = tubes[idx]
        # Determine the variable position.
        inset = Ω.toVar(a,t,is[a]) 
        insert = Ω.insert(a,tube)
        view(M, :, (offset+inset):(offset+inset+size(insert,2)-1)) .= insert
        # advance the offset
        offset += Ω.dimFormula(a, t)
    end
    return M
end;

"""
Constructs multiple chisel constraint equations (spalls) for a list of index tuples.

    - the `chisel`
    - the tensor `t`
    - `many_is`: a collection of index tuples

Returns a (sparse) matrix where each group of `neqns` rows corresponds to the
spall equations for each index tuple in `many_is`.
"""
function spall(chisel::LinearChisel, 
    t::AbstractArray, 
    many_is::AbstractVector{<:Tuple})
    
    cat = chisel.category
    Ω = chisel.operators
    engaged = findall(e -> e != Disengaged, cat)
    
    # find the number of variables from the chisel and tensor size
    nvars = sum(a -> Ω.dimFormula(a, t), engaged)
    neqns = size(chisel.polynomials, 1)
    
    # initialize the equation matrix for all indices
    M = zeros(eltype(t), (neqns * length(many_is), nvars))
    
    # Process each index tuple
    for (eq_idx, is) in enumerate(many_is)
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
            insert = Ω.insert(a, tube)
            view(M, (row_offset+1):(row_offset+neqns), (offset+inset):(offset+inset+size(insert,2)-1)) .= insert
            # advance the offset
            offset += Ω.dimFormula(a, t)
        end
    end
    
    return M
end;

function fullSpall(chisel::LinearChisel, t::AbstractArray)
    all_is = vec([ Tuple(is) for is in CartesianIndices(t) ])
    return spall(chisel, t, all_is)
end;

# t = reshapce(collect(1:24), (2,3,4))
# M = reduce(vcat, [ spall(uc, t, Tuple(is)) for is in CartesianIndices(t) ])

## Hack to top printing message upon loading
;
# println("Chisels.jl loaded.")