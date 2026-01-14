#
# Strata Dleto: Local Operators Abstract
#   Creation and application of local operators for tensors.
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
    LocalOperators

    TO BE WRITTEN
"""

"""
    Operators(local)

    A space of operators acting on each axis
    has functions
        localDim            -- gives domenstion of the representation
        closedUnderStar     -- is star defined
        coordinates : Matrix -> Vector or nothing
        embed       : Vector -> Matrix or Nothing
        generate_random     -- generates a random vector which is valid
        trivial             -- generates vactor which represents trivial element

        most of these function have unsafe varaints which skip checks

    optionally can be closed under an some invloution star represented by a function
        star        : Vector -> Vector

    Properties: 
        coordinates ∘ embed = Identity or Nothing
        embed ∘ coordinates = Identity or Nothing
"""
abstract type Operator  end


"""
    Return the native encoding of an operator in the transverse set,
    or `Nothing` if it is not a member.
"""
function coordinates(Op::Operator, M::AbstractMatrix) ::Union{Vector{<:Number}, Nothing} 
    @assert false "Calling Placeholder Abstract Function"
end;
function unsafe_coordinates(Op::Operator, M::AbstractMatrix) ::Vector{<: Number} 
    @assert false "Calling Placeholder Abstract Function"
end;
coordinates(Op::Operator, ::Nothing)= nothing;
unsafe_coordinates(Op::Operator, ::Nothing)= nothing;


"""
    Convert the native encoding of an operator into matrix.
"""
function embed(Op::Operator, dim::Integer, data::Vector{<:Number} ) ::Union{AbstractMatrix,Nothing}  
    @assert dim > 0 "Dimension needs to be positive"
    @assert length(data)== localDim(Op,dim) "Incompatable Data"
    unsafe_embed(Op,dim,data)
end;

function unsafe_embed(Op::Operator, dim::Integer, data::Vector{<:Number} ) ::AbstractMatrix  
    @assert false "Calling Placeholder Abstract Function"
end;
embed(Op::Operator, dim::Integer, ::Nothing)= nothing;
unsafe_embed(Op::Operator, dim::Integer, ::Nothing)= nothing;


function localDim(Op::Operator, dim::Integer)::Integer 
    @assert dim > 0 "Dimensio needs to be positive"
    @assert false "Calling Placeholder Abstract Function"
end;

"""
Here Star can be essenitally any invloutions
in the Linear case it is transpose (or may be transpose conjugate)
and in the inverible case usually the inverse
"""
function closedUnderStar(Op::Operator)::Bool  
    @assert false "Calling Placeholder Abstract Function"
end;


function star(Op::Operator, dim::Integer, data::Vector{<:Number} ) ::Union{Vector{<:Number},Nothing}
    @assert length(data)== localDim(Op,dim) "Incompatable Data"
    unsafe_star(Op,dim,data)
end;

function unsafe_star(Op::Operator, dim::Integer, data::Vector{<:Number} ) ::Vector{<:Number}
    if closedUnderStar(Op)  
        @assert false "Calling Placeholder Abstract Function"
    else 
        @assert false "Star is not defined"
    end
end;

function generate_random(Op::Operator, dim::Integer)::Vector{<:Number} 
    @assert false "Calling Placeholder Abstract Function"
end;

function trivial(Op::Operator, dim::Integer)::Vector{<:Number} 
    @assert false "Calling Placeholder Abstract Function"
end;

# create operator from symbol -- used to avoid exporting all types of operators
function Operator(s::Symbol)::Operator
    if haskey(LinearOperatorsDict, s)
        return LinearOperatorsDict[s]
    elseif haskey(InvertableOperatorsDict, s)
        return InvertableOperatorsDict[s]
    else
        @assert false "Unkown operator"
    end
end



"""
    Linear Operators(local)

    A space of linear operators acting on each axis
    inherit all functions from operators and also provides
        containScalars        : Bool, weather it contains scalars   
        transposeEmbed       : Matrix -> Vector
    Properties: 
        embed is defined everywhere
        embed and transposeEmbed are transpose to each other
        all maps are linear
"""
abstract type LinearOperator <: Operator  end


function transposeEmbed(Op::LinearOperator, M::AbstractMatrix) :: Vector{<:Number}
    @assert size(M)[1]==size(M)[2] "Incompatable Data"
    return unsafe_transposeEmbed(Op, M)  
end;

function unsafe_transposeEmbed(Op::LinearOperator, M::AbstractMatrix) :: Vector{<:Number} 
    @assert false "Calling Placeholder Abstract Function"
end;
# this is the transpose of the embed map




function containScalars(Op::LinearOperator)::Bool  
    @assert false "Calling Placeholder Abstract Function"
end;


#generic Star for Linear operatros coming from transpose
#for each type it is faster to implement a separate method
function unsafe_star(Op::LinearOperator, dim::Integer, data::Vector{<:Number} ) ::Vector{<:Number}
    if closedUnderStar(Op)  
        unsafe_coordinates(Op,Matrix(LinearAlgebra.transpose(unsafe_embed(Op,dim,data))))
    else 
        @assert false "Not defined"
    end
end;

generate_random(Op::LinearOperator, dim::Integer)::Vector{<:Number} = randn(localDim(Op,dim)); 

trivial(Op::LinearOperator, dim::Integer)::Vector{<:Number} = zeros(localDim(Op,dim));

# create linear operator from symbol -- used to avoid exporting all types of operators
function LinearOperator(s::Symbol)::Operator
    if haskey(LinearOperatorsDict, s)
        return LinearOperatorsDict[s]
    else
        @assert false "Unkown operator"
    end
end



"""
    Dictionary of implemented local linear operators
"""
LinearOperatorsDict = Dict{Symbol, Operator}()





"""
    Invertible Operators(local)

    A space of invertible linear operators acting on each axis
"""
abstract type InvertableOperator <: Operator  end

function embed(Op::InvertableOperator, dim::Integer, data::Vector{<:Number} ) ::Union{AbstractMatrix,Nothing}  
    @assert dim > 0 "Dimension needs to be positive"
    @assert length(data)== localDim(Op,dim) "Incompatable Data"
    M = unsafe_embed(Op,dim,data)
    if !isapprox(LinearAlgebra.det(M),0)
        return M
    else
        return nothing
    end
end;

trivial(Op::InvertableOperator, dim::Integer)::Vector{<:Number} = coordinates(Op, LinearAlgebra.Diagonal([1.0 for i=1:dim]));

#generic Star for invertible operatros coming from inverse transpose (it must ne homomorphism!)
#for each type it is faster to implement a separate method
function unsafe_star(Op::InvertableOperator, dim::Integer, data::Vector{<:Number} ) ::Vector{<:Number}
    if closedUnderStar(Op)  
        return unsafe_coordinates(Op,LinearAlgebra.transpose(inv(unsafe_embed(Op,dim,data))))
    else 
        @assert false "Not defined"
    end
end;


# create invertible operator from symbol -- used to avoid exporting all types of operators
function InvertableOperator(s::Symbol)::Operator
    if haskey(InvertableOperatorsDict, s)
        return InvertableOperatorsDict[s]
    else
        @assert false "Unkown operator"
    end
end




"""
    Dictionary of implemented local linear operators
"""
InvertableOperatorsDict = Dict{Symbol, Operator}()


"""
    Tell us how to simplfy generic linear operator
    for example SymmetricOp simplfy to DiagonalOp via OthogonalOp
        It takes a matrix M into pair of matrices T D such that
        M = T D T^{-1} 

        need to be compaitble with star like 
        if M -> (D,T)  then M* -> (D*,T*)  
"""
function simplifyTo(Op::LinearOperator)::NamedTuple{(:D, :T), Tuple(LinearOperator,InvertableOperator)}  
    @assert false "Calling Placeholder Abstract Function"
end;

"""
Function which does the simplification (ie diagonalization of symetric matrices)
"""
function simplify(Op::LinearOperator,dim::Integer, data::Vector{<:Number} )::NamedTuple{(:d, :t), Tuple(Vector{<:Number},Vector{<:Number}) } 
    @assert false "Calling Placeholder Abstract Function"
end;





"""
    Dictionary of implemented local operators
"""
OperatorsDict = Dict{Symbol,Dict}(
    :linear => LinearOperatorsDict,
    :invertible => InvertableOperatorsDict
)
