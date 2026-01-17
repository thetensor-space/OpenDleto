simplifyTo(::UniversalOp)=(D=TriDiagonalOp(),T=InvertableOp());
# simplifyTo(::UniversalOp)=(D=UniversalOp(),T=InvertableOp());
simplifyTo(::DiagonalOp)=(D=DiagonalOp(),T=InvertableOp());
simplifyTo(::TriDiagonalOp)=(D=TriDiagonalOp(),T=InvertableOp());
simplifyTo(::SymmetricOp)=(D=DiagonalOp(),T=OrthogonalOp());
simplifyTo(::AntiSymmetricOp)=(D=TriDiagonalOp(),T=InvertableOp());
simplifyTo(::ScalarOp)=(D=ScalarOp(),T=OnlyIdOp());
simplifyTo(::EmptyOp)=(D=EmptyOp(),T=OnlyIdOp());



# function simplfy(Op::Union{UniversalOp,AntiSymmetricOp},dim::Integer, data::Vector{<:Number} )::Tuple(Vector{<:Number},Vector{<:Number})  
function simplfy(Op::Union{UniversalOp,AntiSymmetricOp,TriDiagonalOp},dim::Integer, data::Vector{<:Number} )::NamedTuple{(:d, :t), Tuple{Vector{<:Number},Vector{<:Number}} }  
    M = embed(Op, dim, data)
    res = realCanonicalForm(M) 
    return (
        d= unsafe_coordinates(TriDiagonalOp(),res.D),    
        # d= unsafe_coordinates(UniversalOp(),res.D),    
        t= unsafe_coordinates(InvertableOp(),res.T)
    )    
end;


function simplfy(Op::Union{DiagonalOp,SymmetricOp},dim::Integer, data::Vector{<:Number} )::NamedTuple{(:d, :t), Tuple{Vector{<:Number},Vector{<:Number}} }  
# function simplfy(Op::Union{DiagonalOp,SymmetricOp},dim::Integer, data::Vector{<:Number} )::Tuple(Vector{<:Number},Vector{<:Number})  
    M = embed(Op, dim, data)
    eig = LinearAlgebra.eigen(M)
    return (d= eig.values, t= unsafe_coordinates(OrthogonalOp(), eig.vectors) )
end;

# function simplfy(Op::Union{ScalarOp,EmptyOp},dim::Integer, data::Vector{<:Number} )::Tuple(Vector{<:Number},Vector{<:Number})  
#     return (data, zeros(0) )
# end;
function simplfy(Op::Union{ScalarOp,EmptyOp},dim::Integer, data::Vector{<:Number} )::NamedTuple{(:d, :t), Tuple{Vector{<:Number},Vector{<:Number}} } 
    return (d=data, t=zeros(0) )
end;



# to be moved into Utils.jl
"""
    Real canonical form of a matrix
    Group conjugate pairs of complex eigenvalues and eigenvectors into real blocks.

    Returns a named tuple with:
    - `D`: A block diagonal matrix with real blocks.
    - `T`: A matrix whose columns are the real eigenvectors or the real 
    and imaginary parts of complex conjugate pairs.

    LawA
    ```julia
    res = realCanonicalForm(M); isapprox(M * res.T, res.T * res.D)
    ```
"""
function realCanonicalForm( M ::AbstractMatrix; tol::Float64=1e-10):: NamedTuple{(:D, :T), Tuple{AbstractMatrix,AbstractMatrix}}
### this function need to be rewritten since right now it assumes that complex conjugate eigenvectors aprear in pairs next to each other
### i added a hack to gurantee this for most random matrices from reasonable classes but it might fail sometime
    @assert size(M,1)==size(M,2) "Matrix must be square"
    if all( M .|> (x -> abs(x) < tol) )
        # M is zero matrix, return identityh transformation
        return (;D=zeros(eltype(M),size(M)), T = LinearAlgebra.Diagonal([ 1.0 for i = 1:size(M,1)]) )
    end
    ## hack to make get ordering of the eigenvaluescorrect
    eig = LinearAlgebra.eigen(M,sortby = z -> real(z) + imag(z)*imag(z)*0.001)
    evalues = eig.values
    evec = eig.vectors
    if isa(M, LinearAlgebra.Symmetric)              # no need to do anything if the matrix is symmetric
        return (; D= LinearAlgebra.Diagonal(evalues), T=evec) 
    end 
    found_complex=false
    n  = real.(evec)
    nn = real.(evec)
    D = zeros(eltype(n),size(M))
    D[1,1] = real(evalues[1])
    for i = 2: size(M,2)
        if ((((n[:,i] - n[:,i-1]) .|> x -> x*x) |> sum) > tol)
            # nn[:,i] =n[:,i]
            D[i,i] = real(evalues[i])
        else 
            nn[:,i] = imag.(evec[:,i])
            D[i,i] = real(evalues[i])
            D[i,i-1] = -imag(evalues[i])
            D[i-1,i] = imag(evalues[i])
            found_complex=true
        end
    end
    return (;D = found_complex ? D : LinearAlgebra.Diagonal(real.(evalues)) , T = nn)
end
