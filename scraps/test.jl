
using ITensors
using LinearAlgebra: Symmetric

function act(ω::Ω, Γ::ITensor, a::Index{N})::ITensor where {Ω, N}
    @assert false "Calling Placeholder Abstract Function"
end

function act(ω::Ω, Γ::ITensor, a::Index{N})::ITensor where {Ω<:Vector, N}
    println("vector here")
    return Γ  # Return the input tensor for now
end
function act(ω::Ω, Γ::ITensor, a::Index{N})::ITensor where {Ω<:Symmetric, N}
    println("symmetric here")
    return Γ  # Return the input tensor for now
end

# testing
x = Index(3,"x")
act([1,2,3], ITensor(x), x)
act(Symmetric(rand(3,3)), ITensor(x), x)