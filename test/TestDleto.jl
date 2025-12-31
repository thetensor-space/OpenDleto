# 
# TestDleto.jl
#

using Dleto
using ITensors

# Do some very basic tensor contraction yoga to confirm no confusing of frames

function testMultiplication()
    passing = true
    Γ = reshape( 1:8, (2,2,2))

    # If X's are ITensors then promote AbstractArray Γ to ITensor
    x = Index(2,"x"); y = Index(2,"y"); z = Index(2,"z");
    X = ITensor( [ -1.0 0.0; 0.0 1.0], x, x');
    Y = ITensor( [  1.0 0.0; 0.0 11.0], y, y');
    Z = ITensor( [  0.0 1.0; 1.0 0.0], z, z');
    Σ = Γ*[X, Y, Z]
    # @assert asarray(Σ) == reshape( [-5.0 6.0 -77.0 88.0 -1.0 2.0 -33.0 44.0], (2,2,2)) "Failed ITensor promotion multiplication test"
    if Γ*[X, Y, Z] != [X, Y, Z]*Γ 
        println("Index frame mismatch in ambidextrous multiplication")
        passing = false
    end

    return passing
end

function testRandomization()
    passing = true
    Γ = reshape( 1:8, (2,2,2))
    
    Ξ, Xs = randomize_tensor(Γ)
    # Check that Γ * Xs == Ξ
    if !isapprox(Γ * Xs, Ξ)
        println("Randomization test failed: Γ * Xs != Ξ")
        passing = false
    end
    return passing
end