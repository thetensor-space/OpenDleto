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

function testDegeneracy()
    passing = true
    for val in 2:10
        dmax = round(Int, 1000^(1/val))
        es = Tuple(rand(5:dmax) for _ in 1:val)
        ds = Tuple(es[i] + rand(0:10) for i in 1:val)
        Δ = zeros(Float64, ds);  
        Δ[UnitRange.(1, es)...] .= randn(es...)
        Δ_rand, Xs = randomize_tensor(Δ);  
        @assert isapprox(Δ * Xs, Δ_rand) "Randomization test failed: Δ * Xs != Δ_rand"
        Δ_nondeg, Ys = nondeg(Δ_rand)
        @assert isapprox(Δ_rand * Ys, Δ_nondeg) "Nondegeneracy test failed: Δ_nondeg * Y2s != Δ"
        # check dims.
        for a in 1:ndims(Δ)
            if size(Δ_nondeg, a) != es[a]
                println("Nondegeneracy test failed: axis $a size mismatch $(size(Δ_nondeg, a)) != $(es[a])")
                passing = false
                return Δ_rand, Δ
            end
        end
    end
    return passing
end