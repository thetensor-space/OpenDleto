#
# TestTensorSynthesis
#

function testRandomize()
    for _ in 1:10 
        Γ = randn(rand(10:100),rand(10:100),rand(10:100))
        Σ, X = randomize(Γ)
        @assert  isapprox(Σ,Γ*X) "Randomize function failed to perform correct basis change"
    end
end

## TBD do some other tests here of surface etc.