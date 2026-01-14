using Dleto
#
# TestTensorSynthesis
#

function testRandomize()
    for _ in 1:10 
        Γ = randn(rand(10:100),rand(10:100),rand(10:100))
        Σ, X = randomize(Γ)
        @assert  isapprox(Σ,Γ*X) "Randomize function failed to perform correct basis change"
    end
    return true
end

function testSurfaceTensor()
    for _ in 1:10 
        xes = randn(rand(10:50))
        yes = randn(rand(10:50))
        zes = randn(rand(10:50))
        error = 0.1
        Γ = randSurfaceTensor(xes, yes, zes, error)
        d = distSurfaceTensor(Γ, xes, yes, zes)
        @assert abs(d) < 2*error "Surface tensor distillation failed to preserve marginal sums"
    end
    return true
end
## TBD do some other tests here of surface etc.